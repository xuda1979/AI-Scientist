"""
PromptOptimizer — dynamically selects the best prompt for each research stage.

Strategy:
1. **Meta-prompting**: Ask the LLM to generate N candidate prompts for the task.
2. **Scoring**: Ask the LLM to rate each candidate (or use a lightweight eval).
3. **Selection**: Pick the highest-scoring prompt and execute it.

This replaces hard-coded prompt templates with adaptive, stage-aware prompts
that account for the current state of the paper and project files.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from autopilot.config import AutoPilotConfig, StageConfig

logger = logging.getLogger(__name__)


# ── Prompt templates for each research stage ────────────────────────────

STAGE_DESCRIPTIONS: Dict[str, str] = {
    "literature_review": (
        "Conduct a thorough literature review relevant to the research topic. "
        "Identify key papers, summarise findings, and note gaps."
    ),
    "research_gap_analysis": (
        "Analyse the literature review to pinpoint specific research gaps, "
        "open questions, and opportunities for novel contributions."
    ),
    "hypothesis_formation": (
        "Formulate clear, testable hypotheses (or research questions) based "
        "on the identified gaps, explaining the rationale."
    ),
    "methodology_design": (
        "Design the research methodology — experiments, datasets, models, "
        "evaluation metrics — that will test the hypotheses."
    ),
    "experiment_code_generation": (
        "Write working simulation / experiment code (Python) that implements "
        "the methodology. The code must be self-contained and runnable."
    ),
    "experiment_execution": (
        "Execute the experiment code, capture outputs, and generate "
        "result files (CSV, plots, tables)."
    ),
    "results_analysis": (
        "Analyse experiment results: compute statistics, draw conclusions, "
        "compare with baselines, and identify limitations."
    ),
    "initial_draft": (
        "Write a complete first draft of the research paper in LaTeX, "
        "including all standard sections (abstract, intro, related work, "
        "method, experiments, results, discussion, conclusion)."
    ),
    "section_deep_dive": (
        "Deepen one or more sections that are thin: add detail, data, "
        "equations, figures, or discussion. Do NOT shorten other sections."
    ),
    "internal_review": (
        "Conduct a rigorous internal peer review of the paper. "
        "Produce a structured review covering novelty, soundness, clarity, "
        "and completeness with specific actionable suggestions."
    ),
    "revision": (
        "Revise the paper to address every issue from the internal review. "
        "Apply all suggestions without removing existing good content."
    ),
    "bibliography_polish": (
        "Validate and polish the bibliography: ensure all citations are "
        "real, properly formatted, and that every \\cite{} has a matching "
        "\\bibitem or BibTeX entry."
    ),
    "final_compilation": (
        "Perform a final consistency check and produce the publication-ready "
        "LaTeX paper. Fix any remaining compilation issues."
    ),
}


class PromptOptimizer:
    """
    Generates, scores, and selects the best prompt for a given research stage.
    """

    def __init__(
        self,
        config: AutoPilotConfig,
        llm_fn: Callable[..., str],
    ):
        """
        Args:
            config: Global auto-pilot configuration.
            llm_fn: A callable with signature (messages, model, prompt_type, ...) -> str
                    that wraps the existing _universal_chat.
        """
        self.config = config
        self.llm_fn = llm_fn
        # History of what worked well (stage → list of (prompt, score))
        self._history: Dict[str, List[Tuple[str, float]]] = {}

    # ── Public API ───────────────────────────────────────────────────────

    def best_prompt(
        self,
        stage: str,
        workspace_snapshot: str,
        extra_context: str = "",
    ) -> str:
        """
        Return the best prompt to send to the LLM for *stage*.

        1. Generate N candidate prompts via meta-prompting.
        2. Score each candidate.
        3. Return the winner.
        """
        stage_cfg = self.config.get_stage_config(stage)
        n = self.config.prompt_candidates_per_stage

        if not self.config.use_meta_prompt or n <= 1:
            # Single deterministic prompt — no meta-prompting
            return self._static_prompt(stage, workspace_snapshot, extra_context)

        # ── Step 1: Generate candidate prompts ──────────────────────────
        candidates = self._generate_candidates(stage, workspace_snapshot, extra_context, n)

        if len(candidates) <= 1:
            return candidates[0] if candidates else self._static_prompt(stage, workspace_snapshot, extra_context)

        # ── Step 2: Score candidates ────────────────────────────────────
        scored = self._score_candidates(stage, candidates, workspace_snapshot)

        # ── Step 3: Pick best ───────────────────────────────────────────
        scored.sort(key=lambda x: x[1], reverse=True)
        best_prompt, best_score = scored[0]

        # Remember for future reference
        self._history.setdefault(stage, []).append((best_prompt[:200], best_score))

        logger.info(
            "PromptOptimizer | stage=%s | candidates=%d | best_score=%.2f",
            stage, len(candidates), best_score,
        )
        return best_prompt

    # ── Internal: static fallback ────────────────────────────────────────

    def _static_prompt(self, stage: str, snapshot: str, extra: str) -> str:
        """Build a deterministic prompt from templates."""
        desc = STAGE_DESCRIPTIONS.get(stage, f"Perform the '{stage}' stage of the research.")
        history_hint = ""
        if stage in self._history and self._history[stage]:
            best_past = max(self._history[stage], key=lambda x: x[1])
            history_hint = f"\n\nPreviously the highest-scoring approach was: {best_past[0]}"

        return (
            f"## Task: {stage.replace('_', ' ').title()}\n\n"
            f"{desc}\n\n"
            f"### Research Context\n"
            f"- Topic: {self.config.topic}\n"
            f"- Field: {self.config.field_name}\n"
            f"- Research Question: {self.config.research_question}\n\n"
            f"### Current Project Files\n{snapshot}\n\n"
            f"{f'### Additional Context{chr(10)}{extra}' if extra else ''}"
            f"{history_hint}\n\n"
            f"### Output Format\n"
            f"Produce your output, then list ALL file changes using this format:\n\n"
            f"```file_actions\n"
            f'[{{"action": "write", "path": "filename.ext", "content": "...full content..."}}]\n'
            f"```\n"
        )

    # ── Internal: meta-prompt generation ─────────────────────────────────

    def _generate_candidates(
        self, stage: str, snapshot: str, extra: str, n: int
    ) -> List[str]:
        """Ask the LLM to generate N distinct prompt strategies for this stage."""
        desc = STAGE_DESCRIPTIONS.get(stage, stage)
        history_summary = ""
        if stage in self._history and self._history[stage]:
            top3 = sorted(self._history[stage], key=lambda x: x[1], reverse=True)[:3]
            history_summary = "\n".join(
                f"  - (score {s:.2f}) {p}" for p, s in top3
            )
            history_summary = f"\nPrevious successful prompts:\n{history_summary}"

        meta_prompt = (
            f"You are an expert prompt engineer for AI research agents.\n\n"
            f"The agent is about to perform the **{stage.replace('_', ' ')}** stage "
            f"of writing a research paper.\n\n"
            f"Stage description: {desc}\n"
            f"Research topic: {self.config.topic}\n"
            f"Field: {self.config.field_name}\n"
            f"Research question: {self.config.research_question}\n"
            f"{history_summary}\n\n"
            f"Current project file tree:\n"
            f"{self._file_tree_from_snapshot(snapshot)}\n\n"
            f"Generate exactly {n} DIFFERENT prompt strategies for this stage. "
            f"Each prompt should be a complete, self-contained instruction that "
            f"tells the LLM exactly what to do, what to focus on, and how to "
            f"format its output.\n\n"
            f"Return them as a JSON array of strings:\n"
            f'```json\n["prompt 1...", "prompt 2...", "prompt 3..."]\n```'
        )

        try:
            response = self.llm_fn(
                messages=[{"role": "user", "content": meta_prompt}],
                model=self.config.model,
                prompt_type="general",
                request_timeout=120,
            )
            return self._parse_candidate_list(response, n, stage, snapshot, extra)
        except Exception as exc:
            logger.warning("Meta-prompt generation failed: %s — falling back", exc)
            return [self._static_prompt(stage, snapshot, extra)]

    def _parse_candidate_list(
        self, response: str, n: int, stage: str, snapshot: str, extra: str
    ) -> List[str]:
        """Extract a JSON list of prompt strings from the LLM response."""
        # Try JSON block
        m = re.search(r"```(?:json)?\s*\n(\[.*?\])\s*```", response, re.DOTALL)
        if m:
            try:
                items = json.loads(m.group(1))
                if isinstance(items, list) and all(isinstance(x, str) for x in items):
                    # Augment each candidate with the file-action output format
                    suffix = (
                        "\n\n### Output Format\n"
                        "Produce your output, then list ALL file changes:\n\n"
                        "```file_actions\n"
                        '[{"action": "write", "path": "filename.ext", "content": "..."}]\n'
                        "```\n"
                    )
                    return [p + suffix for p in items[:n]]
            except json.JSONDecodeError:
                pass
        # Fallback
        return [self._static_prompt(stage, snapshot, extra)]

    # ── Internal: candidate scoring ──────────────────────────────────────

    def _score_candidates(
        self, stage: str, candidates: List[str], snapshot: str
    ) -> List[Tuple[str, float]]:
        """Ask the LLM to score each candidate prompt for the given stage."""
        numbered = "\n\n".join(
            f"--- Candidate {i+1} ---\n{c[:2000]}" for i, c in enumerate(candidates)
        )
        scoring_prompt = (
            f"You are evaluating {len(candidates)} prompt strategies for the "
            f"**{stage.replace('_', ' ')}** stage of a research paper.\n\n"
            f"Research topic: {self.config.topic}\n"
            f"Field: {self.config.field_name}\n\n"
            f"{numbered}\n\n"
            f"Score each candidate from 0.0 to 1.0 on:\n"
            f"1. Specificity — does it give clear, actionable instructions?\n"
            f"2. Completeness — does it cover everything the stage needs?\n"
            f"3. Quality focus — does it push for high-quality output?\n"
            f"4. Practicality — can it actually be executed with the current project?\n\n"
            f"Return a JSON array of scores (same order as candidates):\n"
            f'```json\n[0.85, 0.72, 0.91]\n```'
        )

        try:
            response = self.llm_fn(
                messages=[{"role": "user", "content": scoring_prompt}],
                model=self.config.model,
                prompt_type="general",
                request_timeout=90,
            )
            scores = self._parse_scores(response, len(candidates))
        except Exception as exc:
            logger.warning("Candidate scoring failed: %s — using equal scores", exc)
            scores = [0.5] * len(candidates)

        return list(zip(candidates, scores))

    @staticmethod
    def _parse_scores(response: str, n: int) -> List[float]:
        """Extract a list of float scores from the LLM response."""
        m = re.search(r"```(?:json)?\s*\n(\[.*?\])\s*```", response, re.DOTALL)
        if m:
            try:
                items = json.loads(m.group(1))
                if isinstance(items, list) and len(items) >= n:
                    return [float(x) for x in items[:n]]
            except (json.JSONDecodeError, ValueError):
                pass
        # Fallback: try to find floats in the response
        floats = re.findall(r"(\d+\.\d+)", response)
        if len(floats) >= n:
            return [float(x) for x in floats[:n]]
        return [0.5] * n

    @staticmethod
    def _file_tree_from_snapshot(snapshot: str) -> str:
        """Extract just the file names from a workspace snapshot string."""
        files = re.findall(r"===\s*FILE:\s*(.+?)\s*===", snapshot)
        return "\n".join(f"  - {f}" for f in files) if files else "(no files yet)"
