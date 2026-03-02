"""
QualityGate — evaluates stage output and decides whether to proceed, retry, or abort.

Uses a combination of:
- LLM self-reflection (ask the LLM to rate its own output)
- Heuristic checks (word count, LaTeX compilability, code syntax)
- Cross-stage consistency checks
"""
from __future__ import annotations

import logging
import re
import subprocess
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QualityGate:
    """
    Evaluates the output of a research stage and returns a quality verdict.
    """

    def __init__(
        self,
        llm_fn: Callable[..., str],
        model: str,
        quality_threshold: float = 0.75,
    ):
        self.llm_fn = llm_fn
        self.model = model
        self.quality_threshold = quality_threshold

    def evaluate(
        self,
        stage: str,
        output_text: str,
        workspace_snapshot: str,
        file_actions_applied: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a stage's output.

        Returns:
            {
                "pass": bool,
                "score": float (0-1),
                "issues": [str],
                "suggestions": [str],
                "verdict": "proceed" | "retry" | "abort"
            }
        """
        results: Dict[str, Any] = {
            "pass": False,
            "score": 0.0,
            "issues": [],
            "suggestions": [],
            "verdict": "retry",
        }

        # ── 1. Heuristic checks ─────────────────────────────────────────
        heuristic_score, heuristic_issues = self._heuristic_checks(stage, workspace_snapshot)
        results["issues"].extend(heuristic_issues)

        # ── 2. LLM self-reflection ──────────────────────────────────────
        llm_score, llm_issues, llm_suggestions = self._llm_reflection(
            stage, output_text, workspace_snapshot
        )

        # ── 3. Combine scores ───────────────────────────────────────────
        combined_score = 0.4 * heuristic_score + 0.6 * llm_score
        results["score"] = round(combined_score, 3)
        results["issues"].extend(llm_issues)
        results["suggestions"].extend(llm_suggestions)

        # ── 4. Verdict ──────────────────────────────────────────────────
        if combined_score >= self.quality_threshold:
            results["pass"] = True
            results["verdict"] = "proceed"
        elif combined_score >= self.quality_threshold * 0.6:
            results["verdict"] = "retry"
        else:
            results["verdict"] = "abort"

        logger.info(
            "QualityGate | stage=%s | score=%.3f | verdict=%s | issues=%d",
            stage, combined_score, results["verdict"], len(results["issues"]),
        )
        return results

    # ── Heuristic checks ─────────────────────────────────────────────────

    def _heuristic_checks(self, stage: str, snapshot: str) -> Tuple[float, List[str]]:
        """Run fast, deterministic quality checks."""
        issues: List[str] = []
        score = 1.0

        # Check that key files exist after certain stages
        file_expectations: Dict[str, List[str]] = {
            "initial_draft": ["paper.tex"],
            "experiment_code_generation": ["simulation.py"],
            "bibliography_polish": ["paper.tex"],
            "final_compilation": ["paper.tex"],
        }

        expected_files = file_expectations.get(stage, [])
        for fname in expected_files:
            if f"FILE: {fname}" not in snapshot and f"FILE: ./{fname}" not in snapshot:
                issues.append(f"Expected file '{fname}' not found after stage '{stage}'")
                score -= 0.3

        # Check LaTeX compilability for paper-related stages
        if stage in ("initial_draft", "revision", "section_deep_dive", "final_compilation"):
            tex_match = re.search(
                r"===\s*FILE:\s*paper\.tex\s*===\s*\n(.*?)(?=\n===\s*FILE:|\Z)",
                snapshot,
                re.DOTALL,
            )
            if tex_match:
                tex_content = tex_match.group(1)
                tex_issues = self._check_latex_structure(tex_content)
                issues.extend(tex_issues)
                score -= 0.1 * len(tex_issues)

        # Check Python syntax for code stages
        if stage in ("experiment_code_generation",):
            py_match = re.search(
                r"===\s*FILE:\s*simulation\.py\s*===\s*\n(.*?)(?=\n===\s*FILE:|\Z)",
                snapshot,
                re.DOTALL,
            )
            if py_match:
                try:
                    compile(py_match.group(1), "simulation.py", "exec")
                except SyntaxError as e:
                    issues.append(f"Python syntax error in simulation.py: {e}")
                    score -= 0.4

        return max(0.0, min(1.0, score)), issues

    @staticmethod
    def _check_latex_structure(tex: str) -> List[str]:
        """Quick structural checks for LaTeX content."""
        issues = []
        if "\\begin{document}" not in tex:
            issues.append("Missing \\begin{document}")
        if "\\end{document}" not in tex:
            issues.append("Missing \\end{document}")
        if "\\title{" not in tex:
            issues.append("Missing \\title{}")
        if "\\begin{abstract}" not in tex:
            issues.append("Missing abstract")

        # Check for unmatched environments
        begins = re.findall(r"\\begin\{(\w+)\}", tex)
        ends = re.findall(r"\\end\{(\w+)\}", tex)
        begin_counts: Dict[str, int] = {}
        end_counts: Dict[str, int] = {}
        for b in begins:
            begin_counts[b] = begin_counts.get(b, 0) + 1
        for e in ends:
            end_counts[e] = end_counts.get(e, 0) + 1
        for env in set(list(begin_counts.keys()) + list(end_counts.keys())):
            bc = begin_counts.get(env, 0)
            ec = end_counts.get(env, 0)
            if bc != ec:
                issues.append(f"Unmatched LaTeX environment: {env} (begin={bc}, end={ec})")

        return issues

    # ── LLM self-reflection ──────────────────────────────────────────────

    def _llm_reflection(
        self, stage: str, output: str, snapshot: str
    ) -> Tuple[float, List[str], List[str]]:
        """Ask the LLM to self-evaluate the stage output."""
        reflection_prompt = (
            f"You are a research quality reviewer. Evaluate the output of the "
            f"**{stage.replace('_', ' ')}** stage.\n\n"
            f"## Stage Output (first 5000 chars)\n"
            f"{output[:5000]}\n\n"
            f"## Evaluation Criteria\n"
            f"Rate the output on each criterion from 0.0 to 1.0:\n"
            f"1. **Completeness**: Does it fulfil all requirements of this stage?\n"
            f"2. **Quality**: Is the content at a high academic standard?\n"
            f"3. **Correctness**: Are there factual or logical errors?\n"
            f"4. **Integration**: Does it fit with the rest of the project?\n\n"
            f"Return your evaluation as JSON:\n"
            f"```json\n"
            f'{{\n'
            f'  "scores": {{"completeness": 0.8, "quality": 0.7, "correctness": 0.9, "integration": 0.8}},\n'
            f'  "overall": 0.8,\n'
            f'  "issues": ["issue 1", "issue 2"],\n'
            f'  "suggestions": ["suggestion 1"]\n'
            f'}}\n'
            f"```"
        )

        try:
            response = self.llm_fn(
                messages=[{"role": "user", "content": reflection_prompt}],
                model=self.model,
                prompt_type="general",
                request_timeout=90,
            )
            return self._parse_reflection(response)
        except Exception as exc:
            logger.warning("LLM reflection failed: %s", exc)
            return 0.5, [], []

    @staticmethod
    def _parse_reflection(response: str) -> Tuple[float, List[str], List[str]]:
        """Parse the LLM's self-evaluation JSON."""
        m = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*```", response, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                overall = float(data.get("overall", 0.5))
                issues = data.get("issues", [])
                suggestions = data.get("suggestions", [])
                if isinstance(issues, list) and isinstance(suggestions, list):
                    return overall, issues, suggestions
            except (json.JSONDecodeError, ValueError):
                pass
        return 0.5, [], []
