"""
Planner — dynamic goal decomposition, stage ordering, and backtracking.

Unlike the fixed linear pipeline, the Planner can:
- Dynamically decide which stage to run next
- Skip stages that aren't needed
- Backtrack to earlier stages if results are poor
- Branch into parallel experiments
- Re-plan based on intermediate results
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.config import AgentConfig, PHASES, ALL_STAGES
from agent.memory import AgentMemory

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Stage Task Descriptions
# ═════════════════════════════════════════════════════════════════════════

STAGE_TASKS: Dict[str, str] = {
    "literature_search": (
        "Search for relevant academic papers using Semantic Scholar and arXiv. "
        "Find at least 10-15 papers covering: (a) foundational work in the area, "
        "(b) recent advances, (c) closely related approaches. "
        "For each paper, record the title, authors, year, venue, and key findings. "
        "Generate BibTeX entries for all papers found."
    ),
    "literature_synthesis": (
        "Synthesize the papers found during literature search into a coherent narrative. "
        "Identify key themes, methodological trends, and consensus points. "
        "Create a structured literature review outline."
    ),
    "gap_analysis": (
        "Based on the literature synthesis, identify specific research gaps: "
        "unexplored areas, limitations of existing approaches, open questions. "
        "Rank gaps by significance and feasibility."
    ),
    "hypothesis_formation": (
        "Formulate clear, testable hypotheses based on the identified gaps. "
        "Each hypothesis should include: statement, rationale, how it will be tested, "
        "and what outcomes would support or refute it."
    ),
    "methodology_design": (
        "Design the research methodology to test the hypotheses. "
        "Specify: experimental setup, independent/dependent variables, controls, "
        "evaluation metrics, and analysis plan."
    ),
    "experiment_planning": (
        "Create a detailed experiment plan: what code needs to be written, "
        "what data is needed, what computations to run, expected outputs. "
        "Plan figures and tables that will present the results."
    ),
    "code_generation": (
        "Write the experiment code (simulation.py) implementing the methodology. "
        "Code must be self-contained, well-documented, and produce clear output. "
        "Include proper random seeding for reproducibility."
    ),
    "code_testing": (
        "Validate the experiment code: check syntax, run with small inputs, "
        "verify output format. Fix any errors."
    ),
    "experiment_execution": (
        "Execute the full experiment code and capture all results. "
        "If execution fails, use the fix_and_retry tool to debug and fix issues."
    ),
    "results_analysis": (
        "Analyze experiment results: compute statistics, compare with baselines, "
        "test hypotheses, identify patterns. Prepare data for figures and tables."
    ),
    "outline_generation": (
        "Generate a detailed paper outline with section structure, key points "
        "for each section, and where figures/tables will go."
    ),
    "initial_draft": (
        "Write the complete first draft of the paper in LaTeX. Include all sections: "
        "abstract, introduction, related work, methodology, experiments, results, "
        "discussion, conclusion, and bibliography. Use real data from experiments."
    ),
    "figure_generation": (
        "Generate all figures for the paper using matplotlib. Each figure should be "
        "publication-quality with proper labels, legends, and captions."
    ),
    "section_deepening": (
        "Review each section and deepen thin areas. Add more detail, data, "
        "equations, analysis, or discussion where needed. Do NOT remove content."
    ),
    "internal_review": (
        "Conduct a thorough internal peer review of the paper. Evaluate novelty, "
        "soundness, clarity, and completeness. Provide specific, actionable feedback."
    ),
    "revision": (
        "Revise the paper to address ALL issues from the internal review. "
        "Improve every identified weakness while preserving existing good content."
    ),
    "bibliography_polish": (
        "Validate and polish the bibliography. Ensure every \\cite{} has a matching "
        "entry, all entries are real and properly formatted, and remove any unused entries."
    ),
    "final_compilation": (
        "Final pass: fix any LaTeX errors, ensure compilation succeeds, "
        "verify formatting, and produce the final PDF."
    ),
}


class Planner:
    """
    Dynamic research planner with adaptive stage ordering and backtracking.
    """

    def __init__(
        self,
        config: AgentConfig,
        memory: AgentMemory,
        llm_fn: Optional[Callable[..., str]] = None,
    ):
        self.config = config
        self.memory = memory
        self.llm_fn = llm_fn

    def get_initial_plan(self) -> List[str]:
        """Get the initial ordered list of stages to execute."""
        return self.config.get_active_stages()

    def get_task_prompt(self, stage: str) -> str:
        """Get the detailed task prompt for a stage."""
        task = STAGE_TASKS.get(stage, f"Perform the '{stage}' stage of the research.")
        return (
            f"## Task: {stage.replace('_', ' ').title()}\n\n"
            f"{task}\n\n"
            f"### Research Context\n"
            f"- **Topic**: {self.config.topic}\n"
            f"- **Field**: {self.config.field}\n"
            f"- **Research Question**: {self.config.research_question}\n"
        )

    def should_backtrack(self, stage: str, quality_score: float) -> Optional[str]:
        """
        Determine if we should backtrack to an earlier stage.

        Returns the stage to backtrack to, or None if we should proceed.
        """
        if quality_score >= self.config.quality_threshold:
            return None

        # Backtracking rules
        backtrack_map = {
            "experiment_execution": "code_generation",  # bad results → rewrite code
            "results_analysis": "experiment_execution",  # can't analyze → rerun
            "initial_draft": "outline_generation",       # bad draft → redo outline
            "revision": "internal_review",               # bad revision → re-review
            "final_compilation": "bibliography_polish",  # can't compile → fix bib
        }

        target = backtrack_map.get(stage)
        if target and quality_score < self.config.quality_threshold * 0.6:
            logger.info("Backtracking from %s to %s (score=%.2f)", stage, target, quality_score)
            self.memory.log_decision(
                stage, f"backtrack to {target}",
                f"Quality score {quality_score:.2f} below threshold",
            )
            return target

        return None

    def should_skip(self, stage: str) -> Tuple[bool, str]:
        """
        Determine if a stage can be skipped based on current state.

        Returns (should_skip, reason).
        """
        state = self.memory.state

        # Skip literature synthesis if no papers found
        if stage == "literature_synthesis" and not state.literature:
            return True, "No papers found yet — will search first"

        # Skip gap analysis if no literature
        if stage == "gap_analysis" and not state.literature:
            return True, "No literature to analyze gaps in"

        # Skip code testing if no code generated
        if stage == "code_testing" and not any(
            e.code_file for e in state.experiments
        ):
            return True, "No experiment code to test"

        # Skip figure generation if no experiment results
        if stage == "figure_generation" and not any(
            e.status == "completed" for e in state.experiments
        ):
            return True, "No experiment results to visualize"

        # Skip section deepening on first pass
        if stage == "section_deepening" and "initial_draft" not in state.stages_completed:
            return True, "Initial draft not yet written"

        return False, ""

    def replan(self, completed_stages: List[str], quality_scores: Dict[str, float]) -> List[str]:
        """
        Dynamically re-plan the remaining stages based on progress so far.

        Uses the LLM to make intelligent decisions about what to do next.
        """
        if not self.llm_fn:
            # Without LLM, fall back to static plan
            all_stages = self.config.get_active_stages()
            return [s for s in all_stages if s not in completed_stages]

        # Ask the LLM to re-plan
        state_summary = self.memory.build_context_for_stage("replanning")
        scores_text = "\n".join(
            f"  - {s}: {q:.2f}" for s, q in quality_scores.items()
        )

        remaining = [s for s in self.config.get_active_stages() if s not in completed_stages]

        prompt = (
            f"You are a research project planner. Based on the current progress, "
            f"decide the optimal order for the remaining stages.\n\n"
            f"## Completed Stages & Quality Scores\n{scores_text}\n\n"
            f"## Current State\n{state_summary}\n\n"
            f"## Remaining Stages\n{json.dumps(remaining)}\n\n"
            f"Return a JSON array of stage names in the optimal order. "
            f"You may also add stages that need to be re-done (e.g., if a score was low).\n\n"
            f"```json\n[\"stage1\", \"stage2\", ...]\n```"
        )

        try:
            response = self.llm_fn(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.model,
                prompt_type="general",
                request_timeout=60,
            )

            # Parse the response
            match = re.search(r"```(?:json)?\s*\n(\[.*?\])\s*```", response, re.DOTALL)
            if match:
                plan = json.loads(match.group(1))
                if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
                    # Validate stages
                    valid = [s for s in plan if s in ALL_STAGES]
                    if valid:
                        return valid

        except Exception as exc:
            logger.warning("Re-planning failed: %s — using default order", exc)

        return remaining
