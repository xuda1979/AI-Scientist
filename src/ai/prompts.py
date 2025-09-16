"""Prompt templates for SciResearch Workflow."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence


PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _template_path(name: str) -> Path:
    """Return the path to a stored template file."""
    return PROMPTS_DIR / f"{name}.txt"


@lru_cache(maxsize=None)
def _load_template(name: str) -> str:
    """Load a template from disk and cache the result."""
    path = _template_path(name)
    return path.read_text(encoding="utf-8")


def _format_quality_issues(issues: Sequence[str] | None) -> str:
    """Format quality issues for template interpolation."""
    if not issues:
        return "SUCCESS: No major quality issues detected"
    return "\n".join(f"- {issue}" for issue in issues)


def _latex_status_message(latex_errors: Optional[str]) -> str:
    """Return a human-readable LaTeX status message."""
    return latex_errors or "SUCCESS: Compilation successful"


class PromptTemplates:
    """Centralized prompt template management."""

    @staticmethod
    def initial_draft(
        topic: str,
        field: str,
        question: str,
        user_prompt: Optional[str] = None,
    ) -> str:
        """Template for initial paper draft generation."""
        additional_requirements = (
            f"\n\nAdditional Requirements: {user_prompt}" if user_prompt else ""
        )
        template = _load_template("initial_draft")
        return template.format(
            topic=topic,
            field=field,
            question=question,
            additional_requirements=additional_requirements,
        )

    @staticmethod
    def combined_ideation_draft(
        topic: str,
        field: str,
        question: str,
        num_ideas: int = 15,
        user_prompt: Optional[str] = None,
    ) -> str:
        """Template for combined ideation and draft generation."""
        template = _load_template("combined_ideation_draft")
        return template.format(
            topic=topic,
            field=field,
            question=question,
            num_ideas=num_ideas,
            user_requirements=user_prompt or "Standard academic quality",
        )

    @staticmethod
    def comprehensive_review_revision(
        current_tex: str,
        sim_summary: str,
        latex_errors: Optional[str],
        quality_issues: Sequence[str] | None,
        user_prompt: Optional[str],
        iteration: int,
        max_iterations: int,
    ) -> str:
        """Template for comprehensive review and revision."""
        template = _load_template("comprehensive_review_revision")
        return template.format(
            iteration=iteration,
            max_iterations=max_iterations,
            current_tex=current_tex,
            sim_summary=sim_summary,
            latex_status=_latex_status_message(latex_errors),
            quality_issues=_format_quality_issues(quality_issues),
            user_requirements=(
                user_prompt or "Standard academic quality requirements"
            ),
        )

    @staticmethod
    def simulation_fixing(code: str, error: str, attempt: int) -> str:
        """Template for simulation code fixing."""
        template = _load_template("simulation_fixing")
        return template.format(code=code, error=error, attempt=attempt)

    @staticmethod
    def research_ideation(
        topic: str,
        field: str,
        question: str,
        num_ideas: int = 15,
    ) -> str:
        """Template for research idea generation."""
        template = _load_template("research_ideation")
        return template.format(
            topic=topic,
            field=field,
            question=question,
            num_ideas=num_ideas,
        )
