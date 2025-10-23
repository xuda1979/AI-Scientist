from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

from src.ai.chat import AIChat
from src.core.config import DEFAULT_MODEL, WorkflowConfig

logger = logging.getLogger(__name__)


class WorkflowParsingError(RuntimeError):
    """Raised when a model response cannot be parsed into JSON."""


_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_stage_payload(raw_response: str) -> Dict[str, Any]:
    """Extract and decode a JSON object embedded in a model response."""
    fenced_match = _JSON_BLOCK_PATTERN.search(raw_response)
    if fenced_match:
        candidate = fenced_match.group(1)
    else:
        bracket_match = _JSON_OBJECT_PATTERN.search(raw_response)
        if not bracket_match:
            raise WorkflowParsingError("No JSON object found in model response")
        candidate = bracket_match.group(0)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise WorkflowParsingError(f"Failed to decode JSON payload: {exc}") from exc


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _ensure_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


@dataclass
class ResearchPreparationResult:
    topics: List[str]
    literature_trends: List[str]
    hypotheses: List[str]
    innovation_points: List[str]
    literature_search_summary: str
    literature_review_sections: List[Dict[str, Any]]


@dataclass
class WritingStageResult:
    paper_outline: List[Dict[str, Any]]
    section_drafts: Dict[str, str]
    significance: str
    limitations: List[str]
    future_directions: List[str]


@dataclass
class DataAnalysisFiguresResult:
    result_descriptions: List[str]
    statistical_statements: List[str]
    figure_table_suggestions: List[Dict[str, Any]]


@dataclass
class LanguageFormattingResult:
    grammar_improvements: List[str]
    tone_recommendations: List[str]
    translation_notes: Dict[str, str]
    reference_formatting: Dict[str, Any]


@dataclass
class ScientificReasoningValidationResult:
    reviewer_questions: List[Dict[str, Any]]
    rebuttal_strategies: List[str]
    experiment_checks: List[str]
    limitations_analysis: List[str]
    improvement_suggestions: List[str]


@dataclass
class SubmissionDisseminationResult:
    target_journals: List[Dict[str, Any]]
    cover_letter_outline: List[str]
    promotion_snippets: List[str]


@dataclass
class ComprehensiveWorkflowResult:
    research_preparation: ResearchPreparationResult
    writing_stage: WritingStageResult
    data_analysis: DataAnalysisFiguresResult
    language_formatting: LanguageFormattingResult
    scientific_validation: ScientificReasoningValidationResult
    submission_dissemination: SubmissionDisseminationResult

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComprehensiveResearchWorkflow:
    """End-to-end research workflow covering preparation through dissemination."""

    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        chat: Optional[AIChat] = None,
    ) -> None:
        self.config = config or WorkflowConfig()
        self.chat = chat or AIChat(DEFAULT_MODEL, self.config.fallback_models)
        self.last_output_dir: Optional[Path] = None

    def run(
        self,
        topic: str,
        field: str,
        objective: str,
        dataset_description: Optional[str] = None,
        output_dir: Optional[Path] = None,
        save_outputs: bool = True,
    ) -> ComprehensiveWorkflowResult:
        logger.info("Starting comprehensive research workflow")
        logger.info("Topic: %s", topic)
        logger.info("Field: %s", field)
        logger.info("Objective: %s", objective)

        research_preparation = self._run_research_preparation_stage(
            topic, field, objective, dataset_description
        )
        writing_stage = self._run_writing_stage(
            topic, field, objective, research_preparation
        )
        data_analysis = self._run_data_analysis_stage(
            topic, field, objective, dataset_description, writing_stage
        )
        language_formatting = self._run_language_formatting_stage(writing_stage)
        scientific_validation = self._run_scientific_validation_stage(
            topic, field, objective, writing_stage, data_analysis
        )
        submission_dissemination = self._run_submission_stage(
            topic, field, objective, writing_stage, scientific_validation
        )

        result = ComprehensiveWorkflowResult(
            research_preparation=research_preparation,
            writing_stage=writing_stage,
            data_analysis=data_analysis,
            language_formatting=language_formatting,
            scientific_validation=scientific_validation,
            submission_dissemination=submission_dissemination,
        )

        if save_outputs:
            destination = self._prepare_output_directory(output_dir, topic)
            self._persist_outputs(destination, result)
            self.last_output_dir = destination
        else:
            self.last_output_dir = None

        return result

    def _invoke_stage(self, prompt: str, stage_name: str) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert research assistant. Respond with strictly valid JSON "
                    "that follows the requested schema. Do not use markdown fences unless "
                    "explicitly requested."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        logger.debug("Invoking stage '%s'", stage_name)
        response = self.chat.chat(messages, prompt_type=stage_name)
        payload = parse_stage_payload(response)
        logger.debug("Stage '%s' completed", stage_name)
        return payload

    def _run_research_preparation_stage(
        self,
        topic: str,
        field: str,
        objective: str,
        dataset_description: Optional[str],
    ) -> ResearchPreparationResult:
        dataset_line = (
            f"- Data context: {dataset_description}" if dataset_description else "- Data context: Not specified"
        )
        prompt = dedent(
            f"""
            Prepare the research foundation for a scientific study.
            Context:
            - Discipline: {field}
            - Core topic: {topic}
            - Research objective: {objective}
            {dataset_line}

            Provide JSON with the following keys:
            - "topics": ordered list of refined research topic statements.
            - "literature_trends": recent publication patterns or emerging areas to track.
            - "hypotheses": testable hypotheses grounded in the objective.
            - "innovation_points": ways this project can contribute novel insights.
            - "literature_search_summary": paragraph synthesizing what the literature search reveals.
            - "literature_review_sections": array where each item has "heading", "core_claims", "representative_papers" (list), and "summary".
            """
        ).strip()

        data = self._invoke_stage(prompt, "research_preparation")
        return ResearchPreparationResult(
            topics=_ensure_list(data.get("topics")),
            literature_trends=_ensure_list(data.get("literature_trends")),
            hypotheses=_ensure_list(data.get("hypotheses")),
            innovation_points=_ensure_list(data.get("innovation_points")),
            literature_search_summary=_ensure_str(data.get("literature_search_summary")),
            literature_review_sections=_ensure_list(data.get("literature_review_sections")),
        )

    def _run_writing_stage(
        self,
        topic: str,
        field: str,
        objective: str,
        research_preparation: ResearchPreparationResult,
    ) -> WritingStageResult:
        prep_summary = json.dumps(
            research_preparation.topics + research_preparation.hypotheses,
            ensure_ascii=False,
        )
        prompt = dedent(
            f"""
            Design the manuscript structure and draft critical sections for a research paper.
            Context:
            - Discipline: {field}
            - Topic: {topic}
            - Objective: {objective}
            - Preparation highlights: {prep_summary}

            Return JSON with keys:
            - "paper_outline": list where each item has "section", "purpose", and "key_points" (list).
            - "section_drafts": object mapping section names to draft paragraphs of LaTeX-ready text.
            - "significance": paragraph summarizing significance of findings.
            - "limitations": list of realistic study limitations.
            - "future_directions": list of concrete follow-up research ideas.
            """
        ).strip()

        data = self._invoke_stage(prompt, "writing_stage")
        section_drafts_raw = data.get("section_drafts") or {}
        return WritingStageResult(
            paper_outline=_ensure_list(data.get("paper_outline")),
            section_drafts={str(k): _ensure_str(v) for k, v in section_drafts_raw.items()},
            significance=_ensure_str(data.get("significance")),
            limitations=_ensure_list(data.get("limitations")),
            future_directions=_ensure_list(data.get("future_directions")),
        )

    def _run_data_analysis_stage(
        self,
        topic: str,
        field: str,
        objective: str,
        dataset_description: Optional[str],
        writing_stage: WritingStageResult,
    ) -> DataAnalysisFiguresResult:
        dataset_line = dataset_description or "No dataset description provided"
        outline_preview = json.dumps(writing_stage.paper_outline[:3], ensure_ascii=False)
        prompt = dedent(
            f"""
            Provide analysis narratives, statistical reporting, and figure/table plans for the manuscript.
            Context:
            - Discipline: {field}
            - Topic: {topic}
            - Objective: {objective}
            - Dataset: {dataset_line}
            - Outline snapshot: {outline_preview}

            Return JSON with keys:
            - "result_descriptions": list of narrative descriptions for major results.
            - "statistical_statements": list of statistical reporting statements (APA-style where applicable).
            - "figure_table_suggestions": list where each item has "type", "title", "description", and "data_requirements" (list).
            """
        ).strip()

        data = self._invoke_stage(prompt, "data_analysis")
        return DataAnalysisFiguresResult(
            result_descriptions=_ensure_list(data.get("result_descriptions")),
            statistical_statements=_ensure_list(data.get("statistical_statements")),
            figure_table_suggestions=_ensure_list(data.get("figure_table_suggestions")),
        )

    def _run_language_formatting_stage(
        self,
        writing_stage: WritingStageResult,
    ) -> LanguageFormattingResult:
        draft_sections = json.dumps(
            {k: v[:120] for k, v in writing_stage.section_drafts.items()},
            ensure_ascii=False,
        )
        prompt = dedent(
            f"""
            Analyze manuscript language quality and formatting requirements.
            Use the following section excerpts as context: {draft_sections}

            Return JSON with keys:
            - "grammar_improvements": list of grammar or clarity fixes to apply.
            - "tone_recommendations": list of ways to improve academic tone and flow.
            - "translation_notes": object with optional keys like "es", "zh", etc. mapping to guidance for academic translations.
            - "reference_formatting": object describing citation style, tools, and outstanding tasks.
            """
        ).strip()

        data = self._invoke_stage(prompt, "language_formatting")
        translation_notes_raw = data.get("translation_notes") or {}
        reference_formatting_raw = data.get("reference_formatting") or {}
        return LanguageFormattingResult(
            grammar_improvements=_ensure_list(data.get("grammar_improvements")),
            tone_recommendations=_ensure_list(data.get("tone_recommendations")),
            translation_notes={str(k): _ensure_str(v) for k, v in translation_notes_raw.items()},
            reference_formatting={str(k): v for k, v in reference_formatting_raw.items()},
        )

    def _run_scientific_validation_stage(
        self,
        topic: str,
        field: str,
        objective: str,
        writing_stage: WritingStageResult,
        data_analysis: DataAnalysisFiguresResult,
    ) -> ScientificReasoningValidationResult:
        outline_preview = json.dumps(writing_stage.paper_outline[:3], ensure_ascii=False)
        stats_preview = json.dumps(data_analysis.statistical_statements[:2], ensure_ascii=False)
        prompt = dedent(
            f"""
            Stress-test the scientific rigor of the project and prepare reviewer-ready responses.
            Context:
            - Discipline: {field}
            - Topic: {topic}
            - Objective: {objective}
            - Outline snapshot: {outline_preview}
            - Statistical claims preview: {stats_preview}

            Return JSON with keys:
            - "reviewer_questions": list where each item has "area" and "question" and optionally "severity".
            - "rebuttal_strategies": list of guidance snippets for drafting responses.
            - "experiment_checks": list of validation or reproducibility checks to perform.
            - "limitations_analysis": list detailing limitation plus mitigation ideas.
            - "improvement_suggestions": list of concrete next actions to strengthen the study.
            """
        ).strip()

        data = self._invoke_stage(prompt, "scientific_validation")
        return ScientificReasoningValidationResult(
            reviewer_questions=_ensure_list(data.get("reviewer_questions")),
            rebuttal_strategies=_ensure_list(data.get("rebuttal_strategies")),
            experiment_checks=_ensure_list(data.get("experiment_checks")),
            limitations_analysis=_ensure_list(data.get("limitations_analysis")),
            improvement_suggestions=_ensure_list(data.get("improvement_suggestions")),
        )

    def _run_submission_stage(
        self,
        topic: str,
        field: str,
        objective: str,
        writing_stage: WritingStageResult,
        scientific_validation: ScientificReasoningValidationResult,
    ) -> SubmissionDisseminationResult:
        significance = writing_stage.significance or objective
        mitigation_notes = json.dumps(
            scientific_validation.improvement_suggestions[:3],
            ensure_ascii=False,
        )
        prompt = dedent(
            f"""
            Plan the submission and dissemination strategy for the project.
            Context:
            - Discipline: {field}
            - Topic: {topic}
            - Objective: {objective}
            - Significance summary: {significance}
            - Key improvement actions: {mitigation_notes}

            Return JSON with keys:
            - "target_journals": list where each item has "name", "fit_reason", and optional "impact_factor".
            - "cover_letter_outline": ordered list of bullet points for a cover letter.
            - "promotion_snippets": list of 1-3 sentence summaries suitable for social media or press.
            """
        ).strip()

        data = self._invoke_stage(prompt, "submission_dissemination")
        return SubmissionDisseminationResult(
            target_journals=_ensure_list(data.get("target_journals")),
            cover_letter_outline=_ensure_list(data.get("cover_letter_outline")),
            promotion_snippets=_ensure_list(data.get("promotion_snippets")),
        )

    def _prepare_output_directory(self, output_dir: Optional[Path], topic: str) -> Path:
        if output_dir is not None:
            destination = Path(output_dir)
        else:
            slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-") or "research-project"
            destination = Path("output") / "comprehensive_workflow" / slug

        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def _persist_outputs(self, destination: Path, result: ComprehensiveWorkflowResult) -> None:
        json_path = destination / "workflow_plan.json"
        json_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        markdown_path = destination / "workflow_plan.md"
        markdown_path.write_text(
            self._format_markdown_summary(result),
            encoding="utf-8",
        )

        logger.info("Saved workflow outputs to %s", destination)

    def _format_markdown_summary(self, result: ComprehensiveWorkflowResult) -> str:
        data = result.to_dict()
        sections = []
        for stage_name, content in data.items():
            title = stage_name.replace("_", " ").title()
            sections.append(f"## {title}")
            sections.append(self._format_content_block(content))
        return "\n\n".join(sections).strip() + "\n"

    def _format_content_block(self, content: Any, indent: int = 0) -> str:
        prefix = " " * indent
        if isinstance(content, dict):
            lines = []
            for key, value in content.items():
                lines.append(f"{prefix}- **{key.replace('_', ' ').title()}**: {self._format_content_block(value, indent + 2).strip()}")
            return "\n".join(lines)
        if isinstance(content, list):
            if not content:
                return "[]"
            lines = []
            for item in content:
                formatted = self._format_content_block(item, indent + 2)
                lines.append(f"{prefix}- {formatted.strip()}")
            return "\n".join(lines)
        return prefix + str(content)


__all__ = [
    "ComprehensiveResearchWorkflow",
    "ComprehensiveWorkflowResult",
    "DataAnalysisFiguresResult",
    "LanguageFormattingResult",
    "ResearchPreparationResult",
    "ScientificReasoningValidationResult",
    "SubmissionDisseminationResult",
    "WorkflowParsingError",
    "WritingStageResult",
    "parse_stage_payload",
]
