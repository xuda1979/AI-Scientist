"""Core workflow components."""

from .workflow import run_workflow, SciResearchWorkflow
from .config import WorkflowConfig, DEFAULT_MODEL
from .quality import QualityAssessment
from .comprehensive_workflow import (
    ComprehensiveResearchWorkflow,
    ComprehensiveWorkflowResult,
    DataAnalysisFiguresResult,
    LanguageFormattingResult,
    ResearchPreparationResult,
    ScientificReasoningValidationResult,
    SubmissionDisseminationResult,
    WorkflowParsingError,
    WritingStageResult,
    parse_stage_payload,
)

__all__ = [
    "run_workflow",
    "SciResearchWorkflow",
    "WorkflowConfig",
    "DEFAULT_MODEL",
    "QualityAssessment",
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
