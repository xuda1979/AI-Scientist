"""SciResearch Workflow - Modular Research Paper Generation System."""

from .core.workflow import run_workflow, SciResearchWorkflow
from .core.config import WorkflowConfig, DEFAULT_MODEL

__version__ = "2.0.0"
__all__ = ["run_workflow", "SciResearchWorkflow", "WorkflowConfig", "DEFAULT_MODEL"]
