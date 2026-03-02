"""
Auto-Pilot Mode for AI-Scientist
=================================

Fully autonomous research paper writing system that:
1. Dynamically selects the best prompt/focus at each research stage
2. Manages all project files (LaTeX, code, data, bibliography) like Copilot
3. Self-evaluates and course-corrects throughout the pipeline
4. Produces publication-ready research papers with minimal human intervention

Architecture:
    AutoPilotAgent  — Top-level orchestrator; runs the full pipeline
    PromptOptimizer — Selects/generates the best prompt for each stage
    WorkspaceManager— Reads, writes, diffs any file in the project folder
    StageExecutor   — Executes individual research stages via LLM
    QualityGate     — Evaluates output quality and decides next action
"""

from autopilot.agent import AutoPilotAgent
from autopilot.config import AutoPilotConfig

__all__ = ["AutoPilotAgent", "AutoPilotConfig"]
__version__ = "0.1.0"
