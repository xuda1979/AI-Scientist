"""
Autonomous Research Agent
=========================

A fully autonomous multi-agent system for end-to-end research paper production.
Covers the complete pipeline: literature research → experimental design →
implementation → analysis → paper writing → review & revision.

Architecture:
    ResearchOrchestrator  — Top-level controller with dynamic planning
    AgentMemory           — Persistent state, conversation history, knowledge base
    ToolRegistry          — Structured function-calling framework for LLM tools
    ResearchTools         — Literature search (Semantic Scholar, arXiv), web search
    ExperimentTools       — Code execution, data analysis, figure generation
    WritingTools          — LaTeX authoring, compilation, bibliography management
    SpecialistAgents      — Role-specific agents (researcher, experimenter, writer, reviewer)

Usage::

    from agent import ResearchOrchestrator, AgentConfig

    config = AgentConfig(
        topic="Quantum error correction with surface codes",
        field="quantum computing",
        research_question="How does code distance affect logical error rates?",
        model="gpt-5-pro",
    )
    orchestrator = ResearchOrchestrator(config)
    result = orchestrator.run()
"""

from agent.config import AgentConfig
from agent.orchestrator import ResearchOrchestrator
from agent.memory import AgentMemory

__all__ = ["ResearchOrchestrator", "AgentConfig", "AgentMemory"]
__version__ = "1.0.0"
