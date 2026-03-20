"""
Specialist Agents — role-specific AI agents for different research phases.

Each specialist has a unique persona, system prompt, and tool access
optimized for its phase of the research pipeline.

Specialists:
- ResearcherAgent: Literature search, gap analysis, hypothesis formation
- ExperimenterAgent: Methodology design, code generation, experiment execution
- WriterAgent: Paper drafting, structure, LaTeX authoring
- ReviewerAgent: Internal peer review, revision suggestions
- EditorAgent: Final polishing, bibliography, formatting
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

from agent.config import AgentConfig
from agent.memory import AgentMemory
from agent.tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class BaseSpecialist:
    """
    Base class for all specialist agents.

    Each specialist wraps LLM calls with:
    - A role-specific system prompt
    - Access to a subset of tools
    - Memory integration for context
    - An agentic loop that can call tools and reason iteratively
    """

    ROLE = "assistant"
    TOOL_CATEGORIES: List[str] = []

    def __init__(
        self,
        config: AgentConfig,
        memory: AgentMemory,
        tool_registry: ToolRegistry,
        llm_fn: Callable[..., str],
    ):
        self.config = config
        self.memory = memory
        self.tools = tool_registry
        self.llm_fn = llm_fn

    def get_system_prompt(self, stage: str) -> str:
        """Build the system prompt for this specialist."""
        raise NotImplementedError

    def run(self, stage: str, user_prompt: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Execute the specialist's agentic loop.

        The specialist can:
        1. Reason about the task
        2. Call tools to gather information or take actions
        3. Iterate until the task is complete

        Returns:
            {
                "response": str,       # Final textual output
                "tool_calls": list,     # All tool calls made
                "iterations": int,      # Number of loop iterations
                "success": bool,
            }
        """
        system_prompt = self.get_system_prompt(stage)
        tools_desc = self.tools.get_tools_description(
            category=None  # give access to all tools
        )

        # Build messages with context
        context = self.memory.build_context_for_stage(stage)
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + tools_desc},
            {"role": "user", "content": f"{context}\n\n---\n\n{user_prompt}"},
        ]

        all_tool_calls = []
        final_response = ""

        for iteration in range(1, max_iterations + 1):
            logger.info(
                "%s | stage=%s | iteration %d/%d",
                self.ROLE, stage, iteration, max_iterations,
            )

            # Call LLM
            try:
                model = self.config.model
                specialist_cfg = self.config.get_specialist_for_stage(stage)
                if specialist_cfg and specialist_cfg.model:
                    model = specialist_cfg.model

                response = self.llm_fn(
                    messages=messages,
                    model=model,
                    prompt_type="general",
                    request_timeout=self.config.request_timeout,
                    fallback_models=self.config.fallback_models,
                )
            except Exception as exc:
                logger.error("%s LLM call failed: %s", self.ROLE, exc)
                return {
                    "response": "",
                    "tool_calls": all_tool_calls,
                    "iterations": iteration,
                    "success": False,
                    "error": str(exc),
                }

            # Check for tool calls
            tool_calls = self.tools.parse_tool_calls(response)

            if tool_calls:
                # Execute tools and feed results back
                results = self.tools.execute_tool_calls(response)
                all_tool_calls.extend(results)

                # Add assistant response and tool results to messages
                messages.append({"role": "assistant", "content": response})

                tool_results_text = "\n\n".join(
                    f"### Tool: {r['tool']}\n{r['result'].to_message()}"
                    for r in results
                )
                messages.append({
                    "role": "user",
                    "content": f"Tool results:\n\n{tool_results_text}\n\n"
                               f"Continue with your task. If you're done, provide your final output "
                               f"WITHOUT any tool_calls block.",
                })

                # Record tool interactions in memory
                for r in results:
                    self.memory.add_message(
                        stage, "tool",
                        f"{r['tool']}: {'OK' if r['result'].success else 'FAIL'}",
                    )
            else:
                # No tool calls — this is the final response
                final_response = response
                break

        # Store in memory
        self.memory.add_message(stage, "assistant", final_response[:2000])

        return {
            "response": final_response,
            "tool_calls": all_tool_calls,
            "iterations": iteration,
            "success": True,
        }


# ═════════════════════════════════════════════════════════════════════════
# Specialist Implementations
# ═════════════════════════════════════════════════════════════════════════

class ResearcherAgent(BaseSpecialist):
    """Specialist for literature review, gap analysis, and hypothesis formation."""

    ROLE = "researcher"
    TOOL_CATEGORIES = ["research"]

    def get_system_prompt(self, stage: str) -> str:
        return (
            "You are an expert academic researcher with deep expertise in "
            f"{self.config.field}. Your task is to conduct rigorous scholarly "
            "research for a research paper.\n\n"
            "## Your Capabilities\n"
            "- Search for and analyze academic papers using Semantic Scholar and arXiv\n"
            "- Identify research gaps and formulate hypotheses\n"
            "- Synthesize findings from multiple papers\n"
            "- Generate properly formatted BibTeX entries\n\n"
            "## Your Standards\n"
            "- Only cite real, verifiable papers — never fabricate references\n"
            "- Provide balanced coverage of the field\n"
            "- Identify both supporting and contradicting evidence\n"
            "- Clearly distinguish established facts from open questions\n\n"
            "## Research Context\n"
            f"- Topic: {self.config.topic}\n"
            f"- Field: {self.config.field}\n"
            f"- Research Question: {self.config.research_question}\n\n"
            "Use the available tools to search for papers, then synthesize your findings. "
            "When you're done gathering information, provide your complete analysis."
        )


class ExperimenterAgent(BaseSpecialist):
    """Specialist for methodology design, code generation, and experiment execution."""

    ROLE = "experimenter"
    TOOL_CATEGORIES = ["experiment"]

    def get_system_prompt(self, stage: str) -> str:
        return (
            "You are an expert computational scientist and software engineer. "
            "Your task is to design experiments, write code, and analyze results "
            "for a research paper.\n\n"
            "## Your Capabilities\n"
            "- Design rigorous experimental methodologies\n"
            "- Write clean, well-documented Python code\n"
            "- Execute code and analyze outputs\n"
            "- Generate publication-quality figures\n"
            "- Perform statistical analysis\n\n"
            "## Your Standards\n"
            "- All code must be syntactically correct and runnable\n"
            "- Include proper error handling and logging\n"
            "- Use appropriate statistical tests\n"
            "- Generate reproducible results (set random seeds)\n"
            "- Create clear, well-labeled figures\n\n"
            "## Research Context\n"
            f"- Topic: {self.config.topic}\n"
            f"- Field: {self.config.field}\n"
            f"- Research Question: {self.config.research_question}\n\n"
            "Use the available tools to write code, execute experiments, and analyze data. "
            "When you need to fix errors, use the fix_and_retry tool. "
            "Always validate code syntax before execution."
        )


class WriterAgent(BaseSpecialist):
    """Specialist for paper drafting and LaTeX authoring."""

    ROLE = "writer"
    TOOL_CATEGORIES = ["writing"]

    def get_system_prompt(self, stage: str) -> str:
        return (
            "You are an expert academic writer with extensive experience publishing "
            f"in {self.config.field}. Your task is to write a high-quality research "
            "paper in LaTeX.\n\n"
            "## Your Capabilities\n"
            "- Write clear, precise academic prose\n"
            "- Structure papers following field conventions\n"
            "- Create proper LaTeX documents with figures, tables, and equations\n"
            "- Manage bibliographies and cross-references\n\n"
            "## Your Standards\n"
            "- Write in formal academic style, avoiding colloquialisms\n"
            "- Every claim must be supported by evidence or citations\n"
            "- Include all required sections: abstract, introduction, related work, "
            "  methodology, experiments, results, discussion, conclusion\n"
            "- Ensure LaTeX compiles without errors\n"
            "- Use proper mathematical notation and notation consistency\n\n"
            "## Research Context\n"
            f"- Topic: {self.config.topic}\n"
            f"- Field: {self.config.field}\n"
            f"- Research Question: {self.config.research_question}\n\n"
            "Use the write_file tool to create/update paper.tex and refs.bib. "
            "Use validate_latex to check your work. "
            "Use compile_latex to test compilation."
        )


class ReviewerAgent(BaseSpecialist):
    """Specialist for internal peer review."""

    ROLE = "reviewer"
    TOOL_CATEGORIES = ["writing", "research"]

    def get_system_prompt(self, stage: str) -> str:
        return (
            "You are an experienced peer reviewer for top-tier academic venues in "
            f"{self.config.field}. Your task is to provide a thorough, constructive "
            "review of a research paper.\n\n"
            "## Review Criteria\n"
            "1. **Novelty**: Is the contribution genuinely new?\n"
            "2. **Soundness**: Is the methodology correct and experiments valid?\n"
            "3. **Significance**: Does this advance the field meaningfully?\n"
            "4. **Clarity**: Is the paper well-written and organized?\n"
            "5. **Completeness**: Are all claims supported? All sections adequate?\n"
            "6. **Reproducibility**: Could someone reproduce these results?\n"
            "7. **References**: Are citations real, relevant, and comprehensive?\n\n"
            "## Review Format\n"
            "Provide:\n"
            "- Overall score (1-10)\n"
            "- Confidence (1-5)\n"
            "- Summary of contributions\n"
            "- Strengths (numbered list)\n"
            "- Weaknesses (numbered list)\n"
            "- Detailed comments (specific, actionable suggestions)\n"
            "- Questions for the authors\n"
            "- Minor issues (typos, formatting)\n\n"
            "Be rigorous but constructive. Every criticism should come with a suggestion."
        )


class EditorAgent(BaseSpecialist):
    """Specialist for final polishing, formatting, and bibliography."""

    ROLE = "editor"
    TOOL_CATEGORIES = ["writing"]

    def get_system_prompt(self, stage: str) -> str:
        return (
            "You are a meticulous academic editor specializing in preparing "
            "papers for publication. Your task is to polish and finalize a "
            "research paper.\n\n"
            "## Your Responsibilities\n"
            "- Fix all LaTeX compilation errors\n"
            "- Ensure consistent formatting throughout\n"
            "- Validate all bibliography entries\n"
            "- Check cross-references and citations\n"
            "- Fix grammar, spelling, and style issues\n"
            "- Ensure figures and tables are properly referenced\n"
            "- Verify the paper meets formatting requirements\n\n"
            "## Your Standards\n"
            "- NEVER remove content — only fix and improve\n"
            "- Maintain the authors' voice and intent\n"
            "- Ensure every \\cite{} has a matching \\bibitem or BibTeX entry\n"
            "- All figures must have captions and be referenced in text\n"
            "- Page numbers, headers, and formatting must be correct\n\n"
            "Read the current paper, validate it, fix any issues, then compile to PDF."
        )


# ═════════════════════════════════════════════════════════════════════════
# Specialist Selector
# ═════════════════════════════════════════════════════════════════════════

STAGE_TO_SPECIALIST = {
    # Research phase
    "literature_search": ResearcherAgent,
    "literature_synthesis": ResearcherAgent,
    "gap_analysis": ResearcherAgent,
    "hypothesis_formation": ResearcherAgent,
    # Design phase
    "methodology_design": ExperimenterAgent,
    "experiment_planning": ExperimenterAgent,
    # Implementation phase
    "code_generation": ExperimenterAgent,
    "code_testing": ExperimenterAgent,
    "experiment_execution": ExperimenterAgent,
    "results_analysis": ExperimenterAgent,
    # Writing phase
    "outline_generation": WriterAgent,
    "initial_draft": WriterAgent,
    "figure_generation": ExperimenterAgent,
    "section_deepening": WriterAgent,
    # Review phase
    "internal_review": ReviewerAgent,
    "revision": WriterAgent,
    "bibliography_polish": EditorAgent,
    "final_compilation": EditorAgent,
}


def get_specialist_for_stage(
    stage: str,
    config: AgentConfig,
    memory: AgentMemory,
    tool_registry: ToolRegistry,
    llm_fn: Callable[..., str],
) -> BaseSpecialist:
    """Get the appropriate specialist agent for a given stage."""
    specialist_cls = STAGE_TO_SPECIALIST.get(stage, WriterAgent)
    return specialist_cls(config, memory, tool_registry, llm_fn)
