"""
StageExecutor — runs a single research stage via the LLM and applies file changes.

Each stage:
1. Receives the best prompt from the PromptOptimizer
2. Builds a context-aware message with the full workspace snapshot
3. Calls the LLM
4. Parses file actions from the response
5. Applies changes to the workspace via WorkspaceManager
6. Returns the raw LLM output for quality evaluation
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from autopilot.config import AutoPilotConfig, StageConfig
from autopilot.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


class StageExecutor:
    """Executes a single research stage."""

    def __init__(
        self,
        config: AutoPilotConfig,
        workspace: WorkspaceManager,
        llm_fn: Callable[..., str],
    ):
        self.config = config
        self.workspace = workspace
        self.llm_fn = llm_fn

    def execute(
        self,
        stage: str,
        prompt: str,
        stage_cfg: StageConfig,
        extra_system: str = "",
    ) -> Dict[str, Any]:
        """
        Execute one stage.

        Returns:
            {
                "stage": str,
                "success": bool,
                "llm_response": str,
                "file_actions": [dict],
                "apply_results": [str],
                "elapsed_seconds": float,
                "error": str | None,
            }
        """
        result: Dict[str, Any] = {
            "stage": stage,
            "success": False,
            "llm_response": "",
            "file_actions": [],
            "apply_results": [],
            "elapsed_seconds": 0.0,
            "error": None,
        }

        t0 = time.time()

        # ── Build messages ───────────────────────────────────────────────
        system_msg = self._build_system_message(stage, extra_system)
        user_msg = prompt  # already crafted by PromptOptimizer

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # ── Call LLM ────────────────────────────────────────────────────
        try:
            response = self.llm_fn(
                messages=messages,
                model=self.config.model,
                prompt_type=self._prompt_type_for_stage(stage),
                request_timeout=stage_cfg.timeout,
                fallback_models=self.config.fallback_models,
            )
            result["llm_response"] = response
        except Exception as exc:
            result["error"] = str(exc)
            result["elapsed_seconds"] = time.time() - t0
            logger.error("LLM call failed for stage %s: %s", stage, exc)
            return result

        # ── Parse file actions ──────────────────────────────────────────
        actions = WorkspaceManager.parse_file_actions_from_response(response)
        result["file_actions"] = actions

        # ── Apply file changes ──────────────────────────────────────────
        if actions:
            apply_results = self.workspace.apply_actions(actions)
            result["apply_results"] = apply_results
            logger.info("Applied %d file actions for stage %s", len(actions), stage)
        else:
            logger.info("No file actions parsed from stage %s response", stage)

        # ── Self-reflection (optional) ──────────────────────────────────
        if stage_cfg.enable_self_reflection:
            reflection = self._self_reflect(stage, response)
            result["self_reflection"] = reflection

        result["success"] = True
        result["elapsed_seconds"] = time.time() - t0
        return result

    # ── Build system message ─────────────────────────────────────────────

    def _build_system_message(self, stage: str, extra: str) -> str:
        """
        Construct the system message that gives the LLM its identity and
        instructions for how to format file changes.
        """
        return (
            "You are an autonomous AI research scientist working on a research paper. "
            "You have full read/write access to all project files.\n\n"
            f"## Current Task Stage: {stage.replace('_', ' ').title()}\n\n"
            "## File Change Protocol\n"
            "When you need to create or modify files, output your changes using "
            "this EXACT format at the end of your response:\n\n"
            "```file_actions\n"
            "[\n"
            '  {"action": "write", "path": "paper.tex", "content": "...full file content..."},\n'
            '  {"action": "write", "path": "simulation.py", "content": "...full file content..."},\n'
            '  {"action": "patch", "path": "refs.bib", "search": "old text", "replace": "new text"},\n'
            '  {"action": "delete", "path": "old_file.txt"}\n'
            "]\n"
            "```\n\n"
            "IMPORTANT RULES:\n"
            "- For 'write' actions, always provide the COMPLETE file content.\n"
            "- For 'patch' actions, provide exact text to search and replace.\n"
            "- Never truncate files. If a file is large, still output the full content.\n"
            "- You can modify ANY file in the project: .tex, .py, .bib, .csv, .md, etc.\n"
            "- Always maintain LaTeX compilability for .tex files.\n"
            "- Always maintain Python syntax correctness for .py files.\n\n"
            f"{extra}"
        )

    # ── Self-reflection ──────────────────────────────────────────────────

    def _self_reflect(self, stage: str, response: str) -> str:
        """Quick self-reflection: ask the LLM to critique its own output."""
        try:
            reflection = self.llm_fn(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"You just completed the '{stage}' stage. "
                            f"Here is your output (first 3000 chars):\n\n"
                            f"{response[:3000]}\n\n"
                            f"Briefly list any weaknesses or things you would improve "
                            f"if you could redo this stage. Be specific and concise."
                        ),
                    }
                ],
                model=self.config.model,
                prompt_type="general",
                request_timeout=60,
            )
            return reflection
        except Exception as exc:
            logger.warning("Self-reflection failed: %s", exc)
            return ""

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _prompt_type_for_stage(stage: str) -> str:
        """Map stage to the prompt_type parameter used by _universal_chat."""
        mapping = {
            "literature_review": "general",
            "research_gap_analysis": "general",
            "hypothesis_formation": "general",
            "methodology_design": "general",
            "experiment_code_generation": "general",
            "experiment_execution": "general",
            "results_analysis": "general",
            "initial_draft": "draft",
            "section_deep_dive": "revise",
            "internal_review": "review",
            "revision": "revise",
            "bibliography_polish": "revise",
            "final_compilation": "revise",
        }
        return mapping.get(stage, "general")
