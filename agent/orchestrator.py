"""
ResearchOrchestrator — the top-level controller for the autonomous research agent.

Ties together:
- Dynamic planning with backtracking
- Multi-agent specialist system
- Tool-use framework
- Persistent memory
- Quality gating with adaptive thresholds

This is the main entry point for running a fully autonomous research pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.config import AgentConfig, PHASES
from agent.memory import AgentMemory, ExperimentResult
from agent.planner import Planner
from agent.specialists import get_specialist_for_stage
from agent.tools import ToolRegistry, ToolResult
from agent.tools.research_tools import register_research_tools
from agent.tools.experiment_tools import register_experiment_tools
from agent.tools.writing_tools import register_writing_tools

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """
    Fully autonomous research paper writing orchestrator.

    Manages the complete pipeline from literature search through final PDF,
    with dynamic planning, multi-agent specialists, and tool-use capabilities.

    Usage::

        from agent import ResearchOrchestrator, AgentConfig

        config = AgentConfig(
            topic="...",
            field="...",
            research_question="...",
        )
        orchestrator = ResearchOrchestrator(config)
        result = orchestrator.run()
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_fn: Optional[Callable[..., str]] = None,
    ):
        self.config = config
        self._setup_logging()

        # Resolve project directory
        if not config.project_dir:
            safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in config.topic)
            safe_topic = safe_topic.strip().replace(" ", "_")[:50]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.project_dir = str(
                Path(config.output_dir) / f"{safe_topic}_{ts}"
            )

        self.project_dir = Path(config.project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Resolve LLM function
        if llm_fn:
            self.llm_fn = llm_fn
        else:
            self.llm_fn = self._resolve_llm_fn()

        # Initialize memory
        memory_dir = self.project_dir / config.memory_dir
        self.memory = AgentMemory(
            memory_dir=memory_dir,
            max_context_tokens=config.max_context_tokens,
        )

        # Initialize tool registry
        self.tools = ToolRegistry()
        self._register_all_tools()

        # Initialize planner
        self.planner = Planner(config, self.memory, self.llm_fn)

        # Trace
        self._trace: List[Dict[str, Any]] = []
        self._start_time: float = 0
        self._trace_dir = self.project_dir / config.trace_dir

    # ═════════════════════════════════════════════════════════════════════
    # Public API
    # ═════════════════════════════════════════════════════════════════════

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete autonomous research pipeline.

        Returns:
            {
                "status": "completed" | "aborted" | "error",
                "project_dir": str,
                "stages_completed": list,
                "quality_scores": dict,
                "total_time": float,
                "pdf_generated": bool,
            }
        """
        self._start_time = time.time()
        self._trace_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "status": "error",
            "project_dir": str(self.project_dir),
            "stages_completed": [],
            "quality_scores": {},
            "total_time": 0,
            "pdf_generated": False,
        }

        self._print_banner()

        try:
            for global_iter in range(1, self.config.max_global_iterations + 1):
                if global_iter > 1:
                    self._print_section(
                        f"GLOBAL ITERATION {global_iter}/{self.config.max_global_iterations}"
                    )
                    # Re-plan based on results so far
                    plan = self.planner.replan(
                        self.memory.state.stages_completed,
                        self.memory.state.quality_scores,
                    )
                else:
                    plan = self.planner.get_initial_plan()

                logger.info("Plan: %s", plan)
                all_passed = True

                stage_idx = 0
                while stage_idx < len(plan):
                    stage = plan[stage_idx]

                    # Check if stage should be skipped
                    should_skip, skip_reason = self.planner.should_skip(stage)
                    if should_skip:
                        self._print_stage_skip(stage, skip_reason)
                        stage_idx += 1
                        continue

                    # Run the stage
                    self._print_stage_start(stage, stage_idx + 1, len(plan))

                    stage_result = self._run_stage(stage)
                    quality_score = stage_result.get("quality_score", 0.0)

                    # Record results
                    summary["stages_completed"].append(stage)
                    summary["quality_scores"][stage] = quality_score
                    self.memory.set_quality_score(stage, quality_score)

                    # Save trace
                    self._save_trace(stage, stage_result)

                    # Check for backtracking
                    backtrack_to = self.planner.should_backtrack(stage, quality_score)
                    if backtrack_to:
                        self._print_backtrack(stage, backtrack_to, quality_score)
                        # Find the backtrack target in the plan
                        if backtrack_to in plan:
                            stage_idx = plan.index(backtrack_to)
                        else:
                            plan.insert(stage_idx + 1, backtrack_to)
                            stage_idx += 1
                        all_passed = False
                        continue

                    if quality_score < self.config.quality_threshold:
                        all_passed = False
                        self._print_stage_warn(stage, quality_score)
                    else:
                        self._print_stage_pass(stage, quality_score)

                    stage_idx += 1

                if all_passed:
                    break

            summary["status"] = "completed"

        except KeyboardInterrupt:
            summary["status"] = "interrupted"
            print("\n\n⚠️  Interrupted by user")
        except Exception as exc:
            summary["status"] = "error"
            logger.error("Pipeline error: %s", exc, exc_info=True)
            print(f"\n\n❌  Pipeline error: {exc}")

        # Save final state
        summary["total_time"] = time.time() - self._start_time
        summary["pdf_generated"] = (self.project_dir / "paper.pdf").exists()
        self.memory.save()
        self._save_summary(summary)
        self._print_summary(summary)

        return summary

    # ═════════════════════════════════════════════════════════════════════
    # Stage Execution
    # ═════════════════════════════════════════════════════════════════════

    def _run_stage(self, stage: str) -> Dict[str, Any]:
        """
        Run a single stage using the appropriate specialist agent.
        """
        phase = self.config.get_phase_for_stage(stage)
        self.memory.update_stage(stage, phase)

        # Get the specialist for this stage
        specialist = get_specialist_for_stage(
            stage, self.config, self.memory, self.tools, self.llm_fn,
        )

        # Build the task prompt
        task_prompt = self.planner.get_task_prompt(stage)

        # Run the specialist's agentic loop
        result = specialist.run(
            stage=stage,
            user_prompt=task_prompt,
            max_iterations=self.config.max_stage_retries + 2,
        )

        # Evaluate quality
        quality_score = self._evaluate_quality(stage, result)
        result["quality_score"] = quality_score

        # Post-process: extract structured data from the response
        self._post_process_stage(stage, result)

        return result

    def _evaluate_quality(self, stage: str, result: Dict[str, Any]) -> float:
        """Evaluate the quality of a stage's output."""
        if not result.get("success"):
            return 0.0

        score = 0.5  # base score

        response = result.get("response", "")

        # Heuristic checks based on stage
        phase = self.config.get_phase_for_stage(stage)

        if phase == "research":
            # Check that papers were found
            if self.memory.state.literature:
                score += 0.2
            if len(response) > 500:
                score += 0.1
            if any(kw in response.lower() for kw in ["hypothesis", "gap", "finding", "contribution"]):
                score += 0.1

        elif phase == "implementation":
            # Check that code was generated/executed successfully
            tool_calls = result.get("tool_calls", [])
            successful_tools = sum(1 for tc in tool_calls if tc.get("result", ToolResult(False)).success)
            if successful_tools > 0:
                score += 0.3
            if any(e.status == "completed" for e in self.memory.state.experiments):
                score += 0.2

        elif phase == "writing":
            # Check LaTeX structure
            paper_path = self.project_dir / "paper.tex"
            if paper_path.exists():
                from agent.tools.writing_tools import validate_latex_structure
                val_result = validate_latex_structure(paper_path.read_text(encoding="utf-8", errors="replace"))
                if val_result.success and val_result.data.get("valid"):
                    score += 0.3
                elif val_result.success:
                    # Partial credit
                    issues = val_result.data.get("issues", [])
                    score += max(0, 0.3 - 0.05 * len(issues))

        elif phase == "review":
            if len(response) > 300:
                score += 0.2
            if "score" in response.lower() or "weakness" in response.lower():
                score += 0.1

        # LLM self-evaluation (if response is long enough)
        if len(response) > 200 and self.llm_fn:
            try:
                llm_score = self._llm_quality_check(stage, response)
                score = 0.4 * score + 0.6 * llm_score
            except Exception:
                pass

        return min(1.0, max(0.0, score))

    def _llm_quality_check(self, stage: str, response: str) -> float:
        """Ask the LLM to evaluate its own output quality."""
        eval_prompt = (
            f"Rate the quality of this '{stage}' output on a scale of 0.0 to 1.0.\n\n"
            f"Output (first 3000 chars):\n{response[:3000]}\n\n"
            f"Criteria: completeness, correctness, academic quality, actionability.\n"
            f"Return ONLY a JSON object: {{\"score\": 0.X, \"reason\": \"...\"}}"
        )

        try:
            eval_response = self.llm_fn(
                messages=[{"role": "user", "content": eval_prompt}],
                model=self.config.model,
                prompt_type="general",
                request_timeout=60,
            )

            import re
            match = re.search(r'"score"\s*:\s*([\d.]+)', eval_response)
            if match:
                return float(match.group(1))
        except Exception:
            pass

        return 0.5

    def _post_process_stage(self, stage: str, result: Dict[str, Any]) -> None:
        """
        Extract structured data from the stage result and update memory.
        """
        response = result.get("response", "")
        tool_calls = result.get("tool_calls", [])

        # Extract papers from research tool calls
        for tc in tool_calls:
            if tc["tool"] in ("search_papers", "search_arxiv") and tc["result"].success:
                data = tc["result"].data
                papers = data.get("papers", [])
                for p in papers:
                    from agent.memory import PaperReference
                    self.memory.add_paper(PaperReference(
                        title=p.get("title", ""),
                        authors=p.get("authors", []),
                        year=p.get("year", 0),
                        venue=p.get("venue", ""),
                        abstract=p.get("abstract", ""),
                        doi=p.get("doi", ""),
                        arxiv_id=p.get("arxiv_id", ""),
                    ))

        # Extract experiment results
        for tc in tool_calls:
            if tc["tool"] == "execute_code" and tc["result"].success:
                data = tc["result"].data
                self.memory.add_experiment(ExperimentResult(
                    experiment_id=f"exp_{len(self.memory.state.experiments) + 1}",
                    description=stage,
                    status="completed" if data.get("success") else "failed",
                    stdout=data.get("stdout", "")[:2000],
                    stderr=data.get("stderr", "")[:1000],
                    output_files=data.get("output_files", []),
                    duration_seconds=data.get("elapsed_seconds", 0),
                ))

        # Check if paper.tex was written (from write_file tool calls)
        for tc in tool_calls:
            if tc["tool"] == "write_file" and tc["result"].success:
                args = tc.get("args", {})
                if args.get("file_path") == "paper.tex":
                    self.memory.state.current_draft_summary = f"Draft written during {stage}"

    # ═════════════════════════════════════════════════════════════════════
    # Tool Registration
    # ═════════════════════════════════════════════════════════════════════

    def _register_all_tools(self) -> None:
        """Register all tool categories."""
        register_research_tools(
            self.tools,
            api_key=self.config.tools.semantic_scholar_api_key,
        )
        register_experiment_tools(
            self.tools,
            working_dir=str(self.project_dir),
            python_executable=self.config.tools.python_executable,
            timeout=self.config.tools.code_execution_timeout,
            llm_fn=self.llm_fn,
            model=self.config.model,
        )
        register_writing_tools(
            self.tools,
            project_dir=str(self.project_dir),
            pdflatex_path=self.config.tools.pdflatex_path,
            bibtex_path=self.config.tools.bibtex_path,
        )

    # ═════════════════════════════════════════════════════════════════════
    # LLM Resolution
    # ═════════════════════════════════════════════════════════════════════

    def _resolve_llm_fn(self) -> Callable[..., str]:
        """
        Resolve the LLM function. Tries:
        1. Import _universal_chat from sciresearch_workflow (existing monolith)
        2. Fall back to a simple OpenAI wrapper
        """
        # Try the existing monolith
        try:
            from sciresearch_workflow import _universal_chat
            logger.info("Using _universal_chat from sciresearch_workflow")
            return _universal_chat
        except ImportError:
            pass

        # Fall back to direct OpenAI API
        logger.info("Using built-in OpenAI LLM wrapper")
        return self._builtin_llm_fn

    def _builtin_llm_fn(
        self,
        messages: List[Dict[str, str]],
        model: str = "",
        prompt_type: str = "general",
        request_timeout: int = 600,
        fallback_models: List[str] = None,
        **kwargs,
    ) -> str:
        """Built-in LLM function using the OpenAI API directly."""
        import openai

        model = model or self.config.model
        fallback_models = fallback_models or self.config.fallback_models

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "No API key configured. Set OPENAI_API_KEY environment variable "
                "or pass api_key in AgentConfig."
            )

        client = openai.OpenAI(api_key=api_key, timeout=request_timeout)

        models_to_try = [model] + (fallback_models or [])

        last_error = None
        for try_model in models_to_try:
            try:
                # Use Responses API for gpt-5-pro
                if "gpt-5-pro" in try_model:
                    response = client.responses.create(
                        model=try_model,
                        input=messages,
                    )
                    # Extract text from response
                    if hasattr(response, 'output'):
                        for item in response.output:
                            if hasattr(item, 'content'):
                                for content in item.content:
                                    if hasattr(content, 'text'):
                                        return content.text
                    return str(response)
                else:
                    response = client.chat.completions.create(
                        model=try_model,
                        messages=messages,
                        temperature=self.config.temperature,
                    )
                    return response.choices[0].message.content or ""

            except Exception as exc:
                last_error = exc
                logger.warning("Model %s failed: %s", try_model, exc)
                continue

        raise RuntimeError(f"All models failed. Last error: {last_error}")

    # ═════════════════════════════════════════════════════════════════════
    # Logging & Output
    # ═════════════════════════════════════════════════════════════════════

    def _setup_logging(self) -> None:
        """Configure logging."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def _print_banner(self) -> None:
        """Print the startup banner."""
        print("\n" + "═" * 80)
        print("  🔬  AUTONOMOUS RESEARCH AGENT")
        print("═" * 80)
        print(f"  Topic:    {self.config.topic}")
        print(f"  Field:    {self.config.field}")
        print(f"  Question: {self.config.research_question}")
        print(f"  Model:    {self.config.model}")
        print(f"  Project:  {self.project_dir}")
        print(f"  Phases:   {' → '.join(self.config.phases)}")

        stages = self.config.get_active_stages()
        print(f"  Stages:   {len(stages)}")
        for i, s in enumerate(stages, 1):
            phase = self.config.get_phase_for_stage(s)
            print(f"    {i:2d}. [{phase}] {s.replace('_', ' ').title()}")
        print("═" * 80 + "\n")

    def _print_section(self, title: str) -> None:
        print(f"\n{'─' * 60}")
        print(f"  🔄  {title}")
        print(f"{'─' * 60}\n")

    def _print_stage_start(self, stage: str, idx: int, total: int) -> None:
        phase = self.config.get_phase_for_stage(stage)
        print(f"\n{'━' * 60}")
        print(f"  Stage {idx}/{total}: {stage.replace('_', ' ').title()}")
        print(f"  Phase: {phase} | Specialist: {self._get_specialist_name(stage)}")
        print(f"{'━' * 60}")

    def _print_stage_pass(self, stage: str, score: float) -> None:
        print(f"  ✅  Stage passed (quality: {score:.2f})")

    def _print_stage_warn(self, stage: str, score: float) -> None:
        print(f"  ⚠️  Stage completed with issues (quality: {score:.2f})")

    def _print_stage_skip(self, stage: str, reason: str) -> None:
        print(f"  ⏭️  Skipping {stage}: {reason}")

    def _print_backtrack(self, from_stage: str, to_stage: str, score: float) -> None:
        print(f"  ↩️  Backtracking from {from_stage} to {to_stage} (score: {score:.2f})")

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        elapsed = summary["total_time"]
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"\n{'═' * 80}")
        print(f"  {'✅' if summary['status'] == 'completed' else '❌'}  RESEARCH AGENT — {summary['status'].upper()}")
        print(f"  Stages completed: {len(summary['stages_completed'])}")
        print(f"  Time: {mins}m {secs}s")
        print(f"  PDF generated: {'Yes' if summary['pdf_generated'] else 'No'}")
        print(f"  Project: {summary['project_dir']}")
        if summary["quality_scores"]:
            avg = sum(summary["quality_scores"].values()) / len(summary["quality_scores"])
            print(f"  Average quality: {avg:.2f}")
        print(f"{'═' * 80}\n")

    @staticmethod
    def _get_specialist_name(stage: str) -> str:
        from agent.specialists import STAGE_TO_SPECIALIST
        cls = STAGE_TO_SPECIALIST.get(stage)
        return cls.ROLE if cls else "unknown"

    # ═════════════════════════════════════════════════════════════════════
    # Trace / Persistence
    # ═════════════════════════════════════════════════════════════════════

    def _save_trace(self, stage: str, result: Dict[str, Any]) -> None:
        """Save a stage trace to disk."""
        if not self.config.save_trace:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_file = self._trace_dir / f"{ts}_{stage}.json"

        # Make serializable
        serializable = {
            "stage": stage,
            "success": result.get("success", False),
            "quality_score": result.get("quality_score", 0),
            "iterations": result.get("iterations", 0),
            "response_preview": result.get("response", "")[:5000],
            "tool_calls_count": len(result.get("tool_calls", [])),
            "tool_calls": [
                {
                    "tool": tc["tool"],
                    "success": tc["result"].success if hasattr(tc.get("result"), "success") else False,
                }
                for tc in result.get("tool_calls", [])
            ],
            "timestamp": ts,
        }

        try:
            trace_file.write_text(
                json.dumps(serializable, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save trace: %s", exc)

    def _save_summary(self, summary: Dict[str, Any]) -> None:
        """Save the run summary to disk."""
        summary_file = self._trace_dir / "run_summary.json"
        try:
            summary_file.write_text(
                json.dumps(summary, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save summary: %s", exc)
