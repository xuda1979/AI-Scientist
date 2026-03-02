"""
AutoPilotAgent — the top-level orchestrator for fully automatic paper writing.

Flow:
    1. Initialise workspace and configuration
    2. For each stage in the research pipeline:
        a. Ask PromptOptimizer for the best prompt
        b. Execute the stage via StageExecutor
        c. Evaluate output via QualityGate
        d. Decide: proceed / retry (with improvements) / abort
    3. After all stages, run final compilation
    4. Output the finished project folder
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from autopilot.config import AutoPilotConfig, STAGES
from autopilot.prompt_optimizer import PromptOptimizer
from autopilot.quality_gate import QualityGate
from autopilot.stage_executor import StageExecutor
from autopilot.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


class AutoPilotAgent:
    """
    Fully autonomous research paper writing agent.

    Usage::

        from autopilot import AutoPilotAgent, AutoPilotConfig

        config = AutoPilotConfig(
            topic="...",
            field_name="...",
            research_question="...",
            project_dir="/path/to/project",
        )
        agent = AutoPilotAgent(config)
        result = agent.run()
    """

    def __init__(
        self,
        config: AutoPilotConfig,
        llm_fn: Optional[Callable[..., str]] = None,
    ):
        """
        Args:
            config:  Auto-pilot configuration.
            llm_fn:  LLM call function. If None, imports _universal_chat from
                     sciresearch_workflow (the existing codebase).
        """
        self.config = config

        # Resolve the LLM function
        if llm_fn is not None:
            self._llm_fn = llm_fn
        else:
            self._llm_fn = self._default_llm_fn()

        # Initialise sub-components
        self.workspace = WorkspaceManager(
            project_dir=Path(config.project_dir) if config.project_dir else Path.cwd(),
            allowed_extensions=config.allowed_extensions,
            max_file_size=config.max_file_size_bytes,
            create_backups=config.create_backups,
            content_protection=config.content_protection,
            max_word_loss_pct=config.max_word_loss_pct,
        )
        self.prompt_optimizer = PromptOptimizer(config, self._llm_fn)
        self.quality_gate = QualityGate(
            self._llm_fn, config.model, config.quality_gate_threshold
        )
        self.stage_executor = StageExecutor(config, self.workspace, self._llm_fn)

        # Execution trace
        self._trace: List[Dict[str, Any]] = []
        self._start_time: float = 0
        self._trace_dir: Optional[Path] = None

    # ── Public API ───────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Execute the full auto-pilot pipeline.

        Returns a summary dict with:
            - status: "completed" | "aborted" | "error"
            - stages_completed: int
            - total_time: float
            - project_dir: str
            - trace: list of per-stage records
        """
        self._start_time = time.time()
        self._setup_trace_dir()

        summary: Dict[str, Any] = {
            "status": "error",
            "stages_completed": 0,
            "total_time": 0,
            "project_dir": str(self.workspace.project_dir),
            "trace": [],
        }

        stages = self.config.stages
        logger.info("=" * 80)
        logger.info("AUTO-PILOT MODE — %d stages", len(stages))
        logger.info("Topic: %s", self.config.topic)
        logger.info("Field: %s", self.config.field_name)
        logger.info("Question: %s", self.config.research_question)
        logger.info("Project dir: %s", self.workspace.project_dir)
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("🤖  AUTO-PILOT MODE ACTIVATED")
        print(f"    Topic: {self.config.topic}")
        print(f"    Field: {self.config.field_name}")
        print(f"    Question: {self.config.research_question}")
        print(f"    Stages: {len(stages)}")
        print(f"    Model: {self.config.model}")
        print(f"    Project: {self.workspace.project_dir}")
        print("=" * 80 + "\n")

        for global_iter in range(1, self.config.max_global_iterations + 1):
            if global_iter > 1:
                print(f"\n{'─' * 60}")
                print(f"🔄  GLOBAL ITERATION {global_iter}/{self.config.max_global_iterations}")
                print(f"{'─' * 60}\n")

            all_passed = True

            for stage_idx, stage in enumerate(stages):
                print(f"\n{'━' * 60}")
                print(f"  Stage {stage_idx + 1}/{len(stages)}: {stage.replace('_', ' ').title()}")
                print(f"{'━' * 60}")

                stage_result = self._run_stage(stage)
                self._trace.append(stage_result)
                self._save_trace_entry(stage, stage_result)

                verdict = stage_result.get("quality_verdict", {}).get("verdict", "proceed")

                if verdict == "proceed":
                    summary["stages_completed"] += 1
                    print(f"  ✅  Stage passed (score: {stage_result.get('quality_verdict', {}).get('score', '?')})")
                elif verdict == "abort":
                    print(f"  ❌  Stage ABORTED — critical quality failure")
                    if self.config.abort_on_critical_failure:
                        summary["status"] = "aborted"
                        summary["total_time"] = time.time() - self._start_time
                        summary["trace"] = self._trace
                        self._save_summary(summary)
                        return summary
                    all_passed = False
                else:  # retry already happened inside _run_stage
                    summary["stages_completed"] += 1
                    all_passed = False
                    print(f"  ⚠️  Stage completed with issues (score: {stage_result.get('quality_verdict', {}).get('score', '?')})")

            if all_passed:
                break  # No need for another global iteration

        # ── Final compilation ────────────────────────────────────────────
        print(f"\n{'═' * 60}")
        print("  📄  FINAL COMPILATION")
        print(f"{'═' * 60}")
        self._final_compile()

        summary["status"] = "completed"
        summary["total_time"] = time.time() - self._start_time
        summary["trace"] = self._trace
        self._save_summary(summary)

        print(f"\n{'═' * 80}")
        print(f"  ✅  AUTO-PILOT COMPLETE")
        print(f"  Stages completed: {summary['stages_completed']}")
        print(f"  Total time: {summary['total_time']:.1f}s")
        print(f"  Project: {self.workspace.project_dir}")
        print(f"{'═' * 80}\n")

        return summary

    # ── Single stage execution with retries ──────────────────────────────

    def _run_stage(self, stage: str) -> Dict[str, Any]:
        """Run a single stage with prompt optimisation, execution, and quality gating."""
        stage_cfg = self.config.get_stage_config(stage)
        last_result: Dict[str, Any] = {}
        extra_context = ""

        for attempt in range(1, stage_cfg.max_attempts + 1):
            if attempt > 1:
                print(f"    🔁  Retry {attempt}/{stage_cfg.max_attempts}")

            # ── 1. Get workspace snapshot ────────────────────────────────
            snapshot = self.workspace.snapshot()

            # ── 2. Optimise prompt ───────────────────────────────────────
            prompt = self.prompt_optimizer.best_prompt(
                stage=stage,
                workspace_snapshot=snapshot,
                extra_context=extra_context,
            )

            # ── 3. Execute stage ─────────────────────────────────────────
            exec_result = self.stage_executor.execute(
                stage=stage,
                prompt=prompt,
                stage_cfg=stage_cfg,
            )

            if not exec_result["success"]:
                logger.warning("Stage %s attempt %d failed: %s", stage, attempt, exec_result.get("error"))
                extra_context = f"Previous attempt failed with error: {exec_result.get('error', 'unknown')}"
                last_result = exec_result
                continue

            # ── 4. Quality gate ──────────────────────────────────────────
            updated_snapshot = self.workspace.snapshot()
            quality = self.quality_gate.evaluate(
                stage=stage,
                output_text=exec_result["llm_response"],
                workspace_snapshot=updated_snapshot,
                file_actions_applied=exec_result.get("apply_results", []),
            )

            exec_result["quality_verdict"] = quality
            last_result = exec_result

            if quality["verdict"] == "proceed":
                return last_result
            elif quality["verdict"] == "abort":
                return last_result  # caller handles abort

            # Retry: feed quality issues back as extra context
            issues_text = "\n".join(f"- {i}" for i in quality.get("issues", []))
            suggestions_text = "\n".join(f"- {s}" for s in quality.get("suggestions", []))
            extra_context = (
                f"Previous attempt scored {quality['score']:.2f} (threshold {self.config.quality_gate_threshold}).\n"
                f"Issues found:\n{issues_text}\n"
                f"Suggestions:\n{suggestions_text}\n"
                f"Please address ALL these issues in your next attempt."
            )

        return last_result

    # ── Final compilation ────────────────────────────────────────────────

    def _final_compile(self) -> None:
        """Attempt to compile the LaTeX paper into a PDF."""
        paper_path = self.workspace.project_dir / "paper.tex"
        if not paper_path.exists():
            logger.warning("No paper.tex found — skipping final compilation")
            return

        try:
            import subprocess
            import shutil

            # Check if pdflatex is available
            if not shutil.which("pdflatex"):
                print("  ⚠️  pdflatex not found — skipping PDF compilation")
                return

            # Extract refs.bib from filecontents if embedded
            tex_content = paper_path.read_text(encoding="utf-8", errors="replace")
            self._extract_filecontents(tex_content)

            # Compile: pdflatex → bibtex → pdflatex × 2
            cwd = self.workspace.project_dir
            for step_name, cmd in [
                ("pdflatex (1)", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"]),
                ("bibtex", ["bibtex", "paper"]),
                ("pdflatex (2)", ["pdflatex", "-interaction=nonstopmode", "paper.tex"]),
                ("pdflatex (3)", ["pdflatex", "-interaction=nonstopmode", "paper.tex"]),
            ]:
                print(f"    Running {step_name}...")
                proc = subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if proc.returncode != 0 and "pdflatex" in step_name:
                    logger.warning("%s returned non-zero: %s", step_name, proc.stderr[:500])

            pdf_path = cwd / "paper.pdf"
            if pdf_path.exists():
                print(f"  ✅  PDF generated: {pdf_path}")
            else:
                print(f"  ⚠️  PDF not generated — check LaTeX errors")

        except subprocess.TimeoutExpired:
            print("  ⚠️  LaTeX compilation timed out")
        except Exception as exc:
            logger.warning("Final compilation error: %s", exc)
            print(f"  ⚠️  Compilation error: {exc}")

    def _extract_filecontents(self, tex_content: str) -> None:
        """Extract filecontents environments from the LaTeX source."""
        import re
        pattern = r"\\begin\{filecontents\*?\}\{(.+?)\}(.*?)\\end\{filecontents\*?\}"
        for match in re.finditer(pattern, tex_content, re.DOTALL):
            filename = match.group(1).strip()
            content = match.group(2).strip()
            fp = self.workspace.project_dir / filename
            fp.write_text(content, encoding="utf-8")
            logger.info("Extracted filecontents: %s", filename)

    # ── Trace / logging ──────────────────────────────────────────────────

    def _setup_trace_dir(self) -> None:
        """Create the trace directory if tracing is enabled."""
        if self.config.save_trace:
            self._trace_dir = self.workspace.project_dir / self.config.trace_dir
            self._trace_dir.mkdir(parents=True, exist_ok=True)

    def _save_trace_entry(self, stage: str, result: Dict[str, Any]) -> None:
        """Save a single stage's trace to disk."""
        if not self._trace_dir:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        entry_file = self._trace_dir / f"{ts}_{stage}.json"

        # Make a serialisable copy
        serialisable = {}
        for k, v in result.items():
            if k == "llm_response":
                serialisable[k] = v[:10000] if isinstance(v, str) else str(v)[:10000]
            elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serialisable[k] = v
            else:
                serialisable[k] = str(v)

        try:
            entry_file.write_text(json.dumps(serialisable, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save trace: %s", exc)

    def _save_summary(self, summary: Dict[str, Any]) -> None:
        """Save the run summary to disk."""
        if not self._trace_dir:
            return
        summary_file = self._trace_dir / "run_summary.json"
        # Strip full trace from saved summary (it's already in individual files)
        save_data = {k: v for k, v in summary.items() if k != "trace"}
        save_data["trace_count"] = len(summary.get("trace", []))
        try:
            summary_file.write_text(json.dumps(save_data, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save summary: %s", exc)

    # ── Default LLM function ─────────────────────────────────────────────

    @staticmethod
    def _default_llm_fn() -> Callable[..., str]:
        """Import and return the existing _universal_chat from sciresearch_workflow."""
        try:
            from sciresearch_workflow import _universal_chat
            return _universal_chat
        except ImportError as exc:
            raise ImportError(
                "Could not import _universal_chat from sciresearch_workflow. "
                "Make sure sciresearch_workflow.py is on the Python path."
            ) from exc
