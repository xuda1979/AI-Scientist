"""Tkinter GUI for the SciResearch workflow."""
from __future__ import annotations

import contextlib
import logging
import queue
import threading
from io import TextIOBase
from pathlib import Path
from typing import Dict, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from core.config import WorkflowConfig
from document_types import get_available_document_types
from sciresearch_workflow import (
    DEFAULT_MODEL,
    WorkflowCancelled,
    run_workflow,
    test_time_compute_scaling,
)


class QueueWriter(TextIOBase):
    """File-like object that forwards writes into a queue."""

    def __init__(self, output_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self._queue = output_queue

    def write(self, message: str) -> int:  # type: ignore[override]
        if message:
            self._queue.put(message)
        return len(message)

    def flush(self) -> None:  # type: ignore[override]
        return None


class QueueLogHandler(logging.Handler):
    """Logging handler that forwards formatted records into a queue."""

    def __init__(self, output_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self._queue = output_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:  # pragma: no cover - safeguard for logging errors
            msg = record.getMessage()
        self._queue.put(msg + "\n")


class WorkflowGUI(tk.Tk):
    """Main Tkinter application window."""

    POLL_INTERVAL_MS = 100
    SENTINEL_DONE = "__THREAD_DONE__"

    def __init__(self) -> None:
        super().__init__()
        self.title("SciResearch Workflow")
        self.geometry("1180x820")
        self.minsize(1040, 720)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.cancel_event: Optional[threading.Event] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.worker_result: str = "idle"
        self.running = False

        self.status_var = tk.StringVar(value="Idle")
        self.vars: Dict[str, tk.Variable] = {}

        self._build_ui()
        self._bind_shortcuts()
        self._update_run_button_state()
        self._poll_queue()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():  # Use a modern theme when available
            style.theme_use("clam")

        main = ttk.Frame(self, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # Top form frames
        form_frame = ttk.Frame(main)
        form_frame.pack(fill=tk.X, expand=False)

        left_column = ttk.Frame(form_frame)
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right_column = ttk.Frame(form_frame)
        right_column.grid(row=0, column=1, sticky="nsew")
        form_frame.columnconfigure(0, weight=1)
        form_frame.columnconfigure(1, weight=1)

        self._build_project_frame(left_column)
        self._build_execution_frame(left_column)
        self._build_quality_frame(right_column)
        self._build_advanced_frame(right_column)

        # Prompt entry spans both columns
        prompt_frame = ttk.LabelFrame(main, text="Custom User Prompt", padding=8)
        prompt_frame.pack(fill=tk.X, expand=False, pady=(12, 0))
        self.prompt_text = ScrolledText(prompt_frame, height=4, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        # Log output frame
        log_frame = ttk.LabelFrame(main, text="Workflow Output", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.log_text = ScrolledText(log_frame, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Status and controls
        control_frame = ttk.Frame(main)
        control_frame.pack(fill=tk.X, expand=False, pady=(12, 0))

        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)

        self.run_button = ttk.Button(control_frame, text="Run Workflow", command=self.start_workflow)
        self.run_button.pack(side=tk.RIGHT)

        self.cancel_button = ttk.Button(control_frame, text="Cancel", command=self.cancel_workflow, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.RIGHT, padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_project_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Project Details", padding=8)
        frame.pack(fill=tk.X, expand=False, pady=(0, 12))

        self._add_entry(frame, "Topic", "topic", row=0)
        self._add_entry(frame, "Field", "field", row=1)
        self._add_entry(frame, "Research Question", "question", row=2)

        ttk.Label(frame, text="Document Type").grid(row=3, column=0, sticky=tk.W, pady=4)
        doc_var = tk.StringVar(value="auto")
        self.vars["document_type"] = doc_var
        doc_combo = ttk.Combobox(frame, textvariable=doc_var, values=get_available_document_types(), state="readonly")
        doc_combo.grid(row=3, column=1, sticky="ew", pady=4)

        self._add_entry(frame, "Model", "model", default=DEFAULT_MODEL, row=4)

        ttk.Label(frame, text="Output Directory").grid(row=5, column=0, sticky=tk.W, pady=4)
        out_var = tk.StringVar()
        self.vars["output_dir"] = out_var
        out_entry = ttk.Entry(frame, textvariable=out_var)
        out_entry.grid(row=5, column=1, sticky="ew", pady=4)
        browse_btn = ttk.Button(frame, text="Browse", command=lambda: self._browse_directory(out_var))
        browse_btn.grid(row=5, column=2, sticky=tk.W, padx=(6, 0))
        out_var.trace_add("write", lambda *_: self._update_run_button_state())

        frame.columnconfigure(1, weight=1)

    def _build_execution_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Execution Settings", padding=8)
        frame.pack(fill=tk.X, expand=False, pady=(0, 12))

        self._add_spinbox(frame, "Request Timeout (s)", "request_timeout", default=3600, from_=0, to=99999, row=0)
        self._add_spinbox(frame, "Max Retries", "max_retries", default=3, from_=0, to=20, row=1)
        self._add_spinbox(frame, "Max Iterations", "max_iterations", default=4, from_=1, to=20, row=2)
        self._add_check(frame, "Disable Early Stopping", "no_early_stopping", default=False, row=3)
        self._add_check(frame, "Modify Existing Project", "modify_existing", default=False, row=4)
        self._add_check(frame, "Enforce Single Files", "strict_singletons", default=True, row=5)
        self._add_entry(frame, "Python Executable", "python_exec", row=6)

        ttk.Label(frame, text="Config File").grid(row=7, column=0, sticky=tk.W, pady=4)
        config_var = tk.StringVar()
        self.vars["config_path"] = config_var
        config_entry = ttk.Entry(frame, textvariable=config_var)
        config_entry.grid(row=7, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="Browse", command=lambda: self._browse_file(config_var)).grid(row=7, column=2, sticky=tk.W, padx=(6, 0))

        ttk.Label(frame, text="Save Config To").grid(row=8, column=0, sticky=tk.W, pady=4)
        save_var = tk.StringVar()
        self.vars["save_config_path"] = save_var
        save_entry = ttk.Entry(frame, textvariable=save_var)
        save_entry.grid(row=8, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="Choose", command=lambda: self._browse_save_file(save_var)).grid(row=8, column=2, sticky=tk.W, padx=(6, 0))

        frame.columnconfigure(1, weight=1)

    def _build_quality_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Quality & Validation", padding=8)
        frame.pack(fill=tk.X, expand=False, pady=(0, 12))

        self._add_spinbox(frame, "Quality Threshold", "quality_threshold", default=1.0, from_=0.0, to=1.0, increment=0.05, row=0, var_type="double")
        self._add_check(frame, "Check References", "check_references", default=True, row=1)
        self._add_check(frame, "Skip Reference Check", "skip_reference_check", default=False, row=2)
        self._add_check(frame, "Validate Figures", "validate_figures", default=True, row=3)
        self._add_check(frame, "Skip Figure Validation", "skip_figure_validation", default=False, row=4)
        self._add_check(frame, "Enable PDF Review", "enable_pdf_review", default=False, row=5)
        self._add_check(frame, "Disable PDF Review", "disable_pdf_review", default=False, row=6)
        self._add_check(frame, "Enable Ideation", "enable_ideation", default=True, row=7)
        self._add_check(frame, "Skip Ideation", "skip_ideation", default=False, row=8)
        self._add_entry(frame, "Specify Idea", "specify_idea", row=9)
        self._add_spinbox(frame, "Number of Ideas", "num_ideas", default=15, from_=1, to=50, row=10)

    def _build_advanced_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Advanced Options", padding=8)
        frame.pack(fill=tk.X, expand=False, pady=(0, 12))

        self._add_check(frame, "Disable Content Protection", "disable_content_protection", default=False, row=0)
        self._add_check(frame, "Auto-Approve Changes", "auto_approve_changes", default=False, row=1)
        self._add_spinbox(frame, "Content Protection Threshold", "content_protection_threshold", default=0.15, from_=0.0, to=1.0, increment=0.01, row=2, var_type="double")
        self._add_check(frame, "Save Output Diffs", "output_diffs", default=True, row=3)
        self._add_check(frame, "Disable Output Diffs", "no_output_diffs", default=False, row=4)
        self._add_check(frame, "Enable Test-Time Scaling Mode", "test_scaling", default=False, row=5)
        self._add_entry(frame, "Scaling Prompt", "scaling_prompt", row=6)
        self._add_entry(frame, "Scaling Candidates", "scaling_candidates", default="3,5,7,10", row=7)
        self._add_spinbox(frame, "Scaling Timeout (s)", "scaling_timeout", default=1800, from_=0, to=99999, row=8)
        self._add_check(frame, "Use Scaling During Workflow", "use_test_time_scaling", default=False, row=9)
        self._add_spinbox(frame, "Revision Candidates", "revision_candidates", default=3, from_=1, to=10, row=10)
        self._add_spinbox(frame, "Draft Candidates", "draft_candidates", default=1, from_=1, to=5, row=11)

    def _add_entry(self, frame: ttk.Frame, label: str, key: str, row: int, default: str = "") -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
        var = tk.StringVar(value=default)
        self.vars[key] = var
        entry = ttk.Entry(frame, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        frame.columnconfigure(1, weight=1)
        var.trace_add("write", lambda *_: self._update_run_button_state())

    def _add_spinbox(
        self,
        frame: ttk.Frame,
        label: str,
        key: str,
        default: float,
        from_: float,
        to: float,
        row: int,
        increment: float = 1.0,
        var_type: str = "int",
    ) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
        if var_type == "double":
            var = tk.DoubleVar(value=default)
        else:
            var = tk.IntVar(value=int(default))
        self.vars[key] = var
        spin = ttk.Spinbox(frame, textvariable=var, from_=from_, to=to, increment=increment)
        spin.grid(row=row, column=1, sticky="ew", pady=4)
        frame.columnconfigure(1, weight=1)
        var.trace_add("write", lambda *_: self._update_run_button_state())

    def _add_check(self, frame: ttk.Frame, label: str, key: str, default: bool, row: int) -> None:
        var = tk.BooleanVar(value=default)
        self.vars[key] = var
        check = ttk.Checkbutton(frame, text=label, variable=var)
        check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=4)
        var.trace_add("write", lambda *_: self._on_flag_change(key))

    # ------------------------------------------------------------------
    # Event bindings and callbacks
    # ------------------------------------------------------------------
    def _bind_shortcuts(self) -> None:
        self.bind("<Return>", self._on_return_pressed)
        self.bind("<Escape>", lambda *_: self._on_close())

    def _on_return_pressed(self, event: tk.Event) -> None:  # type: ignore[override]
        widget = event.widget
        if isinstance(widget, tk.Entry):
            self.start_workflow()

    def _on_flag_change(self, key: str) -> None:
        # Maintain mutually exclusive toggle pairs
        pairs = {
            "enable_pdf_review": "disable_pdf_review",
            "disable_pdf_review": "enable_pdf_review",
            "enable_ideation": "skip_ideation",
            "skip_ideation": "enable_ideation",
            "output_diffs": "no_output_diffs",
            "no_output_diffs": "output_diffs",
            "check_references": "skip_reference_check",
            "validate_figures": "skip_figure_validation",
        }
        skip_pairs = {
            "skip_reference_check": "check_references",
            "skip_figure_validation": "validate_figures",
        }
        if key in pairs and self.vars[key].get():
            counterpart = pairs[key]
            if self.vars[counterpart].get():
                self.vars[counterpart].set(False)
        if key in skip_pairs:
            target = skip_pairs[key]
            if self.vars[key].get():
                self.vars[target].set(False)
            elif target in ("check_references", "validate_figures") and not self.vars[target].get():
                # Restore default when skip unchecked
                default_values = {
                    "check_references": True,
                    "validate_figures": True,
                }
                self.vars[target].set(default_values[target])
        if key == "disable_content_protection" and self.vars[key].get():
            # Content protection disabled implies auto-approve might be risky - no automatic change, but update run button state
            pass
        self._update_run_button_state()

    def _browse_directory(self, var: tk.StringVar) -> None:
        initial_dir = var.get() or str(Path.cwd())
        directory = filedialog.askdirectory(parent=self, initialdir=initial_dir)
        if directory:
            var.set(directory)

    def _browse_file(self, var: tk.StringVar) -> None:
        initial_dir = Path(var.get()).parent if var.get() else Path.cwd()
        file_path = filedialog.askopenfilename(parent=self, initialdir=initial_dir)
        if file_path:
            var.set(file_path)

    def _browse_save_file(self, var: tk.StringVar) -> None:
        initial_dir = Path(var.get()).parent if var.get() else Path.cwd()
        file_path = filedialog.asksaveasfilename(parent=self, initialdir=initial_dir, defaultextension=".json")
        if file_path:
            var.set(file_path)

    def _update_run_button_state(self) -> None:
        output_dir = self.vars.get("output_dir")
        topic = self.vars.get("topic")
        field = self.vars.get("field")
        question = self.vars.get("question")
        modify_existing = bool(self.vars.get("modify_existing").get()) if "modify_existing" in self.vars else False

        required_ready = bool(output_dir and output_dir.get().strip())
        if not modify_existing:
            required_ready = required_ready and all(
                var is not None and bool(var.get().strip()) for var in (topic, field, question)
            )

        state = tk.NORMAL if required_ready and not self.running else tk.DISABLED
        self.run_button.configure(state=state)
        self.cancel_button.configure(state=tk.NORMAL if self.running else tk.DISABLED)

    def start_workflow(self) -> None:
        if self.running:
            return
        if not self._validate_inputs():
            return

        self.running = True
        self.worker_result = "running"
        self.status_var.set("Running...")
        self.cancel_event = threading.Event()
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

        params = self._gather_parameters()
        self.worker_thread = threading.Thread(target=self._run_workflow_thread, args=(params,), daemon=True)
        self.worker_thread.start()
        self._update_run_button_state()

    def _validate_inputs(self) -> bool:
        output_dir = self.vars["output_dir"].get().strip()
        modify_existing = self.vars["modify_existing"].get()
        missing = []
        if not output_dir:
            missing.append("output directory")
        if not modify_existing:
            for field_name in ("topic", "field", "question"):
                if not self.vars[field_name].get().strip():
                    missing.append(field_name.replace("_", " "))
        if missing:
            messagebox.showerror(
                "Missing information",
                "Please provide the following before starting the workflow:\n- " + "\n- ".join(missing),
                parent=self,
            )
            return False
        return True

    def cancel_workflow(self) -> None:
        if self.cancel_event and not self.cancel_event.is_set():
            self.cancel_event.set()
            self.status_var.set("Cancelling...")
            self.log_queue.put("Cancellation requested. Waiting for the current phase to finish...\n")
            self.cancel_button.configure(state=tk.DISABLED)

    def _gather_parameters(self) -> Dict[str, object]:
        params: Dict[str, object] = {
            "topic": self.vars["topic"].get().strip(),
            "field": self.vars["field"].get().strip(),
            "question": self.vars["question"].get().strip(),
            "document_type": self.vars["document_type"].get(),
            "output_dir": self.vars["output_dir"].get().strip(),
            "model": self.vars["model"].get().strip() or DEFAULT_MODEL,
            "request_timeout": int(self.vars["request_timeout"].get()),
            "max_retries": int(self.vars["max_retries"].get()),
            "max_iterations": int(self.vars["max_iterations"].get()),
            "no_early_stopping": bool(self.vars["no_early_stopping"].get()),
            "modify_existing": bool(self.vars["modify_existing"].get()),
            "strict_singletons": bool(self.vars["strict_singletons"].get()),
            "python_exec": self.vars["python_exec"].get().strip() or None,
            "config_path": self.vars["config_path"].get().strip() or None,
            "save_config_path": self.vars["save_config_path"].get().strip() or None,
            "quality_threshold": float(self.vars["quality_threshold"].get()),
            "check_references": bool(self.vars["check_references"].get()),
            "skip_reference_check": bool(self.vars["skip_reference_check"].get()),
            "validate_figures": bool(self.vars["validate_figures"].get()),
            "skip_figure_validation": bool(self.vars["skip_figure_validation"].get()),
            "enable_pdf_review": bool(self.vars["enable_pdf_review"].get()),
            "disable_pdf_review": bool(self.vars["disable_pdf_review"].get()),
            "enable_ideation": bool(self.vars["enable_ideation"].get()),
            "skip_ideation": bool(self.vars["skip_ideation"].get()),
            "specify_idea": self.vars["specify_idea"].get().strip() or None,
            "num_ideas": int(self.vars["num_ideas"].get()),
            "disable_content_protection": bool(self.vars["disable_content_protection"].get()),
            "auto_approve_changes": bool(self.vars["auto_approve_changes"].get()),
            "content_protection_threshold": float(self.vars["content_protection_threshold"].get()),
            "output_diffs": bool(self.vars["output_diffs"].get()),
            "no_output_diffs": bool(self.vars["no_output_diffs"].get()),
            "test_scaling": bool(self.vars["test_scaling"].get()),
            "scaling_prompt": self.vars["scaling_prompt"].get().strip() or None,
            "scaling_candidates": self.vars["scaling_candidates"].get().strip(),
            "scaling_timeout": int(self.vars["scaling_timeout"].get()),
            "use_test_time_scaling": bool(self.vars["use_test_time_scaling"].get()),
            "revision_candidates": int(self.vars["revision_candidates"].get()),
            "draft_candidates": int(self.vars["draft_candidates"].get()),
            "user_prompt": self.prompt_text.get("1.0", tk.END).strip(),
        }
        return params

    def _prepare_config(self, params: Dict[str, object]) -> WorkflowConfig:
        config_path = params.get("config_path")
        if config_path:
            config = WorkflowConfig.from_file(Path(config_path))
            self.log_queue.put(f"Loaded configuration from {config_path}.\n")
        else:
            config = WorkflowConfig()
            self.log_queue.put("Using default workflow configuration.\n")

        config.enable_pdf_review = bool(params["enable_pdf_review"]) and not bool(params["disable_pdf_review"])
        config.reference_validation = bool(params["check_references"]) and not bool(params["skip_reference_check"])
        config.figure_validation = bool(params["validate_figures"]) and not bool(params["skip_figure_validation"])
        config.research_ideation = bool(params["enable_ideation"]) and not bool(params["skip_ideation"])
        config.diff_output_tracking = bool(params["output_diffs"]) and not bool(params["no_output_diffs"])
        config.content_protection = not bool(params["disable_content_protection"])
        config.auto_approve_changes = bool(params["auto_approve_changes"])
        config.content_protection_threshold = float(params["content_protection_threshold"])
        config.no_early_stopping = bool(params["no_early_stopping"])
        config.use_test_time_scaling = bool(params["use_test_time_scaling"])
        config.revision_candidates = int(params["revision_candidates"])
        config.initial_draft_candidates = int(params["draft_candidates"])
        config.quality_threshold = float(params["quality_threshold"])
        config.max_iterations = int(params["max_iterations"])
        config.request_timeout = int(params["request_timeout"])
        config.max_retries = int(params["max_retries"])
        return config

    def _run_workflow_thread(self, params: Dict[str, object]) -> None:
        log_handler = QueueLogHandler(self.log_queue)
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger = logging.getLogger("sciresearch_workflow")
        logger.addHandler(log_handler)

        queue_writer = QueueWriter(self.log_queue)

        self.worker_result = "completed"
        try:
            with contextlib.redirect_stdout(queue_writer), contextlib.redirect_stderr(queue_writer):
                config = self._prepare_config(params)

                save_path = params.get("save_config_path")
                if save_path:
                    config.save_to_file(Path(save_path))
                    self.log_queue.put(f"Configuration saved to {save_path}.\n")
                    self.worker_result = "completed"
                    return

                if params.get("test_scaling"):
                    candidate_text = params.get("scaling_candidates", "") or ""
                    try:
                        candidates = [int(value.strip()) for value in str(candidate_text).split(",") if value.strip()]
                    except ValueError as exc:
                        raise ValueError(f"Invalid scaling candidates: {candidate_text}") from exc
                    if not candidates:
                        raise ValueError("At least one scaling candidate is required when test scaling is enabled.")

                    self.log_queue.put("Running test-time compute scaling analysis...\n")
                    result = test_time_compute_scaling(
                        model=str(params["model"] or DEFAULT_MODEL),
                        test_prompt=params.get("scaling_prompt"),
                        candidate_counts=candidates,
                        timeout_base=int(params["scaling_timeout"]),
                    )
                    if result:
                        self.log_queue.put("Test-time compute scaling completed successfully.\n")
                    else:
                        self.worker_result = "failed"
                        self.log_queue.put("Test-time compute scaling failed.\n")
                    return

                output_dir = Path(str(params["output_dir"]))
                user_prompt = str(params.get("user_prompt", ""))

                result_dir = run_workflow(
                    topic=str(params["topic"] or ""),
                    field=str(params["field"] or ""),
                    question=str(params["question"] or ""),
                    output_dir=output_dir,
                    model=str(params["model"] or DEFAULT_MODEL),
                    request_timeout=(None if int(params["request_timeout"]) == 0 else int(params["request_timeout"])),
                    max_retries=int(params["max_retries"]),
                    max_iterations=int(params["max_iterations"]),
                    modify_existing=bool(params["modify_existing"]),
                    strict_singletons=bool(params["strict_singletons"]),
                    python_exec=params.get("python_exec"),
                    quality_threshold=float(params["quality_threshold"]),
                    check_references=config.reference_validation,
                    validate_figures=config.figure_validation,
                    user_prompt=user_prompt,
                    config=config,
                    enable_ideation=config.research_ideation,
                    specify_idea=params.get("specify_idea"),
                    num_ideas=int(params["num_ideas"]),
                    output_diffs=config.diff_output_tracking,
                    document_type=str(params["document_type"]),
                    cancel_event=self.cancel_event,
                )
                self.log_queue.put(f"Workflow completed. Results stored in: {result_dir}\n")
        except WorkflowCancelled as exc:
            self.worker_result = "cancelled"
            self.log_queue.put(f"Workflow cancelled: {exc}\n")
        except Exception as exc:
            self.worker_result = "failed"
            self.log_queue.put(f"Error: {exc}\n")
        finally:
            logger.removeHandler(log_handler)
            self.log_queue.put(self.SENTINEL_DONE)

    def _poll_queue(self) -> None:
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message == self.SENTINEL_DONE:
                    self._on_worker_done()
                else:
                    self._append_log(message)
        except queue.Empty:
            pass
        finally:
            self.after(self.POLL_INTERVAL_MS, self._poll_queue)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _on_worker_done(self) -> None:
        self.running = False
        if self.worker_result == "completed":
            self.status_var.set("Completed")
        elif self.worker_result == "failed":
            self.status_var.set("Failed - see log")
        elif self.worker_result == "cancelled":
            self.status_var.set("Cancelled")
        else:
            self.status_var.set("Idle")
        self.cancel_event = None
        self.worker_thread = None
        self._update_run_button_state()

    def _on_close(self) -> None:
        if self.running:
            if not messagebox.askyesno(
                "Workflow running",
                "A workflow run is still in progress. Cancel and exit?",
                parent=self,
            ):
                return
            self.cancel_workflow()
        self.destroy()


def main() -> None:
    app = WorkflowGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
