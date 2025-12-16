"""Enhanced Tkinter GUI for the SciResearch workflow with all features."""
from __future__ import annotations

import contextlib
import logging
import queue
import threading
import webbrowser
import os
from io import TextIOBase
from pathlib import Path
from typing import Dict, Optional, List, Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from core.config import WorkflowConfig
from core.openai_connection import (
    OpenAIConnectionError,
    get_shared_connection_manager,
)
from core.yunwu_connection import (
    configure_yunwu,
    is_yunwu_enabled,
    YunwuConnectionError,
)
from document_types import get_available_document_types
from sciresearch_workflow import (
    DEFAULT_MODEL,
    WorkflowCancelled,
)
from workflow_wrapper import run_from_gui


class QueueWriter(TextIOBase):
    """File-like object that forwards writes into a queue."""

    def __init__(self, output_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self._queue = output_queue

    def write(self, message: str) -> int:
        if message:
            self._queue.put(message)
        return len(message)

    def flush(self) -> None:
        return None


class QueueLogHandler(logging.Handler):
    """Logging handler that forwards formatted records into a queue."""

    def __init__(self, output_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self._queue = output_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self._queue.put(msg + "\n")


class EnhancedWorkflowGUI(tk.Tk):
    """Enhanced GUI with API management, review options, chat window, and error log."""

    POLL_INTERVAL_MS = 100
    SENTINEL_DONE = "__THREAD_DONE__"

    def __init__(self) -> None:
        super().__init__()
        self.title("AI Scientist - Enhanced Research Workflow")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        # Queues and threading
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.error_queue: "queue.Queue[str]" = queue.Queue()
        self.cancel_event: Optional[threading.Event] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.worker_result: str = "idle"
        self.running = False

        # Status
        self.status_var = tk.StringVar(value="Idle")
        self.vars: Dict[str, tk.Variable] = {}

        # API connections
        self.connection_manager = get_shared_connection_manager()
        self.openai_status_var = tk.StringVar(value="Status: Disconnected")
        self.openai_details_var = tk.StringVar(value="Connect your API key to run the workflow.")
        self._openai_status = "disconnected"
        
        # Yunwu API
        self.yunwu_enabled = tk.BooleanVar(value=False)
        self.yunwu_status_var = tk.StringVar(value="Not configured")

        # Build UI
        self._build_ui()
        self._bind_shortcuts()
        self._refresh_openai_status()
        self._update_run_button_state()
        self._poll_queue()

    def _build_ui(self) -> None:
        """Build the main UI with all components."""
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Create notebook for different sections
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Workflow Configuration
        workflow_frame = ttk.Frame(notebook)
        notebook.add(workflow_frame, text="Workflow")
        self._build_workflow_tab(workflow_frame)

        # Tab 2: Review & Revision Options
        review_frame = ttk.Frame(notebook)
        notebook.add(review_frame, text="Review Options")
        self._build_review_tab(review_frame)

        # Tab 3: Chat with LLM
        chat_frame = ttk.Frame(notebook)
        notebook.add(chat_frame, text="Chat")
        self._build_chat_tab(chat_frame)

        # Tab 4: API Configuration
        api_frame = ttk.Frame(notebook)
        notebook.add(api_frame, text="API Config")
        self._build_api_tab(api_frame)

        # Bottom status bar and controls
        self._build_status_bar()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_workflow_tab(self, parent: ttk.Frame) -> None:
        """Build the main workflow configuration tab."""
        # Create scrollable canvas
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main = ttk.Frame(scrollable_frame, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # Project Details
        project_frame = ttk.LabelFrame(main, text="Project Details", padding=8)
        project_frame.pack(fill=tk.X, expand=False, pady=(0, 12))
        self._build_project_section(project_frame)

        # Execution Settings
        exec_frame = ttk.LabelFrame(main, text="Execution Settings", padding=8)
        exec_frame.pack(fill=tk.X, expand=False, pady=(0, 12))
        self._build_execution_section(exec_frame)

        # Quality & Validation
        quality_frame = ttk.LabelFrame(main, text="Quality & Validation", padding=8)
        quality_frame.pack(fill=tk.X, expand=False, pady=(0, 12))
        self._build_quality_section(quality_frame)

        # Advanced Options
        advanced_frame = ttk.LabelFrame(main, text="Advanced Options", padding=8)
        advanced_frame.pack(fill=tk.X, expand=False, pady=(0, 12))
        self._build_advanced_section(advanced_frame)

        # Custom Prompt
        prompt_frame = ttk.LabelFrame(main, text="Custom User Prompt", padding=8)
        prompt_frame.pack(fill=tk.X, expand=False, pady=(0, 12))
        self.prompt_text = ScrolledText(prompt_frame, height=4, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        # Workflow Output Log
        log_frame = ttk.LabelFrame(main, text="Workflow Output", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.log_text = ScrolledText(log_frame, state=tk.DISABLED, wrap=tk.WORD, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_review_tab(self, parent: ttk.Frame) -> None:
        """Build the review options configuration tab."""
        main = ttk.Frame(parent, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # Review Items Selection
        items_frame = ttk.LabelFrame(main, text="Review Items to Check", padding=8)
        items_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        info_label = ttk.Label(
            items_frame,
            text="Select which aspects the reviewer should focus on. If nothing is selected, a general review will be performed.",
            wraplength=700,
            foreground="#555555"
        )
        info_label.pack(anchor=tk.W, pady=(0, 10))

        # Create checkboxes for review items
        self.review_items = {}
        review_options = [
            ("check_structure", "Paper Structure & Organization"),
            ("check_content", "Content Quality & Depth"),
            ("check_methodology", "Methodology & Approach"),
            ("check_results", "Results & Analysis"),
            ("check_references", "References & Citations"),
            ("check_figures", "Figures & Tables"),
            ("check_writing", "Writing Quality & Clarity"),
            ("check_novelty", "Novelty & Contribution"),
            ("check_reproducibility", "Reproducibility"),
            ("check_statistical_rigor", "Statistical Rigor"),
        ]

        # Create a frame for the grid layout
        checkbox_container = ttk.Frame(items_frame)
        checkbox_container.pack(fill=tk.BOTH, expand=True, pady=5)

        for i, (key, label) in enumerate(review_options):
            var = tk.BooleanVar(value=False)
            self.review_items[key] = var
            check = ttk.Checkbutton(checkbox_container, text=label, variable=var)
            check.grid(row=i // 2, column=i % 2, sticky=tk.W, padx=10, pady=5)

        checkbox_container.columnconfigure(0, weight=1)
        checkbox_container.columnconfigure(1, weight=1)

        # Review/Revision Mode
        mode_frame = ttk.LabelFrame(main, text="Review/Revision Execution Mode", padding=8)
        mode_frame.pack(fill=tk.X, expand=False, pady=(0, 12))

        self.review_mode = tk.StringVar(value="combined")
        
        ttk.Radiobutton(
            mode_frame,
            text="Combined (Single API Call) - Review and revision in one message",
            variable=self.review_mode,
            value="combined"
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Radiobutton(
            mode_frame,
            text="Separated (Two API Calls) - Review first, then revision in separate calls",
            variable=self.review_mode,
            value="separated"
        ).pack(anchor=tk.W, pady=5)

        # Review Prompt Customization
        custom_frame = ttk.LabelFrame(main, text="Custom Review Instructions", padding=8)
        custom_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            custom_frame,
            text="Add specific instructions for the reviewer (optional):",
            foreground="#555555"
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.review_custom_text = ScrolledText(custom_frame, height=6, wrap=tk.WORD)
        self.review_custom_text.pack(fill=tk.BOTH, expand=True)

    def _build_chat_tab(self, parent: ttk.Frame) -> None:
        """Build the chat interface tab."""
        main = ttk.Frame(parent, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # Chat history
        history_frame = ttk.LabelFrame(main, text="Chat History", padding=8)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        
        self.chat_history = ScrolledText(history_frame, state=tk.DISABLED, wrap=tk.WORD)
        self.chat_history.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for styling
        self.chat_history.tag_config("user", foreground="blue")
        self.chat_history.tag_config("assistant", foreground="green")
        self.chat_history.tag_config("system", foreground="gray")

        # Chat input
        input_frame = ttk.Frame(main)
        input_frame.pack(fill=tk.X, expand=False)

        ttk.Label(input_frame, text="Your message:").pack(anchor=tk.W)
        
        self.chat_input = ScrolledText(input_frame, height=4, wrap=tk.WORD)
        self.chat_input.pack(fill=tk.BOTH, expand=True, pady=5)

        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X)

        self.chat_send_button = ttk.Button(button_frame, text="Send", command=self._send_chat_message)
        self.chat_send_button.pack(side=tk.RIGHT)

        ttk.Button(button_frame, text="Clear History", command=self._clear_chat_history).pack(side=tk.RIGHT, padx=5)

        # Model selection for chat
        model_frame = ttk.Frame(button_frame)
        model_frame.pack(side=tk.LEFT)
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.chat_model_var = tk.StringVar(value=DEFAULT_MODEL)
        chat_model_entry = ttk.Entry(model_frame, textvariable=self.chat_model_var, width=20)
        chat_model_entry.pack(side=tk.LEFT)

    def _build_api_tab(self, parent: ttk.Frame) -> None:
        """Build the API configuration tab."""
        main = ttk.Frame(parent, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # OpenAI Configuration
        openai_frame = ttk.LabelFrame(main, text="OpenAI API Configuration", padding=8)
        openai_frame.pack(fill=tk.X, expand=False, pady=(0, 12))
        self._build_openai_section(openai_frame)

        # Yunwu API Configuration
        yunwu_frame = ttk.LabelFrame(main, text="Yunwu API Configuration (OpenAI-compatible)", padding=8)
        yunwu_frame.pack(fill=tk.X, expand=False, pady=(0, 12))
        self._build_yunwu_section(yunwu_frame)

        # Environment Variables
        env_frame = ttk.LabelFrame(main, text="Environment Variables", padding=8)
        env_frame.pack(fill=tk.X, expand=False, pady=(0, 12))
        self._build_env_section(env_frame)

    def _build_openai_section(self, parent: ttk.Frame) -> None:
        """Build OpenAI API configuration section."""
        self.openai_status_label = ttk.Label(parent, textvariable=self.openai_status_var)
        self.openai_status_label.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        self.openai_details_label = ttk.Label(
            parent,
            textvariable=self.openai_details_var,
            wraplength=500,
            foreground="#555555"
        )
        self.openai_details_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 8))

        connect_btn = ttk.Button(parent, text="Connect OpenAI…", command=self._open_openai_dialog)
        connect_btn.grid(row=2, column=0, sticky=tk.W)

        self.openai_disconnect_button = ttk.Button(parent, text="Disconnect", command=self._disconnect_openai)
        self.openai_disconnect_button.grid(row=2, column=1, sticky=tk.W, padx=(8, 0))

        ttk.Label(
            parent,
            text="API key will be validated and stored encrypted. You can disconnect anytime.",
            foreground="#555555"
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))

    def _build_yunwu_section(self, parent: ttk.Frame) -> None:
        """Build Yunwu API configuration section."""
        ttk.Label(parent, text="Status:").grid(row=0, column=0, sticky=tk.W)
        status_label = ttk.Label(parent, textvariable=self.yunwu_status_var, foreground="#555555")
        status_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))

        ttk.Checkbutton(
            parent,
            text="Enable Yunwu API (OpenAI-compatible endpoint)",
            variable=self.yunwu_enabled,
            command=self._on_yunwu_toggle
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 10))

        ttk.Label(parent, text="API Key:").grid(row=2, column=0, sticky=tk.W)
        self.yunwu_key_var = tk.StringVar(value=os.getenv("YUNWU_API_KEY", ""))
        yunwu_key_entry = ttk.Entry(parent, textvariable=self.yunwu_key_var, show="*", width=40)
        yunwu_key_entry.grid(row=2, column=1, sticky="ew", padx=(5, 0))

        ttk.Label(parent, text="Base URL:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.yunwu_base_var = tk.StringVar(value=os.getenv("YUNWU_API_BASE", "https://yunwu.ai/v1"))
        yunwu_base_entry = ttk.Entry(parent, textvariable=self.yunwu_base_var, width=40)
        yunwu_base_entry.grid(row=3, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))

        ttk.Button(parent, text="Test Connection", command=self._test_yunwu_connection).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 0)
        )

        parent.columnconfigure(1, weight=1)

    def _build_env_section(self, parent: ttk.Frame) -> None:
        """Build environment variables section."""
        info_text = (
            "The GUI will automatically use these environment variables if set:\n"
            "• OPENAI_API_KEY - OpenAI API key\n"
            "• YUNWU_API_KEY - Yunwu API key\n"
            "• YUNWU_API_BASE - Yunwu API base URL\n"
            "• SCI_MODEL - Default model to use\n\n"
            "You can override them using the fields in this GUI."
        )
        
        info_label = ttk.Label(parent, text=info_text, foreground="#555555", justify=tk.LEFT)
        info_label.pack(anchor=tk.W)

        # Show current environment variables
        current_frame = ttk.Frame(parent)
        current_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.env_text = ScrolledText(current_frame, height=8, state=tk.DISABLED, wrap=tk.WORD)
        self.env_text.pack(fill=tk.BOTH, expand=True)

        ttk.Button(parent, text="Refresh Environment Info", command=self._refresh_env_info).pack(
            anchor=tk.W, pady=(10, 0)
        )

        self._refresh_env_info()

    def _build_project_section(self, parent: ttk.Frame) -> None:
        """Build project details section."""
        self._add_entry(parent, "Topic", "topic", row=0)
        self._add_entry(parent, "Field", "field", row=1)
        self._add_entry(parent, "Research Question", "question", row=2)

        ttk.Label(parent, text="Document Type").grid(row=3, column=0, sticky=tk.W, pady=4)
        doc_var = tk.StringVar(value="auto")
        self.vars["document_type"] = doc_var
        doc_combo = ttk.Combobox(parent, textvariable=doc_var, values=get_available_document_types(), state="readonly")
        doc_combo.grid(row=3, column=1, sticky="ew", pady=4)

        self._add_entry(parent, "Model", "model", default=DEFAULT_MODEL, row=4)

        ttk.Label(parent, text="Output Directory").grid(row=5, column=0, sticky=tk.W, pady=4)
        out_var = tk.StringVar()
        self.vars["output_dir"] = out_var
        out_entry = ttk.Entry(parent, textvariable=out_var)
        out_entry.grid(row=5, column=1, sticky="ew", pady=4)
        browse_btn = ttk.Button(parent, text="Browse", command=lambda: self._browse_directory(out_var))
        browse_btn.grid(row=5, column=2, sticky=tk.W, padx=(6, 0))
        out_var.trace_add("write", lambda *_: self._update_run_button_state())

        parent.columnconfigure(1, weight=1)

    def _build_execution_section(self, parent: ttk.Frame) -> None:
        """Build execution settings section."""
        self._add_spinbox(parent, "Request Timeout (s)", "request_timeout", default=3600, from_=0, to=99999, row=0)
        self._add_spinbox(parent, "Max Retries", "max_retries", default=3, from_=0, to=20, row=1)
        self._add_spinbox(parent, "Max Iterations", "max_iterations", default=4, from_=1, to=20, row=2)
        self._add_check(parent, "Modify Existing Project", "modify_existing", default=False, row=3)
        self._add_check(parent, "Enforce Single Files", "strict_singletons", default=True, row=4)
        self._add_check(parent, "Disable Blueprint Planning", "disable_blueprint_planning", default=False, row=5)
        self._add_entry(parent, "Python Executable", "python_exec", row=6)

        ttk.Label(parent, text="Config File").grid(row=7, column=0, sticky=tk.W, pady=4)
        config_var = tk.StringVar()
        self.vars["config_path"] = config_var
        config_entry = ttk.Entry(parent, textvariable=config_var)
        config_entry.grid(row=7, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", command=lambda: self._browse_file(config_var)).grid(
            row=7, column=2, sticky=tk.W, padx=(6, 0)
        )

        parent.columnconfigure(1, weight=1)

    def _build_quality_section(self, parent: ttk.Frame) -> None:
        """Build quality & validation section."""
        self._add_spinbox(parent, "Quality Threshold", "quality_threshold", default=1.0, from_=0.0, to=1.0, increment=0.05, row=0, var_type="double")
        self._add_check(parent, "Check References", "check_references", default=True, row=1)
        self._add_check(parent, "Skip Reference Check", "skip_reference_check", default=False, row=2)
        self._add_check(parent, "Validate Figures", "validate_figures", default=True, row=3)
        self._add_check(parent, "Skip Figure Validation", "skip_figure_validation", default=False, row=4)
        self._add_check(parent, "Enable PDF Review", "enable_pdf_review", default=False, row=5)
        self._add_check(parent, "Enable Ideation", "enable_ideation", default=True, row=6)
        self._add_check(parent, "Skip Ideation", "skip_ideation", default=False, row=7)
        self._add_entry(parent, "Specify Idea", "specify_idea", row=8)
        self._add_spinbox(parent, "Number of Ideas", "num_ideas", default=15, from_=1, to=50, row=9)

    def _build_advanced_section(self, parent: ttk.Frame) -> None:
        """Build advanced options section."""
        self._add_check(parent, "Disable Content Protection", "disable_content_protection", default=False, row=0)
        self._add_check(parent, "Auto-Approve Changes", "auto_approve_changes", default=False, row=1)
        self._add_spinbox(parent, "Content Protection Threshold", "content_protection_threshold", default=0.15, from_=0.0, to=1.0, increment=0.01, row=2, var_type="double")
        self._add_check(parent, "Save Output Diffs", "output_diffs", default=True, row=3)
        self._add_check(parent, "Enable Test-Time Scaling", "use_test_time_scaling", default=False, row=4)
        self._add_spinbox(parent, "Revision Candidates", "revision_candidates", default=3, from_=1, to=10, row=5)
        self._add_spinbox(parent, "Draft Candidates", "draft_candidates", default=1, from_=1, to=5, row=6)
        self._add_check(parent, "All-Code Mode", "all_code", default=False, row=7)
        self._add_entry(parent, "Code Output Dir", "code_output_dir", default="code", row=8)
        self._add_check(parent, "Science-Only Mode", "science_only", default=False, row=9)

    def _build_status_bar(self) -> None:
        """Build status bar and control buttons."""
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, expand=False, padx=12, pady=(0, 12))

        # Left side - status and logs
        left_frame = ttk.Frame(status_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(left_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)

        # Error log button
        self.error_log_button = ttk.Button(left_frame, text="Show Error Log", command=self._show_error_log)
        self.error_log_button.pack(side=tk.LEFT, padx=(20, 0))

        # Right side - control buttons
        right_frame = ttk.Frame(status_frame)
        right_frame.pack(side=tk.RIGHT)

        self.cancel_button = ttk.Button(right_frame, text="Cancel", command=self.cancel_workflow, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.RIGHT, padx=(0, 8))

        self.run_button = ttk.Button(right_frame, text="Run Workflow", command=self.start_workflow)
        self.run_button.pack(side=tk.RIGHT)

    # -------------------------------------------------------------------------
    # Helper methods for building UI components
    # -------------------------------------------------------------------------

    def _add_entry(self, frame: ttk.Frame, label: str, key: str, row: int, default: str = "") -> None:
        """Add an entry field to a frame."""
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
        """Add a spinbox to a frame."""
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
        if var_type == "double":
            var = tk.DoubleVar(value=default)
        else:
            var = tk.IntVar(value=int(default))
        self.vars[key] = var
        spin = ttk.Spinbox(frame, textvariable=var, from_=from_, to=to, increment=increment)
        spin.grid(row=row, column=1, sticky="ew", pady=4)
        frame.columnconfigure(1, weight=1)

    def _add_check(self, frame: ttk.Frame, label: str, key: str, default: bool, row: int) -> None:
        """Add a checkbox to a frame."""
        var = tk.BooleanVar(value=default)
        self.vars[key] = var
        check = ttk.Checkbutton(frame, text=label, variable=var)
        check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=4)

    # -------------------------------------------------------------------------
    # API Connection Methods
    # -------------------------------------------------------------------------

    def _open_openai_dialog(self) -> None:
        """Open dialog to connect OpenAI API."""
        dialog = tk.Toplevel(self)
        dialog.title("Connect OpenAI")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(
            dialog,
            text="Paste your OpenAI API key. We validate it and store it encrypted.",
            wraplength=360,
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(12, 8))

        # Check for environment variable
        env_key = os.getenv("OPENAI_API_KEY", "")
        if env_key:
            ttk.Label(
                dialog,
                text=f"Found in environment: {env_key[:8]}...{env_key[-4:]}",
                foreground="green"
            ).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(0, 8))

        key_var = tk.StringVar(value=env_key)
        ttk.Label(dialog, text="OpenAI API key").grid(row=2, column=0, sticky=tk.W, padx=12)
        entry = ttk.Entry(dialog, textvariable=key_var, show="*", width=50)
        entry.grid(row=2, column=1, sticky="ew", padx=12)
        dialog.columnconfigure(1, weight=1)

        def open_docs(*_: object) -> None:
            webbrowser.open_new_tab("https://platform.openai.com/account/api-keys")

        link = ttk.Label(dialog, text="Create or manage keys", foreground="#1a73e8", cursor="hand2")
        link.grid(row=3, column=1, sticky=tk.W, padx=12, pady=(0, 8))
        link.bind("<Button-1>", open_docs)

        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, padx=12, pady=(0, 12))

        def submit() -> None:
            api_key = key_var.get().strip()
            if not api_key:
                messagebox.showerror("Missing key", "Enter your OpenAI API key.", parent=dialog)
                return
            try:
                self.connection_manager.connect(api_key)
            except OpenAIConnectionError as exc:
                messagebox.showerror("Connection failed", str(exc), parent=dialog)
                return
            messagebox.showinfo("Connected", "OpenAI API key validated and stored.", parent=dialog)
            dialog.destroy()
            self._refresh_openai_status()

        def cancel() -> None:
            dialog.destroy()

        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Connect", command=submit).pack(side=tk.LEFT, padx=5)

        entry.focus_set()
        dialog.bind("<Return>", lambda event: submit())
        dialog.bind("<Escape>", lambda event: cancel())

    def _disconnect_openai(self) -> None:
        """Disconnect OpenAI API."""
        if not self.connection_manager.is_connected():
            return
        if not messagebox.askyesno("Disconnect OpenAI", "Remove the stored API key?", parent=self):
            return
        try:
            self.connection_manager.disconnect()
        except OpenAIConnectionError as exc:
            messagebox.showerror("Error", str(exc), parent=self)
            return
        messagebox.showinfo("Disconnected", "OpenAI API key removed.", parent=self)
        self._refresh_openai_status()

    def _refresh_openai_status(self) -> None:
        """Refresh OpenAI connection status display."""
        status = self.connection_manager.get_status()
        status_value = status.get("status", "disconnected")
        self._openai_status = status_value

        if status_value == "disconnected":
            self.openai_status_var.set("Status: Disconnected")
            self.openai_details_var.set("Connect your OpenAI API key to run the workflow.")
            disconnect_state = tk.DISABLED
        else:
            display_status = status_value.replace("_", " ").title()
            last4 = status.get("last4") or "----"
            self.openai_status_var.set(f"Status: {display_status} (••••{last4})")
            details = []
            if created := status.get("created_at"):
                details.append(f"Added: {created}")
            details.append(f"Last used: {status.get('last_used') or 'Never'}")
            if model_count := status.get("model_count"):
                details.append(f"Models: {model_count}")
            self.openai_details_var.set(" • ".join(details))
            disconnect_state = tk.NORMAL

        if self.openai_disconnect_button:
            self.openai_disconnect_button.configure(state=disconnect_state)
        self._update_run_button_state()

    def _on_yunwu_toggle(self) -> None:
        """Handle Yunwu API toggle."""
        if self.yunwu_enabled.get():
            self._test_yunwu_connection()
        else:
            self.yunwu_status_var.set("Disabled")

    def _test_yunwu_connection(self) -> None:
        """Test Yunwu API connection."""
        if not self.yunwu_enabled.get():
            return

        api_key = self.yunwu_key_var.get().strip()
        base_url = self.yunwu_base_var.get().strip()

        if not api_key:
            messagebox.showerror("Missing Key", "Enter Yunwu API key first.", parent=self)
            self.yunwu_enabled.set(False)
            return

        try:
            configure_yunwu(api_key=api_key, base_url=base_url)
            self.yunwu_status_var.set("Connected and Ready")
            messagebox.showinfo("Success", "Yunwu API connection successful!", parent=self)
        except YunwuConnectionError as exc:
            self.yunwu_status_var.set("Connection Failed")
            messagebox.showerror("Connection Failed", str(exc), parent=self)
            self.yunwu_enabled.set(False)

    def _refresh_env_info(self) -> None:
        """Refresh environment variables display."""
        env_vars = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "Not set"),
            "YUNWU_API_KEY": os.getenv("YUNWU_API_KEY", "Not set"),
            "YUNWU_API_BASE": os.getenv("YUNWU_API_BASE", "Not set"),
            "SCI_MODEL": os.getenv("SCI_MODEL", "Not set"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", "Not set"),
        }

        text = "Current Environment Variables:\n" + "="*60 + "\n\n"
        for key, value in env_vars.items():
            if value != "Not set" and "KEY" in key:
                # Mask API keys
                masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                text += f"{key}: {masked}\n"
            else:
                text += f"{key}: {value}\n"

        self.env_text.configure(state=tk.NORMAL)
        self.env_text.delete("1.0", tk.END)
        self.env_text.insert("1.0", text)
        self.env_text.configure(state=tk.DISABLED)

    # -------------------------------------------------------------------------
    # Chat Methods
    # -------------------------------------------------------------------------

    def _send_chat_message(self) -> None:
        """Send a message in the chat."""
        message = self.chat_input.get("1.0", tk.END).strip()
        if not message:
            return

        # Add user message to history
        self._append_chat("user", f"You: {message}\n\n")
        self.chat_input.delete("1.0", tk.END)

        # Disable send button while processing
        self.chat_send_button.configure(state=tk.DISABLED)

        # Run chat in background thread
        def chat_thread():
            try:
                from sciresearch_workflow import _universal_chat
                
                messages = [{"role": "user", "content": message}]
                model = self.chat_model_var.get() or DEFAULT_MODEL
                
                response = _universal_chat(
                    messages,
                    model=model,
                    request_timeout=300,
                    prompt_type="chat",
                )
                
                self._append_chat("assistant", f"Assistant: {response}\n\n")
            except Exception as e:
                self._append_chat("system", f"Error: {str(e)}\n\n")
                self.error_queue.put(f"Chat error: {str(e)}\n")
            finally:
                self.chat_send_button.configure(state=tk.NORMAL)

        threading.Thread(target=chat_thread, daemon=True).start()

    def _append_chat(self, role: str, text: str) -> None:
        """Append text to chat history."""
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.insert(tk.END, text, role)
        self.chat_history.see(tk.END)
        self.chat_history.configure(state=tk.DISABLED)

    def _clear_chat_history(self) -> None:
        """Clear chat history."""
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.delete("1.0", tk.END)
        self.chat_history.configure(state=tk.DISABLED)

    # -------------------------------------------------------------------------
    # Error Log Methods
    # -------------------------------------------------------------------------

    def _show_error_log(self) -> None:
        """Show error log window."""
        error_window = tk.Toplevel(self)
        error_window.title("Error Log")
        error_window.geometry("800x600")

        error_text = ScrolledText(error_window, state=tk.DISABLED, wrap=tk.WORD)
        error_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Get all errors from queue
        errors = []
        try:
            while True:
                error = self.error_queue.get_nowait()
                errors.append(error)
        except queue.Empty:
            pass

        # Re-add errors to queue
        for error in errors:
            self.error_queue.put(error)

        # Display errors
        error_text.configure(state=tk.NORMAL)
        if errors:
            error_text.insert("1.0", "".join(errors))
        else:
            error_text.insert("1.0", "No errors recorded.")
        error_text.configure(state=tk.DISABLED)

        ttk.Button(error_window, text="Clear Errors", command=lambda: self._clear_error_log(error_text)).pack(pady=10)
        ttk.Button(error_window, text="Close", command=error_window.destroy).pack(pady=10)

    def _clear_error_log(self, error_text: ScrolledText) -> None:
        """Clear error log."""
        # Clear queue
        try:
            while True:
                self.error_queue.get_nowait()
        except queue.Empty:
            pass

        # Clear display
        error_text.configure(state=tk.NORMAL)
        error_text.delete("1.0", tk.END)
        error_text.insert("1.0", "Error log cleared.")
        error_text.configure(state=tk.DISABLED)

    # -------------------------------------------------------------------------
    # Workflow Execution Methods
    # -------------------------------------------------------------------------

    def _browse_directory(self, var: tk.StringVar) -> None:
        """Browse for directory."""
        initial_dir = var.get() or str(Path.cwd())
        directory = filedialog.askdirectory(parent=self, initialdir=initial_dir)
        if directory:
            var.set(directory)

    def _browse_file(self, var: tk.StringVar) -> None:
        """Browse for file."""
        initial_dir = Path(var.get()).parent if var.get() else Path.cwd()
        file_path = filedialog.askopenfilename(parent=self, initialdir=initial_dir)
        if file_path:
            var.set(file_path)

    def _bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts."""
        self.bind("<Control-Return>", lambda *_: self.start_workflow())
        self.bind("<Escape>", lambda *_: self.cancel_workflow() if self.running else None)

    def _update_run_button_state(self) -> None:
        """Update run button state based on configuration."""
        output_dir = self.vars.get("output_dir")
        topic = self.vars.get("topic")
        field = self.vars.get("field")
        question = self.vars.get("question")
        modify_existing = bool(self.vars.get("modify_existing", tk.BooleanVar()).get())

        required_ready = bool(output_dir and output_dir.get().strip())
        if not modify_existing:
            required_ready = required_ready and all(
                var is not None and bool(var.get().strip()) for var in (topic, field, question)
            )

        # Check API connection
        api_ready = self._openai_status == "valid" or self.yunwu_enabled.get()

        state = tk.NORMAL if required_ready and api_ready and not self.running else tk.DISABLED
        self.run_button.configure(state=state)
        self.cancel_button.configure(state=tk.NORMAL if self.running else tk.DISABLED)

    def start_workflow(self) -> None:
        """Start the workflow."""
        if self.running:
            return
        if not self._validate_inputs():
            return

        self.running = True
        self.worker_result = "running"
        self.status_var.set("Running...")
        self.cancel_event = threading.Event()
        
        # Clear logs
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

        params = self._gather_parameters()
        self.worker_thread = threading.Thread(target=self._run_workflow_thread, args=(params,), daemon=True)
        self.worker_thread.start()
        self._update_run_button_state()

    def _validate_inputs(self) -> bool:
        """Validate inputs before starting workflow."""
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
                "Please provide:\n- " + "\n- ".join(missing),
                parent=self,
            )
            return False
        
        # Check API connection
        if self._openai_status != "valid" and not self.yunwu_enabled.get():
            messagebox.showerror(
                "No API Connection",
                "Please connect OpenAI or enable Yunwu API before running.",
                parent=self,
            )
            return False
        
        return True

    def cancel_workflow(self) -> None:
        """Cancel the running workflow."""
        if self.cancel_event and not self.cancel_event.is_set():
            self.cancel_event.set()
            self.status_var.set("Cancelling...")
            self.log_queue.put("Cancellation requested...\n")
            self.cancel_button.configure(state=tk.DISABLED)

    def _gather_parameters(self) -> Dict[str, Any]:
        """Gather all parameters from the GUI."""
        # Get selected review items
        selected_review_items = [
            key for key, var in self.review_items.items() if var.get()
        ]
        
        # Get review mode
        review_mode = self.review_mode.get()
        
        # Get custom review instructions
        review_custom = self.review_custom_text.get("1.0", tk.END).strip()

        params: Dict[str, Any] = {
            "topic": self.vars["topic"].get().strip(),
            "field": self.vars["field"].get().strip(),
            "question": self.vars["question"].get().strip(),
            "document_type": self.vars["document_type"].get(),
            "output_dir": self.vars["output_dir"].get().strip(),
            "model": self.vars["model"].get().strip() or DEFAULT_MODEL,
            "request_timeout": int(self.vars["request_timeout"].get()),
            "max_retries": int(self.vars["max_retries"].get()),
            "max_iterations": int(self.vars["max_iterations"].get()),
            "modify_existing": bool(self.vars["modify_existing"].get()),
            "strict_singletons": bool(self.vars["strict_singletons"].get()),
            "disable_blueprint_planning": bool(self.vars["disable_blueprint_planning"].get()),
            "python_exec": self.vars["python_exec"].get().strip() or None,
            "config_path": self.vars["config_path"].get().strip() or None,
            "quality_threshold": float(self.vars["quality_threshold"].get()),
            "check_references": bool(self.vars["check_references"].get()),
            "skip_reference_check": bool(self.vars["skip_reference_check"].get()),
            "validate_figures": bool(self.vars["validate_figures"].get()),
            "skip_figure_validation": bool(self.vars["skip_figure_validation"].get()),
            "enable_pdf_review": bool(self.vars["enable_pdf_review"].get()),
            "enable_ideation": bool(self.vars["enable_ideation"].get()),
            "skip_ideation": bool(self.vars["skip_ideation"].get()),
            "specify_idea": self.vars["specify_idea"].get().strip() or None,
            "num_ideas": int(self.vars["num_ideas"].get()),
            "disable_content_protection": bool(self.vars["disable_content_protection"].get()),
            "auto_approve_changes": bool(self.vars["auto_approve_changes"].get()),
            "content_protection_threshold": float(self.vars["content_protection_threshold"].get()),
            "output_diffs": bool(self.vars["output_diffs"].get()),
            "use_test_time_scaling": bool(self.vars["use_test_time_scaling"].get()),
            "revision_candidates": int(self.vars["revision_candidates"].get()),
            "draft_candidates": int(self.vars["draft_candidates"].get()),
            "all_code": bool(self.vars["all_code"].get()),
            "code_output_dir": self.vars["code_output_dir"].get().strip(),
            "science_only": bool(self.vars["science_only"].get()),
            "user_prompt": self.prompt_text.get("1.0", tk.END).strip(),
            # Review options
            "review_items": selected_review_items,
            "review_mode": review_mode,
            "review_custom_instructions": review_custom,
            # Yunwu API
            "yunwu_enabled": self.yunwu_enabled.get(),
            "yunwu_api_key": self.yunwu_key_var.get().strip() if self.yunwu_enabled.get() else None,
            "yunwu_api_base": self.yunwu_base_var.get().strip() if self.yunwu_enabled.get() else None,
        }
        return params

    def _run_workflow_thread(self, params: Dict[str, Any]) -> None:
        """Run workflow in background thread."""
        log_handler = QueueLogHandler(self.log_queue)
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger = logging.getLogger("sciresearch_workflow")
        logger.addHandler(log_handler)

        queue_writer = QueueWriter(self.log_queue)

        self.worker_result = "completed"
        try:
            # Configure Yunwu if enabled
            if params.get("yunwu_enabled"):
                try:
                    configure_yunwu(
                        api_key=params.get("yunwu_api_key"),
                        base_url=params.get("yunwu_api_base"),
                    )
                except YunwuConnectionError as e:
                    self.error_queue.put(f"Yunwu configuration error: {e}\n")
                    raise

            with contextlib.redirect_stdout(queue_writer), contextlib.redirect_stderr(queue_writer):
                result_dir = run_from_gui(params, cancel_event=self.cancel_event)
                self.log_queue.put(f"\n{'='*80}\n")
                self.log_queue.put(f"✓ Workflow completed successfully!\n")
                self.log_queue.put(f"Results: {result_dir}\n")
                self.log_queue.put(f"{'='*80}\n")
        except WorkflowCancelled as exc:
            self.worker_result = "cancelled"
            self.log_queue.put(f"\nWorkflow cancelled: {exc}\n")
        except Exception as exc:
            self.worker_result = "failed"
            error_msg = f"\n{'!'*80}\nERROR: {exc}\n{'!'*80}\n"
            self.log_queue.put(error_msg)
            self.error_queue.put(error_msg)
            import traceback
            tb = traceback.format_exc()
            self.error_queue.put(tb)
        finally:
            logger.removeHandler(log_handler)
            self.log_queue.put(self.SENTINEL_DONE)

    def _poll_queue(self) -> None:
        """Poll log queue for updates."""
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
        """Append message to log."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _on_worker_done(self) -> None:
        """Handle workflow completion."""
        self.running = False
        if self.worker_result == "completed":
            self.status_var.set("✓ Completed")
        elif self.worker_result == "failed":
            self.status_var.set("✗ Failed - check error log")
        elif self.worker_result == "cancelled":
            self.status_var.set("Cancelled")
        else:
            self.status_var.set("Idle")
        
        self.cancel_event = None
        self.worker_thread = None
        self._update_run_button_state()

    def _on_close(self) -> None:
        """Handle window close."""
        if self.running:
            if not messagebox.askyesno(
                "Workflow running",
                "A workflow is in progress. Cancel and exit?",
                parent=self,
            ):
                return
            self.cancel_workflow()
        self.destroy()


def main() -> None:
    """Main entry point."""
    app = EnhancedWorkflowGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
