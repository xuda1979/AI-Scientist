"""VSCode-style GUI for AI Scientist - Modern, Professional Interface.

Features:
- Auto Mode: Fully autonomous research agent that completes the entire workflow
- Interactive Mode: User can input prompts and control iterations (like VS Code + Copilot)
- Iteration Control: Specify the number of research/revision iterations
- Real-time Progress: See agent progress and outputs in real-time
"""
from __future__ import annotations

import os
import sys
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font
from tkinter.scrolledtext import ScrolledText

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import WorkflowConfig
from core.openai_connection import get_shared_connection_manager, OpenAIConnectionError
from core.yunwu_connection import configure_yunwu, YunwuConnectionError
from document_types import get_available_document_types
from sciresearch_workflow import DEFAULT_MODEL, WorkflowCancelled
from workflow_wrapper import run_from_gui
from ai.chat import chat


# VSCode-like color scheme
COLORS = {
    'bg_dark': '#1e1e1e',
    'bg_medium': '#252526',
    'bg_light': '#2d2d30',
    'bg_hover': '#37373d',
    'fg_primary': '#cccccc',
    'fg_secondary': '#808080',
    'accent_blue': '#007acc',
    'accent_green': '#4ec9b0',
    'accent_yellow': '#dcdcaa',
    'accent_orange': '#ce9178',
    'accent_purple': '#c586c0',
    'border': '#3e3e42',
    'error': '#f48771',
    'success': '#89d185',
    'warning': '#cca700',
}

# Research modes
MODE_AUTO = "auto"
MODE_INTERACTIVE = "interactive"


class QueueWriter:
    """File-like object for redirecting output to queue."""
    def __init__(self, output_queue: queue.Queue) -> None:
        self._queue = output_queue

    def write(self, message: str) -> int:
        if message:
            self._queue.put(message)
        return len(message)

    def flush(self) -> None:
        pass


class FileTreeView(ttk.Frame):
    """VSCode-style file tree view."""
    
    def __init__(self, parent, on_file_select=None, on_folder_select=None):
        super().__init__(parent)
        self.on_file_select = on_file_select
        self.on_folder_select = on_folder_select
        self.current_root = None
        
        # Configure style
        style = ttk.Style()
        style.configure("Treeview", background=COLORS['bg_medium'], 
                       foreground=COLORS['fg_primary'], fieldbackground=COLORS['bg_medium'])
        
        # Toolbar
        toolbar = tk.Frame(self, bg=COLORS['bg_dark'], height=35)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        
        tk.Label(toolbar, text="📁 EXPLORER", bg=COLORS['bg_dark'], 
                fg=COLORS['fg_primary'], font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=10, pady=7)
        
        btn_open = tk.Button(toolbar, text="Open Folder", command=self.open_folder,
                            bg=COLORS['accent_blue'], fg='white', relief=tk.FLAT,
                            font=('Segoe UI', 9), cursor='hand2', padx=10, pady=3)
        btn_open.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Tree view with scrollbar
        tree_frame = tk.Frame(self, bg=COLORS['bg_medium'])
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree = ttk.Treeview(tree_frame, yscrollcommand=scrollbar.set, selectmode='browse')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.tree.yview)
        
        # Configure columns
        self.tree.heading('#0', text='Name', anchor=tk.W)
        
        # Bind events
        self.tree.bind('<<TreeviewOpen>>', self.on_open_node)
        self.tree.bind('<<TreeviewSelect>>', self.on_select)
        self.tree.bind('<Double-1>', self.on_double_click)
        
        # File icons mapping
        self.file_icons = {
            '.py': '🐍',
            '.txt': '📄',
            '.md': '📝',
            '.json': '⚙️',
            '.tex': '📐',
            '.pdf': '📕',
            '.png': '🖼️',
            '.jpg': '🖼️',
            '.csv': '📊',
            'folder': '📁',
            'folder_open': '📂',
        }
    
    def open_folder(self):
        """Open a folder in the file tree."""
        folder = filedialog.askdirectory(title="Select Project Folder")
        if folder:
            self.load_folder(folder)
            if self.on_folder_select:
                self.on_folder_select(folder)
    
    def load_folder(self, folder_path: str):
        """Load folder structure into tree."""
        self.current_root = Path(folder_path)
        self.tree.delete(*self.tree.get_children())
        
        root_id = self.tree.insert('', 'end', text=self.current_root.name, 
                                   values=[str(self.current_root)], open=True)
        self.tree.item(root_id, image='', tags=('folder',))
        
        self.populate_tree(root_id, self.current_root)
    
    def populate_tree(self, parent_id: str, path: Path):
        """Populate tree with folder contents."""
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            
            for item in items:
                # Skip hidden files and common build directories
                if item.name.startswith('.') or item.name in ['__pycache__', 'node_modules', '.git']:
                    continue
                
                icon = self.file_icons.get('folder' if item.is_dir() else item.suffix, '📄')
                display_name = f"{icon} {item.name}"
                
                node_id = self.tree.insert(parent_id, 'end', text=display_name,
                                          values=[str(item)], open=False)
                
                if item.is_dir():
                    # Add dummy child to make folder expandable
                    self.tree.insert(node_id, 'end', text='Loading...')
        except PermissionError:
            pass
    
    def on_open_node(self, event):
        """Handle node expansion."""
        node_id = self.tree.focus()
        if not node_id:
            return
        
        # Check if already populated
        children = self.tree.get_children(node_id)
        if children and self.tree.item(children[0])['text'] == 'Loading...':
            self.tree.delete(children[0])
            path = Path(self.tree.item(node_id)['values'][0])
            self.populate_tree(node_id, path)
    
    def on_select(self, event):
        """Handle selection change."""
        node_id = self.tree.focus()
        if not node_id:
            return
        
        path_str = self.tree.item(node_id)['values'][0]
        path = Path(path_str)
        
        if path.is_file() and self.on_file_select:
            self.on_file_select(path)
    
    def on_double_click(self, event):
        """Handle double-click on file."""
        node_id = self.tree.focus()
        if not node_id:
            return
        
        path_str = self.tree.item(node_id)['values'][0]
        path = Path(path_str)
        
        if path.is_file() and self.on_file_select:
            self.on_file_select(path)


class FileEditor(ttk.Frame):
    """VSCode-style file editor with tabs."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.open_files = {}  # path -> (tab_id, text_widget)
        self.current_file = None
        
        # Tab bar
        self.tab_frame = tk.Frame(self, bg=COLORS['bg_dark'], height=35)
        self.tab_frame.pack(fill=tk.X, side=tk.TOP)
        
        self.tabs_container = tk.Frame(self.tab_frame, bg=COLORS['bg_dark'])
        self.tabs_container.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Editor area
        self.editor_frame = tk.Frame(self, bg=COLORS['bg_medium'])
        self.editor_frame.pack(fill=tk.BOTH, expand=True)
        
        # Welcome message
        self.show_welcome()
    
    def show_welcome(self):
        """Show welcome message when no file is open."""
        for widget in self.editor_frame.winfo_children():
            widget.destroy()
        
        welcome = tk.Frame(self.editor_frame, bg=COLORS['bg_medium'])
        welcome.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(welcome, text="AI Scientist", font=('Segoe UI', 24, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['accent_blue']).pack(pady=(100, 20))
        
        tk.Label(welcome, text="Open a folder to start editing files", font=('Segoe UI', 12),
                bg=COLORS['bg_medium'], fg=COLORS['fg_secondary']).pack()
    
    def open_file(self, file_path: Path):
        """Open a file in the editor."""
        if file_path in self.open_files:
            # File already open, just switch to it
            self.switch_to_file(file_path)
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file:\n{e}")
            return
        
        # Create tab
        tab_btn = tk.Button(self.tabs_container, text=file_path.name,
                           bg=COLORS['bg_light'], fg=COLORS['fg_primary'],
                           relief=tk.FLAT, padx=15, pady=5,
                           font=('Segoe UI', 9), cursor='hand2')
        tab_btn.pack(side=tk.LEFT)
        tab_btn.bind('<Button-1>', lambda e: self.switch_to_file(file_path))
        
        # Create close button on tab
        close_btn = tk.Label(tab_btn, text=" ✕", bg=COLORS['bg_light'],
                            fg=COLORS['fg_secondary'], cursor='hand2')
        close_btn.pack(side=tk.RIGHT)
        close_btn.bind('<Button-1>', lambda e: self.close_file(file_path))
        close_btn.bind('<Enter>', lambda e: close_btn.config(fg=COLORS['error']))
        close_btn.bind('<Leave>', lambda e: close_btn.config(fg=COLORS['fg_secondary']))
        
        # Create editor
        editor = ScrolledText(self.editor_frame, wrap=tk.NONE, undo=True,
                             bg=COLORS['bg_medium'], fg=COLORS['fg_primary'],
                             insertbackground=COLORS['fg_primary'],
                             selectbackground=COLORS['accent_blue'],
                             font=('Consolas', 11))
        editor.insert('1.0', content)
        
        # Store
        self.open_files[file_path] = (tab_btn, editor)
        self.switch_to_file(file_path)
    
    def switch_to_file(self, file_path: Path):
        """Switch to a different file."""
        # Hide all editors
        for widget in self.editor_frame.winfo_children():
            widget.pack_forget()
        
        # Reset all tab colors
        for path, (tab_btn, _) in self.open_files.items():
            if path == file_path:
                tab_btn.config(bg=COLORS['bg_medium'])
            else:
                tab_btn.config(bg=COLORS['bg_light'])
        
        # Show selected editor
        if file_path in self.open_files:
            _, editor = self.open_files[file_path]
            editor.pack(fill=tk.BOTH, expand=True)
            self.current_file = file_path
    
    def close_file(self, file_path: Path):
        """Close a file."""
        if file_path not in self.open_files:
            return
        
        tab_btn, editor = self.open_files[file_path]
        
        # TODO: Check for unsaved changes
        
        tab_btn.destroy()
        editor.destroy()
        del self.open_files[file_path]
        
        if self.current_file == file_path:
            self.current_file = None
            if self.open_files:
                # Switch to another file
                next_file = next(iter(self.open_files))
                self.switch_to_file(next_file)
            else:
                self.show_welcome()
    
    def save_current_file(self):
        """Save the currently open file."""
        if not self.current_file or self.current_file not in self.open_files:
            return False
        
        try:
            _, editor = self.open_files[self.current_file]
            content = editor.get('1.0', 'end-1c')
            
            with open(self.current_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")
            return False


class WorkflowPanel(ttk.Frame):
    """Right panel with workflow controls."""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.vars = {}
        
        # Configure frame
        self.configure(style='Dark.TFrame')
        
        # Create notebook for different sections
        style = ttk.Style()
        style.configure('Dark.TNotebook', background=COLORS['bg_dark'])
        style.configure('Dark.TNotebook.Tab', background=COLORS['bg_light'], 
                       foreground=COLORS['fg_primary'], padding=[20, 10])
        
        self.notebook = ttk.Notebook(self, style='Dark.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Add tabs
        self._create_workflow_tab()
        self._create_models_tab()
        self._create_chat_tab()
        self._create_settings_tab()
    
    def _create_workflow_tab(self):
        """Create workflow configuration tab with Auto/Interactive mode selection."""
        frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(frame, text='Workflow')
        
        # Initialize default variables
        self._init_default_vars()
        
        # Scrollable container
        canvas = tk.Canvas(frame, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=COLORS['bg_dark'])
        
        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mode Selection Section
        self._add_section_header(scrollable, "🤖 Research Mode")
        
        mode_frame = tk.Frame(scrollable, bg=COLORS['bg_dark'])
        mode_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Mode variable
        self.vars['research_mode'] = tk.StringVar(value=MODE_AUTO)
        
        # Auto Mode Radio Button
        auto_frame = tk.Frame(mode_frame, bg=COLORS['bg_light'], padx=15, pady=15)
        auto_frame.pack(fill=tk.X, pady=(0, 10))
        
        auto_radio = tk.Radiobutton(
            auto_frame, text="🚀 Auto Mode", 
            variable=self.vars['research_mode'], value=MODE_AUTO,
            bg=COLORS['bg_light'], fg=COLORS['accent_green'],
            activebackground=COLORS['bg_light'], activeforeground=COLORS['accent_green'],
            selectcolor=COLORS['bg_dark'], font=('Segoe UI', 11, 'bold'),
            command=self._on_mode_change
        )
        auto_radio.pack(anchor=tk.W)
        
        tk.Label(auto_frame, 
                text="Agent autonomously completes the entire research workflow.\n"
                     "Just set your topic and let the AI do everything.",
                bg=COLORS['bg_light'], fg=COLORS['fg_secondary'],
                font=('Segoe UI', 9), justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(5, 0))
        
        # Interactive Mode Radio Button
        interactive_frame = tk.Frame(mode_frame, bg=COLORS['bg_light'], padx=15, pady=15)
        interactive_frame.pack(fill=tk.X)
        
        interactive_radio = tk.Radiobutton(
            interactive_frame, text="💬 Interactive Mode",
            variable=self.vars['research_mode'], value=MODE_INTERACTIVE,
            bg=COLORS['bg_light'], fg=COLORS['accent_blue'],
            activebackground=COLORS['bg_light'], activeforeground=COLORS['accent_blue'],
            selectcolor=COLORS['bg_dark'], font=('Segoe UI', 11, 'bold'),
            command=self._on_mode_change
        )
        interactive_radio.pack(anchor=tk.W)
        
        tk.Label(interactive_frame,
                text="Guide the research with custom prompts.\n"
                     "Control each iteration and provide feedback.",
                bg=COLORS['bg_light'], fg=COLORS['fg_secondary'],
                font=('Segoe UI', 9), justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(5, 0))
        
        # Iteration Control Section
        self._add_section_header(scrollable, "🔄 Iteration Control")
        
        iter_frame = tk.Frame(scrollable, bg=COLORS['bg_dark'])
        iter_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(iter_frame, text="Number of Iterations:", 
                bg=COLORS['bg_dark'], fg=COLORS['fg_primary'],
                font=('Segoe UI', 10)).pack(anchor=tk.W)
        
        iter_control_frame = tk.Frame(iter_frame, bg=COLORS['bg_dark'])
        iter_control_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Iteration slider
        self.iter_slider = tk.Scale(
            iter_control_frame, from_=1, to=20, orient=tk.HORIZONTAL,
            variable=self.vars['max_iterations'],
            bg=COLORS['bg_dark'], fg=COLORS['fg_primary'],
            activebackground=COLORS['accent_blue'],
            highlightthickness=0, troughcolor=COLORS['bg_light'],
            font=('Segoe UI', 10), length=250
        )
        self.iter_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Quick iteration buttons
        quick_iter_frame = tk.Frame(iter_frame, bg=COLORS['bg_dark'])
        quick_iter_frame.pack(fill=tk.X, pady=(10, 0))
        
        for label, value in [("Quick", 2), ("Standard", 4), ("Thorough", 8), ("Deep", 15)]:
            btn = tk.Button(
                quick_iter_frame, text=f"{label} ({value})",
                command=lambda v=value: self.vars['max_iterations'].set(v),
                bg=COLORS['bg_light'], fg=COLORS['fg_primary'],
                activebackground=COLORS['accent_blue'], activeforeground='white',
                relief=tk.FLAT, font=('Segoe UI', 9), padx=10, pady=5, cursor='hand2'
            )
            btn.pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Label(iter_frame,
                text="More iterations = more refined results but longer runtime",
                bg=COLORS['bg_dark'], fg=COLORS['fg_secondary'],
                font=('Segoe UI', 8)).pack(anchor=tk.W, pady=(5, 0))
        
        # User Prompt Section (for Interactive Mode)
        self._add_section_header(scrollable, "📝 Research Prompt")
        
        self.prompt_frame = tk.Frame(scrollable, bg=COLORS['bg_dark'])
        self.prompt_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(self.prompt_frame,
                text="Enter your research question or instructions:",
                bg=COLORS['bg_dark'], fg=COLORS['fg_secondary'],
                font=('Segoe UI', 9)).pack(anchor=tk.W)
        
        self.user_prompt_text = ScrolledText(
            self.prompt_frame, height=5, wrap=tk.WORD,
            bg=COLORS['bg_light'], fg=COLORS['fg_primary'],
            insertbackground=COLORS['fg_primary'],
            font=('Consolas', 10), relief=tk.FLAT
        )
        self.user_prompt_text.pack(fill=tk.X, pady=(5, 0))
        self.user_prompt_text.insert('1.0', "How can we improve the performance of large language models for scientific research?")
        
        # Prompt templates
        template_frame = tk.Frame(self.prompt_frame, bg=COLORS['bg_dark'])
        template_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(template_frame, text="Templates:",
                bg=COLORS['bg_dark'], fg=COLORS['fg_secondary'],
                font=('Segoe UI', 9)).pack(side=tk.LEFT)
        
        templates = [
            ("Survey", "Write a comprehensive survey on [topic] covering recent advances and future directions."),
            ("Novel Method", "Propose a novel method to address [problem] in [field]. Include theoretical analysis."),
            ("Benchmark", "Create a benchmark to evaluate [capability] in [domain] with appropriate metrics."),
            ("Analysis", "Perform a critical analysis of [approach] and identify limitations and improvements."),
        ]
        
        for name, template in templates:
            btn = tk.Button(
                template_frame, text=name,
                command=lambda t=template: self._set_prompt_template(t),
                bg=COLORS['bg_medium'], fg=COLORS['fg_primary'],
                activebackground=COLORS['accent_blue'], activeforeground='white',
                relief=tk.FLAT, font=('Segoe UI', 8), padx=8, pady=2, cursor='hand2'
            )
            btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Configuration Button
        self._add_section_header(scrollable, "⚙️ Advanced Configuration")
        
        config_btn = tk.Button(scrollable, text="📋 Paper Generation/Revision Configuration", 
                              command=self._open_config_dialog,
                              bg=COLORS['bg_light'], fg=COLORS['fg_primary'],
                              font=('Segoe UI', 10), relief=tk.FLAT,
                              cursor='hand2', padx=20, pady=12)
        config_btn.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        # Run Section
        run_frame = tk.Frame(scrollable, bg=COLORS['bg_dark'])
        run_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Run button with status
        self.run_btn = tk.Button(run_frame, text="▶️ Start Research", 
                                command=self.app.start_workflow,
                                bg=COLORS['accent_green'], fg='white',
                                font=('Segoe UI', 12, 'bold'), relief=tk.FLAT,
                                cursor='hand2', padx=30, pady=15)
        self.run_btn.pack(fill=tk.X)
        
        self.cancel_btn = tk.Button(run_frame, text="⏹️ Cancel",
                                    command=self.app.cancel_workflow,
                                    bg=COLORS['error'], fg='white',
                                    font=('Segoe UI', 10), relief=tk.FLAT,
                                    cursor='hand2', padx=20, pady=8, state=tk.DISABLED)
        self.cancel_btn.pack(fill=tk.X, pady=(10, 0))
        
        # Progress indicator
        self.progress_frame = tk.Frame(run_frame, bg=COLORS['bg_dark'])
        self.progress_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.progress_label = tk.Label(
            self.progress_frame, text="Ready to start research",
            bg=COLORS['bg_dark'], fg=COLORS['fg_secondary'],
            font=('Segoe UI', 9)
        )
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, mode='indeterminate', length=300
        )
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
    
    def _set_prompt_template(self, template: str):
        """Set a prompt template in the user prompt text box."""
        self.user_prompt_text.delete('1.0', tk.END)
        self.user_prompt_text.insert('1.0', template)
    
    def _on_mode_change(self):
        """Handle mode change between Auto and Interactive."""
        mode = self.vars['research_mode'].get()
        if mode == MODE_AUTO:
            self.run_btn.config(text="▶️ Start Auto Research", bg=COLORS['accent_green'])
        else:
            self.run_btn.config(text="▶️ Start Interactive Research", bg=COLORS['accent_blue'])
    
    def get_user_prompt(self) -> str:
        """Get the current user prompt text."""
        return self.user_prompt_text.get('1.0', 'end-1c').strip()
    
    def update_progress(self, message: str, running: bool = True):
        """Update the progress indicator."""
        self.progress_label.config(text=message)
        if running:
            self.progress_bar.start(10)
        else:
            self.progress_bar.stop()

    def _init_default_vars(self):
        """Initialize variables with default values."""
        defaults = {
            "topic": "Large Language Models",
            "field": "Computer Science",
            "question": "How to improve LLMs?",
            "model": DEFAULT_MODEL,
            "output_dir": "output",
            "max_iterations": 4,
            "request_timeout": 3600,
            "max_retries": 3,
            "modify_existing": False,
            "strict_singletons": True,
            "disable_blueprint_planning": False,
            "python_exec": "",
            "config_path": "",
            "enable_ideation": True,
            "skip_ideation": False,
            "specify_idea": "",
            "num_ideas": 15,
            "check_references": True,
            "skip_reference_check": False,
            "validate_figures": True,
            "skip_figure_validation": False,
            "enable_pdf_review": False,
            "disable_content_protection": False,
            "auto_approve_changes": False,
            "content_protection_threshold": 0.15,
            "output_diffs": True,
            "use_test_time_scaling": False,
            "revision_candidates": 3,
            "draft_candidates": 1,
            "all_code": False,
            "code_output_dir": "code",
            "science_only": False,
        }
        
        for key, val in defaults.items():
            if key not in self.vars:
                if isinstance(val, bool):
                    self.vars[key] = tk.BooleanVar(value=val)
                elif isinstance(val, int):
                    self.vars[key] = tk.IntVar(value=val)
                elif isinstance(val, float):
                    self.vars[key] = tk.DoubleVar(value=val)
                else:
                    self.vars[key] = tk.StringVar(value=val)

    def _open_config_dialog(self):
        """Open configuration dialog."""
        dialog = tk.Toplevel(self)
        dialog.title("Paper Generation/Revision Configuration")
        dialog.geometry("600x800")
        dialog.configure(bg=COLORS['bg_dark'])
        
        # Scrollable content
        canvas = tk.Canvas(dialog, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=COLORS['bg_dark'])
        
        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self._add_section_header(scrollable, "📋 Project Details")
        self._add_entry(scrollable, "Topic", "topic")
        self._add_entry(scrollable, "Field", "field")
        self._add_entry(scrollable, "Question", "question")
        
        self._add_section_header(scrollable, "🎯 Model Configuration")
        self._add_model_selector(scrollable)
        
        self._add_section_header(scrollable, "📁 Project Workspace")
        self._add_directory_selector(scrollable, "Project Directory", "output_dir")
        
        self._add_section_header(scrollable, "⚙️ Execution")
        self._add_spinbox(scrollable, "Max Iterations", "max_iterations", 4, 1, 20)
        self._add_spinbox(scrollable, "Request Timeout (s)", "request_timeout", 3600, 60, 7200)
        self._add_spinbox(scrollable, "Max Retries", "max_retries", 3, 0, 20)
        self._add_checkbox(scrollable, "Modify Existing Project", "modify_existing")
        self._add_checkbox(scrollable, "Enforce Single Files", "strict_singletons")
        self._add_checkbox(scrollable, "Disable Blueprint Planning", "disable_blueprint_planning")
        self._add_entry(scrollable, "Python Executable", "python_exec")
        self._add_entry(scrollable, "Config File", "config_path")
        
        self._add_section_header(scrollable, "✨ Options")
        self._add_checkbox(scrollable, "Enable Ideation", "enable_ideation")
        self._add_checkbox(scrollable, "Skip Ideation", "skip_ideation")
        self._add_entry(scrollable, "Specify Idea", "specify_idea")
        self._add_spinbox(scrollable, "Number of Ideas", "num_ideas", 15, 1, 50)
        self._add_checkbox(scrollable, "Check References", "check_references")
        self._add_checkbox(scrollable, "Skip Reference Check", "skip_reference_check")
        self._add_checkbox(scrollable, "Validate Figures", "validate_figures")
        self._add_checkbox(scrollable, "Skip Figure Validation", "skip_figure_validation")
        self._add_checkbox(scrollable, "Enable PDF Review", "enable_pdf_review")
        
        self._add_section_header(scrollable, "🔧 Advanced")
        self._add_checkbox(scrollable, "Disable Content Protection", "disable_content_protection")
        self._add_checkbox(scrollable, "Auto-Approve Changes", "auto_approve_changes")
        self._add_spinbox(scrollable, "Content Protection Threshold", "content_protection_threshold", 0.15, 0.0, 1.0)
        self._add_checkbox(scrollable, "Save Output Diffs", "output_diffs")
        self._add_checkbox(scrollable, "Enable Test-Time Scaling", "use_test_time_scaling")
        self._add_spinbox(scrollable, "Revision Candidates", "revision_candidates", 3, 1, 10)
        self._add_spinbox(scrollable, "Draft Candidates", "draft_candidates", 1, 1, 5)
        self._add_checkbox(scrollable, "All-Code Mode", "all_code")
        self._add_entry(scrollable, "Code Output Dir", "code_output_dir")
        self._add_checkbox(scrollable, "Science-Only Mode", "science_only")
        
        # Add Close button
        tk.Button(scrollable, text="Close", command=dialog.destroy,
                 bg=COLORS['accent_blue'], fg='white', relief=tk.FLAT,
                 font=('Segoe UI', 10), padx=20, pady=10).pack(pady=20)

    
    def _create_models_tab(self):
        """Create models and API configuration tab."""
        frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(frame, text='Models & API')
        
        # Scrollable content
        canvas = tk.Canvas(frame, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=COLORS['bg_dark'])
        
        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # OpenAI Configuration
        self._add_section_header(scrollable, "🔑 OpenAI API")
        self._add_api_config(scrollable, "OpenAI", "openai_key", 
                            env_var="OPENAI_API_KEY")
        
        # Yunwu Configuration
        self._add_section_header(scrollable, "🌐 Yunwu API")
        self._add_api_config(scrollable, "Yunwu", "yunwu_key",
                            env_var="YUNWU_API_KEY")
        self._add_entry(scrollable, "Yunwu Base URL", "yunwu_base", 
                       "https://yunwu.ai/v1")
        
        # Google API
        self._add_section_header(scrollable, "🔮 Google AI")
        self._add_api_config(scrollable, "Google", "google_key",
                            env_var="GOOGLE_API_KEY")
        
        # Model List
        self._add_section_header(scrollable, "📚 Available Models")
        self._create_model_list(scrollable)
    
    def _create_chat_tab(self):
        """Create enhanced chat interface tab with LLM integration."""
        frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(frame, text='💬 Chat')
        
        # Header with model selector
        header_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(header_frame, text="🤖 AI Research Assistant",
                bg=COLORS['bg_dark'], fg=COLORS['accent_purple'],
                font=('Segoe UI', 12, 'bold')).pack(side=tk.LEFT)
        
        # Model selector for chat
        chat_model_frame = tk.Frame(header_frame, bg=COLORS['bg_dark'])
        chat_model_frame.pack(side=tk.RIGHT)
        
        tk.Label(chat_model_frame, text="Model:",
                bg=COLORS['bg_dark'], fg=COLORS['fg_secondary'],
                font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 5))
        
        chat_models = ["gpt-4o", "gpt-5-pro", "claude-sonnet-4", "gemini-2.0-flash-exp"]
        self.vars['chat_model'] = tk.StringVar(value=chat_models[0])
        
        chat_model_combo = ttk.Combobox(
            chat_model_frame, textvariable=self.vars['chat_model'],
            values=chat_models, state='readonly', width=20
        )
        chat_model_combo.pack(side=tk.LEFT)
        
        # Chat history with system message
        history_frame = tk.Frame(frame, bg=COLORS['bg_medium'])
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        
        self.chat_history = ScrolledText(
            history_frame, wrap=tk.WORD, state=tk.DISABLED,
            bg=COLORS['bg_medium'], fg=COLORS['fg_primary'],
            font=('Segoe UI', 10), relief=tk.FLAT,
            padx=10, pady=10
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for different message types
        self.chat_history.tag_config("system", foreground=COLORS['fg_secondary'], font=('Segoe UI', 9, 'italic'))
        self.chat_history.tag_config("user", foreground=COLORS['accent_blue'], font=('Segoe UI', 10, 'bold'))
        self.chat_history.tag_config("assistant", foreground=COLORS['accent_green'], font=('Segoe UI', 10, 'bold'))
        self.chat_history.tag_config("error", foreground=COLORS['error'], font=('Segoe UI', 10))
        self.chat_history.tag_config("code", background=COLORS['bg_light'], font=('Consolas', 10))
        
        # Add welcome message
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert('end', "🤖 AI Research Assistant\n", "assistant")
        self.chat_history.insert('end', "I'm here to help with your research. You can:\n", "system")
        self.chat_history.insert('end', "• Ask questions about your research topic\n", "system")
        self.chat_history.insert('end', "• Get suggestions for improving your paper\n", "system")
        self.chat_history.insert('end', "• Discuss methodology and approaches\n", "system")
        self.chat_history.insert('end', "• Generate ideas and hypotheses\n\n", "system")
        self.chat_history.config(state=tk.DISABLED)
        
        # Quick actions
        actions_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        actions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(actions_frame, text="Quick Actions:",
                bg=COLORS['bg_dark'], fg=COLORS['fg_secondary'],
                font=('Segoe UI', 9)).pack(side=tk.LEFT)
        
        quick_prompts = [
            ("💡 Ideas", "Generate 5 innovative research ideas related to my topic."),
            ("📊 Methods", "Suggest appropriate research methodologies for my study."),
            ("📚 Related Work", "What are the key related works I should cite?"),
            ("🔍 Gaps", "Identify potential research gaps in the current literature."),
        ]
        
        for label, prompt in quick_prompts:
            btn = tk.Button(
                actions_frame, text=label,
                command=lambda p=prompt: self._send_quick_prompt(p),
                bg=COLORS['bg_light'], fg=COLORS['fg_primary'],
                activebackground=COLORS['accent_blue'], activeforeground='white',
                relief=tk.FLAT, font=('Segoe UI', 9), padx=10, pady=3, cursor='hand2'
            )
            btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Input area
        input_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        input_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        # Context toggle
        context_frame = tk.Frame(input_frame, bg=COLORS['bg_dark'])
        context_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.vars['include_context'] = tk.BooleanVar(value=True)
        tk.Checkbutton(
            context_frame, text="Include research context",
            variable=self.vars['include_context'],
            bg=COLORS['bg_dark'], fg=COLORS['fg_secondary'],
            activebackground=COLORS['bg_dark'], activeforeground=COLORS['fg_secondary'],
            selectcolor=COLORS['bg_light'], font=('Segoe UI', 9)
        ).pack(side=tk.LEFT)
        
        # Text input
        self.chat_input = tk.Text(
            input_frame, height=3, wrap=tk.WORD,
            bg=COLORS['bg_light'], fg=COLORS['fg_primary'],
            insertbackground=COLORS['fg_primary'],
            font=('Segoe UI', 10), relief=tk.FLAT,
            padx=10, pady=10
        )
        self.chat_input.pack(fill=tk.X, pady=(0, 10))
        self.chat_input.bind('<Control-Return>', lambda e: self.app.send_chat())
        self.chat_input.bind('<Shift-Return>', lambda e: None)  # Allow newlines
        
        # Buttons
        btn_frame = tk.Frame(input_frame, bg=COLORS['bg_dark'])
        btn_frame.pack(fill=tk.X)
        
        self.chat_send_btn = tk.Button(
            btn_frame, text="📤 Send (Ctrl+Enter)",
            command=self.app.send_chat,
            bg=COLORS['accent_blue'], fg='white',
            activebackground=COLORS['accent_green'], activeforeground='white',
            relief=tk.FLAT, font=('Segoe UI', 10, 'bold'),
            cursor='hand2', padx=20, pady=8
        )
        self.chat_send_btn.pack(side=tk.RIGHT)
        
        tk.Button(
            btn_frame, text="🗑️ Clear",
            command=self.app.clear_chat,
            bg=COLORS['bg_light'], fg=COLORS['fg_primary'],
            activebackground=COLORS['error'], activeforeground='white',
            relief=tk.FLAT, font=('Segoe UI', 9),
            cursor='hand2', padx=15, pady=5
        ).pack(side=tk.RIGHT, padx=(0, 10))
        
        # Chat state
        self.chat_messages: List[Dict[str, str]] = []
    
    def _send_quick_prompt(self, prompt: str):
        """Send a quick prompt to the chat."""
        self.chat_input.delete('1.0', tk.END)
        self.chat_input.insert('1.0', prompt)
        self.app.send_chat()
    
    def add_chat_message(self, role: str, content: str):
        """Add a message to the chat history."""
        self.chat_history.config(state=tk.NORMAL)
        
        if role == "user":
            self.chat_history.insert('end', "\n👤 You: ", "user")
        elif role == "assistant":
            self.chat_history.insert('end', "\n🤖 Assistant: ", "assistant")
        elif role == "error":
            self.chat_history.insert('end', "\n⚠️ Error: ", "error")
        else:
            self.chat_history.insert('end', f"\n{role}: ", "system")
        
        self.chat_history.insert('end', f"{content}\n")
        self.chat_history.see('end')
        self.chat_history.config(state=tk.DISABLED)
        
        # Store message
        if role in ["user", "assistant"]:
            self.chat_messages.append({"role": role, "content": content})
    
    def set_chat_loading(self, loading: bool):
        """Set the chat loading state."""
        if loading:
            self.chat_send_btn.config(text="⏳ Thinking...", state=tk.DISABLED)
        else:
            self.chat_send_btn.config(text="📤 Send (Ctrl+Enter)", state=tk.NORMAL)
    
    def _create_settings_tab(self):
        """Create settings tab."""
        frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(frame, text='Settings')
        
        # Scrollable content
        canvas = tk.Canvas(frame, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=COLORS['bg_dark'])
        
        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self._add_section_header(scrollable, "🎨 Review Options")
        self._add_checkbox(scrollable, "Check Structure", "check_structure", False)
        self._add_checkbox(scrollable, "Check Content", "check_content", False)
        self._add_checkbox(scrollable, "Check Methodology", "check_methodology", False)
        self._add_checkbox(scrollable, "Check Results", "check_results", False)
        self._add_checkbox(scrollable, "Check References", "check_references_review", False)
        self._add_checkbox(scrollable, "Check Figures", "check_figures_review", False)
        self._add_checkbox(scrollable, "Check Writing", "check_writing", False)
        self._add_checkbox(scrollable, "Check Novelty", "check_novelty", False)
        self._add_checkbox(scrollable, "Check Reproducibility", "check_reproducibility", False)
        self._add_checkbox(scrollable, "Check Statistical Rigor", "check_statistical_rigor", False)
        
        self._add_section_header(scrollable, "🔒 Content Protection")
        self._add_checkbox(scrollable, "Enable Content Protection", "content_protection", True)
        self._add_spinbox(scrollable, "Protection Threshold", "protection_threshold", 0.15, 0.0, 1.0)
        
        self._add_section_header(scrollable, "📊 Output")
        self._add_checkbox(scrollable, "Save Diffs", "save_diffs", True)
        self._add_checkbox(scrollable, "Enable PDF Review", "pdf_review", False)
    
    # Helper methods for creating UI elements
    
    def _add_section_header(self, parent, text):
        """Add a section header."""
        frame = tk.Frame(parent, bg=COLORS['bg_dark'])
        frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(frame, text=text, bg=COLORS['bg_dark'], fg=COLORS['accent_yellow'],
                font=('Segoe UI', 11, 'bold')).pack(anchor=tk.W)
    
    def _add_entry(self, parent, label, key, default=""):
        """Add a labeled entry field."""
        frame = tk.Frame(parent, bg=COLORS['bg_dark'])
        frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(frame, text=label, bg=COLORS['bg_dark'], fg=COLORS['fg_secondary'],
                font=('Segoe UI', 9)).pack(anchor=tk.W)
        
        if key in self.vars:
            var = self.vars[key]
        else:
            var = tk.StringVar(value=default)
            self.vars[key] = var
        
        entry = tk.Entry(frame, textvariable=var, bg=COLORS['bg_light'],
                        fg=COLORS['fg_primary'], relief=tk.FLAT,
                        insertbackground=COLORS['fg_primary'],
                        font=('Segoe UI', 10))
        entry.pack(fill=tk.X, pady=(3, 0), ipady=5)
        
        return var
    
    def _add_directory_selector(self, parent, label, key, default=""):
        """Add a directory selector."""
        frame = tk.Frame(parent, bg=COLORS['bg_dark'])
        frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(frame, text=label, bg=COLORS['bg_dark'], fg=COLORS['fg_secondary'],
                font=('Segoe UI', 9)).pack(anchor=tk.W)
        
        inner = tk.Frame(frame, bg=COLORS['bg_dark'])
        inner.pack(fill=tk.X, pady=(3, 0))
        
        if key in self.vars:
            var = self.vars[key]
        else:
            var = tk.StringVar(value=default)
            self.vars[key] = var
        
        entry = tk.Entry(inner, textvariable=var, bg=COLORS['bg_light'],
                        fg=COLORS['fg_primary'], relief=tk.FLAT,
                        insertbackground=COLORS['fg_primary'],
                        font=('Segoe UI', 10))
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        
        btn = tk.Button(inner, text="📁", command=lambda: self._browse_directory(var),
                       bg=COLORS['accent_blue'], fg='white', relief=tk.FLAT,
                       font=('Segoe UI', 10), cursor='hand2', padx=10)
        btn.pack(side=tk.RIGHT, padx=(5, 0))
    
    def _browse_directory(self, var):
        """Browse for directory."""
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
    
    def _add_checkbox(self, parent, label, key, default=False):
        """Add a checkbox."""
        if key in self.vars:
            var = self.vars[key]
        else:
            var = tk.BooleanVar(value=default)
            self.vars[key] = var
        
        frame = tk.Frame(parent, bg=COLORS['bg_dark'])
        frame.pack(fill=tk.X, padx=20, pady=3)
        
        cb = tk.Checkbutton(frame, text=label, variable=var,
                           bg=COLORS['bg_dark'], fg=COLORS['fg_primary'],
                           activebackground=COLORS['bg_dark'],
                           activeforeground=COLORS['fg_primary'],
                           selectcolor=COLORS['bg_light'],
                           font=('Segoe UI', 9))
        cb.pack(anchor=tk.W)
    
    def _add_spinbox(self, parent, label, key, default, from_, to):
        """Add a spinbox."""
        frame = tk.Frame(parent, bg=COLORS['bg_dark'])
        frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(frame, text=label, bg=COLORS['bg_dark'], fg=COLORS['fg_secondary'],
                font=('Segoe UI', 9)).pack(anchor=tk.W)
        
        if key in self.vars:
            var = self.vars[key]
            if isinstance(var.get(), float):
                increment = 0.01
            else:
                increment = 1
        else:
            if isinstance(default, float):
                var = tk.DoubleVar(value=default)
                increment = 0.01
            else:
                var = tk.IntVar(value=default)
                increment = 1
            self.vars[key] = var
        
        # Use a frame to hold spinbox and manual entry
        container = tk.Frame(frame, bg=COLORS['bg_dark'])
        container.pack(fill=tk.X, pady=(3, 0))
        
        spinbox = tk.Spinbox(container, from_=from_, to=to, textvariable=var,
                            bg=COLORS['bg_light'], fg=COLORS['fg_primary'],
                            relief=tk.FLAT, font=('Segoe UI', 10), increment=increment)
        spinbox.pack(fill=tk.X, expand=True)
    
    def _add_model_selector(self, parent):
        """Add model selection dropdown."""
        frame = tk.Frame(parent, bg=COLORS['bg_dark'])
        frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(frame, text="Model", bg=COLORS['bg_dark'], fg=COLORS['fg_secondary'],
                font=('Segoe UI', 9)).pack(anchor=tk.W)
        
        models = [
            "gpt-5-pro",
            "gpt-5",
            "gpt-4o",
            "gpt-4",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
        ]
        
        if 'model' in self.vars:
            var = self.vars['model']
        else:
            var = tk.StringVar(value=DEFAULT_MODEL)
            self.vars['model'] = var
        
        combo = ttk.Combobox(frame, textvariable=var, values=models,
                            state='readonly', font=('Segoe UI', 10))
        combo.pack(fill=tk.X, pady=(3, 0))
    
    def _add_api_config(self, parent, name, key, env_var=""):
        """Add API configuration section."""
        frame = tk.Frame(parent, bg=COLORS['bg_dark'])
        frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Show if env var is set
        env_value = os.getenv(env_var, "")
        if env_value:
            status_text = f"✓ Found in environment: {env_value[:10]}...{env_value[-4:]}"
            status_color = COLORS['success']
        else:
            status_text = "✗ Not set in environment"
            status_color = COLORS['error']
        
        tk.Label(frame, text=status_text, bg=COLORS['bg_dark'],
                fg=status_color, font=('Segoe UI', 8)).pack(anchor=tk.W)
        
        tk.Label(frame, text=f"{name} API Key", bg=COLORS['bg_dark'],
                fg=COLORS['fg_secondary'], font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        
        var = tk.StringVar(value=env_value)
        self.vars[key] = var
        
        entry = tk.Entry(frame, textvariable=var, show="*",
                        bg=COLORS['bg_light'], fg=COLORS['fg_primary'],
                        relief=tk.FLAT, insertbackground=COLORS['fg_primary'],
                        font=('Segoe UI', 10))
        entry.pack(fill=tk.X, pady=(3, 0), ipady=5)
    
    def _create_model_list(self, parent):
        """Create a list of available models."""
        frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        models_info = [
            ("OpenAI", ["gpt-5-pro", "gpt-5", "gpt-4o", "gpt-4"]),
            ("Anthropic", ["claude-opus-4", "claude-sonnet-4", "claude-3-5-sonnet"]),
            ("Google", ["gemini-2.0-flash-exp", "gemini-1.5-pro"]),
        ]
        
        for provider, models in models_info:
            tk.Label(frame, text=provider, bg=COLORS['bg_medium'],
                    fg=COLORS['accent_yellow'], font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
            
            for model in models:
                tk.Label(frame, text=f"  • {model}", bg=COLORS['bg_medium'],
                        fg=COLORS['fg_secondary'], font=('Segoe UI', 9)).pack(anchor=tk.W)


class VSCodeStyleGUI(tk.Tk):
    """Main VSCode-style GUI application with Auto/Interactive research modes."""
    
    POLL_INTERVAL_MS = 100
    
    def __init__(self):
        super().__init__()
        
        self.title("AI Scientist - Research Workflow")
        self.geometry("1600x900")
        self.minsize(1400, 800)
        
        # Configure style
        self.configure(bg=COLORS['bg_dark'])
        
        # State
        self.running = False
        self.cancel_event: Optional[threading.Event] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.log_queue: queue.Queue = queue.Queue()
        self.error_queue: queue.Queue = queue.Queue()
        self.connection_manager = get_shared_connection_manager()
        
        # Build UI
        self._build_ui()
        
        # Keyboard shortcuts
        self.bind('<Control-s>', lambda e: self.save_file())
        self.bind('<Control-o>', lambda e: self.file_tree.open_folder())
        self.bind('<F5>', lambda e: self.start_workflow())
        
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Start polling
        self._poll_queue()
    
    def _build_ui(self):
        """Build the main UI layout."""
        # Menu bar
        menubar = tk.Menu(self, bg=COLORS['bg_dark'], fg=COLORS['fg_primary'])
        self.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_dark'], fg=COLORS['fg_primary'])
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Folder", command=lambda: self.file_tree.open_folder())
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        
        run_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_dark'], fg=COLORS['fg_primary'])
        menubar.add_cascade(label="Run", menu=run_menu)
        run_menu.add_command(label="Start Research", command=self.start_workflow, accelerator="F5")
        run_menu.add_command(label="Cancel", command=self.cancel_workflow)
        
        # Main container
        main_container = tk.Frame(self, bg=COLORS['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - File tree (20%)
        self.file_tree = FileTreeView(main_container, 
                                      on_file_select=self.open_file,
                                      on_folder_select=self.on_folder_opened)
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.file_tree.config(width=300)
        
        # Separator
        ttk.Separator(main_container, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)
        
        # Middle panel - Editor and Output (50%)
        middle_panel = tk.Frame(main_container, bg=COLORS['bg_dark'])
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Split middle panel vertically
        self.editor = FileEditor(middle_panel)
        self.editor.pack(fill=tk.BOTH, expand=True)
        
        # Output panel at bottom
        output_frame = tk.Frame(middle_panel, bg=COLORS['bg_dark'], height=200)
        output_frame.pack(fill=tk.X, side=tk.BOTTOM)
        output_frame.pack_propagate(False)
        
        output_header = tk.Frame(output_frame, bg=COLORS['bg_medium'])
        output_header.pack(fill=tk.X)
        
        tk.Label(output_header, text="📋 Output",
                bg=COLORS['bg_medium'], fg=COLORS['fg_primary'],
                font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Clear output button
        tk.Button(output_header, text="Clear",
                 command=self.clear_output,
                 bg=COLORS['bg_light'], fg=COLORS['fg_secondary'],
                 relief=tk.FLAT, font=('Segoe UI', 8), cursor='hand2',
                 padx=5).pack(side=tk.RIGHT, padx=5, pady=3)
        
        self.output_text = ScrolledText(
            output_frame, wrap=tk.WORD,
            bg=COLORS['bg_dark'], fg=COLORS['fg_primary'],
            font=('Consolas', 9), relief=tk.FLAT,
            height=10
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure output tags
        self.output_text.tag_config("info", foreground=COLORS['fg_primary'])
        self.output_text.tag_config("success", foreground=COLORS['success'])
        self.output_text.tag_config("error", foreground=COLORS['error'])
        self.output_text.tag_config("warning", foreground=COLORS['warning'])
        self.output_text.tag_config("step", foreground=COLORS['accent_blue'], font=('Consolas', 9, 'bold'))
        
        # Separator
        ttk.Separator(main_container, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)
        
        # Right panel - Workflow controls (30%)
        self.workflow_panel = WorkflowPanel(main_container, self)
        self.workflow_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.workflow_panel.config(width=450)
        
        # Bottom status bar
        self.status_bar = tk.Frame(self, bg=COLORS['accent_blue'], height=25)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(self.status_bar, text="Ready", 
                                     bg=COLORS['accent_blue'], fg='white',
                                     font=('Segoe UI', 9), anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Mode indicator
        self.mode_label = tk.Label(self.status_bar, text="Mode: Auto",
                                   bg=COLORS['accent_blue'], fg='white',
                                   font=('Segoe UI', 9))
        self.mode_label.pack(side=tk.RIGHT, padx=10)
        
        # Iteration indicator
        self.iter_label = tk.Label(self.status_bar, text="Iterations: 4",
                                   bg=COLORS['accent_blue'], fg='white',
                                   font=('Segoe UI', 9))
        self.iter_label.pack(side=tk.RIGHT, padx=10)
    
    def _poll_queue(self):
        """Poll the log queue and update the output."""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self._append_output(message)
        except queue.Empty:
            pass
        
        try:
            while True:
                error = self.error_queue.get_nowait()
                self._append_output(error, "error")
        except queue.Empty:
            pass
        
        # Update mode and iteration labels
        if hasattr(self.workflow_panel, 'vars'):
            mode = self.workflow_panel.vars.get('research_mode', tk.StringVar()).get()
            self.mode_label.config(text=f"Mode: {mode.capitalize()}")
            
            iters = self.workflow_panel.vars.get('max_iterations', tk.IntVar()).get()
            self.iter_label.config(text=f"Iterations: {iters}")
        
        # Schedule next poll
        self.after(self.POLL_INTERVAL_MS, self._poll_queue)
    
    def _append_output(self, message: str, tag: str = "info"):
        """Append a message to the output text."""
        self.output_text.config(state=tk.NORMAL)
        
        # Detect message type
        msg_lower = message.lower()
        if "error" in msg_lower or "failed" in msg_lower:
            tag = "error"
        elif "success" in msg_lower or "completed" in msg_lower:
            tag = "success"
        elif "warning" in msg_lower:
            tag = "warning"
        elif "step" in msg_lower or "iteration" in msg_lower or "phase" in msg_lower:
            tag = "step"
        
        self.output_text.insert(tk.END, message + "\n", tag)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
    
    def clear_output(self):
        """Clear the output panel."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete('1.0', tk.END)
        self.output_text.config(state=tk.DISABLED)
    
    def on_folder_opened(self, folder_path):
        """Handle folder opening."""
        # Auto-set output directory to the opened folder
        if 'output_dir' in self.workflow_panel.vars:
            self.workflow_panel.vars['output_dir'].set(folder_path)
            self.status_label.config(text=f"Project loaded: {Path(folder_path).name}")
            
            # Auto-enable modify_existing since user wants to work on imported folder
            if 'modify_existing' in self.workflow_panel.vars:
                self.workflow_panel.vars['modify_existing'].set(True)
        
        self._append_output(f"📁 Opened project folder: {folder_path}", "step")

    def open_file(self, file_path: Path):
        """Open a file in the editor."""
        self.editor.open_file(file_path)
        self.status_label.config(text=f"Opened: {file_path.name}")
    
    def save_file(self):
        """Save the current file."""
        if self.editor.save_current_file():
            self.status_label.config(text="File saved")
        return "break"  # Prevent default handler
    
    def start_workflow(self):
        """Start the research workflow."""
        if self.running:
            messagebox.showwarning("Warning", "Workflow is already running!")
            return
        
        # Get parameters from workflow panel
        vars_dict = self.workflow_panel.vars
        output_dir = vars_dict.get('output_dir', tk.StringVar()).get()
        
        if not output_dir:
            messagebox.showerror("Error", "Please specify a project directory")
            return
        
        # Build parameters
        params = {}
        for key, var in vars_dict.items():
            try:
                params[key] = var.get()
            except Exception:
                pass
        
        # Add user prompt from the text widget
        user_prompt = self.workflow_panel.get_user_prompt()
        if user_prompt:
            params['user_prompt'] = user_prompt
            params['specify_idea'] = user_prompt  # Also use as idea specification
        
        # Get mode
        mode = params.get('research_mode', MODE_AUTO)
        max_iters = params.get('max_iterations', 4)
        
        # Update UI
        self.running = True
        self.cancel_event = threading.Event()
        
        self.workflow_panel.run_btn.config(state=tk.DISABLED)
        self.workflow_panel.cancel_btn.config(state=tk.NORMAL)
        self.workflow_panel.update_progress(f"Starting {mode} research...", True)
        
        self.status_label.config(text="Running workflow...", bg=COLORS['accent_green'])
        
        self._append_output(f"\n{'='*60}", "step")
        self._append_output(f"🚀 Starting {mode.upper()} Mode Research", "step")
        self._append_output(f"📊 Max Iterations: {max_iters}", "info")
        self._append_output(f"📁 Output Directory: {output_dir}", "info")
        if user_prompt:
            self._append_output(f"📝 Research Prompt: {user_prompt[:100]}...", "info")
        self._append_output(f"{'='*60}\n", "step")
        
        # Start workflow in background thread
        self.worker_thread = threading.Thread(
            target=self._run_workflow_thread,
            args=(params,),
            daemon=True
        )
        self.worker_thread.start()
    
    def _run_workflow_thread(self, params: Dict[str, Any]):
        """Run the workflow in a background thread."""
        import sys
        
        # Redirect stdout/stderr to queue
        class QueueWriter:
            def __init__(self, q: queue.Queue):
                self.queue = q
            def write(self, msg: str):
                if msg.strip():
                    self.queue.put(msg)
            def flush(self):
                pass
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = QueueWriter(self.log_queue)
        sys.stderr = QueueWriter(self.error_queue)
        
        try:
            # Run the workflow
            result_dir = run_from_gui(params, self.cancel_event)
            
            self.log_queue.put(f"\n{'='*60}")
            self.log_queue.put(f"✅ Research completed successfully!")
            self.log_queue.put(f"📁 Results saved to: {result_dir}")
            self.log_queue.put(f"{'='*60}\n")
            
            # Schedule UI update on main thread
            self.after(0, lambda: self._workflow_completed(True, str(result_dir)))
            
        except WorkflowCancelled:
            self.log_queue.put("\n⚠️ Workflow cancelled by user")
            self.after(0, lambda: self._workflow_completed(False, "Cancelled"))
            
        except Exception as e:
            import traceback
            self.error_queue.put(f"\n❌ Workflow failed: {str(e)}")
            self.error_queue.put(traceback.format_exc())
            self.after(0, lambda: self._workflow_completed(False, str(e)))
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _workflow_completed(self, success: bool, message: str):
        """Handle workflow completion."""
        self.running = False
        self.cancel_event = None
        
        self.workflow_panel.run_btn.config(state=tk.NORMAL)
        self.workflow_panel.cancel_btn.config(state=tk.DISABLED)
        self.workflow_panel.update_progress(
            "Research completed!" if success else f"Failed: {message[:50]}",
            False
        )
        
        if success:
            self.status_label.config(text=f"Completed: {message}", bg=COLORS['accent_green'])
            messagebox.showinfo("Success", f"Research workflow completed!\n\nResults saved to:\n{message}")
            
            # Refresh file tree if output dir is set
            output_dir = self.workflow_panel.vars.get('output_dir', tk.StringVar()).get()
            if output_dir:
                self.file_tree.load_folder(output_dir)
        else:
            self.status_label.config(text="Failed", bg=COLORS['error'])
        
        # Reset status bar color after delay
        self.after(3000, lambda: self.status_label.config(bg=COLORS['accent_blue']))
    
    def cancel_workflow(self):
        """Cancel the running workflow."""
        if self.cancel_event and self.running:
            self.cancel_event.set()
            self.status_label.config(text="Cancelling...", bg=COLORS['warning'])
            self._append_output("⏹️ Cancellation requested...", "warning")
    
    def send_chat(self):
        """Send a chat message to the LLM."""
        message = self.workflow_panel.chat_input.get('1.0', 'end-1c').strip()
        if not message:
            return
        
        # Clear input
        self.workflow_panel.chat_input.delete('1.0', 'end')
        
        # Add user message
        self.workflow_panel.add_chat_message("user", message)
        
        # Set loading state
        self.workflow_panel.set_chat_loading(True)
        
        # Get context if enabled
        context = ""
        if self.workflow_panel.vars.get('include_context', tk.BooleanVar()).get():
            topic = self.workflow_panel.vars.get('topic', tk.StringVar()).get()
            field = self.workflow_panel.vars.get('field', tk.StringVar()).get()
            if topic or field:
                context = f"Research context - Topic: {topic}, Field: {field}. "
        
        # Get model
        model = self.workflow_panel.vars.get('chat_model', tk.StringVar()).get() or "gpt-4o"
        
        # Build messages
        system_prompt = """You are an AI research assistant helping with scientific research. 
You provide insightful, well-reasoned responses about research methodology, literature, 
and scientific writing. Be concise but thorough."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in self.workflow_panel.chat_messages[-10:]:  # Last 10 messages
            messages.append(msg)
        
        # Add current message with context
        messages.append({"role": "user", "content": context + message})
        
        # Run chat in background thread
        def chat_thread():
            try:
                response, _ = chat(
                    messages=messages,
                    model=model,
                    temperature=0.7,
                    max_tokens=2000
                )
                self.after(0, lambda: self._handle_chat_response(response))
            except Exception as e:
                self.after(0, lambda: self._handle_chat_error(str(e)))
        
        threading.Thread(target=chat_thread, daemon=True).start()
    
    def _handle_chat_response(self, response: str):
        """Handle a successful chat response."""
        self.workflow_panel.set_chat_loading(False)
        self.workflow_panel.add_chat_message("assistant", response)
    
    def _handle_chat_error(self, error: str):
        """Handle a chat error."""
        self.workflow_panel.set_chat_loading(False)
        self.workflow_panel.add_chat_message("error", f"Failed to get response: {error}")
    
    def clear_chat(self):
        """Clear chat history."""
        self.workflow_panel.chat_history.config(state=tk.NORMAL)
        self.workflow_panel.chat_history.delete('1.0', 'end')
        self.workflow_panel.chat_history.config(state=tk.DISABLED)
        self.workflow_panel.chat_messages.clear()
        
        # Re-add welcome message
        self.workflow_panel.chat_history.config(state=tk.NORMAL)
        self.workflow_panel.chat_history.insert('end', "🤖 AI Research Assistant\n", "assistant")
        self.workflow_panel.chat_history.insert('end', "Chat history cleared. How can I help?\n\n", "system")
        self.workflow_panel.chat_history.config(state=tk.DISABLED)
    
    def _on_close(self):
        """Handle window close."""
        if self.running:
            if not messagebox.askyesno("Confirm", "Workflow is running. Exit anyway?"):
                return
            if self.cancel_event:
                self.cancel_event.set()
        self.destroy()


def main():
    """Main entry point."""
    app = VSCodeStyleGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
