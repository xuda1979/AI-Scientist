"""VSCode-style GUI for AI Scientist - Modern, Professional Interface."""
from __future__ import annotations

import os
import sys
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Dict, Optional, List, Any

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
    'border': '#3e3e42',
    'error': '#f48771',
    'success': '#89d185',
}


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
    
    def __init__(self, parent, on_file_select=None):
        super().__init__(parent)
        self.on_file_select = on_file_select
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
        """Create workflow configuration tab."""
        frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(frame, text='Workflow')
        
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
        
        # Content
        self._add_section_header(scrollable, "📋 Project Details")
        self._add_entry(scrollable, "Topic", "topic", "Large Language Models")
        self._add_entry(scrollable, "Field", "field", "Computer Science")
        self._add_entry(scrollable, "Question", "question", "How to improve LLMs?")
        
        self._add_section_header(scrollable, "🎯 Model Configuration")
        self._add_model_selector(scrollable)
        
        self._add_section_header(scrollable, "📁 Output")
        self._add_directory_selector(scrollable, "Output Directory", "output_dir", "output")
        
        self._add_section_header(scrollable, "⚙️ Execution")
        self._add_spinbox(scrollable, "Max Iterations", "max_iterations", 4, 1, 20)
        self._add_spinbox(scrollable, "Request Timeout (s)", "request_timeout", 3600, 60, 7200)
        
        self._add_section_header(scrollable, "✨ Options")
        self._add_checkbox(scrollable, "Enable Ideation", "enable_ideation", True)
        self._add_checkbox(scrollable, "Check References", "check_references", True)
        self._add_checkbox(scrollable, "Validate Figures", "validate_figures", True)
        
        # Run button
        btn_frame = tk.Frame(scrollable, bg=COLORS['bg_dark'])
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.run_btn = tk.Button(btn_frame, text="▶ Run Workflow", 
                                command=self.app.start_workflow,
                                bg=COLORS['accent_green'], fg='white',
                                font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
                                cursor='hand2', padx=30, pady=12)
        self.run_btn.pack(fill=tk.X)
        
        self.cancel_btn = tk.Button(btn_frame, text="⬛ Cancel",
                                    command=self.app.cancel_workflow,
                                    bg=COLORS['error'], fg='white',
                                    font=('Segoe UI', 10), relief=tk.FLAT,
                                    cursor='hand2', padx=20, pady=8, state=tk.DISABLED)
        self.cancel_btn.pack(fill=tk.X, pady=(10, 0))
    
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
        """Create chat interface tab."""
        frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(frame, text='Chat')
        
        # Chat history
        history_frame = tk.Frame(frame, bg=COLORS['bg_medium'])
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
        
        self.chat_history = ScrolledText(history_frame, wrap=tk.WORD, state=tk.DISABLED,
                                         bg=COLORS['bg_medium'], fg=COLORS['fg_primary'],
                                         font=('Segoe UI', 10))
        self.chat_history.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags
        self.chat_history.tag_config("user", foreground=COLORS['accent_blue'], font=('Segoe UI', 10, 'bold'))
        self.chat_history.tag_config("assistant", foreground=COLORS['accent_green'], font=('Segoe UI', 10, 'bold'))
        
        # Input area
        input_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        input_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        self.chat_input = tk.Text(input_frame, height=3, wrap=tk.WORD,
                                  bg=COLORS['bg_light'], fg=COLORS['fg_primary'],
                                  insertbackground=COLORS['fg_primary'],
                                  font=('Segoe UI', 10))
        self.chat_input.pack(fill=tk.X, pady=(0, 10))
        self.chat_input.bind('<Control-Return>', lambda e: self.app.send_chat())
        
        btn_frame = tk.Frame(input_frame, bg=COLORS['bg_dark'])
        btn_frame.pack(fill=tk.X)
        
        tk.Button(btn_frame, text="Send (Ctrl+Enter)", command=self.app.send_chat,
                 bg=COLORS['accent_blue'], fg='white', relief=tk.FLAT,
                 font=('Segoe UI', 9), cursor='hand2', padx=15, pady=5).pack(side=tk.RIGHT)
        
        tk.Button(btn_frame, text="Clear", command=self.app.clear_chat,
                 bg=COLORS['bg_light'], fg=COLORS['fg_primary'], relief=tk.FLAT,
                 font=('Segoe UI', 9), cursor='hand2', padx=15, pady=5).pack(side=tk.RIGHT, padx=(0, 10))
    
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
        self._add_checkbox(scrollable, "Check Structure", "review_structure", False)
        self._add_checkbox(scrollable, "Check Methodology", "review_methodology", False)
        self._add_checkbox(scrollable, "Check Results", "review_results", False)
        
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
        
        if isinstance(default, float):
            var = tk.DoubleVar(value=default)
        else:
            var = tk.IntVar(value=default)
        self.vars[key] = var
        
        spinbox = tk.Spinbox(frame, from_=from_, to=to, textvariable=var,
                            bg=COLORS['bg_light'], fg=COLORS['fg_primary'],
                            relief=tk.FLAT, font=('Segoe UI', 10))
        spinbox.pack(fill=tk.X, pady=(3, 0))
    
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
    """Main VSCode-style GUI application."""
    
    def __init__(self):
        super().__init__()
        
        self.title("AI Scientist - VSCode Style")
        self.geometry("1600x900")
        self.minsize(1400, 800)
        
        # Configure style
        self.configure(bg=COLORS['bg_dark'])
        
        # State
        self.running = False
        self.cancel_event = None
        self.log_queue = queue.Queue()
        self.connection_manager = get_shared_connection_manager()
        
        # Build UI
        self._build_ui()
        
        # Keyboard shortcuts
        self.bind('<Control-s>', lambda e: self.save_file())
        self.bind('<Control-o>', lambda e: self.file_tree.open_folder())
        
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
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
        
        # Main container
        main_container = tk.Frame(self, bg=COLORS['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - File tree (25%)
        self.file_tree = FileTreeView(main_container, on_file_select=self.open_file)
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.file_tree.config(width=350)
        
        # Separator
        ttk.Separator(main_container, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)
        
        # Middle panel - File editor (50%)
        self.editor = FileEditor(main_container)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Separator
        ttk.Separator(main_container, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)
        
        # Right panel - Workflow controls (25%)
        self.workflow_panel = WorkflowPanel(main_container, self)
        self.workflow_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.workflow_panel.config(width=400)
        
        # Bottom status bar
        self.status_bar = tk.Frame(self, bg=COLORS['accent_blue'], height=25)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(self.status_bar, text="Ready", 
                                     bg=COLORS['accent_blue'], fg='white',
                                     font=('Segoe UI', 9), anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
    
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
        """Start the workflow."""
        if self.running:
            return
        
        # Validate
        if not self.workflow_panel.vars.get('output_dir', tk.StringVar()).get():
            messagebox.showerror("Error", "Please specify an output directory")
            return
        
        self.running = True
        self.status_label.config(text="Running workflow...", bg=COLORS['accent_green'])
        self.workflow_panel.run_btn.config(state=tk.DISABLED)
        self.workflow_panel.cancel_btn.config(state=tk.NORMAL)
        
        # Start workflow in background
        # TODO: Implement actual workflow execution
        messagebox.showinfo("Info", "Workflow started! (Implementation pending)")
        
        self.running = False
        self.status_label.config(text="Ready", bg=COLORS['accent_blue'])
        self.workflow_panel.run_btn.config(state=tk.NORMAL)
        self.workflow_panel.cancel_btn.config(state=tk.DISABLED)
    
    def cancel_workflow(self):
        """Cancel the running workflow."""
        if self.cancel_event:
            self.cancel_event.set()
            self.status_label.config(text="Cancelling...", bg=COLORS['error'])
    
    def send_chat(self):
        """Send a chat message."""
        message = self.workflow_panel.chat_input.get('1.0', 'end-1c').strip()
        if not message:
            return
        
        # Add to history
        self.workflow_panel.chat_history.config(state=tk.NORMAL)
        self.workflow_panel.chat_history.insert('end', "You: ", "user")
        self.workflow_panel.chat_history.insert('end', f"{message}\n\n")
        self.workflow_panel.chat_history.config(state=tk.DISABLED)
        
        # Clear input
        self.workflow_panel.chat_input.delete('1.0', 'end')
        
        # TODO: Send to LLM
        self.workflow_panel.chat_history.config(state=tk.NORMAL)
        self.workflow_panel.chat_history.insert('end', "Assistant: ", "assistant")
        self.workflow_panel.chat_history.insert('end', "Chat functionality coming soon!\n\n")
        self.workflow_panel.chat_history.see('end')
        self.workflow_panel.chat_history.config(state=tk.DISABLED)
    
    def clear_chat(self):
        """Clear chat history."""
        self.workflow_panel.chat_history.config(state=tk.NORMAL)
        self.workflow_panel.chat_history.delete('1.0', 'end')
        self.workflow_panel.chat_history.config(state=tk.DISABLED)
    
    def _on_close(self):
        """Handle window close."""
        if self.running:
            if not messagebox.askyesno("Confirm", "Workflow is running. Exit anyway?"):
                return
        self.destroy()


def main():
    """Main entry point."""
    app = VSCodeStyleGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
