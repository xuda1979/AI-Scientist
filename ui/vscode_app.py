"""
VS Code-like GUI for AI Scientist.
Replicates the Visual Studio Code interface for a familiar user experience.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
import threading
import queue
import json
from typing import Optional, Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workflow_wrapper import run_from_gui, WorkflowParameters
from sciresearch_workflow import DEFAULT_MODEL

# VS Code Colors
COLORS = {
    'activity_bar': '#333333',
    'sidebar': '#252526',
    'editor_bg': '#1e1e1e',
    'panel_bg': '#1e1e1e',
    'status_bar': '#007acc',
    'status_bar_fg': '#ffffff',
    'tab_bg': '#2d2d2d',
    'tab_active_bg': '#1e1e1e',
    'tab_fg': '#969696',
    'tab_active_fg': '#ffffff',
    'text_fg': '#cccccc',
    'accent': '#007acc',
    'border': '#3e3e42',
    'list_hover': '#2a2d2e',
    'list_select': '#37373d',
    'button_bg': '#0e639c',
    'button_fg': '#ffffff',
    'button_hover': '#1177bb',
}

class VSCodeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Scientist - VS Code Mode")
        self.geometry("1200x800")
        self.configure(bg=COLORS['editor_bg'])
        
        # State
        self.current_workspace: Optional[Path] = None
        self.open_files: Dict[str, Any] = {} # path -> editor widget
        self.active_file: Optional[str] = None
        self.workflow_running = False
        self.log_queue = queue.Queue()
        
        # Layout
        self._setup_layout()
        self._setup_styles()
        
        # Components
        self._create_activity_bar()
        self._create_sidebar()
        self._create_main_area()
        self._create_status_bar()
        
        # Bindings
        self.bind('<Control-o>', self.open_folder_dialog)
        self.bind('<Control-s>', self.save_current_file)
        self.bind('<Control-Shift-P>', self.show_command_palette)
        self.bind('<Control-p>', self.show_command_palette) # Alternative
        
        # Start log poller
        self.after(100, self._poll_logs)
        
        # Start file monitor
        self.after(2000, self._monitor_files)

    def show_command_palette(self, event=None):
        """Show a simple command palette."""
        # Create a popup window
        palette = tk.Toplevel(self)
        palette.title("Command Palette")
        palette.geometry("600x300")
        palette.configure(bg=COLORS['activity_bar'])
        palette.transient(self)
        palette.grab_set()
        
        # Center it
        x = self.winfo_x() + (self.winfo_width() // 2) - 300
        y = self.winfo_y() + 50
        palette.geometry(f"+{x}+{y}")
        
        # Search box
        entry = tk.Entry(palette, bg='#3c3c3c', fg='white', font=('Segoe UI', 12), 
                        relief=tk.FLAT, insertbackground='white')
        entry.pack(fill=tk.X, padx=5, pady=5)
        entry.focus_set()
        
        # List
        listbox = tk.Listbox(palette, bg=COLORS['sidebar'], fg=COLORS['text_fg'],
                            font=('Segoe UI', 10), relief=tk.FLAT, highlightthickness=0)
        listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        commands = [
            ("File: Open Folder", self.open_folder_dialog),
            ("File: Save", self.save_current_file),
            ("AI Scientist: Start Workflow", self._start_workflow),
            ("AI Scientist: Stop Workflow", self._stop_workflow),
            ("View: Toggle Sidebar", lambda: self.sidebar.pack_forget() if self.sidebar.winfo_ismapped() else self.sidebar.pack(side=tk.LEFT, fill=tk.Y, before=self.content_area)),
            ("View: Toggle Panel", lambda: self.panel.pack_forget() if self.panel.winfo_ismapped() else self.panel.pack(side=tk.BOTTOM, fill=tk.X)),
        ]
        
        def filter_commands(event=None):
            query = entry.get().lower()
            listbox.delete(0, tk.END)
            for name, cmd in commands:
                if query in name.lower():
                    listbox.insert(tk.END, name)
            if listbox.size() > 0:
                listbox.selection_set(0)
                
        entry.bind('<KeyRelease>', filter_commands)
        filter_commands()
        
        def execute_command(event=None):
            selection = listbox.curselection()
            if selection:
                cmd_name = listbox.get(selection[0])
                for name, cmd in commands:
                    if name == cmd_name:
                        palette.destroy()
                        cmd()
                        break
        
        entry.bind('<Return>', execute_command)
        listbox.bind('<Double-1>', execute_command)
        palette.bind('<Escape>', lambda e: palette.destroy())

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('ActivityBar.TFrame', background=COLORS['activity_bar'])
        style.configure('Sidebar.TFrame', background=COLORS['sidebar'])
        style.configure('Editor.TFrame', background=COLORS['editor_bg'])
        style.configure('Panel.TFrame', background=COLORS['panel_bg'])
        style.configure('StatusBar.TFrame', background=COLORS['status_bar'])
        
        style.configure('TButton', background=COLORS['button_bg'], foreground=COLORS['button_fg'], borderwidth=0)
        style.map('TButton', background=[('active', COLORS['button_hover'])])
        
        style.configure('Treeview', 
                        background=COLORS['sidebar'], 
                        foreground=COLORS['text_fg'],
                        fieldbackground=COLORS['sidebar'],
                        borderwidth=0)
        style.map('Treeview', background=[('selected', COLORS['list_select'])])

    def _setup_layout(self):
        # Main container
        self.main_container = tk.Frame(self, bg=COLORS['editor_bg'])
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Activity Bar (Leftmost)
        self.activity_bar = tk.Frame(self.main_container, bg=COLORS['activity_bar'], width=50)
        self.activity_bar.pack(side=tk.LEFT, fill=tk.Y)
        self.activity_bar.pack_propagate(False)
        
        # Sidebar (Left)
        self.sidebar = tk.Frame(self.main_container, bg=COLORS['sidebar'], width=250)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        
        # Content Area (Right)
        self.content_area = tk.Frame(self.main_container, bg=COLORS['editor_bg'])
        self.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Editor Group (Top of Content)
        self.editor_group = tk.Frame(self.content_area, bg=COLORS['editor_bg'])
        self.editor_group.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Panel (Bottom of Content)
        self.panel = tk.Frame(self.content_area, bg=COLORS['panel_bg'], height=200)
        self.panel.pack(side=tk.BOTTOM, fill=tk.X)
        self.panel.pack_propagate(False)
        
        # Status Bar (Bottom of Window)
        self.status_bar = tk.Frame(self, bg=COLORS['status_bar'], height=22)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar.pack_propagate(False)

    def _create_activity_bar(self):
        # Icons (using text for now, could be images)
        self.activity_buttons = {}
        
        activities = [
            ("Explorer", "📁", self._show_explorer),
            ("Search", "🔍", self._show_search),
            ("Source Control", "gd", self._show_git),
            ("Run AI Scientist", "▶", self._show_run),
            ("Extensions", "🧩", self._show_extensions),
        ]
        
        for name, icon, command in activities:
            btn = tk.Button(self.activity_bar, text=icon, command=command,
                           bg=COLORS['activity_bar'], fg='#858585',
                           activebackground=COLORS['activity_bar'], activeforeground='white',
                           relief=tk.FLAT, font=('Segoe UI', 14), pady=10)
            btn.pack(fill=tk.X)
            self.activity_buttons[name] = btn
            
        # Settings at bottom
        settings_btn = tk.Button(self.activity_bar, text="⚙",
                                bg=COLORS['activity_bar'], fg='#858585',
                                activebackground=COLORS['activity_bar'], activeforeground='white',
                                relief=tk.FLAT, font=('Segoe UI', 14), pady=10)
        settings_btn.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_sidebar(self):
        # Sidebar Title
        self.sidebar_title = tk.Label(self.sidebar, text="EXPLORER", 
                                     bg=COLORS['sidebar'], fg=COLORS['text_fg'],
                                     font=('Segoe UI', 8), anchor='w', padx=10, pady=10)
        self.sidebar_title.pack(fill=tk.X)
        
        # Sidebar Content Container
        self.sidebar_content = tk.Frame(self.sidebar, bg=COLORS['sidebar'])
        self.sidebar_content.pack(fill=tk.BOTH, expand=True)
        
        # Initialize views
        self.views = {}
        self._init_explorer_view()
        self._init_run_view()
        
        # Default view
        self._show_explorer()

    def _init_explorer_view(self):
        view = tk.Frame(self.sidebar_content, bg=COLORS['sidebar'])
        self.views['Explorer'] = view
        
        # Treeview
        self.file_tree = ttk.Treeview(view, show='tree', selectmode='browse')
        self.file_tree.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(view, orient="vertical", command=self.file_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        
        self.file_tree.bind('<Double-1>', self._on_file_double_click)

    def _init_run_view(self):
        view = tk.Frame(self.sidebar_content, bg=COLORS['sidebar'])
        self.views['Run AI Scientist'] = view
        
        # Configuration Form
        tk.Label(view, text="AI SCIENTIST CONFIG", bg=COLORS['sidebar'], fg=COLORS['text_fg'], 
                 font=('Segoe UI', 8, 'bold')).pack(fill=tk.X, padx=10, pady=5)
        
        self.config_vars = {
            'topic': tk.StringVar(value="Machine Learning"),
            'model': tk.StringVar(value=DEFAULT_MODEL),
            'iterations': tk.IntVar(value=4),
            'quality': tk.DoubleVar(value=0.8),
        }
        
        # Form fields
        self._add_config_field(view, "Research Topic", self.config_vars['topic'])
        self._add_config_field(view, "Model", self.config_vars['model'])
        self._add_config_field(view, "Max Iterations", self.config_vars['iterations'])
        self._add_config_field(view, "Quality Threshold", self.config_vars['quality'])
        
        # Run Button
        self.run_btn = tk.Button(view, text="Start Workflow", command=self._start_workflow,
                                bg=COLORS['button_bg'], fg=COLORS['button_fg'],
                                relief=tk.FLAT, font=('Segoe UI', 10))
        self.run_btn.pack(fill=tk.X, padx=10, pady=20)
        
        # Stop Button
        self.stop_btn = tk.Button(view, text="Stop", command=self._stop_workflow,
                                 bg='#a10000', fg='white',
                                 relief=tk.FLAT, font=('Segoe UI', 10), state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, padx=10)

    def _add_config_field(self, parent, label, variable):
        frame = tk.Frame(parent, bg=COLORS['sidebar'])
        frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame, text=label, bg=COLORS['sidebar'], fg=COLORS['text_fg'], font=('Segoe UI', 9)).pack(anchor='w')
        tk.Entry(frame, textvariable=variable, bg='#3c3c3c', fg='white', relief=tk.FLAT, insertbackground='white').pack(fill=tk.X)

    def _create_main_area(self):
        # Tabs
        self.tab_bar = tk.Frame(self.editor_group, bg=COLORS['activity_bar'], height=35)
        self.tab_bar.pack(side=tk.TOP, fill=tk.X)
        self.tab_bar.pack_propagate(False)
        
        # Editor Container
        self.editor_container = tk.Frame(self.editor_group, bg=COLORS['editor_bg'])
        self.editor_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Welcome Screen
        self.welcome_label = tk.Label(self.editor_container, text="AI Scientist\nVS Code Mode", 
                                     bg=COLORS['editor_bg'], fg='#3e3e42', font=('Segoe UI', 32, 'bold'))
        self.welcome_label.pack(expand=True)
        
        # Panel (Terminal/Output)
        panel_header = tk.Frame(self.panel, bg=COLORS['activity_bar'], height=30)
        panel_header.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(panel_header, text="OUTPUT", bg=COLORS['activity_bar'], fg='white', padx=10).pack(side=tk.LEFT)
        
        self.output_text = ScrolledText(self.panel, bg=COLORS['panel_bg'], fg=COLORS['text_fg'],
                                       font=('Consolas', 10), relief=tk.FLAT)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def _create_status_bar(self):
        self.status_label = tk.Label(self.status_bar, text="Ready", bg=COLORS['status_bar'], fg='white', padx=10)
        self.status_label.pack(side=tk.LEFT)
        
        self.workspace_label = tk.Label(self.status_bar, text="No Workspace", bg=COLORS['status_bar'], fg='white', padx=10)
        self.workspace_label.pack(side=tk.RIGHT)

    # --- Logic ---

    def _switch_sidebar_view(self, name):
        # Hide all views
        for view in self.views.values():
            view.pack_forget()
        
        # Show selected
        if name in self.views:
            self.views[name].pack(fill=tk.BOTH, expand=True)
            self.sidebar_title.config(text=name.upper())
            
        # Update activity bar selection
        for btn_name, btn in self.activity_buttons.items():
            if btn_name == name:
                btn.config(fg='white', borderwidth=0, relief=tk.FLAT) # Highlight logic could be better
            else:
                btn.config(fg='#858585')

    def _show_explorer(self): self._switch_sidebar_view('Explorer')
    def _show_search(self): pass # TODO
    def _show_git(self): pass # TODO
    def _show_run(self): self._switch_sidebar_view('Run AI Scientist')
    def _show_extensions(self): pass # TODO

    def open_folder_dialog(self, event=None):
        folder = filedialog.askdirectory()
        if folder:
            self.load_workspace(folder)

    def load_workspace(self, folder):
        self.current_workspace = Path(folder)
        self.workspace_label.config(text=f"Workspace: {self.current_workspace.name}")
        self._refresh_file_tree()
        
        # Auto-switch to Explorer
        self._show_explorer()

    def _refresh_file_tree(self):
        self.file_tree.delete(*self.file_tree.get_children())
        if not self.current_workspace:
            return
            
        def add_node(parent, path):
            try:
                for p in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                    if p.name.startswith('.'): continue
                    oid = self.file_tree.insert(parent, 'end', text=f" {p.name}", open=False, values=[str(p)])
                    if p.is_dir():
                        # Dummy node for lazy loading or just recurse if small
                        add_node(oid, p)
            except PermissionError:
                pass

        add_node('', self.current_workspace)

    def _on_file_double_click(self, event):
        item = self.file_tree.selection()[0]
        path_str = self.file_tree.item(item, 'values')[0]
        path = Path(path_str)
        if path.is_file():
            self.open_file(path)

    def open_file(self, path):
        # Simple single editor for now
        if self.welcome_label:
            self.welcome_label.pack_forget()
            
        # Check if already open
        # For now, just replace content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clear existing editor (TODO: Tabs)
            for widget in self.editor_container.winfo_children():
                widget.destroy()
                
            editor = ScrolledText(self.editor_container, bg=COLORS['editor_bg'], fg=COLORS['text_fg'],
                                 font=('Consolas', 11), insertbackground='white', undo=True)
            editor.pack(fill=tk.BOTH, expand=True)
            editor.insert('1.0', content)
            
            self.active_file = path
            self.open_files[str(path)] = editor
            
            # Add tab (visual only for now)
            # TODO: Real tab management
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {e}")

    def save_current_file(self, event=None):
        if self.active_file and str(self.active_file) in self.open_files:
            editor = self.open_files[str(self.active_file)]
            content = editor.get('1.0', 'end-1c')
            try:
                with open(self.active_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.status_label.config(text=f"Saved {self.active_file.name}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")

    def _start_workflow(self):
        if not self.current_workspace:
            messagebox.showwarning("Warning", "Please open a workspace folder first.")
            return
            
        self.workflow_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Workflow Running...")
        
        # Prepare params
        params = {
            'output_dir': str(self.current_workspace),
            'topic': self.config_vars['topic'].get(),
            'model': self.config_vars['model'].get(),
            'max_iterations': self.config_vars['iterations'].get(),
            'quality_threshold': self.config_vars['quality'].get(),
            'modify_existing': True, # Always modify existing in this mode
            'enable_ideation': True,
        }
        
        # Run in thread
        self.cancel_event = threading.Event()
        self.workflow_thread = threading.Thread(target=self._run_workflow_thread, args=(params,))
        self.workflow_thread.start()

    def _run_workflow_thread(self, params):
        try:
            # Redirect stdout/stderr to queue
            class QueueWriter:
                def __init__(self, q): self.q = q
                def write(self, msg): self.q.put(msg)
                def flush(self): pass
            
            sys.stdout = QueueWriter(self.log_queue)
            sys.stderr = QueueWriter(self.log_queue)
            
            run_from_gui(params, self.cancel_event)
            
        except Exception as e:
            self.log_queue.put(f"\nError: {e}\n")
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.log_queue.put("WORKFLOW_FINISHED")

    def _stop_workflow(self):
        if self.cancel_event:
            self.cancel_event.set()
            self.log_queue.put("\nStopping workflow...\n")

    def _poll_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg == "WORKFLOW_FINISHED":
                    self.workflow_running = False
                    self.run_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.status_label.config(text="Workflow Finished")
                else:
                    self.output_text.insert(tk.END, msg)
                    self.output_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_logs)

    def _monitor_files(self):
        """Periodically check for file changes."""
        if not self.current_workspace:
            self.after(2000, self._monitor_files)
            return

        # Refresh tree if directory structure changed (simplified: just refresh every few seconds if running)
        if self.workflow_running:
            # Ideally we'd check for actual changes, but a full refresh is safer for now
            # to show new files generated by the AI
            # We save the open states to restore them
            open_nodes = []
            def get_open_nodes(item):
                if self.file_tree.item(item, 'open'):
                    open_nodes.append(self.file_tree.item(item, 'values')[0])
                for child in self.file_tree.get_children(item):
                    get_open_nodes(child)
            
            for child in self.file_tree.get_children(''):
                get_open_nodes(child)
                
            self._refresh_file_tree()
            
            # Restore open nodes (best effort)
            def restore_open(item):
                val = self.file_tree.item(item, 'values')
                if val and val[0] in open_nodes:
                    self.file_tree.item(item, open=True)
                for child in self.file_tree.get_children(item):
                    restore_open(child)
            
            for child in self.file_tree.get_children(''):
                restore_open(child)

        # Check active file for external changes
        if self.active_file and str(self.active_file) in self.open_files:
            try:
                mtime = os.path.getmtime(self.active_file)
                # Store last mtime in open_files dict? 
                # For now, just reload if it's not dirty in editor (simplified)
                # This is tricky without a proper dirty state tracking.
                # Let's just leave the editor alone to avoid overwriting user changes
                pass 
            except FileNotFoundError:
                pass

        self.after(2000, self._monitor_files)

if __name__ == "__main__":
    app = VSCodeApp()
    app.mainloop()
