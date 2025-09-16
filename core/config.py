#!/usr/bin/env python3
"""
Core configuration for the sciresearch workflow.
"""
from __future__ import annotations
import json
import logging
import signal
import sys
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List


def timeout_input(prompt: str, timeout: int = 30, default: str = "") -> str:
    """Get user input with timeout, returning default if no input within timeout."""
    result = [default]
    
    def input_handler():
        try:
            result[0] = input(prompt)
        except EOFError:
            result[0] = default
    
    thread = threading.Thread(target=input_handler)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"Timeout reached. Using default: {default}")
        return default
    
    return result[0]


@dataclass
class WorkflowConfig:
    """Configuration class for workflow settings."""
    quality_threshold: float = 1.0
    max_iterations: int = 5
    request_timeout: int = 3600
    max_retries: int = 3
    no_early_stopping: bool = False
    enable_pdf_review: bool = False
    reference_validation: bool = True
    figure_validation: bool = True
    research_ideation: bool = True
    diff_output_tracking: bool = True
    content_protection: bool = True
    content_protection_threshold: float = 0.15  # Changed from 15.0 to 0.15 (15%)
    auto_approve_changes: bool = False
    fallback_models: List[str] = None
    max_quality_history_size: int = 100
    
    def __post_init__(self):
        """Initialize default fallback models if not provided."""
        if self.fallback_models is None:
            self.fallback_models = ["gpt-4o", "gpt-4"]
            
    @classmethod
    def from_file(cls, config_path: Path) -> 'WorkflowConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return cls()  # Return default config
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)


def setup_workflow_logging(log_level=logging.INFO, log_dir: Optional[Path] = None) -> logging.Logger:
    """Set up comprehensive logging for the workflow."""
    logger = logging.getLogger('sciresearch_workflow')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log directory is specified
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"workflow_{_nowstamp()}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    return logger


def _nowstamp() -> str:
    """Generate timestamp string for filenames."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _prepare_project_dir(output_dir: Path, modify_existing: bool) -> Path:
    """Prepare the project directory for workflow execution."""
    if modify_existing and output_dir.exists():
        return output_dir
    else:
        # Create new directory with timestamp
        timestamp = _nowstamp()
        project_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir


def _collect_project_files(project_dir: Path) -> str:
    """Collect all relevant project files for context."""
    if not project_dir.exists():
        return "No project directory found."
    
    content = ""
    
    # Define file extensions and limits
    text_extensions = {'.tex', '.py', '.md', '.txt', '.yaml', '.yml', '.json'}
    data_extensions = {'.csv', '.dat'}
    max_file_size = 50000  # 50KB limit per file
    
    try:
        for file_path in sorted(project_dir.rglob('*')):
            if not file_path.is_file():
                continue
                
            # Skip large files and certain file types
            if file_path.stat().st_size > max_file_size:
                continue
                
            if file_path.suffix.lower() in text_extensions:
                try:
                    file_content = file_path.read_text(encoding='utf-8', errors='ignore')
                    content += f"\n--- {file_path.name} ---\n{file_content}\n"
                except (UnicodeDecodeError, PermissionError):
                    content += f"\n--- {file_path.name} ---\n[Binary or inaccessible file]\n"
                    
            elif file_path.suffix.lower() in data_extensions:
                try:
                    file_content = file_path.read_text(encoding='utf-8', errors='ignore')
                    # Truncate large data files
                    if len(file_content) > 2000:
                        file_content = file_content[:2000] + "\n... [truncated]"
                    content += f"\n--- {file_path.name} ---\n{file_content}\n"
                except (UnicodeDecodeError, PermissionError):
                    content += f"\n--- {file_path.name} ---\n[Binary or inaccessible file]\n"
                    
    except Exception as e:
        content += f"\nError collecting project files: {str(e)}\n"
    
    return content or "No relevant project files found."


class SecurityError(Exception):
    """Custom exception for security violations."""
    pass


def _validate_code_security(code: str) -> None:
    """Validate simulation code for security risks before execution."""
    import re
    
    dangerous_patterns = [
        (r'import\s+os\b', "OS module import detected"),
        (r'import\s+subprocess\b', "Subprocess module import detected"),
        (r'exec\s*\(', "exec() function call detected"),
        (r'eval\s*\(', "eval() function call detected"),
        (r'__import__\s*\(', "__import__() function call detected"),
        (r'open\s*\(["\'][^"\']*["\'][^)]*["\']w', "File writing operation detected"),
        (r'urllib\.request', "Network request detected"),
        (r'socket\.', "Socket operation detected"),
        (r'shutil\.rmtree', "Directory deletion detected"),
        (r'os\.system', "System command execution detected"),
        (r'os\.popen', "Process execution detected"),
    ]
    
    security_issues = []
    for pattern, description in dangerous_patterns:
        if re.search(pattern, code):
            security_issues.append(description)
    
    if security_issues:
        raise SecurityError(f"Security risks detected: {'; '.join(security_issues)}")
