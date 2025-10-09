"""
Configuration management for SciResearch Workflow.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class WorkflowConfig:
    """Configuration for the research workflow."""
    
    # Core workflow settings
    max_iterations: int = 4
    quality_threshold: float = 0.8
    
    # AI model settings
    fallback_models: List[str] = field(default_factory=lambda: ["gpt-4", "gpt-3.5-turbo"])
    use_test_time_scaling: bool = False
    initial_draft_candidates: int = 1
    
    # Validation settings
    reference_validation: bool = True
    figure_validation: bool = True
    enable_pdf_review: bool = True
    
    # Feature flags
    enable_ideation: bool = True
    enable_optimization: bool = True
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'WorkflowConfig':
        """Load configuration from JSON file."""
        import json
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return cls(**data)
        return cls()
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2)


# Default models and constants
DEFAULT_MODEL = "gpt-4"
SUPPORTED_MODELS = [
    "gpt-4", "gpt-4-turbo", "gpt-5", "gpt-5-pro",
    "claude-3-opus", "claude-3-sonnet",
    "gemini-pro"
]

# Timeout settings
DEFAULT_TIMEOUT = 3600
MAX_TIMEOUT = 7200

# Quality thresholds
MIN_QUALITY_THRESHOLD = 0.1
MAX_QUALITY_THRESHOLD = 1.0

# File size limits (in characters)
MAX_PAPER_LENGTH = 100000
MIN_PAPER_LENGTH = 1000
