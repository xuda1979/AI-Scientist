"""
AgentConfig — unified configuration for the autonomous research agent.

Consolidates all configuration into one place with sensible defaults.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as dataclass_field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Research phases (high-level) and their constituent stages
# ---------------------------------------------------------------------------
PHASES = {
    "research": [
        "literature_search",
        "literature_synthesis",
        "gap_analysis",
        "hypothesis_formation",
    ],
    "design": [
        "methodology_design",
        "experiment_planning",
    ],
    "implementation": [
        "code_generation",
        "code_testing",
        "experiment_execution",
        "results_analysis",
    ],
    "writing": [
        "outline_generation",
        "initial_draft",
        "figure_generation",
        "section_deepening",
    ],
    "review": [
        "internal_review",
        "revision",
        "bibliography_polish",
        "final_compilation",
    ],
}

ALL_STAGES = [stage for stages in PHASES.values() for stage in stages]


@dataclass
class ToolConfig:
    """Configuration for external tool integrations."""
    # Literature search
    semantic_scholar_api_key: str = ""
    arxiv_max_results: int = 20
    enable_web_search: bool = True

    # Code execution
    python_executable: str = "python"
    code_execution_timeout: int = 300
    max_retries_on_error: int = 3
    sandbox_mode: bool = True  # run code in isolated subprocess

    # LaTeX
    pdflatex_path: str = "pdflatex"
    bibtex_path: str = "bibtex"
    latex_timeout: int = 180

    # Figures
    matplotlib_backend: str = "Agg"
    figure_dpi: int = 300
    figure_format: str = "pdf"


@dataclass
class SpecialistConfig:
    """Configuration for specialist agent personas."""
    name: str = ""
    system_prompt: str = ""
    model: str = ""  # empty = use default
    temperature: float = 0.4
    max_tokens: int = 16000


@dataclass
class AgentConfig:
    """Master configuration for the autonomous research agent."""

    # ── Research topic ──────────────────────────────────────────────────
    topic: str = ""
    field: str = ""
    research_question: str = ""

    # ── LLM settings ────────────────────────────────────────────────────
    model: str = dataclass_field(default_factory=lambda: os.environ.get("SCI_MODEL", "gpt-5-pro"))
    fallback_models: List[str] = dataclass_field(default_factory=lambda: ["gpt-5", "gpt-4o", "gpt-4"])
    api_key: str = dataclass_field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    temperature: float = 0.4
    max_tokens: int = 16000
    request_timeout: int = 600

    # ── Pipeline control ────────────────────────────────────────────────
    phases: List[str] = dataclass_field(default_factory=lambda: list(PHASES.keys()))
    skip_stages: List[str] = dataclass_field(default_factory=list)
    max_global_iterations: int = 2
    max_stage_retries: int = 3
    quality_threshold: float = 0.75

    # ── Project / workspace ─────────────────────────────────────────────
    project_dir: str = ""
    output_dir: str = "output"

    # ── Tool configuration ──────────────────────────────────────────────
    tools: ToolConfig = dataclass_field(default_factory=ToolConfig)

    # ── Specialist agents ───────────────────────────────────────────────
    specialists: Dict[str, SpecialistConfig] = dataclass_field(default_factory=dict)

    # ── Memory & context ────────────────────────────────────────────────
    max_context_tokens: int = 120000
    enable_memory_compression: bool = True
    memory_dir: str = "agent_memory"

    # ── Content protection ──────────────────────────────────────────────
    content_protection: bool = True
    max_word_loss_pct: float = 15.0

    # ── Logging / tracing ───────────────────────────────────────────────
    verbose: bool = True
    save_trace: bool = True
    trace_dir: str = "agent_trace"
    log_level: str = "INFO"

    # ── Document type ───────────────────────────────────────────────────
    document_type: str = "research_paper"

    # ── Helpers ──────────────────────────────────────────────────────────

    def get_active_stages(self) -> List[str]:
        """Return the ordered list of stages to execute, minus skipped ones."""
        stages = []
        for phase in self.phases:
            for stage in PHASES.get(phase, []):
                if stage not in self.skip_stages:
                    stages.append(stage)
        return stages

    def get_phase_for_stage(self, stage: str) -> str:
        """Return which phase a stage belongs to."""
        for phase, stages in PHASES.items():
            if stage in stages:
                return phase
        return "unknown"

    def get_specialist_for_stage(self, stage: str) -> Optional[SpecialistConfig]:
        """Return specialist config for a stage's phase, if any."""
        phase = self.get_phase_for_stage(stage)
        return self.specialists.get(phase)

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentConfig":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle nested tool config
        if "tools" in data and isinstance(data["tools"], dict):
            data["tools"] = ToolConfig(**data["tools"])

        # Handle nested specialist configs
        if "specialists" in data and isinstance(data["specialists"], dict):
            data["specialists"] = {
                k: SpecialistConfig(**v) if isinstance(v, dict) else v
                for k, v in data["specialists"].items()
            }

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
