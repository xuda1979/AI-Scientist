"""
Auto-Pilot configuration — all tunable knobs in one place.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any


# ── Research pipeline stages (canonical order) ──────────────────────────
STAGES = [
    "literature_review",
    "research_gap_analysis",
    "hypothesis_formation",
    "methodology_design",
    "experiment_code_generation",
    "experiment_execution",
    "results_analysis",
    "initial_draft",
    "section_deep_dive",
    "internal_review",
    "revision",
    "bibliography_polish",
    "final_compilation",
]


@dataclass
class StageConfig:
    """Per-stage tuning knobs."""
    max_attempts: int = 3
    temperature: float = 0.4
    quality_threshold: float = 0.7
    max_tokens: int = 16000
    prompt_candidates: int = 3          # how many prompt variants to score
    enable_self_reflection: bool = True  # LLM rates its own output
    timeout: int = 600                  # seconds


@dataclass
class AutoPilotConfig:
    """Master configuration for the auto-pilot agent."""

    # ── LLM settings ────────────────────────────────────────────────────
    model: str = "gpt-5-pro"
    fallback_models: List[str] = field(default_factory=lambda: ["gpt-5", "gpt-4o"])
    request_timeout: int = 3600

    # ── Pipeline settings ───────────────────────────────────────────────
    stages: List[str] = field(default_factory=lambda: list(STAGES))
    max_global_iterations: int = 3      # full pipeline reruns
    max_stage_retries: int = 3
    quality_gate_threshold: float = 0.75
    abort_on_critical_failure: bool = True

    # ── Prompt optimiser ────────────────────────────────────────────────
    prompt_candidates_per_stage: int = 3
    use_meta_prompt: bool = True        # LLM generates its own prompts
    prompt_selection_strategy: str = "score"  # "score" | "tournament" | "ensemble"

    # ── Workspace / file management ─────────────────────────────────────
    project_dir: Optional[str] = None   # filled at runtime
    allowed_extensions: List[str] = field(default_factory=lambda: [
        ".tex", ".bib", ".py", ".md", ".txt", ".csv", ".json",
        ".yaml", ".yml", ".toml", ".r", ".R", ".sh", ".bat",
        ".html", ".js", ".cpp", ".c", ".h", ".java",
    ])
    max_file_size_bytes: int = 500_000  # 500 KB per file
    create_backups: bool = True

    # ── Content protection ──────────────────────────────────────────────
    content_protection: bool = True
    max_word_loss_pct: float = 15.0     # reject revision that loses > 15 % words

    # ── Per-stage overrides (stage_name → StageConfig fields) ──────────
    stage_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ── Logging / tracing ───────────────────────────────────────────────
    verbose: bool = True
    save_trace: bool = True             # dump every LLM call & response
    trace_dir: str = "autopilot_trace"

    # ── Document type ───────────────────────────────────────────────────
    document_type: str = "auto"

    # ── Research topic info (filled at runtime) ─────────────────────────
    topic: str = ""
    field_name: str = ""
    research_question: str = ""

    # ── Helpers ──────────────────────────────────────────────────────────

    def get_stage_config(self, stage_name: str) -> StageConfig:
        """Return a StageConfig with any per-stage overrides applied."""
        base = StageConfig()
        overrides = self.stage_overrides.get(stage_name, {})
        for key, val in overrides.items():
            if hasattr(base, key):
                setattr(base, key, val)
        return base

    @classmethod
    def from_file(cls, path: Path) -> "AutoPilotConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
