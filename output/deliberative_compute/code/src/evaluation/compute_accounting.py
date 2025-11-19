
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ComputeAccountant:
    """Tracks compute usage and metareasoning overhead.

    We count micro-actions (token-equivalent units) for:
      - generation
      - verification
      - retrieval
      - other (catch-all)

    We also measure *overhead_seconds* spent in scheduling/control code (excluding env actions).
    Use the overhead_timer context manager around scheduling sections.
    """
    generation: int = 0
    verification: int = 0
    retrieval: int = 0
    other: int = 0
    overhead_seconds: float = 0.0
    decisions: int = 0

    def charge(self, kind: str, units: int = 1) -> None:
        if kind not in {"generation", "verification", "retrieval", "other"}:
            raise ValueError(f"unknown compute kind: {kind}")
        setattr(self, kind, getattr(self, kind) + int(units))

    def count_decision(self, n: int = 1) -> None:
        self.decisions += int(n)

    def snapshot(self) -> Dict[str, float]:
        return {
            "generation": float(self.generation),
            "verification": float(self.verification),
            "retrieval": float(self.retrieval),
            "other": float(self.other),
            "overhead_seconds": float(self.overhead_seconds),
            "decisions": float(self.decisions),
            "total_units": float(self.generation + self.verification + self.retrieval + self.other),
        }

class overhead_timer:
    """Context manager to measure pure scheduling overhead in seconds."""
    def __init__(self, accountant: ComputeAccountant):
        self.acc = accountant
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        self.acc.overhead_seconds += dt
        return False
