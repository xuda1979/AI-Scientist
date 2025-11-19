
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol
import numpy as np
from ..evaluation.compute_accounting import ComputeAccountant, overhead_timer

@dataclass
class AlgoResult:
    utility: float
    details: Dict[str, Any]
    accountant_snapshot: Dict[str, float]

class Environment(Protocol):
    def reset(self, rng: np.random.Generator) -> None: ...
    def clone(self) -> 'Environment': ...
    def step_extend_thread(self, thread_id: int, rng: np.random.Generator) -> float: ...
    def step_verify(self, rng: np.random.Generator) -> float: ...
    def utility(self) -> float: ...
    def n_threads(self) -> int: ...

class Algorithm(Protocol):
    name: str
    def run(self, env: Environment, budget: int, rng: np.random.Generator, acc: ComputeAccountant) -> AlgoResult: ...

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))
