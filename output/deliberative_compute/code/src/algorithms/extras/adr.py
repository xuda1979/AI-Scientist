
from __future__ import annotations
import numpy as np
from typing import Dict, Any
from ..base import Algorithm, AlgoResult, Environment
from ...evaluation.compute_accounting import ComputeAccountant, overhead_timer

class ADR(Algorithm):
    """Abduction–Deduction–Refutation: lightweight synthetic variant.
    - Abduction: pick promising threads (by probe gains)
    - Deduction: extend them
    - Refutation: occasionally call a 'verify' action when projected info gain is high
    """
    name = "adr"

    def __init__(self, verify_every: int = 5, verify_weight: float = 0.2):
        self.verify_every = int(verify_every)
        self.verify_weight = float(verify_weight)

    def run(self, env: Environment, budget: int, rng: np.random.Generator, acc: ComputeAccountant) -> AlgoResult:
        env = env.clone()
        m = env.n_threads()
        scores = np.zeros(m, dtype=float)

        for t in range(1, budget + 1):
            with overhead_timer(acc):
                # Abduction: softmax over scores to pick a thread
                logits = scores - scores.max()
                probs = np.exp(logits)
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones(m)/m
                i = int(rng.choice(m, p=probs))
                acc.count_decision(1)
            acc.charge("generation", 1)
            g = env.step_extend_thread(i, rng)
            # Deduction update
            scores[i] = 0.7 * scores[i] + 0.3 * g
            # Refutation/verification
            if t % self.verify_every == 0:
                with overhead_timer(acc):
                    expected_gain = self.verify_weight * np.abs(scores).mean()
                if expected_gain > 0.0:
                    acc.charge("verification", 1)
                    env.step_verify(rng)

        return AlgoResult(env.utility(), {"scores": scores.tolist()}, acc.snapshot())
