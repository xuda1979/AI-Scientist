
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List
from ..algorithms.base import Environment

@dataclass
class ThreadSpec:
    amplitude: float  # base gain magnitude
    decay: float      # per-depth decay in [0,1)
    noise: float      # noise std for each step

class SyntheticEnv(Environment):
    """A simple, stationary environment with m threads.
    Extending thread i at depth d yields a random incremental *raw* gain:
        g_i(d) = amplitude_i * (decay_i ** (d-1)) + Normal(0, noise_i)
    The final utility maps total raw gain G to [0,1] via: U = 1 - exp(-max(0, G)).
    A 'verify' action slightly denoises the accumulated gain (toy for ADR/PSV).
    """
    def __init__(self, threads: List[ThreadSpec]):
        self._spec = [ThreadSpec(t.amplitude, t.decay, t.noise) for t in threads]
        self._depths = [0 for _ in threads]
        self._raw_total = 0.0
        self._rng = None  # set on reset

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._depths = [0 for _ in self._spec]
        self._raw_total = 0.0

    def clone(self) -> 'SyntheticEnv':
        env = SyntheticEnv(self._spec)
        env._depths = list(self._depths)
        env._raw_total = float(self._raw_total)
        env._rng = self._rng
        return env

    def n_threads(self) -> int:
        return len(self._spec)

    def step_extend_thread(self, thread_id: int, rng: np.random.Generator) -> float:
        i = int(thread_id)
        self._depths[i] += 1
        t = self._spec[i]
        d = self._depths[i]
        mean = t.amplitude * (t.decay ** max(0, d - 1))
        gain = float(rng.normal(mean, t.noise))
        self._raw_total += max(0.0, gain)  # gains cannot be negative in our toy
        return gain

    def step_verify(self, rng: np.random.Generator) -> float:
        # small denoising effect: boost total by a fraction of recent expected gain
        delta = 0.01 * sum(ts.amplitude for ts in self._spec) / max(1, len(self._spec))
        self._raw_total += delta
        return delta

    def utility(self) -> float:
        G = max(0.0, self._raw_total)
        return float(1.0 - np.exp(-G))

def make_synthetic_instance(m: int, rng: np.random.Generator) -> SyntheticEnv:
    threads = []
    for _ in range(m):
        amp = float(rng.uniform(0.04, 0.24))
        decay = float(rng.uniform(0.6, 0.95))
        noise = float(rng.uniform(0.01, 0.06))
        threads.append(ThreadSpec(amp, decay, noise))
    env = SyntheticEnv(threads)
    env.reset(rng)
    return env
