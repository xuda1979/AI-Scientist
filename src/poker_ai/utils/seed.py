from __future__ import annotations

import logging
import os
import random
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Set RNG seeds for Python, NumPy, and Torch (if available), including CUDA/MPS."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        warnings.warn("PyTorch is not installed; skipping torch seeding.", stacklevel=2)
        return

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "mps") and torch.mps is not None and torch.mps.is_available():
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        logger.exception("Encountered an error while configuring torch seeds")
        raise


def forked_worker_seed(base_seed: int, rank: int) -> int:
    """Derive a distinct child seed (e.g., for self-play workers or DataLoader workers)."""

    return (base_seed * 0x9E3779B1 + rank) & 0xFFFFFFFF
