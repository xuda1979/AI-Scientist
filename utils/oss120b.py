"""Utilities for loading the OpenAI OSS 120B model with optional NPU support."""
from __future__ import annotations
from typing import Any

try:
    import torch
except ImportError as e:  # pragma: no cover - dependency management
    torch = None

from models.openai_120b import OpenAI120B


def load_model(device: str = "cpu") -> Any:
    """Load the placeholder 120B model on the requested device.

    Parameters
    ----------
    device: str
        Target device ("cpu", "cuda", or "npu").
    """
    model = OpenAI120B()
    dev = device.lower()
    if dev == "npu":
        if torch is None or not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU runtime unavailable; install torch-npu and drivers")
        model.to("npu")
    elif dev in {"cuda", "gpu"}:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available")
        model.to("cuda")
    else:
        model.to("cpu")
    return model

__all__ = ["load_model"]
