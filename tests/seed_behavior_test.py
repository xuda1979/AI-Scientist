import builtins
import os
import sys
import types
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

from poker_ai.utils import seed as seed_module
import self_play


def test_seed_everything_warns_when_torch_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch is not installed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    original_hash_seed = os.environ.get("PYTHONHASHSEED")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            seed_module.seed_everything(123)
    finally:
        if original_hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = original_hash_seed

    assert any("PyTorch" in str(w.message) for w in caught), "Expected torch import warning"


def test_seed_everything_reraises_on_torch_errors(monkeypatch):
    fake_torch = types.ModuleType("torch")

    def fail_seed(_seed):
        raise RuntimeError("failure seeding torch")

    fake_torch.manual_seed = fail_seed
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=True)
    )
    fake_torch.mps = types.SimpleNamespace(is_available=lambda: False)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    original_hash_seed = os.environ.get("PYTHONHASHSEED")
    try:
        with pytest.raises(RuntimeError):
            seed_module.seed_everything(456)
    finally:
        if original_hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = original_hash_seed


def test_maybe_seed_warns_when_seed_function_missing(monkeypatch):
    monkeypatch.setenv("POKER_AI_SEED", "99")
    monkeypatch.setattr(self_play, "seed_everything", None, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        self_play._maybe_seed()

    assert any("seed_everything" in str(w.message) for w in caught)


def test_worker_init_warns_when_utilities_missing(monkeypatch):
    monkeypatch.setenv("POKER_AI_SEED", "100")
    monkeypatch.setattr(self_play, "forked_worker_seed", None, raising=False)
    monkeypatch.setattr(self_play, "seed_everything", None, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        self_play._worker_init(rank=3)

    assert caught, "Expected a warning when worker seeding utilities are missing"
    assert any("forked_worker_seed" in str(w.message) or "seed_everything" in str(w.message) for w in caught)
