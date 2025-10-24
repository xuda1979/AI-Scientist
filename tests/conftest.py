"""Test configuration to ensure local packages take precedence over similarly named dependencies."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

src_path_str = str(SRC_PATH)
if SRC_PATH.exists() and src_path_str not in sys.path:
    try:
        project_index = sys.path.index(project_root_str)
    except ValueError:
        project_index = -1

    insert_at = project_index + 1 if project_index >= 0 else 0
    sys.path.insert(insert_at, src_path_str)

existing = sys.modules.get("core")
if existing is not None:
    module_file = getattr(existing, "__file__", "") or ""
    package_paths = getattr(existing, "__path__", [])
    try:
        root = PROJECT_ROOT.resolve()
    except Exception:  # pragma: no cover - resolution failure is unexpected
        root = PROJECT_ROOT

    def _is_within_root(path: str) -> bool:
        try:
            return Path(path).resolve().is_relative_to(root)
        except AttributeError:  # pragma: no cover - Python < 3.9 fallback
            path_obj = Path(path).resolve()
            return str(path_obj).startswith(str(root))
        except FileNotFoundError:
            # If the path no longer exists we treat it as external to force a reload.
            return False

    in_root = False
    if module_file:
        in_root = _is_within_root(module_file)
    if not in_root and package_paths:
        for entry in package_paths:
            if _is_within_root(entry):
                in_root = True
                break

    if not in_root:
        sys.modules.pop("core", None)

importlib.import_module("core")
