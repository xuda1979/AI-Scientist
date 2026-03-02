"""
WorkspaceManager — Copilot-like file management for the research project folder.

Reads, writes, patches, diffs, and backs up any file in the project directory.
The LLM can request arbitrary file operations through a structured action schema,
and this module executes them safely with content-protection checks.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages all files inside a research project directory."""

    def __init__(
        self,
        project_dir: Path,
        allowed_extensions: Optional[List[str]] = None,
        max_file_size: int = 500_000,
        create_backups: bool = True,
        content_protection: bool = True,
        max_word_loss_pct: float = 15.0,
    ):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_extensions = allowed_extensions or [
            ".tex", ".bib", ".py", ".md", ".txt", ".csv", ".json",
            ".yaml", ".yml", ".toml",
        ]
        self.max_file_size = max_file_size
        self.create_backups = create_backups
        self.content_protection = content_protection
        self.max_word_loss_pct = max_word_loss_pct
        self._backup_dir = self.project_dir / ".autopilot_backups"

    # ── Reading ──────────────────────────────────────────────────────────

    def read_file(self, relative_path: str) -> str:
        """Read a single file. Raises FileNotFoundError if missing."""
        fp = self._resolve(relative_path)
        return fp.read_text(encoding="utf-8", errors="replace")

    def list_files(self, pattern: str = "*") -> List[str]:
        """Return relative paths matching *pattern* (glob)."""
        results = []
        for fp in sorted(self.project_dir.rglob(pattern)):
            if fp.is_file() and not self._is_excluded(fp):
                results.append(str(fp.relative_to(self.project_dir)))
        return results

    def snapshot(self, include_content: bool = True, max_per_file: int = 0) -> str:
        """
        Build a textual snapshot of **every** relevant file in the project.
        Used to give the LLM full workspace context (like Copilot).

        Args:
            include_content: If True, include file contents; otherwise just names.
            max_per_file: If > 0, truncate each file to this many characters.
        """
        parts: List[str] = []
        for rel in self.list_files():
            fp = self.project_dir / rel
            if fp.stat().st_size > self.max_file_size:
                parts.append(f"=== FILE: {rel} === [skipped — {fp.stat().st_size:,} bytes]")
                continue
            if not include_content:
                parts.append(f"=== FILE: {rel} ===")
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                if max_per_file and len(text) > max_per_file:
                    text = text[:max_per_file] + f"\n... [truncated at {max_per_file:,} chars]"
                parts.append(f"=== FILE: {rel} ===\n{text}")
            except Exception as exc:
                parts.append(f"=== FILE: {rel} === [error: {exc}]")
        return "\n\n".join(parts) if parts else "(empty project)"

    def file_tree(self) -> str:
        """Pretty-print the project file tree."""
        lines = [str(self.project_dir.name) + "/"]
        for rel in self.list_files():
            depth = rel.count(os.sep) + rel.count("/")
            indent = "  " * depth
            lines.append(f"{indent}{Path(rel).name}")
        return "\n".join(lines)

    # ── Writing ──────────────────────────────────────────────────────────

    def write_file(self, relative_path: str, content: str, force: bool = False) -> str:
        """
        Write (create or overwrite) a file. Returns a status message.
        Content-protection is applied to .tex files unless *force* is True.
        """
        fp = self._resolve(relative_path, must_exist=False)
        fp.parent.mkdir(parents=True, exist_ok=True)

        # Back up existing file
        if fp.exists() and self.create_backups:
            self._backup(fp)

        # Content protection for LaTeX
        if fp.suffix == ".tex" and self.content_protection and not force and fp.exists():
            old = fp.read_text(encoding="utf-8", errors="replace")
            ok, reason = self._check_content_protection(old, content)
            if not ok:
                logger.warning("Content protection rejected write to %s: %s", relative_path, reason)
                return f"REJECTED ({reason})"

        fp.write_text(content, encoding="utf-8")
        logger.info("Wrote %s (%d bytes)", relative_path, len(content))
        return f"OK — wrote {relative_path} ({len(content):,} bytes)"

    def patch_file(self, relative_path: str, search: str, replace: str) -> str:
        """
        Apply a search-and-replace patch to a file.
        Only replaces the FIRST occurrence.
        """
        fp = self._resolve(relative_path)
        old_text = fp.read_text(encoding="utf-8", errors="replace")
        if search not in old_text:
            return f"PATCH FAILED — search string not found in {relative_path}"
        if self.create_backups:
            self._backup(fp)
        new_text = old_text.replace(search, replace, 1)
        fp.write_text(new_text, encoding="utf-8")
        return f"OK — patched {relative_path}"

    def delete_file(self, relative_path: str) -> str:
        """Delete a file (with backup)."""
        fp = self._resolve(relative_path)
        if self.create_backups:
            self._backup(fp)
        fp.unlink()
        return f"OK — deleted {relative_path}"

    def rename_file(self, old_path: str, new_path: str) -> str:
        """Rename / move a file within the project."""
        src = self._resolve(old_path)
        dst = self._resolve(new_path, must_exist=False)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self.create_backups:
            self._backup(src)
        src.rename(dst)
        return f"OK — renamed {old_path} → {new_path}"

    # ── Batch operations (called by LLM) ─────────────────────────────────

    def apply_actions(self, actions: List[Dict[str, Any]]) -> List[str]:
        """
        Execute a list of file actions from the LLM.

        Each action is a dict with:
            {"action": "write"|"patch"|"delete"|"rename",
             "path": "relative/path",
             ...action-specific keys...}
        """
        results: List[str] = []
        for act in actions:
            try:
                kind = act.get("action", "").lower()
                path = act.get("path", "")
                if kind == "write":
                    res = self.write_file(path, act.get("content", ""), force=act.get("force", False))
                elif kind == "patch":
                    res = self.patch_file(path, act.get("search", ""), act.get("replace", ""))
                elif kind == "delete":
                    res = self.delete_file(path)
                elif kind == "rename":
                    res = self.rename_file(path, act.get("new_path", ""))
                else:
                    res = f"UNKNOWN ACTION: {kind}"
                results.append(res)
            except Exception as exc:
                results.append(f"ERROR on {act}: {exc}")
        return results

    # ── Diffing ──────────────────────────────────────────────────────────

    def diff(self, relative_path: str, new_content: str) -> str:
        """Return a unified diff between current file and proposed new content."""
        fp = self._resolve(relative_path)
        old_lines = fp.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        return "".join(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{relative_path}", tofile=f"b/{relative_path}"))

    # ── Internal helpers ─────────────────────────────────────────────────

    def _resolve(self, relative_path: str, must_exist: bool = True) -> Path:
        """Resolve a relative path inside the project dir (with safety check)."""
        fp = (self.project_dir / relative_path).resolve()
        # Prevent directory traversal
        if not str(fp).startswith(str(self.project_dir.resolve())):
            raise PermissionError(f"Path escapes project directory: {relative_path}")
        if must_exist and not fp.exists():
            raise FileNotFoundError(f"Not found: {relative_path}")
        return fp

    def _is_excluded(self, fp: Path) -> bool:
        """Check whether a file should be excluded from listing."""
        exclude_dirs = {"__pycache__", ".git", "node_modules", ".autopilot_backups", "autopilot_trace"}
        exclude_exts = {".pyc", ".pyo", ".aux", ".bbl", ".blg", ".out", ".pdf",
                        ".npy", ".npz", ".pkl", ".cache", ".o", ".obj", ".so", ".dll", ".exe"}
        if any(part in exclude_dirs for part in fp.parts):
            return True
        if fp.suffix.lower() in exclude_exts:
            return True
        return False

    def _backup(self, fp: Path) -> None:
        """Create a timestamped backup of a file."""
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{fp.stem}_{ts}{fp.suffix}"
        shutil.copy2(fp, self._backup_dir / backup_name)

    def _check_content_protection(self, old_text: str, new_text: str) -> Tuple[bool, str]:
        """Simple content-protection: reject if word count drops too much."""
        old_wc = len(old_text.split())
        new_wc = len(new_text.split())
        if old_wc == 0:
            return True, ""
        loss_pct = (old_wc - new_wc) / old_wc * 100
        if loss_pct > self.max_word_loss_pct:
            return False, f"word count dropped {loss_pct:.1f}% (>{self.max_word_loss_pct}%)"
        return True, ""

    # ── Parse LLM response into file actions ─────────────────────────────

    @staticmethod
    def parse_file_actions_from_response(response: str) -> List[Dict[str, Any]]:
        """
        Parse structured file-change blocks from an LLM response.

        Expected format in the LLM response::

            ```file_actions
            [
              {"action": "write", "path": "paper.tex", "content": "..."},
              {"action": "patch", "path": "simulation.py", "search": "old", "replace": "new"},
              {"action": "delete", "path": "old_data.csv"}
            ]
            ```

        Also supports the simpler per-file format::

            === FILE: paper.tex ===
            <full file content>

        Returns a list of action dicts.
        """
        actions: List[Dict[str, Any]] = []

        # Try structured JSON block first
        json_match = re.search(r"```file_actions\s*\n(.*?)```", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Try JSON array block (without file_actions label)
        json_array_match = re.search(r"```json\s*\n(\[.*?\])\s*```", response, re.DOTALL)
        if json_array_match:
            try:
                parsed = json.loads(json_array_match.group(1))
                if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: parse === FILE: <path> === blocks
        file_blocks = re.findall(
            r"===\s*FILE:\s*(.+?)\s*===\s*\n(.*?)(?=\n===\s*FILE:|\Z)",
            response,
            re.DOTALL,
        )
        for path, content in file_blocks:
            actions.append({"action": "write", "path": path.strip(), "content": content.rstrip()})

        return actions
