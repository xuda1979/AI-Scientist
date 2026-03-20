"""
Writing Tools — LaTeX authoring, compilation, bibliography, and file management.

Provides tools for:
- Reading and writing project files
- LaTeX compilation with error recovery
- Bibliography management
- LaTeX structure validation
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools import ToolDefinition, ToolParameter, ToolResult, ToolRegistry

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# File Management
# ═════════════════════════════════════════════════════════════════════════

def read_project_file(file_path: str, project_dir: str = ".") -> ToolResult:
    """
    Read a file from the project directory.

    Args:
        file_path: Relative path within the project.
        project_dir: Project root directory.
    """
    try:
        fp = Path(project_dir) / file_path
        if not fp.exists():
            return ToolResult(False, error=f"File not found: {file_path}")

        content = fp.read_text(encoding="utf-8", errors="replace")
        return ToolResult(True, data={
            "path": file_path,
            "content": content,
            "size_bytes": fp.stat().st_size,
            "lines": content.count("\n") + 1,
        })
    except Exception as exc:
        return ToolResult(False, error=f"Failed to read {file_path}: {exc}")


def write_project_file(
    file_path: str,
    content: str,
    project_dir: str = ".",
    create_backup: bool = True,
) -> ToolResult:
    """
    Write content to a file in the project directory.

    Args:
        file_path: Relative path within the project.
        content: File content to write.
        project_dir: Project root directory.
        create_backup: Whether to back up existing files before overwriting.
    """
    try:
        fp = Path(project_dir) / file_path
        fp.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing file
        if fp.exists() and create_backup:
            backup_dir = Path(project_dir) / ".backups"
            backup_dir.mkdir(exist_ok=True)
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(fp, backup_dir / f"{fp.stem}_{ts}{fp.suffix}")

        fp.write_text(content, encoding="utf-8")
        return ToolResult(True, data={
            "path": file_path,
            "bytes_written": len(content.encode("utf-8")),
        })
    except Exception as exc:
        return ToolResult(False, error=f"Failed to write {file_path}: {exc}")


def list_project_files(project_dir: str = ".", pattern: str = "*") -> ToolResult:
    """
    List files in the project directory.

    Args:
        project_dir: Project root directory.
        pattern: Glob pattern to filter files.
    """
    try:
        pdir = Path(project_dir)
        exclude = {"__pycache__", ".git", ".backups", "agent_trace", "agent_memory", ".autopilot_backups"}

        files = []
        for fp in sorted(pdir.rglob(pattern)):
            if fp.is_file() and not any(part in exclude for part in fp.parts):
                rel = str(fp.relative_to(pdir))
                files.append({
                    "path": rel,
                    "size": fp.stat().st_size,
                    "extension": fp.suffix,
                })

        return ToolResult(True, data={"files": files, "count": len(files)})
    except Exception as exc:
        return ToolResult(False, error=f"Failed to list files: {exc}")


def get_project_snapshot(project_dir: str = ".", max_file_size: int = 100000) -> ToolResult:
    """
    Get a complete snapshot of all project files and their contents.

    Args:
        project_dir: Project root directory.
        max_file_size: Maximum file size to include content for.
    """
    try:
        pdir = Path(project_dir)
        exclude_dirs = {"__pycache__", ".git", ".backups", "agent_trace", "agent_memory", ".autopilot_backups"}
        exclude_exts = {".pyc", ".pyo", ".aux", ".bbl", ".blg", ".out", ".pdf", ".log",
                        ".npy", ".npz", ".pkl", ".png", ".jpg", ".svg"}

        snapshot_parts = []
        for fp in sorted(pdir.rglob("*")):
            if not fp.is_file():
                continue
            if any(part in exclude_dirs for part in fp.parts):
                continue
            if fp.suffix.lower() in exclude_exts:
                continue

            rel = str(fp.relative_to(pdir))
            size = fp.stat().st_size

            if size > max_file_size:
                snapshot_parts.append(f"=== FILE: {rel} === [{size:,} bytes — too large to include]")
            else:
                try:
                    content = fp.read_text(encoding="utf-8", errors="replace")
                    snapshot_parts.append(f"=== FILE: {rel} ===\n{content}")
                except Exception:
                    snapshot_parts.append(f"=== FILE: {rel} === [unreadable]")

        snapshot = "\n\n".join(snapshot_parts) if snapshot_parts else "(empty project)"
        return ToolResult(True, data={"snapshot": snapshot})
    except Exception as exc:
        return ToolResult(False, error=f"Snapshot failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════
# LaTeX Compilation
# ═════════════════════════════════════════════════════════════════════════

def compile_latex(
    project_dir: str = ".",
    tex_file: str = "paper.tex",
    pdflatex_path: str = "pdflatex",
    bibtex_path: str = "bibtex",
    timeout: int = 180,
) -> ToolResult:
    """
    Compile a LaTeX document to PDF.

    Runs the standard pdflatex → bibtex → pdflatex × 2 pipeline.

    Args:
        project_dir: Project directory containing the .tex file.
        tex_file: Name of the main .tex file.
        pdflatex_path: Path to pdflatex executable.
        bibtex_path: Path to bibtex executable.
        timeout: Timeout for each compilation step.
    """
    try:
        cwd = Path(project_dir)
        tex_path = cwd / tex_file
        if not tex_path.exists():
            return ToolResult(False, error=f"TeX file not found: {tex_file}")

        # Extract filecontents environments first
        _extract_filecontents(tex_path)

        # Determine the base name (without extension)
        base = tex_file.replace(".tex", "")

        results = []
        errors = []

        steps = [
            ("pdflatex (1)", [pdflatex_path, "-interaction=nonstopmode", "-halt-on-error", tex_file]),
            ("bibtex", [bibtex_path, base]),
            ("pdflatex (2)", [pdflatex_path, "-interaction=nonstopmode", tex_file]),
            ("pdflatex (3)", [pdflatex_path, "-interaction=nonstopmode", tex_file]),
        ]

        for step_name, cmd in steps:
            if not shutil.which(cmd[0]):
                results.append(f"{step_name}: SKIPPED ({cmd[0]} not found)")
                continue

            try:
                proc = subprocess.run(
                    cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
                )
                results.append(f"{step_name}: {'OK' if proc.returncode == 0 else 'WARN'}")

                if proc.returncode != 0 and "pdflatex" in step_name:
                    # Extract LaTeX errors from log
                    log_errors = _extract_latex_errors(proc.stdout + proc.stderr)
                    errors.extend(log_errors)

            except subprocess.TimeoutExpired:
                results.append(f"{step_name}: TIMEOUT")
                errors.append(f"{step_name} timed out after {timeout}s")

        pdf_path = cwd / f"{base}.pdf"
        success = pdf_path.exists()

        return ToolResult(success, data={
            "pdf_generated": success,
            "pdf_path": str(pdf_path) if success else "",
            "compilation_steps": results,
            "errors": errors,
        })

    except Exception as exc:
        return ToolResult(False, error=f"LaTeX compilation failed: {exc}")


def validate_latex_structure(content: str) -> ToolResult:
    """
    Validate LaTeX document structure.

    Checks for common issues: missing environments, unmatched braces,
    missing sections, etc.

    Args:
        content: LaTeX source code to validate.
    """
    issues = []
    warnings = []

    # Required elements
    if "\\documentclass" not in content:
        issues.append("Missing \\documentclass")
    if "\\begin{document}" not in content:
        issues.append("Missing \\begin{document}")
    if "\\end{document}" not in content:
        issues.append("Missing \\end{document}")
    if "\\title{" not in content:
        warnings.append("Missing \\title{}")
    if "\\begin{abstract}" not in content:
        warnings.append("Missing abstract environment")

    # Check for unmatched environments
    begins = re.findall(r"\\begin\{(\w+)\}", content)
    ends = re.findall(r"\\end\{(\w+)\}", content)
    begin_counts: Dict[str, int] = {}
    end_counts: Dict[str, int] = {}
    for b in begins:
        begin_counts[b] = begin_counts.get(b, 0) + 1
    for e in ends:
        end_counts[e] = end_counts.get(e, 0) + 1
    for env in set(list(begin_counts.keys()) + list(end_counts.keys())):
        bc = begin_counts.get(env, 0)
        ec = end_counts.get(env, 0)
        if bc != ec:
            issues.append(f"Unmatched environment '{env}': {bc} begins, {ec} ends")

    # Check for common section structure
    sections = re.findall(r"\\section\{(.+?)\}", content)
    if len(sections) < 3:
        warnings.append(f"Only {len(sections)} sections found — academic papers typically have 5+")

    # Check citations
    cites = re.findall(r"\\cite\{(.+?)\}", content)
    if not cites:
        warnings.append("No \\cite{} commands found — paper should cite sources")

    # Word count estimate
    doc_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", content, re.DOTALL)
    if doc_match:
        body = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", doc_match.group(1))
        body = re.sub(r"[{}\\%$&]", " ", body)
        word_count = len(body.split())
    else:
        word_count = 0

    valid = len(issues) == 0

    return ToolResult(True, data={
        "valid": valid,
        "issues": issues,
        "warnings": warnings,
        "sections": sections,
        "citation_count": len(cites),
        "word_count": word_count,
    })


# ═════════════════════════════════════════════════════════════════════════
# Bibliography Management
# ═════════════════════════════════════════════════════════════════════════

def validate_bibliography(
    tex_content: str,
    bib_content: str = "",
) -> ToolResult:
    """
    Validate that all citations in the LaTeX file have matching bibliography entries.

    Args:
        tex_content: LaTeX source code.
        bib_content: BibTeX file content (if using external .bib file).
    """
    # Find all \cite{} references
    cite_keys = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex_content):
        for key in match.group(1).split(","):
            cite_keys.add(key.strip())

    # Find all bib entries (from .bib file or filecontents)
    bib_keys = set()

    # Check external .bib content
    if bib_content:
        for match in re.finditer(r"@\w+\{(\w+)", bib_content):
            bib_keys.add(match.group(1))

    # Check embedded bibitem
    for match in re.finditer(r"\\bibitem\{(\w+)\}", tex_content):
        bib_keys.add(match.group(1))

    # Check filecontents bib
    fc_match = re.search(
        r"\\begin\{filecontents\*?\}\{.*?\.bib\}(.*?)\\end\{filecontents\*?\}",
        tex_content, re.DOTALL,
    )
    if fc_match:
        for match in re.finditer(r"@\w+\{(\w+)", fc_match.group(1)):
            bib_keys.add(match.group(1))

    missing = cite_keys - bib_keys
    unused = bib_keys - cite_keys

    return ToolResult(True, data={
        "valid": len(missing) == 0,
        "cited_keys": sorted(cite_keys),
        "bib_keys": sorted(bib_keys),
        "missing_entries": sorted(missing),
        "unused_entries": sorted(unused),
    })


# ═════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═════════════════════════════════════════════════════════════════════════

def _extract_filecontents(tex_path: Path) -> None:
    """Extract filecontents environments from LaTeX source."""
    content = tex_path.read_text(encoding="utf-8", errors="replace")
    pattern = r"\\begin\{filecontents\*?\}\{(.+?)\}(.*?)\\end\{filecontents\*?\}"
    for match in re.finditer(pattern, content, re.DOTALL):
        filename = match.group(1).strip()
        file_content = match.group(2).strip()
        out_path = tex_path.parent / filename
        out_path.write_text(file_content, encoding="utf-8")


def _extract_latex_errors(log_text: str) -> List[str]:
    """Extract error messages from LaTeX log output."""
    errors = []
    for line in log_text.split("\n"):
        line = line.strip()
        if line.startswith("!"):
            errors.append(line)
        elif "Error:" in line or "Fatal error" in line:
            errors.append(line)
        elif "Undefined control sequence" in line:
            errors.append(line)
    return errors[:10]  # limit to 10 errors


# ═════════════════════════════════════════════════════════════════════════
# Registration
# ═════════════════════════════════════════════════════════════════════════

def register_writing_tools(
    registry: ToolRegistry,
    project_dir: str = ".",
    pdflatex_path: str = "pdflatex",
    bibtex_path: str = "bibtex",
) -> None:
    """Register all writing tools with the tool registry."""

    # File reading
    registry.register(ToolDefinition(
        name="read_file",
        description="Read a file from the project directory. Returns the file content.",
        parameters=[
            ToolParameter("file_path", "string", "Relative path to the file"),
        ],
        handler=lambda file_path: read_project_file(file_path, project_dir),
        category="writing",
    ))

    # File writing
    registry.register(ToolDefinition(
        name="write_file",
        description="Write content to a file in the project. Creates the file if it doesn't exist, overwrites if it does.",
        parameters=[
            ToolParameter("file_path", "string", "Relative path to the file"),
            ToolParameter("content", "string", "Complete file content to write"),
        ],
        handler=lambda file_path, content: write_project_file(file_path, content, project_dir),
        category="writing",
    ))

    # File listing
    registry.register(ToolDefinition(
        name="list_files",
        description="List all files in the project directory.",
        parameters=[
            ToolParameter("pattern", "string", "Glob pattern to filter", required=False, default="*"),
        ],
        handler=lambda pattern="*": list_project_files(project_dir, pattern),
        category="writing",
    ))

    # Project snapshot
    registry.register(ToolDefinition(
        name="get_snapshot",
        description="Get a complete snapshot of all project files with their contents. Useful for understanding the full project state.",
        parameters=[],
        handler=lambda: get_project_snapshot(project_dir),
        category="writing",
    ))

    # LaTeX compilation
    registry.register(ToolDefinition(
        name="compile_latex",
        description="Compile the LaTeX paper to PDF. Runs pdflatex and bibtex. Returns compilation status and any errors.",
        parameters=[
            ToolParameter("tex_file", "string", "Name of the .tex file", required=False, default="paper.tex"),
        ],
        handler=lambda tex_file="paper.tex": compile_latex(
            project_dir, tex_file, pdflatex_path, bibtex_path,
        ),
        category="writing",
    ))

    # LaTeX validation
    registry.register(ToolDefinition(
        name="validate_latex",
        description="Validate LaTeX document structure. Checks for missing environments, unmatched braces, sections, citations.",
        parameters=[
            ToolParameter("content", "string", "LaTeX source code to validate"),
        ],
        handler=validate_latex_structure,
        category="writing",
    ))

    # Bibliography validation
    registry.register(ToolDefinition(
        name="validate_bibliography",
        description="Check that all \\cite{} references have matching bibliography entries.",
        parameters=[
            ToolParameter("tex_content", "string", "LaTeX source with citations"),
            ToolParameter("bib_content", "string", "BibTeX file content", required=False, default=""),
        ],
        handler=validate_bibliography,
        category="writing",
    ))
