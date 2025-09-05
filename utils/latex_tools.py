"""
Utilities for sanitizing LaTeX, constraining figure/table widths, and compiling.
"""
from __future__ import annotations
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Optional, Tuple

_PKGS = [
    ("graphicx", None),
    ("adjustbox", None),
    ("booktabs", None),
    ("tabularx", None),
    ("caption", None),
    ("subcaption", None),
    ("geometry", "margin=1in"),
]

BEGIN_AUTO = "% === BEGIN AUTO-PREAMBLE ==="
END_AUTO = "% === END AUTO-PREAMBLE ==="

def _has_package(tex: str, pkg: str) -> bool:
    return re.search(r"\\usepackage(?:\[[^\]]*\])?\{"+re.escape(pkg)+r"\}", tex) is not None

def _insert_preamble(tex: str) -> str:
    # Ensure packages are present after \documentclass
    m = re.search(r"(\\documentclass[^\n]*\n)", tex)
    if not m:
        # If \documentclass missing, inject a minimal header
        header = (
            "\\documentclass{article}\n"
            f"{BEGIN_AUTO}\n"
        )
        for pkg, opt in _PKGS:
            if opt:
                header += f"\\usepackage[{opt}]{{{pkg}}}\n"
            else:
                header += f"\\usepackage{{{pkg}}}\n"
        header += f"{END_AUTO}\n\\begin{document}\n"
        # Assume the rest of the file is body
        if "\\end{document}" not in tex:
            tex = header + tex + "\n\\end{document}\n"
        else:
            tex = header + tex
        return tex

    insert_point = m.end(1)
    auto_block = f"{BEGIN_AUTO}\n"
    for pkg, opt in _PKGS:
        if not _has_package(tex, pkg):
            if opt:
                auto_block += f"\\usepackage[{opt}]{{{pkg}}}\n"
            else:
                auto_block += f"\\usepackage{{{pkg}}}\n"
    auto_block += f"{END_AUTO}\n"
    return tex[:insert_point] + auto_block + tex[insert_point:]

_inc_pat = re.compile(
    r"\\includegraphics(?P<opts>\[[^\]]*\])?\{(?P<path>[^\}]+)\}",
    flags=re.IGNORECASE,
)

def _normalize_includegraphics(tex: str) -> str:
    def repl(m: re.Match) -> str:
        opts = m.group("opts") or ""
        # Remove any existing width=...; keep other options.
        if opts:
            # strip brackets
            raw = opts[1:-1]
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            parts = [p for p in parts if not re.match(r"^\s*width\s*=", p)]
            if parts:
                new = "[" + ",".join(parts) + ",width=\\linewidth]"
            else:
                new = "[width=\\linewidth]"
        else:
            new = "[width=\\linewidth]"
        return f"\\includegraphics{new}{{{m.group('path')}}}"
    return _inc_pat.sub(repl, tex)

def _wrap_tabular(tex: str) -> str:
    """
    Wrap tabular blocks inside tables with adjustbox{width=\\linewidth}
    when they are not already wrapped.
    """
    # Find table environments and wrap inner tabular
    table_pat = re.compile(
        r"(\\begin\{table\}.*?)(\\begin\{tabular\}.*?\\end\{tabular\})(.*?\\end\{table\})",
        flags=re.DOTALL,
    )
    def wrap_table(m: re.Match) -> str:
        head, tab, tail = m.group(1), m.group(2), m.group(3)
        if "\\centering" not in head:
            head = head.replace("\\begin{table}", "\\begin{table}[htbp]\n\\centering")
        if "adjustbox" in tab or "resizebox" in tab:
            return head + tab + tail
        wrapped = "\\begin{adjustbox}{width=\\linewidth}\n" + tab + "\n\\end{adjustbox}"
        return head + wrapped + tail
    return table_pat.sub(wrap_table, tex)

def _wrap_tikz_in_figure(tex: str) -> str:
    fig_pat = re.compile(
        r"(\\begin\{figure\}.*?)(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})(.*?\\end\{figure\})",
        flags=re.DOTALL,
    )
    def wrap_fig(m: re.Match) -> str:
        head, tikz, tail = m.group(1), m.group(2), m.group(3)
        if "\\centering" not in head:
            head = head.replace("\\begin{figure}", "\\begin{figure}[htbp]\n\\centering")
        if "adjustbox" in tikz or "resizebox" in tikz:
            return head + tikz + tail
        wrapped = "\\begin{adjustbox}{width=\\linewidth}\n" + tikz + "\n\\end{adjustbox}"
        return head + wrapped + tail
    return fig_pat.sub(wrap_fig, tex)

def sanitize_and_constrain_file(tex_path: Path) -> None:
    tex = tex_path.read_text(encoding="utf-8", errors="ignore")
    tex = _insert_preamble(tex)
    tex = _normalize_includegraphics(tex)
    tex = _wrap_tabular(tex)
    tex = _wrap_tikz_in_figure(tex)
    tex_path.write_text(tex, encoding="utf-8")

def _run_cmd(cmd: list[str], cwd: Path) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def compile_latex(project_dir: Path, tex_file: str = "paper.tex") -> Tuple[bool, str]:
    """
    Attempt to compile with latexmk; fallback to pdflatex twice.
    Returns (success, log_text).
    """
    if shutil.which("latexmk"):
        rc, out, err = _run_cmd(["latexmk", "-pdf", "-halt-on-error", "-interaction=nonstopmode", tex_file], project_dir)
        log = out + "\n" + err
        if rc == 0:
            return True, log
        # try clean + one more attempt
        _run_cmd(["latexmk", "-C"], project_dir)
        rc, out, err = _run_cmd(["latexmk", "-pdf", "-halt-on-error", "-interaction=nonstopmode", tex_file], project_dir)
        log = out + "\n" + err
        return rc == 0, log

    # Fallback to pdflatex (twice)
    ok = True
    all_logs = []
    for _ in range(2):
        rc, out, err = _run_cmd(["pdflatex", "-interaction=nonstopmode", tex_file], project_dir)
        all_logs.append(out + "\n" + err)
        if rc != 0:
            ok = False
    return ok, "\n".join(all_logs)

def compile_with_autofix(
    project_dir: Path,
    tex_file: str = "paper.tex",
    max_attempts: int = 2,
    no_autofix: bool = False,
    llm_fix: Optional[callable] = None,
) -> bool:
    """
    Sanitize, compile; on error, request a fix from llm_fix(tex, log)->str and retry.
    """
    tex_path = project_dir / tex_file
    for attempt in range(max_attempts + 1):
        sanitize_and_constrain_file(tex_path)
        ok, log_text = compile_latex(project_dir, tex_file=tex_file)
        if ok:
            return True
        if no_autofix or llm_fix is None or attempt == max_attempts:
            return False
        # Ask the model to fix and overwrite
        fixed_tex = llm_fix(tex_path.read_text(encoding="utf-8", errors="ignore"), log_text)
        if fixed_tex:
            tex_path.write_text(fixed_tex, encoding="utf-8")
        else:
            return False
    return False


def latex_fix_cycle(project_dir: Path, model: str, tex_file: str = "paper.tex", llm_fix: Optional[callable] = None, max_attempts: int = 3) -> Tuple[bool, str]:
    """Iteratively sanitize, compile, and fix LaTeX sources.

    Parameters
    ----------
    project_dir: Path
        Directory containing the LaTeX project.
    model: str
        Name of the model used for fixing (for logging only).
    tex_file: str
        Target LaTeX file name.
    llm_fix: callable
        Function accepting (tex, log) and returning fixed tex.
    max_attempts: int
        Maximum number of fix attempts.

    Returns
    -------
    success: bool
        True if compilation succeeded.
    log: str
        Compiler log of the final attempt.
    """
    tex_path = project_dir / tex_file
    logs_dir = project_dir / "latex_logs"
    logs_dir.mkdir(exist_ok=True)
    last_log = ""
    for attempt in range(1, max_attempts + 1):
        sanitize_and_constrain_file(tex_path)
        ok, log_text = compile_latex(project_dir, tex_file=tex_file)
        log_path = logs_dir / f"attempt_{attempt}.log"
        log_path.write_text(log_text, encoding="utf-8")
        last_log = log_text
        if ok:
            return True, log_text
        if llm_fix is None:
            continue
        fixed_tex = llm_fix(tex_path.read_text(encoding="utf-8", errors="ignore"), log_text)
        if fixed_tex:
            patch_path = logs_dir / f"attempt_{attempt}_fix.tex"
            patch_path.write_text(fixed_tex, encoding="utf-8")
            tex_path.write_text(fixed_tex, encoding="utf-8")
        else:
            break
    return False, last_log
