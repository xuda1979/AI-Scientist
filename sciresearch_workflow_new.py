#!/usr/bin/env python3
"""
Extended workflow:
 - Enforce single paper.tex and simulation.py per project
 - Extract simulation code from LaTeX; run it; pass results to LLM during review/revision
 - Sanitize LaTeX to prevent overflow; compile-check; auto-fix on failure
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Local helpers
from utils.sim_runner import ensure_single_tex_py, extract_simulation_from_tex, run_simulation, summarize_simulation_outputs
from utils.latex_tools import sanitize_and_constrain_file, compile_with_autofix

DEFAULT_MODEL = os.environ.get("SCI_MODEL", "gpt-5")

def _openai_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None) -> str:
    """
    A minimal OpenAI chat wrapper that works with both old and new SDKs.
    """
    try:
        # Newer SDK
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, timeout=request_timeout)
        return resp.choices[0].message.content
    except Exception:
        # Fallback to legacy SDK
        import openai
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.2, timeout=request_timeout)
        return resp["choices"][0]["message"]["content"]

def _llm_fix_latex(model: str, request_timeout: Optional[int] = None):
    """
    Returns a callable: (tex_content, log_text) -> fixed_tex_content
    """
    def _fix(tex: str, log: str) -> str:
        sys_prompt = (
            "You are a LaTeX expert. Given a LaTeX source and a LaTeX compile log, "
            "fix ONLY the LaTeX so it compiles cleanly. Keep intent and content unchanged. "
            "Never introduce oversized figures/tables; keep figures width=\\linewidth and wrap wide tables."
        )
        user = (
            "Here is the current LaTeX file:\n\n"
            "----- BEGIN LATEX -----\n" + tex + "\n----- END LATEX -----\n\n"
            "Here is the compile log and errors:\n\n"
            "----- BEGIN LOG -----\n" + log + "\n----- END LOG -----\n\n"
            "Return ONLY the corrected LaTeX source (no explanations)."
        )
        return _openai_chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
            model=model,
            request_timeout=request_timeout,
        )
    return _fix

def _nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _prepare_project_dir(output_dir: Path, modify_existing: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if modify_existing and (output_dir / "paper.tex").exists():
        return output_dir
    project_dir = output_dir / _nowstamp()
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir

def _initial_draft_prompt(topic: str, field: str, question: str) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are a meticulous scientist writing a LaTeX paper suitable for a top journal. "
        "Always produce compilable LaTeX. Figures must use \\includegraphics[width=\\linewidth]{...}. "
        "Wrap wide tables using adjustbox width=\\linewidth. Include Abstract, Intro, Related Work, "
        "Methodology, Experiments, Results, Discussion, Conclusion, References. "
        "If you need code, include Python blocks using lstlisting or minted."
    )
    user_prompt = f"Topic: {topic}\nField: {field}\nResearch Question: {question}\n\nDraft the full LaTeX paper."
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

def _review_prompt(paper_tex: str, sim_summary: str) -> List[Dict[str, str]]:
    sys_prompt = (
        "Act as a top journal reviewer. Be constructive. Do not recommend rejection; instead give specific fixes. "
        "Ensure LaTeX compiles cleanly, figures/tables fit within margins, and results align with the provided simulation outputs."
    )
    user = (
        "Here is the current paper (LaTeX):\n\n"
        "----- BEGIN LATEX -----\n" + paper_tex + "\n----- END LATEX -----\n\n"
        "Here are the latest simulation outputs to ground your critique:\n\n"
        "----- BEGIN SIM OUTPUTS -----\n" + sim_summary + "\n----- END SIM OUTPUTS -----\n\n"
        "Write a constructive review with bullet points and suggested concrete edits."
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]

def _editor_prompt(review_text: str) -> List[Dict[str, str]]:
    sys_prompt = "You are the handling editor. Decide if the paper is ready."
    user = (
        "Given the review below, answer strictly YES if the paper is ready for submission; otherwise NO.\n\n"
        "----- REVIEW -----\n" + review_text + "\n----- END REVIEW -----\n"
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]

def _revise_prompt(paper_tex: str, sim_summary: str, review_text: str) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are the author revising the paper. Produce a corrected FULL LaTeX file only (no explanations). "
        "Keep figures \\includegraphics[width=\\linewidth], wrap wide tables with adjustbox width=\\linewidth, "
        "and ensure LaTeX compiles. If simulation results require changes, update the text/tables accordingly."
    )
    user = (
        "----- CURRENT PAPER (LATEX) -----\n" + paper_tex + "\n"
        "----- SIMULATION OUTPUTS -----\n" + sim_summary + "\n"
        "----- REVIEW FEEDBACK -----\n" + review_text + "\n"
        "Return ONLY the full revised LaTeX file."
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]

def run_workflow(
    topic: str,
    field: str,
    question: str,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    request_timeout: Optional[int] = 3600,
    max_retries: int = 3,
    max_iterations: int = 4,
    modify_existing: bool = False,
    max_compile_fixes: int = 2,
    no_latex_autofix: bool = False,
    strict_singletons: bool = True,
    python_exec: Optional[str] = None,
) -> Path:
    """
    End-to-end workflow with added LaTeX QC and simulation integration.
    """
    project_dir = _prepare_project_dir(output_dir, modify_existing)
    paper_path, sim_path = ensure_single_tex_py(project_dir, strict=strict_singletons)

    # If no paper yet (fresh), draft one.
    if (paper_path.read_text(encoding="utf-8").strip() == "\\documentclass{article}\\begin{document}\\end{document}"):
        draft = _openai_chat(_initial_draft_prompt(topic, field, question), model=model, request_timeout=request_timeout)
        paper_path.write_text(draft, encoding="utf-8")

    # Extract/refresh simulation.py from LaTeX
    extract_simulation_from_tex(paper_path, sim_path)
    # Run simulation and capture outputs
    sim_out = run_simulation(sim_path, python_exec=python_exec, cwd=project_dir)
    sim_summary = summarize_simulation_outputs(sim_out)

    # First sanitize & compile, with autofix as needed
    ok = compile_with_autofix(
        project_dir,
        tex_file="paper.tex",
        max_attempts=max_compile_fixes,
        no_autofix=no_latex_autofix,
        llm_fix=None if no_latex_autofix else _llm_fix_latex(model, request_timeout=request_timeout),
    )
    if not ok:
        print("ERROR: LaTeX failed to compile even after auto-fix attempts.", file=sys.stderr)

    # Review-Revise loop
    for i in range(1, max_iterations + 1):
        current_tex = paper_path.read_text(encoding="utf-8", errors="ignore")
        review = _openai_chat(_review_prompt(current_tex, sim_summary), model=model, request_timeout=request_timeout)
        decision = _openai_chat(_editor_prompt(review), model=model, request_timeout=request_timeout)
        if decision.strip().upper().startswith("YES"):
            print(f"[OK] Editor accepted at iteration {i}.")
            break
        # Revise
        revised = _openai_chat(_revise_prompt(current_tex, sim_summary, review), model=model, request_timeout=request_timeout)
        paper_path.write_text(revised, encoding="utf-8")

        # Refresh simulation from LaTeX; rerun
        extract_simulation_from_tex(paper_path, sim_path)
        sim_out = run_simulation(sim_path, python_exec=python_exec, cwd=project_dir)
        sim_summary = summarize_simulation_outputs(sim_out)

        # Sanitize+compile again; autofix on error
        ok = compile_with_autofix(
            project_dir,
            tex_file="paper.tex",
            max_attempts=max_compile_fixes,
            no_autofix=no_latex_autofix,
            llm_fix=None if no_latex_autofix else _llm_fix_latex(model, request_timeout=request_timeout),
        )
        if not ok:
            print(f"WARNING: Compile still failing after iteration {i}. Continuing revisions.", file=sys.stderr)

    return project_dir

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--topic", required=False, help="Research topic")
    p.add_argument("--field", required=False, help="Field")
    p.add_argument("--question", required=False, help="Research question")
    p.add_argument("--output-dir", default="output", help="Output directory root (contains project subfolder)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use (default: gpt-5)")
    p.add_argument("--request-timeout", type=int, default=3600, help="Per-request timeout seconds (0 means no timeout)")
    p.add_argument("--max-retries", type=int, default=3, help="Max OpenAI retries")
    p.add_argument("--max-iterations", type=int, default=4, help="Max review->revise iterations")
    p.add_argument("--modify-existing", action="store_true", help="If output dir already has paper.tex, modify in place")
    p.add_argument("--max-compile-fixes", type=int, default=2, help="Max LaTeX auto-fix attempts")
    p.add_argument("--no-latex-autofix", action="store_true", help="Disable model-based LaTeX auto-fix")
    p.add_argument("--strict-singletons", action="store_true", default=True, help="Keep only paper.tex & simulation.py (others archived)")
    p.add_argument("--python-exec", default=None, help="Python interpreter for running simulation.py")
    args = p.parse_args(argv)
    # Interactive prompts if missing
    if not args.topic:
        args.topic = input("Topic: ").strip()
    if not args.field:
        args.field = input("Field: ").strip()
    if not args.question:
        args.question = input("Research question: ").strip()
    return args

if __name__ == "__main__":
    ns = parse_args()
    out = run_workflow(
        topic=ns.topic,
        field=ns.field,
        question=ns.question,
        output_dir=Path(ns.output_dir),
        model=ns.model,
        request_timeout=(None if ns.request_timeout == 0 else ns.request_timeout),
        max_retries=ns.max_retries,
        max_iterations=ns.max_iterations,
        modify_existing=ns.modify_existing,
        max_compile_fixes=ns.max_compile_fixes,
        no_latex_autofix=ns.no_latex_autofix,
        strict_singletons=ns.strict_singletons,
        python_exec=ns.python_exec,
    )
    print(f"Project at: {out}")
