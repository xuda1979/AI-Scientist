"""
Helpers to enforce single paper/simulation files, extract Python code, and run the simulation.
"""
from __future__ import annotations
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, Optional, Tuple

PY_BLOCKS = [
    # minted
    re.compile(r"\\begin\{minted\}(?:\[[^\]]*\])?\{python\}(.*?)\\end\{minted\}", re.DOTALL | re.IGNORECASE),
    # lstlisting with language=Python in optional arg
    re.compile(r"\\begin\{lstlisting\}(?:\[(?:[^\]]*language\s*=\s*Python[^\]]*)\])?(.*?)\\end\{lstlisting\}", re.DOTALL | re.IGNORECASE),
]

def ensure_single_tex_py(project_dir: Path, strict: bool = True) -> Tuple[Path, Path]:
    """
    Keep exactly one paper.tex and one simulation.py in project_dir.
    Simply rename/delete others without archiving.
    """
    project_dir.mkdir(parents=True, exist_ok=True)

    # Find existing .tex files and rename the first one to paper.tex
    paper = project_dir / "paper.tex"
    existing_tex_files = list(project_dir.glob("*.tex"))
    
    if not paper.exists() and existing_tex_files:
        # Rename the first .tex file to paper.tex
        first_tex = existing_tex_files[0]
        if first_tex.name != "paper.tex":
            first_tex.rename(paper)
            existing_tex_files.remove(first_tex)
    elif not paper.exists():
        # create a minimal paper to be filled
        paper.write_text("\\documentclass{article}\\begin{document}\\end{document}\n", encoding="utf-8")
    
    sim = project_dir / "simulation.py"
    if not sim.exists():
        sim.write_text("# Auto-generated simulation scaffold\nif __name__ == '__main__':\n    print('No simulation yet')\n", encoding="utf-8")

    if not strict:
        return paper, sim

    # Delete other .tex files (no archiving)
    for p in existing_tex_files:
        if p.exists() and p.name != "paper.tex":
            p.unlink()
    
    # Delete other .py files except simulation.py (no archiving)
    for p in project_dir.glob("*.py"):
        if p.name != "simulation.py":
            p.unlink()
    
    return paper, sim

def _extract_python_blocks(tex: str) -> str:
    blocks = []
    for pat in PY_BLOCKS:
        for m in pat.finditer(tex):
            code = m.group(1)
            # strip leading/trailing whitespace
            code = code.strip("\n")
            if code:
                blocks.append(code)
    if not blocks:
        return ""
    sections = []
    for i, b in enumerate(blocks, start=1):
        sections.append(f"# === Begin extracted block {i} ===\n{b}\n# === End block {i} ===\n")
    header = "# Auto-generated from LaTeX code blocks; consolidate all simulation here.\n"
    return header + "\n".join(sections)

def extract_simulation_from_tex(tex_path: Path, sim_path: Path) -> bool:
    """
    Parse LaTeX for Python code blocks and write/append to simulation.py.
    Returns True if any code was found.
    """
    tex = tex_path.read_text(encoding="utf-8", errors="ignore")
    py = _extract_python_blocks(tex)
    if not py:
        return False
    sim_path.write_text(py, encoding="utf-8")
    return True

def run_simulation_with_smart_fixing(
    sim_path: Path, 
    python_exec: Optional[str] = None, 
    cwd: Optional[Path] = None, 
    timeout: Optional[int] = 300,
    llm_fixer: Optional[callable] = None,
    max_fix_attempts: int = 3
) -> Dict[str, str]:
    """
    Run simulation with intelligent error handling via LLM.
    If the simulation fails, the LLM decides whether to:
    1. Accept results as-is
    2. Fix the code
    3. Install missing modules
    """
    python_exec = python_exec or sys.executable
    cwd = cwd or sim_path.parent
    
    for attempt in range(max_fix_attempts + 1):
        # Run the simulation
        proc = subprocess.Popen(
            [python_exec, sim_path.name], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            out, err = proc.communicate(timeout=timeout)
        except Exception:
            proc.kill()
            out, err = proc.communicate()
        
        res = {"stdout": out or "", "stderr": err or "", "return_code": proc.returncode}
        
        # Check for results.json
        results_file = cwd / "results.json"
        if results_file.exists():
            try:
                res["results_json"] = results_file.read_text(encoding="utf-8")
            except Exception:
                pass
        
        # If successful or no LLM fixer, return results
        if proc.returncode == 0 or llm_fixer is None:
            break
            
        # If failed and we have attempts left, ask LLM for help
        if attempt < max_fix_attempts:
            current_code = sim_path.read_text(encoding="utf-8", errors="ignore")
            action = llm_fixer(current_code, out, err, proc.returncode)
            
            if action.get("action") == "accept":
                print("LLM decided to accept results despite errors")
                break
            elif action.get("action") == "fix_code":
                if action.get("fixed_code"):
                    sim_path.write_text(action["fixed_code"], encoding="utf-8")
                    print(f"LLM fixed code (attempt {attempt + 1})")
                    continue
            elif action.get("action") == "install_modules":
                modules = action.get("modules", [])
                if modules:
                    _install_modules(modules, python_exec)
                    print(f"Installed modules: {', '.join(modules)}")
                    continue
        
        # If we get here, either no more attempts or LLM couldn't help
        break
    
    # Always persist output for audit
    (cwd / "simulation_output.txt").write_text((out or "") + ("\n" + err if err else ""), encoding="utf-8")
    return res

def _install_modules(modules: list, python_exec: str) -> None:
    """Install Python modules using pip."""
    for module in modules:
        try:
            subprocess.run([python_exec, "-m", "pip", "install", module], check=True)
            print(f"Successfully installed {module}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {module}")

def run_simulation(sim_path: Path, python_exec: Optional[str] = None, cwd: Optional[Path] = None, timeout: Optional[int] = 300) -> Dict[str, str]:
    """
    Run simulation.py and capture outputs.
    Also read results.json if it exists after run.
    """
    python_exec = python_exec or sys.executable
    cwd = cwd or sim_path.parent
    proc = subprocess.Popen(
        [python_exec, sim_path.name], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except Exception:
        proc.kill()
        out, err = proc.communicate()
    res = {"stdout": out or "", "stderr": err or ""}
    results_file = cwd / "results.json"
    if results_file.exists():
        try:
            res["results_json"] = results_file.read_text(encoding="utf-8")
        except Exception:
            pass
    # Always persist a text copy for audit
    (cwd / "simulation_output.txt").write_text((out or "") + ("\n" + err if err else ""), encoding="utf-8")
    return res

def summarize_simulation_outputs(outputs: Dict[str, str], simulation_code: str = "", max_chars: int = 6000) -> str:
    """
    Compact string passed to the model with both simulation code and execution results.
    """
    parts = []
    
    # Always include the simulation code first
    if simulation_code.strip():
        parts.append("SIMULATION CODE:\n" + simulation_code)
    
    # Then include execution outputs
    if outputs.get("stdout"):
        parts.append("STDOUT:\n" + outputs["stdout"])
    if outputs.get("stderr"):
        parts.append("STDERR:\n" + outputs["stderr"])
    if outputs.get("results_json"):
        parts.append("results.json:\n" + outputs["results_json"])
    
    s = "\n\n".join(parts)
    if len(s) > max_chars:
        s = s[:max_chars] + "\n...[truncated]..."
    return s
