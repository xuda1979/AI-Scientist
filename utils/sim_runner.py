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
    # filecontents for simulation.py (case insensitive)
    re.compile(r"\\begin\{filecontents\*?\}\{simulation\.py\}(.*?)\\end\{filecontents\*?\}", re.DOTALL | re.IGNORECASE),
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

def evaluate_simulation_compatibility(existing_sim: str, extracted_sim: str) -> Dict[str, any]:
    """
    Evaluate compatibility between existing and extracted simulation code.
    
    Args:
        existing_sim: Content of existing simulation.py
        extracted_sim: Content extracted from LaTeX
        
    Returns:
        Dictionary with compatibility analysis and recommendations
    """
    analysis = {
        'existing_quality': 0.0,
        'extracted_quality': 0.0,
        'compatibility_score': 0.0,
        'recommendation': 'preserve_existing',
        'reason': '',
        'merge_possible': False
    }
    
    # Analyze existing simulation
    existing_lines = len([line for line in existing_sim.split('\n') if line.strip() and not line.strip().startswith('#')])
    existing_classes = existing_sim.count('class ')
    existing_functions = existing_sim.count('def ')
    existing_imports = existing_sim.count('import ')
    
    # Quality indicators for existing simulation
    existing_indicators = {
        'test_time_scaling': any(term in existing_sim.lower() for term in [
            'testtimecomputescaling', 'candidate_generation', 'selection_strategy', 'quality_evaluation'
        ]),
        'comprehensive': existing_lines > 100 and existing_classes > 0 and existing_functions > 5,
        'experimental': any(term in existing_sim.lower() for term in [
            'experiment', 'trial', 'simulation', 'benchmark', 'evaluation'
        ]),
        'visualization': any(term in existing_sim.lower() for term in [
            'matplotlib', 'plt.', 'figure', 'plot'
        ]),
        'research_quality': any(term in existing_sim.lower() for term in [
            'algorithm', 'optimization', 'scaling', 'performance', 'analysis'
        ])
    }
    
    analysis['existing_quality'] = (
        (existing_lines / 200) * 0.3 +
        min(existing_classes / 3, 1.0) * 0.2 +
        min(existing_functions / 10, 1.0) * 0.2 +
        sum(existing_indicators.values()) * 0.3 / len(existing_indicators)
    )
    
    # Analyze extracted simulation
    extracted_lines = len([line for line in extracted_sim.split('\n') if line.strip() and not line.strip().startswith('#')])
    extracted_classes = extracted_sim.count('class ')
    extracted_functions = extracted_sim.count('def ')
    
    # Quality indicators for extracted simulation
    extracted_indicators = {
        'structured': extracted_classes > 0 and extracted_functions > 3,
        'experimental': any(term in extracted_sim.lower() for term in [
            'experiment', 'trial', 'seed', 'random'
        ]),
        'comprehensive': extracted_lines > 50,
        'research_content': any(term in extracted_sim.lower() for term in [
            'algorithm', 'optimization', 'evaluation', 'performance'
        ])
    }
    
    analysis['extracted_quality'] = (
        (extracted_lines / 100) * 0.4 +
        min(extracted_classes / 2, 1.0) * 0.2 +
        min(extracted_functions / 5, 1.0) * 0.2 +
        sum(extracted_indicators.values()) * 0.2 / len(extracted_indicators)
    )
    
    # Determine recommendation
    if analysis['existing_quality'] > 0.7 and existing_indicators['test_time_scaling']:
        analysis['recommendation'] = 'preserve_existing'
        analysis['reason'] = 'High-quality test-time compute scaling implementation'
    elif analysis['existing_quality'] > 0.6 and analysis['extracted_quality'] < 0.4:
        analysis['recommendation'] = 'preserve_existing'
        analysis['reason'] = 'Existing simulation significantly better'
    elif analysis['extracted_quality'] > analysis['existing_quality'] + 0.3:
        analysis['recommendation'] = 'use_extracted'
        analysis['reason'] = 'Extracted simulation significantly better'
    elif abs(analysis['existing_quality'] - analysis['extracted_quality']) < 0.2:
        analysis['recommendation'] = 'merge_possible'
        analysis['reason'] = 'Similar quality, consider merging'
        analysis['merge_possible'] = True
    else:
        analysis['recommendation'] = 'preserve_existing'
        analysis['reason'] = 'Default to preserving existing work'
    
    return analysis

def extract_simulation_from_tex(tex_path: Path, sim_path: Path) -> bool:
    """
    Parse LaTeX for Python code blocks and write/append to simulation.py.
    Returns True if any code was found.
    """
    tex = tex_path.read_text(encoding="utf-8", errors="ignore")
    py = _extract_python_blocks(tex)
    if not py:
        return False
    
    # Check if existing simulation.py is substantial and comprehensive
    if sim_path.exists():
        existing_content = sim_path.read_text(encoding="utf-8", errors="ignore")
        
        # Perform intelligent compatibility analysis
        compatibility = evaluate_simulation_compatibility(existing_content, py)
        
        print(f"📊 Simulation Compatibility Analysis:")
        print(f"   Existing quality: {compatibility['existing_quality']:.3f}")
        print(f"   Extracted quality: {compatibility['extracted_quality']:.3f}")
        print(f"   Recommendation: {compatibility['recommendation']}")
        print(f"   Reason: {compatibility['reason']}")
        
        if compatibility['recommendation'] == 'preserve_existing':
            # Create backup of extracted content for reference
            backup_path = sim_path.parent / "simulation_extracted_backup.py"
            backup_path.write_text(py, encoding="utf-8")
            print(f"💾 Saved extracted LaTeX code to {backup_path.name} for reference")
            return True
            
        elif compatibility['recommendation'] == 'merge_possible':
            # For now, preserve existing but suggest manual review
            backup_path = sim_path.parent / "simulation_extracted_candidate.py"
            backup_path.write_text(py, encoding="utf-8")
            print(f"🔄 Saved extracted code as merge candidate to {backup_path.name}")
            print("💡 Consider manually reviewing and merging the best features from both")
            return True
    
    # Use extracted simulation if no existing file or extraction is better
    sim_path.write_text(py, encoding="utf-8")
    print(f"✅ Used extracted simulation code")
    return True
            backup_path.write_text(py, encoding="utf-8")
            print(f"💾 Saved extracted LaTeX code to {backup_path.name} for reference")
            return True
            
        elif compatibility['recommendation'] == 'merge_possible':
            # For now, preserve existing but suggest manual review
            backup_path = sim_path.parent / "simulation_extracted_candidate.py"
            backup_path.write_text(py, encoding="utf-8")
            print(f"� Saved extracted code as merge candidate to {backup_path.name}")
            print("💡 Consider manually reviewing and merging the best features from both")
            return True
    
    # Use extracted simulation if no existing file or extraction is better
    sim_path.write_text(py, encoding="utf-8")
    print(f"✅ Used extracted simulation code")
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
