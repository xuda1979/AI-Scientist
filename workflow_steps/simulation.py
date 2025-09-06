from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Any
import os


def run_simulation_step(
    paper_path: Path,
    sim_path: Path,
    project_dir: Path,
    model: str,
    request_timeout: int,
    python_exec: Optional[str],
) -> Tuple[str, Any]:
    """Extract and run simulation, returning summary and raw output."""
    from sciresearch_workflow import _create_simulation_fixer, _extract_simulation_code_with_validation
    from utils.sim_runner import run_simulation_with_smart_fixing, summarize_simulation_outputs

    extract_success, extract_message = _extract_simulation_code_with_validation(paper_path, sim_path)
    if not extract_success:
        print(f" Simulation extraction issues: {extract_message}")
    if os.getenv("OFFLINE_MODE") == "1":
        print("Skipping simulation in offline mode")
        simulation_code = sim_path.read_text(encoding="utf-8", errors="ignore") if sim_path.exists() else ""
        return "Simulation skipped in offline mode.", {"stdout": "", "stderr": "", "return_code": 0}

    simulation_fixer = _create_simulation_fixer(model, request_timeout)
    sim_out = run_simulation_with_smart_fixing(
        sim_path,
        python_exec=python_exec,
        cwd=project_dir,
        llm_fixer=simulation_fixer,
        max_fix_attempts=2,
    )
    simulation_code = sim_path.read_text(encoding="utf-8", errors="ignore")
    sim_summary = summarize_simulation_outputs(sim_out, simulation_code)
    return sim_summary, sim_out
