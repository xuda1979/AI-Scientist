"""
Bridge between the existing sciresearch_workflow and the new autopilot system.

Provides helper functions to:
1. Convert between WorkflowConfig ↔ AutoPilotConfig
2. Hook autopilot into the existing GUI and CLI
3. Run experiment code using the existing simulation runner
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def workflow_config_to_autopilot(
    wf_config,
    topic: str = "",
    field: str = "",
    question: str = "",
    output_dir: Optional[str] = None,
    model: str = "gpt-5-pro",
):
    """
    Convert an existing WorkflowConfig + CLI args into an AutoPilotConfig.
    """
    from autopilot.config import AutoPilotConfig

    return AutoPilotConfig(
        topic=topic,
        field_name=field,
        research_question=question,
        project_dir=output_dir or "",
        model=model,
        fallback_models=list(wf_config.fallback_models) if wf_config.fallback_models else ["gpt-4o"],
        request_timeout=wf_config.request_timeout,
        max_global_iterations=max(1, wf_config.max_iterations // 4),
        quality_gate_threshold=wf_config.quality_threshold,
        content_protection=wf_config.content_protection,
        max_word_loss_pct=wf_config.content_protection_threshold * 100
        if wf_config.content_protection_threshold < 1
        else wf_config.content_protection_threshold,
        create_backups=True,
        save_trace=True,
    )


def run_autopilot_from_workflow(
    topic: str,
    field: str,
    question: str,
    output_dir: Path,
    model: str = "gpt-5-pro",
    config=None,
    **kwargs,
):
    """
    Drop-in replacement for run_workflow() that uses the autopilot system.
    Can be called from the GUI or CLI.
    """
    from core.config import WorkflowConfig
    from autopilot.agent import AutoPilotAgent
    from autopilot.config import AutoPilotConfig

    wf_config = config or WorkflowConfig()

    ap_config = workflow_config_to_autopilot(
        wf_config,
        topic=topic,
        field=field,
        question=question,
        output_dir=str(output_dir),
        model=model,
    )

    agent = AutoPilotAgent(ap_config)
    result = agent.run()

    # Return the project dir Path (compatible with run_workflow return type)
    return Path(result["project_dir"])


def run_experiment_in_project(project_dir: Path, python_exec: str = "python") -> str:
    """
    Run simulation.py inside a project directory using the existing sim runner.
    Returns the simulation output text.
    """
    sim_path = project_dir / "simulation.py"
    if not sim_path.exists():
        return "(no simulation.py found)"

    try:
        from utils.sim_runner import run_simulation
        success, output = run_simulation(sim_path, timeout=600, python_exec=python_exec)
        return output if success else f"SIMULATION FAILED:\n{output}"
    except ImportError:
        # Fallback: run directly
        import subprocess
        try:
            proc = subprocess.run(
                [python_exec, str(sim_path)],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=600,
            )
            return proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        except Exception as e:
            return f"SIMULATION ERROR: {e}"
