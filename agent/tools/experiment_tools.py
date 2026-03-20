"""
Experiment Tools — code execution, data analysis, and figure generation.

Provides tools for:
- Running Python experiments in sandboxed subprocesses
- Parsing and analyzing experiment outputs
- Generating publication-quality figures
- Data processing and statistical analysis
"""
from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools import ToolDefinition, ToolParameter, ToolResult, ToolRegistry

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Code Execution
# ═════════════════════════════════════════════════════════════════════════

def execute_python_code(
    code: str,
    working_dir: str = ".",
    timeout: int = 300,
    python_executable: str = "python",
    save_as: str = "",
) -> ToolResult:
    """
    Execute Python code in a sandboxed subprocess.

    Args:
        code: Python source code to execute.
        working_dir: Working directory for the subprocess.
        timeout: Execution timeout in seconds.
        python_executable: Path to Python executable.
        save_as: If provided, save the code to this filename before executing.
    """
    try:
        work_dir = Path(working_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Optionally save to file
        if save_as:
            code_path = work_dir / save_as
            code_path.write_text(code, encoding="utf-8")
            cmd = [python_executable, str(code_path)]
        else:
            # Write to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", dir=str(work_dir),
                delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                temp_path = f.name
            cmd = [python_executable, temp_path]

        # Execute
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir),
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        elapsed = time.time() - t0

        # Clean up temp file
        if not save_as and os.path.exists(temp_path):
            os.unlink(temp_path)

        # Collect output files generated
        output_files = []
        for ext in [".png", ".pdf", ".svg", ".csv", ".json", ".dat", ".txt"]:
            for fp in work_dir.glob(f"*{ext}"):
                if fp.stat().st_mtime >= t0:
                    output_files.append(str(fp.relative_to(work_dir)))

        result_data = {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[:10000] if proc.stdout else "",
            "stderr": proc.stderr[:5000] if proc.stderr else "",
            "elapsed_seconds": round(elapsed, 2),
            "output_files": output_files,
        }

        if proc.returncode != 0:
            return ToolResult(
                False,
                data=result_data,
                error=f"Code exited with return code {proc.returncode}:\n{proc.stderr[:2000]}",
            )

        return ToolResult(True, data=result_data)

    except subprocess.TimeoutExpired:
        return ToolResult(False, error=f"Code execution timed out after {timeout}s")
    except Exception as exc:
        return ToolResult(False, error=f"Execution failed: {exc}")


def validate_python_syntax(code: str) -> ToolResult:
    """
    Validate Python code for syntax errors without executing it.

    Args:
        code: Python source code to validate.
    """
    try:
        ast.parse(code)
        return ToolResult(True, data={"valid": True, "message": "Syntax is valid"})
    except SyntaxError as e:
        return ToolResult(True, data={
            "valid": False,
            "error": str(e),
            "line": e.lineno,
            "offset": e.offset,
            "text": e.text,
        })


def fix_and_retry_code(
    code: str,
    error_message: str,
    working_dir: str = ".",
    timeout: int = 300,
    python_executable: str = "python",
    max_retries: int = 3,
    llm_fn=None,
    model: str = "gpt-5-pro",
) -> ToolResult:
    """
    Attempt to fix broken code using LLM-powered error analysis.

    Args:
        code: The code that failed.
        error_message: The error output from the failed execution.
        working_dir: Working directory.
        timeout: Execution timeout.
        python_executable: Python executable path.
        max_retries: Maximum fix attempts.
        llm_fn: LLM function for generating fixes.
        model: Model to use for fixes.
    """
    if not llm_fn:
        return ToolResult(False, error="No LLM function provided for code fixing")

    current_code = code
    current_error = error_message

    for attempt in range(max_retries):
        # Ask LLM to fix the code
        fix_prompt = (
            f"The following Python code failed with an error. Fix it.\n\n"
            f"## Code\n```python\n{current_code}\n```\n\n"
            f"## Error\n```\n{current_error}\n```\n\n"
            f"Return ONLY the corrected Python code, with no explanation, "
            f"wrapped in ```python ... ``` markers."
        )

        try:
            response = llm_fn(
                messages=[{"role": "user", "content": fix_prompt}],
                model=model,
                prompt_type="general",
                request_timeout=120,
            )

            # Extract code from response
            import re
            code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
            if code_match:
                fixed_code = code_match.group(1)
            else:
                fixed_code = response.strip()

            # Try executing the fixed code
            result = execute_python_code(
                fixed_code, working_dir, timeout, python_executable
            )

            if result.success:
                result.data["fix_attempts"] = attempt + 1
                result.data["fixed_code"] = fixed_code
                return result

            current_code = fixed_code
            current_error = result.error or result.data.get("stderr", "")

        except Exception as exc:
            current_error = str(exc)

    return ToolResult(
        False,
        error=f"Failed to fix code after {max_retries} attempts. Last error: {current_error}",
    )


# ═════════════════════════════════════════════════════════════════════════
# Data Analysis
# ═════════════════════════════════════════════════════════════════════════

def analyze_data_file(
    file_path: str,
    analysis_type: str = "summary",
) -> ToolResult:
    """
    Analyze a data file (CSV, JSON, or text) and return statistics.

    Args:
        file_path: Path to the data file.
        analysis_type: Type of analysis — "summary", "statistics", "preview".
    """
    try:
        fp = Path(file_path)
        if not fp.exists():
            return ToolResult(False, error=f"File not found: {file_path}")

        content = fp.read_text(encoding="utf-8", errors="replace")

        if fp.suffix == ".csv":
            return _analyze_csv(content, analysis_type)
        elif fp.suffix == ".json":
            return _analyze_json(content, analysis_type)
        else:
            return ToolResult(True, data={
                "file": str(fp),
                "size_bytes": fp.stat().st_size,
                "lines": content.count("\n") + 1,
                "preview": content[:2000],
            })

    except Exception as exc:
        return ToolResult(False, error=f"Analysis failed: {exc}")


def _analyze_csv(content: str, analysis_type: str) -> ToolResult:
    """Analyze CSV data."""
    lines = content.strip().split("\n")
    if not lines:
        return ToolResult(True, data={"empty": True})

    headers = lines[0].split(",") if lines else []
    num_rows = len(lines) - 1  # subtract header

    data = {
        "format": "csv",
        "columns": headers,
        "num_rows": num_rows,
        "num_columns": len(headers),
    }

    if analysis_type == "preview":
        data["preview_rows"] = lines[:6]
    elif analysis_type == "statistics" and num_rows > 0:
        # Try to compute basic stats for numeric columns
        col_values: Dict[str, List[float]] = {h: [] for h in headers}
        for line in lines[1:]:
            values = line.split(",")
            for i, h in enumerate(headers):
                if i < len(values):
                    try:
                        col_values[h].append(float(values[i].strip()))
                    except ValueError:
                        pass

        stats = {}
        for col, vals in col_values.items():
            if vals:
                stats[col] = {
                    "count": len(vals),
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                }
        data["statistics"] = stats

    return ToolResult(True, data=data)


def _analyze_json(content: str, analysis_type: str) -> ToolResult:
    """Analyze JSON data."""
    try:
        parsed = json.loads(content)
        data: Dict[str, Any] = {"format": "json"}

        if isinstance(parsed, list):
            data["type"] = "array"
            data["length"] = len(parsed)
            if parsed and isinstance(parsed[0], dict):
                data["keys"] = list(parsed[0].keys())
                data["sample"] = parsed[:3]
        elif isinstance(parsed, dict):
            data["type"] = "object"
            data["keys"] = list(parsed.keys())
            data["sample"] = {k: str(v)[:100] for k, v in list(parsed.items())[:10]}

        return ToolResult(True, data=data)
    except json.JSONDecodeError as exc:
        return ToolResult(False, error=f"Invalid JSON: {exc}")


# ═════════════════════════════════════════════════════════════════════════
# Figure Generation
# ═════════════════════════════════════════════════════════════════════════

def generate_figure(
    code: str,
    filename: str = "figure.pdf",
    working_dir: str = ".",
    python_executable: str = "python",
) -> ToolResult:
    """
    Generate a publication-quality figure by executing matplotlib code.

    The code should use matplotlib to create a figure and save it.
    The tool will ensure proper DPI and format settings.

    Args:
        code: Python code that generates and saves a figure using matplotlib.
        filename: Output filename for the figure.
        working_dir: Working directory.
        python_executable: Python executable path.
    """
    # Wrap the code with proper matplotlib setup
    wrapper = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality defaults
plt.rcParams.update({{
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}})

# User figure code
{code}

# Ensure the figure is saved
if plt.get_fignums():
    plt.savefig({filename!r}, bbox_inches='tight', dpi=300)
    print(f"Figure saved: {filename}")
    plt.close('all')
"""

    result = execute_python_code(
        wrapper,
        working_dir=working_dir,
        timeout=120,
        python_executable=python_executable,
    )

    if result.success:
        fig_path = Path(working_dir) / filename
        if fig_path.exists():
            result.data["figure_path"] = str(fig_path)
            result.data["figure_size_bytes"] = fig_path.stat().st_size
        else:
            result.success = False
            result.error = f"Figure file not created: {filename}"

    return result


# ═════════════════════════════════════════════════════════════════════════
# Registration
# ═════════════════════════════════════════════════════════════════════════

def register_experiment_tools(
    registry: ToolRegistry,
    working_dir: str = ".",
    python_executable: str = "python",
    timeout: int = 300,
    llm_fn=None,
    model: str = "gpt-5-pro",
) -> None:
    """Register all experiment tools with the tool registry."""

    # Code execution
    registry.register(ToolDefinition(
        name="execute_code",
        description="Execute Python code in a sandboxed subprocess. Returns stdout, stderr, and any output files created.",
        parameters=[
            ToolParameter("code", "string", "Python source code to execute"),
            ToolParameter("save_as", "string", "Optional filename to save the code as before running", required=False, default=""),
        ],
        handler=lambda code, save_as="": execute_python_code(
            code, working_dir, timeout, python_executable, save_as
        ),
        category="experiment",
    ))

    # Syntax validation
    registry.register(ToolDefinition(
        name="validate_code",
        description="Validate Python code for syntax errors without executing it.",
        parameters=[
            ToolParameter("code", "string", "Python source code to validate"),
        ],
        handler=validate_python_syntax,
        category="experiment",
    ))

    # Fix and retry
    registry.register(ToolDefinition(
        name="fix_and_retry",
        description="Attempt to automatically fix broken Python code using AI-powered error analysis. Iteratively fixes and retries.",
        parameters=[
            ToolParameter("code", "string", "The Python code that failed"),
            ToolParameter("error_message", "string", "The error output from the failed execution"),
        ],
        handler=lambda code, error_message: fix_and_retry_code(
            code, error_message, working_dir, timeout, python_executable,
            llm_fn=llm_fn, model=model,
        ),
        category="experiment",
    ))

    # Data analysis
    registry.register(ToolDefinition(
        name="analyze_data",
        description="Analyze a data file (CSV, JSON, or text) and return statistics, preview, or summary.",
        parameters=[
            ToolParameter("file_path", "string", "Path to the data file"),
            ToolParameter("analysis_type", "string", "Type: summary, statistics, or preview", required=False, default="summary"),
        ],
        handler=analyze_data_file,
        category="experiment",
    ))

    # Figure generation
    registry.register(ToolDefinition(
        name="generate_figure",
        description="Generate a publication-quality figure by executing matplotlib code. Returns the path to the saved figure.",
        parameters=[
            ToolParameter("code", "string", "Python matplotlib code that creates a figure"),
            ToolParameter("filename", "string", "Output filename (e.g. 'figure1.pdf')", required=False, default="figure.pdf"),
        ],
        handler=lambda code, filename="figure.pdf": generate_figure(
            code, filename, working_dir, python_executable
        ),
        category="experiment",
    ))
