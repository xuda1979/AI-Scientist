#!/usr/bin/env python3
"""
Iterative File Modification Agent

A Copilot-like tool that reads all files in a folder, sends them to an LLM
with a user prompt, applies the returned modifications, and repeats for
N iterations.

Usage:
    python iterative_agent.py --folder ./my_paper --prompt "Review this paper and improve it" --iterations 10
    python iterative_agent.py --folder ./my_paper --prompt "Fix all grammar issues" --iterations 3 --model gpt-4o
    python iterative_agent.py --folder ./src --prompt "Refactor for readability" --iterations 5 --extensions .py .js
    python iterative_agent.py --config agent_run_config.json
"""
from __future__ import annotations

import argparse
import copy
import datetime
import difflib
import json
import logging
import os
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Setup path so we can import from the project
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.chat import chat  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("iterative_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4o"
DEFAULT_ITERATIONS = 1
DEFAULT_TIMEOUT = 600  # 10 minutes per LLM call
MAX_FOLDER_SIZE_CHARS = 500_000  # safety limit for context window

# File extensions to include by default
DEFAULT_TEXT_EXTENSIONS: Set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".r",
    ".tex", ".bib", ".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".json",
    ".xml", ".html", ".css", ".scss", ".less", ".sql", ".sh", ".bash", ".bat",
    ".ps1", ".cfg", ".ini", ".conf", ".env", ".gitignore", ".dockerfile",
    ".makefile", ".cmake", ".gradle",
}

# Directories to always skip
SKIP_DIRS: Set[str] = {
    ".git", ".svn", ".hg", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
    "*.egg-info", ".idea", ".vscode",
}

# Files to always skip
SKIP_FILES: Set[str] = {
    ".DS_Store", "Thumbs.db", "desktop.ini",
}


# ---------------------------------------------------------------------------
# System prompt for the LLM
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert AI coding and writing assistant. The user will give you
the full contents of every file in a project folder, plus an instruction.

Your job is to carry out the instruction by modifying, creating, or deleting
files as needed.

IMPORTANT RULES:
1. Return ONLY a JSON array of file operations. Do NOT include any other text,
   markdown fences, or commentary outside the JSON.
2. Each element in the array is an object with these fields:
   - "action": one of "modify", "create", or "delete"
   - "path": the relative file path (use forward slashes)
   - "content": the FULL new content of the file (required for "modify" and
     "create"; omit for "delete")
3. If no changes are needed, return an empty array: []
4. For "modify", always return the COMPLETE file content, not a partial diff.
5. Preserve files you do not need to change — simply omit them from the array.
6. Think carefully before making changes. Explain your reasoning inside a
   top-level "reasoning" field (a string) if you wish, but the "changes"
   field MUST be the array of operations.

RESPONSE FORMAT (strict JSON):
{
  "reasoning": "optional free-text explanation of what you did and why",
  "changes": [
    {"action": "modify", "path": "relative/path/to/file.tex", "content": "...full file..."},
    {"action": "create", "path": "new_file.py", "content": "...full file..."},
    {"action": "delete", "path": "old_file.txt"}
  ]
}
""")


# ===================================================================
# File I/O helpers
# ===================================================================

def _should_skip_dir(dirname: str) -> bool:
    """Return True if this directory should be skipped."""
    return dirname in SKIP_DIRS or dirname.startswith(".")


def _should_skip_file(filename: str) -> bool:
    """Return True if this file should be skipped."""
    return filename in SKIP_FILES


def read_folder(
    folder: Path,
    extensions: Optional[Set[str]] = None,
    max_chars: int = MAX_FOLDER_SIZE_CHARS,
) -> Dict[str, str]:
    """
    Recursively read all text files in *folder*.

    Returns a dict mapping relative paths (forward-slash) to file contents.
    Stops adding files once *max_chars* total characters is reached.
    """
    extensions = extensions or DEFAULT_TEXT_EXTENSIONS
    files: Dict[str, str] = {}
    total_chars = 0

    for root, dirs, filenames in os.walk(folder):
        # Prune skipped directories in-place
        dirs[:] = [d for d in dirs if not _should_skip_dir(d)]
        dirs.sort()

        for fname in sorted(filenames):
            if _should_skip_file(fname):
                continue

            ext = os.path.splitext(fname)[1].lower()
            # If no extension, check if file itself matches (e.g. Makefile)
            if ext not in extensions and fname.lower() not in {e.lstrip(".") for e in extensions}:
                continue

            filepath = Path(root) / fname
            rel = filepath.relative_to(folder).as_posix()

            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError) as exc:
                logger.warning("Skipping %s: %s", rel, exc)
                continue

            if total_chars + len(content) > max_chars:
                logger.warning(
                    "Reached %d-char limit after %d files; remaining files skipped.",
                    max_chars, len(files),
                )
                return files

            files[rel] = content
            total_chars += len(content)

    return files


def format_files_for_prompt(files: Dict[str, str]) -> str:
    """Format file contents into a single string for the LLM context."""
    parts: List[str] = []
    for path, content in sorted(files.items()):
        parts.append(f"===== FILE: {path} =====")
        parts.append(content)
        parts.append(f"===== END FILE: {path} =====\n")
    return "\n".join(parts)


# ===================================================================
# Response parsing
# ===================================================================

def parse_llm_response(response: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse the LLM JSON response into (reasoning, changes_list).

    Handles common LLM quirks like wrapping JSON in markdown fences.
    """
    text = response.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try to parse as JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("Could not parse LLM response as JSON.")
                logger.debug("Raw response:\n%s", response[:2000])
                return ("", [])
        else:
            # Try to find a JSON array
            match = re.search(r"\[[\s\S]*\]", text)
            if match:
                try:
                    arr = json.loads(match.group())
                    data = {"changes": arr}
                except json.JSONDecodeError:
                    logger.error("Could not parse LLM response as JSON.")
                    return ("", [])
            else:
                logger.error("No JSON found in LLM response.")
                return ("", [])

    # Normalize the structure
    if isinstance(data, list):
        return ("", data)

    reasoning = data.get("reasoning", "")
    changes = data.get("changes", [])
    if not isinstance(changes, list):
        changes = []

    return (reasoning, changes)


# ===================================================================
# Apply changes
# ===================================================================

def apply_changes(
    folder: Path,
    changes: List[Dict[str, Any]],
    dry_run: bool = False,
) -> List[str]:
    """
    Apply file changes to the folder.

    Returns a list of human-readable summaries of what was done.
    """
    summaries: List[str] = []

    for change in changes:
        action = change.get("action", "").lower()
        rel_path = change.get("path", "")
        content = change.get("content", "")

        if not rel_path:
            logger.warning("Skipping change with empty path: %s", change)
            continue

        # Normalise path separators and prevent path traversal
        rel_path = rel_path.replace("\\", "/")
        if rel_path.startswith("/") or ".." in rel_path.split("/"):
            logger.warning("Skipping suspicious path: %s", rel_path)
            continue

        abs_path = folder / rel_path

        if action == "delete":
            if abs_path.exists():
                if not dry_run:
                    abs_path.unlink()
                summaries.append(f"  DELETED  {rel_path}")
            else:
                summaries.append(f"  SKIP-DEL {rel_path} (not found)")

        elif action in ("modify", "create"):
            existed = abs_path.exists()
            old_content = ""
            if existed:
                try:
                    old_content = abs_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass

            if action == "modify" and not existed:
                # Treat as create if file doesn't exist
                action = "create"

            if action == "modify" and content == old_content:
                summaries.append(f"  NO-CHANGE {rel_path}")
                continue

            if not dry_run:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(content, encoding="utf-8")

            if action == "create" and not existed:
                summaries.append(f"  CREATED  {rel_path} ({len(content):,} chars)")
            else:
                # Show a compact diff summary
                added = 0
                removed = 0
                for line in difflib.unified_diff(
                    old_content.splitlines(), content.splitlines(), lineterm=""
                ):
                    if line.startswith("+") and not line.startswith("+++"):
                        added += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        removed += 1
                summaries.append(
                    f"  MODIFIED {rel_path} (+{added} -{removed} lines)"
                )
        else:
            logger.warning("Unknown action '%s' for %s", action, rel_path)

    return summaries


# ===================================================================
# Backup / restore helpers
# ===================================================================

def create_backup(folder: Path, backup_dir: Path) -> Path:
    """Create a timestamped backup of the folder."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"backup_{timestamp}"
    shutil.copytree(folder, backup_path, dirs_exist_ok=False)
    logger.info("Backup created: %s", backup_path)
    return backup_path


# ===================================================================
# Main iteration loop
# ===================================================================

def run_iterative_agent(
    folder: str | Path,
    prompt: str,
    iterations: int = DEFAULT_ITERATIONS,
    model: str = DEFAULT_MODEL,
    extensions: Optional[Set[str]] = None,
    dry_run: bool = False,
    backup: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    fallback_models: Optional[List[str]] = None,
    log_dir: Optional[str | Path] = None,
    max_chars: int = MAX_FOLDER_SIZE_CHARS,
    delay_between_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Run the iterative file-modification agent.

    Args:
        folder: Path to the target folder.
        prompt: The instruction to repeat each iteration.
        iterations: How many times to run the prompt.
        model: LLM model name.
        extensions: Set of file extensions to include (e.g. {".py", ".tex"}).
        dry_run: If True, don't actually write changes.
        backup: If True, create a backup before starting.
        timeout: Per-call timeout in seconds.
        fallback_models: Fallback models if primary fails.
        log_dir: Directory to save per-iteration logs. Defaults to folder/.agent_logs.
        max_chars: Max total characters to read from folder.
        delay_between_iterations: Seconds to wait between iterations.

    Returns:
        A summary dict with iteration results.
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    fallback_models = fallback_models or ["gpt-4o", "gpt-4"]

    # Log directory
    if log_dir is None:
        log_dir = folder / ".agent_logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Backup
    if backup and not dry_run:
        backup_dir = folder.parent / f"{folder.name}_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        create_backup(folder, backup_dir)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results: Dict[str, Any] = {
        "run_id": run_id,
        "folder": str(folder),
        "prompt": prompt,
        "model": model,
        "iterations_requested": iterations,
        "iterations_completed": 0,
        "dry_run": dry_run,
        "iteration_details": [],
    }

    print("=" * 70)
    print(f"  Iterative Agent — {iterations} iteration(s)")
    print(f"  Folder : {folder}")
    print(f"  Model  : {model}")
    print(f"  Prompt : {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"  Dry run: {dry_run}")
    print("=" * 70)

    for i in range(1, iterations + 1):
        iter_start = time.time()
        print(f"\n{'─' * 60}")
        print(f"  Iteration {i}/{iterations}")
        print(f"{'─' * 60}")

        # 1) Read current folder state
        files = read_folder(folder, extensions=extensions, max_chars=max_chars)
        if not files:
            logger.error("No files found in %s — aborting.", folder)
            break

        file_listing = format_files_for_prompt(files)
        total_chars = sum(len(c) for c in files.values())
        print(f"  Read {len(files)} file(s)  ({total_chars:,} chars total)")

        # 2) Build messages
        user_message = (
            f"Here are all the files in the project folder:\n\n"
            f"{file_listing}\n\n"
            f"--- INSTRUCTION ---\n"
            f"{prompt}\n\n"
            f"This is iteration {i} of {iterations}. "
            f"Please review the current state of the files and apply the instruction. "
            f"Return your changes as the specified JSON format."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # 3) Call LLM
        print(f"  Calling {model}...")
        try:
            response_text, tokens_used = chat(
                messages=messages,
                model=model,
                temperature=0.7,
                request_timeout=timeout,
                prompt_type="iterative_agent",
                fallback_models=fallback_models,
            )
        except Exception as exc:
            logger.error("LLM call failed on iteration %d: %s", i, exc)
            results["iteration_details"].append({
                "iteration": i,
                "status": "error",
                "error": str(exc),
                "duration_s": round(time.time() - iter_start, 1),
            })
            continue

        print(f"  Response received ({len(response_text):,} chars, ~{tokens_used or '?'} tokens)")

        # 4) Parse response
        reasoning, changes = parse_llm_response(response_text)

        if reasoning:
            print(f"  Reasoning: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")

        if not changes:
            print("  No changes returned by the model.")
            results["iteration_details"].append({
                "iteration": i,
                "status": "no_changes",
                "reasoning": reasoning,
                "duration_s": round(time.time() - iter_start, 1),
            })
            results["iterations_completed"] = i
            # If no changes, the model thinks it's done
            if i < iterations:
                print("  (Model returned no changes — continuing to next iteration)")
            continue

        print(f"  Applying {len(changes)} change(s)...")

        # 5) Apply changes
        summaries = apply_changes(folder, changes, dry_run=dry_run)
        for s in summaries:
            print(s)

        # 6) Save iteration log
        iter_log = {
            "iteration": i,
            "status": "applied",
            "reasoning": reasoning,
            "num_changes": len(changes),
            "summaries": summaries,
            "tokens_used": tokens_used,
            "duration_s": round(time.time() - iter_start, 1),
        }
        results["iteration_details"].append(iter_log)
        results["iterations_completed"] = i

        # Save full response to log file
        iter_log_file = log_dir / f"iteration_{i:03d}.json"
        with open(iter_log_file, "w", encoding="utf-8") as f:
            json.dump({
                "iteration": i,
                "reasoning": reasoning,
                "changes": changes,
                "response_raw": response_text,
            }, f, indent=2, ensure_ascii=False)

        elapsed = round(time.time() - iter_start, 1)
        print(f"  Done in {elapsed}s")

        # Delay between iterations (except after the last one)
        if i < iterations and delay_between_iterations > 0:
            print(f"  Waiting {delay_between_iterations}s before next iteration...")
            time.sleep(delay_between_iterations)

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  Run complete: {results['iterations_completed']}/{iterations} iterations")
    total_changes = sum(
        d.get("num_changes", 0) for d in results["iteration_details"]
    )
    print(f"  Total changes applied: {total_changes}")
    print(f"  Logs saved to: {log_dir}")
    print(f"{'=' * 70}\n")

    # Save overall summary
    summary_file = log_dir / f"run_summary_{run_id}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterative File Modification Agent — like Copilot, but runs N times.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Review and improve a paper 10 times
              python iterative_agent.py --folder ./paper --prompt "Review this paper and improve it" --iterations 10

              # Refactor Python code 5 times
              python iterative_agent.py --folder ./src --prompt "Refactor for readability" -n 5 --extensions .py

              # Dry run (preview changes without writing)
              python iterative_agent.py --folder ./project --prompt "Add docstrings" -n 3 --dry-run

              # Use a config file
              python iterative_agent.py --config my_config.json
        """),
    )

    parser.add_argument(
        "--folder", "-f",
        type=str,
        help="Path to the target folder containing files to modify.",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="The instruction to send to the LLM each iteration.",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of iterations to run (default: {DEFAULT_ITERATIONS}).",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--extensions", "-e",
        nargs="+",
        type=str,
        default=None,
        help="File extensions to include (e.g. .py .tex .md). Defaults to common text files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to disk.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup before starting.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per LLM call in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--fallback-models",
        nargs="+",
        type=str,
        default=None,
        help="Fallback models if primary model fails.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for iteration logs (default: <folder>/.agent_logs).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_FOLDER_SIZE_CHARS,
        help=f"Max total characters to read from folder (default: {MAX_FOLDER_SIZE_CHARS:,}).",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Seconds to wait between iterations (default: 5).",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to a JSON config file (overrides CLI args).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Config values override defaults; CLI args override config
        folder = args.folder or config.get("folder")
        prompt = args.prompt or config.get("prompt")
        iterations = args.iterations if args.iterations != DEFAULT_ITERATIONS else config.get("iterations", DEFAULT_ITERATIONS)
        model = args.model if args.model != DEFAULT_MODEL else config.get("model", DEFAULT_MODEL)
        ext_list = args.extensions or config.get("extensions")
        dry_run = args.dry_run or config.get("dry_run", False)
        no_backup = args.no_backup or config.get("no_backup", False)
        timeout = args.timeout if args.timeout != DEFAULT_TIMEOUT else config.get("timeout", DEFAULT_TIMEOUT)
        fallback_models = args.fallback_models or config.get("fallback_models")
        log_dir = args.log_dir or config.get("log_dir")
        max_chars = args.max_chars if args.max_chars != MAX_FOLDER_SIZE_CHARS else config.get("max_chars", MAX_FOLDER_SIZE_CHARS)
        delay = args.delay if args.delay != 5 else config.get("delay", 5)
    else:
        folder = args.folder
        prompt = args.prompt
        iterations = args.iterations
        model = args.model
        ext_list = args.extensions
        dry_run = args.dry_run
        no_backup = args.no_backup
        timeout = args.timeout
        fallback_models = args.fallback_models
        log_dir = args.log_dir
        max_chars = args.max_chars
        delay = args.delay

    # Validate required args
    if not folder:
        print("Error: --folder is required.")
        parser.print_help()
        sys.exit(1)
    if not prompt:
        print("Error: --prompt is required.")
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse extensions
    extensions: Optional[Set[str]] = None
    if ext_list:
        extensions = set()
        for ext in ext_list:
            if not ext.startswith("."):
                ext = "." + ext
            extensions.add(ext.lower())

    # Run
    try:
        run_iterative_agent(
            folder=folder,
            prompt=prompt,
            iterations=iterations,
            model=model,
            extensions=extensions,
            dry_run=dry_run,
            backup=not no_backup,
            timeout=timeout,
            fallback_models=fallback_models,
            log_dir=log_dir,
            max_chars=max_chars,
            delay_between_iterations=delay,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
