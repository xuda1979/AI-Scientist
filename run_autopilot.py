#!/usr/bin/env python3
"""
run_autopilot.py — CLI entry point for the auto-pilot research paper mode.

Usage:
    python run_autopilot.py \
        --topic "Transformer attention mechanisms" \
        --field "Machine Learning" \
        --question "How does sparse attention compare to full attention for long documents?" \
        --output-dir ./output/my_paper \
        --model gpt-5-pro

    # With an existing project folder:
    python run_autopilot.py \
        --project-dir ./output/existing_project \
        --model gpt-4o

    # Custom config file:
    python run_autopilot.py \
        --config autopilot_config.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from autopilot import AutoPilotAgent, AutoPilotConfig
from autopilot.config import STAGES


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-Pilot Mode: fully automatic research paper writing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Required (unless --project-dir or --config is given) ────────────
    p.add_argument("--topic", type=str, default="",
                   help="Research topic / title")
    p.add_argument("--field", type=str, default="",
                   help="Research field (e.g. 'Machine Learning')")
    p.add_argument("--question", type=str, default="",
                   help="Central research question")

    # ── Output / project ────────────────────────────────────────────────
    p.add_argument("--output-dir", type=str, default="./output/autopilot",
                   help="Base output directory for new projects")
    p.add_argument("--project-dir", type=str, default=None,
                   help="Existing project folder to continue working on")

    # ── Model ───────────────────────────────────────────────────────────
    p.add_argument("--model", type=str, default="gpt-5-pro",
                   help="Primary LLM model")
    p.add_argument("--fallback-models", type=str, nargs="+",
                   default=["gpt-5", "gpt-4o"],
                   help="Fallback models")
    p.add_argument("--timeout", type=int, default=3600,
                   help="Request timeout in seconds")

    # ── Pipeline control ────────────────────────────────────────────────
    p.add_argument("--stages", type=str, nargs="+", default=None,
                   help=f"Stages to run (default: all). Choices: {', '.join(STAGES)}")
    p.add_argument("--skip-stages", type=str, nargs="+", default=None,
                   help="Stages to skip")
    p.add_argument("--max-global-iterations", type=int, default=3,
                   help="Max full pipeline reruns")
    p.add_argument("--quality-threshold", type=float, default=0.75,
                   help="Minimum quality score to pass a stage (0-1)")
    p.add_argument("--no-abort", action="store_true",
                   help="Continue even on critical failures")

    # ── Prompt optimiser ────────────────────────────────────────────────
    p.add_argument("--prompt-candidates", type=int, default=3,
                   help="Number of prompt candidates per stage")
    p.add_argument("--no-meta-prompt", action="store_true",
                   help="Disable meta-prompting (use static prompts)")
    p.add_argument("--prompt-strategy", type=str, default="score",
                   choices=["score", "tournament", "ensemble"],
                   help="Prompt selection strategy")

    # ── File management ─────────────────────────────────────────────────
    p.add_argument("--no-backups", action="store_true",
                   help="Disable file backups")
    p.add_argument("--no-content-protection", action="store_true",
                   help="Disable content protection for .tex files")

    # ── Logging ─────────────────────────────────────────────────────────
    p.add_argument("--verbose", action="store_true", default=True,
                   help="Verbose output")
    p.add_argument("--quiet", action="store_true",
                   help="Minimal output")
    p.add_argument("--no-trace", action="store_true",
                   help="Disable trace saving")

    # ── Config file ─────────────────────────────────────────────────────
    p.add_argument("--config", type=str, default=None,
                   help="Path to autopilot config JSON file")

    # ── Document type ───────────────────────────────────────────────────
    p.add_argument("--document-type", type=str, default="auto",
                   help="Document type (auto, research_paper, survey_paper, etc.)")

    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> AutoPilotConfig:
    """Build an AutoPilotConfig from CLI args (with optional config file merge)."""
    # Start from file or defaults
    if args.config:
        config = AutoPilotConfig.from_file(Path(args.config))
    else:
        config = AutoPilotConfig()

    # Override with CLI args
    if args.topic:
        config.topic = args.topic
    if args.field:
        config.field_name = args.field
    if args.question:
        config.research_question = args.question

    config.model = args.model
    config.fallback_models = args.fallback_models
    config.request_timeout = args.timeout

    # Stages
    if args.stages:
        config.stages = args.stages
    if args.skip_stages:
        config.stages = [s for s in config.stages if s not in args.skip_stages]

    config.max_global_iterations = args.max_global_iterations
    config.quality_gate_threshold = args.quality_threshold
    config.abort_on_critical_failure = not args.no_abort

    # Prompt optimiser
    config.prompt_candidates_per_stage = args.prompt_candidates
    config.use_meta_prompt = not args.no_meta_prompt
    config.prompt_selection_strategy = args.prompt_strategy

    # File management
    config.create_backups = not args.no_backups
    config.content_protection = not args.no_content_protection

    # Logging
    config.verbose = args.verbose and not args.quiet
    config.save_trace = not args.no_trace

    # Document type
    config.document_type = args.document_type

    # Project directory
    if args.project_dir:
        config.project_dir = str(Path(args.project_dir).resolve())
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in config.topic)[:50]
        config.project_dir = str(
            Path(args.output_dir).resolve() / f"{safe_topic}_{timestamp}"
        )

    return config


def main(argv=None):
    args = parse_args(argv)

    # ── Logging ──────────────────────────────────────────────────────────
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Validate inputs ──────────────────────────────────────────────────
    if not args.project_dir and not args.topic:
        print("ERROR: Either --topic or --project-dir is required.", file=sys.stderr)
        sys.exit(1)

    # ── Build config ─────────────────────────────────────────────────────
    config = build_config(args)

    print(f"\n📋  Configuration:")
    print(f"    Topic:     {config.topic}")
    print(f"    Field:     {config.field_name}")
    print(f"    Question:  {config.research_question}")
    print(f"    Model:     {config.model}")
    print(f"    Stages:    {len(config.stages)}")
    print(f"    Project:   {config.project_dir}")
    print()

    # ── Run ──────────────────────────────────────────────────────────────
    agent = AutoPilotAgent(config)
    result = agent.run()

    # ── Report ───────────────────────────────────────────────────────────
    status = result["status"]
    if status == "completed":
        print(f"\n🎉  Paper generation completed successfully!")
    elif status == "aborted":
        print(f"\n⚠️  Paper generation was aborted due to quality issues.")
    else:
        print(f"\n❌  Paper generation encountered an error.")

    print(f"    Stages completed: {result['stages_completed']}")
    print(f"    Total time: {result['total_time']:.1f}s")
    print(f"    Output: {result['project_dir']}")

    sys.exit(0 if status == "completed" else 1)


if __name__ == "__main__":
    main()
