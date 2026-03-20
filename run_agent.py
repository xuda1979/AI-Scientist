#!/usr/bin/env python3
"""
run_agent.py — CLI entry point for the autonomous research agent.

Usage:
    python run_agent.py --topic "Topic" --field "Field" --question "Question"
    python run_agent.py --config agent_config.json
    python run_agent.py --topic "..." --field "..." --question "..." --model gpt-4o --output-dir ./papers

Examples:
    python run_agent.py \\
        --topic "Quantum error correction with surface codes" \\
        --field "quantum computing" \\
        --question "How does code distance affect logical error rates under circuit-level noise?"

    python run_agent.py \\
        --topic "Transformer attention mechanisms" \\
        --field "machine learning" \\
        --question "Can sparse attention achieve comparable performance to full attention?" \\
        --model gpt-4o \\
        --skip-stages figure_generation \\
        --max-iterations 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent — fully automatic research paper writing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments (unless --config is provided)
    parser.add_argument("--topic", type=str, default="",
                        help="Research topic")
    parser.add_argument("--field", type=str, default="",
                        help="Research field/domain")
    parser.add_argument("--question", type=str, default="",
                        help="Specific research question")

    # Configuration
    parser.add_argument("--config", type=str, default="",
                        help="Path to JSON configuration file")
    parser.add_argument("--model", type=str, default="",
                        help="LLM model to use (default: from config or SCI_MODEL env)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Base output directory (default: output)")
    parser.add_argument("--project-dir", type=str, default="",
                        help="Specific project directory (default: auto-generated)")

    # Pipeline control
    parser.add_argument("--phases", nargs="+", default=None,
                        help="Phases to run: research design implementation writing review")
    parser.add_argument("--skip-stages", nargs="+", default=None,
                        help="Stages to skip")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Maximum global iterations")
    parser.add_argument("--quality-threshold", type=float, default=None,
                        help="Quality threshold (0-1)")

    # Options
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress most output")
    parser.add_argument("--no-trace", action="store_true",
                        help="Disable trace logging")

    # Document type
    parser.add_argument("--doc-type", type=str, default="research_paper",
                        choices=["research_paper", "conference_paper", "journal_article",
                                 "technical_report", "survey_paper", "white_paper"],
                        help="Document type")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Import here to avoid slow startup for --help
    from agent.config import AgentConfig
    from agent.orchestrator import ResearchOrchestrator

    # Build configuration
    if args.config:
        config = AgentConfig.from_file(args.config)
        # Override with CLI args
        if args.topic:
            config.topic = args.topic
        if args.field:
            config.field = args.field
        if args.question:
            config.research_question = args.question
    else:
        config = AgentConfig(
            topic=args.topic,
            field=args.field,
            research_question=args.question,
        )

    # Apply CLI overrides
    if args.model:
        config.model = args.model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.project_dir:
        config.project_dir = args.project_dir
    if args.phases:
        config.phases = args.phases
    if args.skip_stages:
        config.skip_stages = args.skip_stages
    if args.max_iterations is not None:
        config.max_global_iterations = args.max_iterations
    if args.quality_threshold is not None:
        config.quality_threshold = args.quality_threshold
    if args.quiet:
        config.verbose = False
        config.log_level = "WARNING"
    if args.no_trace:
        config.save_trace = False
    config.document_type = args.doc_type

    # Validate
    if not config.topic:
        print("ERROR: --topic is required (or provide via --config)")
        return 1
    if not config.field:
        print("ERROR: --field is required (or provide via --config)")
        return 1
    if not config.research_question:
        print("ERROR: --question is required (or provide via --config)")
        return 1

    # Run
    orchestrator = ResearchOrchestrator(config)
    result = orchestrator.run()

    # Exit code
    if result["status"] == "completed":
        return 0
    elif result["status"] == "interrupted":
        return 130  # SIGINT convention
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
