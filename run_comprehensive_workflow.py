#!/usr/bin/env python3
"""Command-line runner for the comprehensive research workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core import ComprehensiveResearchWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the comprehensive research workflow to scaffold a full paper pipeline.",
    )
    parser.add_argument("topic", help="Working research topic or title")
    parser.add_argument("field", help="Scientific field or discipline")
    parser.add_argument(
        "objective",
        help="Primary research objective, question, or contribution to pursue",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_description",
        help="Optional dataset or experimental context description",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where workflow artifacts should be written",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist workflow outputs to disk",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the workflow plan as JSON to stdout",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    workflow = ComprehensiveResearchWorkflow()
    result = workflow.run(
        topic=args.topic,
        field=args.field,
        objective=args.objective,
        dataset_description=args.dataset_description,
        output_dir=args.output_dir,
        save_outputs=not args.no_save,
    )

    if args.print_json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    if workflow.last_output_dir:
        print(f"Workflow artifacts saved to {workflow.last_output_dir}")


if __name__ == "__main__":
    main()
