#!/usr/bin/env python3
"""Lightweight offline simulation for AI-Scientist demo runs."""
import json
from pathlib import Path


def generate_curves():
    steps = list(range(0, 11))
    baseline = [round(0.62 + 0.018 * s, 3) for s in steps]
    demo = [round(min(0.94, 0.60 + 0.024 * s + 0.0015 * s * s), 3) for s in steps]
    return {
        "steps": steps,
        "baseline_accuracy": baseline,
        "demo_accuracy": demo,
    }


def main():
    results = generate_curves()
    output_dir = Path("simulation_outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
