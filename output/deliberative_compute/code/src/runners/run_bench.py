
from __future__ import annotations
import argparse, os, csv, time
import numpy as np
from typing import Dict, Any

from ..algorithms.igd import IGD
from ..algorithms.rs_mctt import RS_MCTT
from ..algorithms.baselines import CoT, SelfConsistency, ToT_BFS
from ..algorithms.extras.dpg import DPG

from ..evaluation.compute_accounting import ComputeAccountant
from ..synthetic.env import make_synthetic_instance  # used as a fallback environment

from ..datasets.gsm8k import load_gsm8k_jsonl, simple_math_verifier
from ..datasets.humaneval import load_humaneval_jsonl, humaneval_verify
from ..datasets.math_bench import load_math_jsonl
from ..models.drivers import OpenAIChatDriver, EchoDriver

def get_algo(name: str):
    name = name.lower()
    if name == 'igd': return IGD()
    if name == 'rs_mctt': return RS_MCTT()
    if name == 'cot': return CoT()
    if name == 'self_consistency': return SelfConsistency()
    if name == 'tot_bfs': return ToT_BFS()
    if name == 'dpg': return DPG()
    raise ValueError(f"unknown algo: {name}")

def get_driver(name: str):
    name = name.lower()
    if name == 'openai': return OpenAIChatDriver()
    if name == 'echo': return EchoDriver()
    raise ValueError(f"unknown driver: {name}")

def bench_dataset(args):
    # Minimal prompt templates
    if args.dataset == 'gsm8k':
        data = load_gsm8k_jsonl(args.dataset_path, args.max_items)
        prompt_tpl = """Solve the math word problem carefully. Show reasoning, then give the final answer after 'Final answer:'.\n\nProblem: {question}\n"""
        verifier = simple_math_verifier
    elif args.dataset == 'humaneval':
        data = load_humaneval_jsonl(args.dataset_path, args.max_items)
        prompt_tpl = """Write a Python function as specified.\n\n{prompt}\n"""
        verifier = humaneval_verify
    elif args.dataset == 'math':
        data = load_math_jsonl(args.dataset_path, args.max_items)
        prompt_tpl = """Solve the math problem. Show reasoning, then final answer after 'Final answer:'.\n\n{problem}\n"""
        verifier = simple_math_verifier
    else:
        raise ValueError("unknown dataset: " + args.dataset)

    driver = get_driver(args.model_driver)
    algo = get_algo(args.algo)
    rng = np.random.default_rng(123)

    out_rows = []
    os.makedirs(args.out_dir, exist_ok=True)

    # For simplicity in this CLI, we treat the LLM calls as part of 'generation' budget; the scheduler overhead is measured by the accountant.
    for i, item in enumerate(data):
        # A tiny synthetic env is used to abstract compute choices (which thread to extend);
        # in a real system, 'step_extend_thread' would call the model with different thoughts/prompts.
        env = make_synthetic_instance(args.threads, rng)
        acc = ComputeAccountant()
        res = algo.run(env, args.budget, rng, acc)

        # Single model call to get a text (placeholder). A production integration would allocate budget tokens per step.
        prompt = prompt_tpl.format(**item)
        r = driver.generate(prompt, max_tokens=min(256, args.budget))
        text = r.get('text','')
        correct = bool(verifier(text))  # weak/pass-fail check

        row = {
            'i': i,
            'algo': args.algo,
            'dataset': args.dataset,
            'budget': args.budget,
            'utility_synth': float(res.utility),  # synthetic proxy utility from scheduler decisions
            'correct': int(correct),
            **{f'cc_{k}': v for k, v in acc.snapshot().items()},
        }
        out_rows.append(row)

    # write
    out_path = os.path.join(args.out_dir, f'{args.dataset}_{args.algo}_summary.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        import csv
        w = csv.DictWriter(f, fieldnames=sorted(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print("Wrote:", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True, choices=['gsm8k','humaneval','math'])
    ap.add_argument('--dataset_path', type=str, required=True)
    ap.add_argument('--algo', type=str, default='igd')
    ap.add_argument('--budget', type=int, default=128)
    ap.add_argument('--model_driver', type=str, default='echo', choices=['echo','openai'])
    ap.add_argument('--out_dir', type=str, default='results/realworld')
    ap.add_argument('--max_items', type=int, default=100)
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    bench_dataset(args)

if __name__ == '__main__':
    main()
