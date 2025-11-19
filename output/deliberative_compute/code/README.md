# Beyond CoT — Empirical Experiments

This repository provides a **reproducible experimental suite** for the paper *"Beyond Chain-of-Thought: Test-Time Compute Scaling for Deliberative Large Language Models"* (code for synthetic + real-world harness). It addresses reviewer feedback by:

- Adding **fully reproducible synthetic experiments** (BPF, MVC, ablations) with matched budgets.
- Providing a **real-world benchmark harness** for GSM8K / MATH-style math and HumanEval-style code tasks (bring your own model and API key).
- Implementing **instrumentation** to measure the **metareasoning overhead** (scheduler and control flow) separately from model generation cost.
- Extending evaluation beyond two algorithms with **baseline** methods (CoT, Self-Consistency, ToT-BFS) and **our core algorithms** (IGD, RS-MCTT). Stubs (and light-weight variants) are included for ADR, APT, CSC, PSV, and DPG to encourage broader empirical coverage.

> **Citation target** in the manuscript: Synthetic protocol, BPF/MVC estimation, and algorithms: IGD & RS-MCTT. See paper for details.

---

## Quick start (Synthetic)

```bash
# 1) Create/activate a Python 3.10+ env
python -V  # 3.10+
python -m venv .venv && source .venv/bin/activate

# 2) Install requirements (numpy, pandas, matplotlib only)
pip install -r requirements.txt

# 3) Run the synthetic benchmark (50 evaluation seeds by default)
python -m src.synthetic.run_synthetic --out_dir results/synth_default

# (Optional) Smaller run for a smoke test
python -m src.synthetic.run_synthetic --out_dir results/synth_smoke --n_eval_seeds 10

# 4) Aggregate & plot
python -m src.evaluation.aggregate --in_dir results/synth_default --out_dir results/synth_default/agg

# Outputs:
# - TSV summaries per method and per budget
# - BPF and MVC plots (PNG)
# - effect sizes, Welch tests (BH-FDR, when SciPy available; falls back to normal approx otherwise)
```

## Real-world harness (GSM8K / MATH / HumanEval)

The harness is **model-agnostic**. Plug in an LLM driver (OpenAI, local HF, or custom). Bring datasets locally.

```bash
# Example (GSM8K), requires an LLM driver and dataset JSON files.
# See src/datasets/gsm8k.py for format hints.
python -m src.runners.run_bench \
  --dataset gsm8k --dataset_path /path/to/gsm8k.jsonl \
  --algo igd --budget 128 --model_driver openai --max_items 100

# HumanEval-style:
python -m src.runners.run_bench \
  --dataset humaneval --dataset_path /path/to/HumanEval.jsonl \
  --algo rs_mctt --budget 1024 --model_driver openai --max_items 50
```

- **Overhead instrumentation** is always on: wall-clock for scheduling, counts of decisions, and compute accounting for generation/verification tokens.
- To reproduce paper plots on real-world data, run `src/evaluation/aggregate.py` on the output directory.

> **Security note:** The repository never sends data unless your selected driver does. All keys are read from environment variables (e.g., `OPENAI_API_KEY`) and are *not* logged.

---

## Repository map

```
src/
  algorithms/
    base.py                 # Interfaces + utilities
    baselines.py            # CoT, Self-Consistency, ToT-BFS
    igd.py                  # Index-Guided Deliberation (ours)
    rs_mctt.py              # Risk-Sensitive Monte Carlo ToT (ours)
    extras/
      adr.py                # Abduction–Deduction–Refutation (light impl for synth)
      apt.py                # Annealed Population of Thoughts (light impl)
      csc.py                # Counterfactual Self-Consistency (filters chains)
      psv.py                # Probabilistic Self-Verification (checks + stopping)
      dpg.py                # Dual-Process Gating (fast-path -> deliberation)
  evaluation/
    aggregate.py            # TSVs, FA, MVC, tests, effect sizes, FDR
    stats.py                # Welch test, Hedges' g, BH-FDR
    plotting.py             # Matplotlib plots (no seaborn)
    compute_accounting.py   # Unified compute accounting + overhead timers
  synthetic/
    env.py                  # Synthetic environment (threads + tree generator)
    run_synthetic.py        # End-to-end synthetic runs w/ matched budgets
  datasets/
    gsm8k.py                # Reader + simple verifier hooks
    humaneval.py            # Reader + unit test executor
    math_bench.py           # Reader stub for MATH-like problems
  runners/
    run_bench.py            # Real-world harness (dataset + algorithm + model driver)
  models/
    drivers.py              # OpenAI + stub drivers (pluggable)
requirements.txt
```

---

## Notes on design choices

- **Matched budgets** across all methods ensure fairness. Budgets are measured in *micro-actions* (e.g., token-equivalent cost units) configurable via `compute_accounting`.
- **Overhead reporting** separates pure scheduling time from model I/O, as requested by the review.
- **Reproducibility**: Deterministic synthetic generator with fixed evaluation seeds; pilot seeds are disjoint (configurable).
- **Ablations** implement toggles for IGD risk-LCB and discount, and RS-MCTT depth-wise vs constant risk schedules.

---

## License

MIT License.
