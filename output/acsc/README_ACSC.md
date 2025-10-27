
# ACSC: Anytime Conformal Self-Consistency (Low-Compute Decoding)

This package contains:
- `simulation.py` — reference implementation + CLI
- `paper.tex` — LaTeX draft (figures and table already populated)
- `results.csv`, `results.json` — default run outputs
- `samples_used_hist.png` — figure referenced by the paper

## Quick start
```bash
python3 simulation.py --alpha 0.1 --Kmax 16 --n_total 1500 --n_calib 300 --C 8 --seed 42 --gamma 1.2 --noise 0.6
pdflatex paper.tex
```

## Default results in this bundle
- Coverage (ACSC): 0.723 (target 0.9)
- Avg. set size: 1.00
- Avg. samples used: 1.00 (median 1.0)
- Fixed-K majority accuracy (K=16): 0.948

## What to change
- Try different `--alpha` or `--Kmax`
- Replace the synthetic solver with real model calls; keep the same interface and calibration code
