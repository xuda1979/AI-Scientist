# ACSC: Anytime Conformal Self-Consistency (Low-Compute Decoding)

This package contains:
- `simulation.py` — reference implementation + CLI (now with grid/ablation support)
- `paper.tex` — LaTeX draft (figures and tables already populated)
- `results.csv`, `results.json` — default run outputs
- `samples_used_hist.png` — figure referenced by the paper
- `grid_results.csv`, `grid_results.json` — saved when running grid ablations
- `ablation_errorbars.png` — optional ablation error-bar plot

## Quick start

Single configuration (multi-seed aggregation):
bash
python3 simulation.py --alpha 0.1 --Kmax 16 --n_total 1500 --n_calib 300 --C 8 \
  --seed 42 --gamma 1.2 --noise 0.6 --n_seeds 5 --method acsc_ub --m_stop 1

Grid across (alpha, Kmax) with 5 seeds and ablation plot:
bash
python3 simulation.py --grid_alphas 0.05,0.1,0.2 --grid_kmax 8,16,32 --n_seeds 5 \
  --method acsc_ub --m_stop 1 --make_ablation_plot 1 --out_dir out_acsc

Compile the paper:
bash
pdflatex paper.tex

## Notes
- Reproducibility: Deterministic seeds for Python/NumPy; method RNGs seeded per seed.
- Significance: We output z-tests vs target coverage, McNemar paired tests, and Benjamini–Hochberg adjusted q-values across multiple hypotheses for grid runs.
- Real datasets: Use `--dataset digits|iris|wine` (requires scikit-learn). We train a logistic regression with CV and sample labels from its predictive distribution for i.i.d. draws, preserving the split-conformal framework.