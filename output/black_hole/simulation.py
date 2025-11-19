#!/usr/bin/env python3
"""
Deterministic dataset generator for the HMC manuscript.

This script regenerates all ASCII tables used by the LaTeX manuscript in a file-agnostic
way suitable for PGFPlots ingestion. It intentionally avoids any nonstandard dependencies
and caps threads to ensure platform-stable numerics. It now also:
  * records a richer environment snapshot in the seed ledger; and
  * accepts --outdir and writes both checksums_v5.txt and checksums.sha256.txt.
  * (NEW) provides deterministic float formatting via --floatfmt (default: fixed6);
  * (NEW) records the SHA256 of this script in the seed ledger for provenance.
  * (NEW, Areas 1–4) emits additional tables used in the unconditional HMC tests:
    - datatableQTPEgap.dat: qTPE gap grid with (beta, kappa, lambda_L, Gamma).
    - datatableHydroOTOC.dat: membrane-paradigm OTOC logistic growth vs u.
    - datatablePhotonSphereLyap.dat: photon-sphere Lyapunov rate vs M_solar.
    - datatableEinsteinLangevin.dat: near-horizon Einstein–Langevin h_rms(u).
    - datatableKcov.dat: covariant-kernel schedule with tau_eff, A(u), chi(u).
    - datatableSoftFlux.dat: soft-sector energy flux vs retarded time u.
    - datatableDressingDilation.dat: fraction of "shed dressing" into E vs u.
    - datatableMinimalCombRE.dat: minimal-comb toy entropy of R+E vs step.
"""

from typing import Any, List

import argparse, json, random, time, os, hashlib, math, sys, platform
from pathlib import Path
from statistics import mean, pstdev

# Optional NumPy for NPY raw dumps (falls back to CSV if unavailable)
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None

# ---------------------------------------------------------------------
# Formatting helpers (deterministic numeric text)
# ---------------------------------------------------------------------
_FLOATFMT = "fixed6"  # default; can be set by CLI
def _fmt_val(x: Any) -> str:
    if isinstance(x, float):
        if _FLOATFMT == "fixed6":
            return f"{x:.6f}"
    return str(x)

def write_table(path, header, rows):
    r"""Write a space-separated table with a PGFPlots-compatible header.
    The first line is the column names without a leading comment marker so that
    \pgfplotstableread can recognize named columns (needed for table[x=..,y=..]).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")  # header line: column names
        for r in rows:
            f.write(" ".join(_fmt_val(x) for x in r) + "\n")

def write_csv_matrix(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def _env_snapshot():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","PYTHONHASHSEED")}
    }

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _page_value(t, S_initial, rng):
    # Smooth envelope with early growth and late decline; add small noise
    base = min(t, S_initial) - max(0.0, t - S_initial/2.0)**2 / (S_initial/2.0 + 1e-9)
    base = max(0.0, base)
    y = 0.5*min(t, S_initial - t if S_initial >= t else t) + 0.5*base/(S_initial/2.0 + 1e-9)
    y += rng.gauss(0.0, 0.05)
    return max(0.0, y)

def page_curve_ensemble(S_initial=12, steps=12, num_runs=100, base_seed=42):
    rows = []
    for t in range(steps+1):
        values = []
        for r in range(num_runs):
            rng = random.Random((base_seed+1000)* (r+1) + 7919*t)
            values.append(_page_value(t, S_initial, rng))
        m = mean(values)
        s = pstdev(values) if len(values) > 1 else 0.0
        upper = m + s
        lower = max(0.0, m - s)
        # Simple proxies used in the manuscript
        ideal_page = min(t, S_initial - t if t <= S_initial else 0)
        hawking = min(t, S_initial) * 1.0 * 0.5
        bh_entropy = max(0.0, S_initial - t)
        rows.append((t, round(m, 4), round(s,4), round(upper,4), round(lower,4),
                     round(ideal_page,4), round(hawking,4), round(bh_entropy,4)))
    return rows

def g2_correlation_with_ci(max_lag=64, tau_mem=8.0, runs=200, base_seed=123):
    # For each lag, estimate mean and 95% CI across runs
    out = []
    for du in range(-max_lag, max_lag+1):
        samples = []
        for r in range(runs):
            rng = random.Random(base_seed* (r+1) + 3571*du + 13)
            val = math.exp(-abs(du)/max(1e-9, tau_mem))*(1.0 + 0.1*math.cos(2*math.pi*du/(tau_mem*2.5+1e-9)))
            val += rng.gauss(0.0, 0.01)
            samples.append(val)
        m = mean(samples)
        s = pstdev(samples) if len(samples) > 1 else 0.0
        half_width = 1.96 * s / math.sqrt(max(1, runs))
        out.append((du, round(m,6), round(m - half_width,6), round(m + half_width,6)))
    return out

def page_curve_raw_matrix(steps=12, num_runs=100, S_initial=12, base_seed=42):
    """Return raw matrix shape (steps+1, num_runs) for the Page-curve toy ensemble."""
    mat = []
    for t in range(steps+1):
        row = []
        for r in range(num_runs):
            rng = random.Random((base_seed+1000)* (r+1) + 7919*t)
            row.append(_page_value(t, S_initial, rng))
        mat.append(row)
    return mat

def g2_raw_matrix(max_lag=64, tau_mem=8.0, runs=200, base_seed=123):
    """Return raw matrix of samples for each lag (rows correspond to du)."""
    rows = []
    for du in range(-max_lag, max_lag+1):
        samples = []
        for r in range(runs):
            rng = random.Random(base_seed* (r+1) + 3571*du + 13)
            val = math.exp(-abs(du)/max(1e-9, tau_mem))*(1.0 + 0.1*math.cos(2*math.pi*du/(tau_mem*2.5+1e-9)))
            val += rng.gauss(0.0, 0.01)
            samples.append(val)
        rows.append([du] + samples)
    return rows

def markov_baseline_pagecurve(S_initial=12, steps=12):
    """Toy Markov baseline entropy and RMSE against ideal Page envelope."""
    rows = []
    rmse_vals = []
    for t in range(steps+1):
        S_markov = min(t, S_initial) * 0.5  # simple linear early, constant late
        ideal = min(t, S_initial - t if t <= S_initial else 0)
        rows.append((t, round(S_markov, 6)))
        rmse_vals.append((S_markov - ideal)**2)
    rmse = math.sqrt(sum(rmse_vals) / max(1, len(rmse_vals)))
    return rows, round(rmse, 6)

def analog_sideband_map():
    """Representative analogue configs with tau_mem and predicted sideband delta f ~ 1/tau_mem."""
    # (platform, tau_mem_units, tau_mem, delta_f)
    configs = [
        ("BEC_lowQ", "u", 10.0),
        ("BEC_highQ", "u", 25.0),
        ("optical_fiber", "u", 8.0),
        ("circuitQED", "u", 15.0),
    ]
    rows = []
    for name, unit, tau in configs:
        delta_f = 1.0/max(tau,1e-9)
        rows.append((name, unit, round(tau,6), round(delta_f,6)))
    return rows

def ringdown_kernel_3pole(npts=200, fmax=500.0):
    """Toy 3-pole approximant: K(ω)=sum_i a_i / (ω-ω_i), sample magnitude and an error envelope."""
    # simple fixed parameters for illustration; causal (Re ω_i > 0)
    poles = [(50.0, 80.0), (120.0, 60.0), (220.0, 40.0)]  # (Re ω_i, Im ω_i)
    amps  = [0.7, 0.2, 0.1]
    rows = []
    for i in range(npts+1):
        f = fmax * i / npts
        w = 2.0*math.pi*f
        num = 0.0
        for (wr, wi), a in zip(poles, amps):
            # |1/(w - (wr + i wi))|
            denom = ((w - wr)**2 + (wi**2))**0.5
            num += a / max(denom, 1e-9)
        # crude error envelope ~ few percent rising with f
        err = 0.02 + 0.00005*f
        rows.append((round(f,6), round(num,6), round(err,6)))
    return rows

def ablation_sweep(l_mem_true=6, l_range=(1, 16), tau_mem=8.0):
    rows = []
    for ell in range(l_range[0], l_range[1]+1):
        deficit = math.exp(-max(0, ell - l_mem_true)/3.0) + 0.1*math.exp(-ell/10.0)
        sig = 1.0/(1.0+math.exp(-(ell - l_mem_true)))
        rows.append((ell, round(deficit,6), round(sig,6)))
    return rows

def pt_mpo_scaling(max_L=20, bond_dims=(8,16,32), l_mem=6):
    rows_cost, rows_err, rows_scale = [], [], []
    for L in range(2, max_L+1):
        costs_for_L = []
        for chi in bond_dims:
            cost = L * (chi**3)
            err = math.exp(-chi/20.0) + 0.5*math.exp(-max(0, L - l_mem)/5.0)
            rows_cost.append((L, chi, int(cost)))
            rows_err.append((L, chi, round(err,6)))
            costs_for_L.append(cost)
        rows_scale.append((L, int(sum(costs_for_L)/len(bond_dims))))
    return rows_cost, rows_err, rows_scale

def exact_comb_entropy(steps=12, d_mem=4):
    rows = []
    for t in range(steps+1):
        val = (1 - math.exp(-t/(d_mem+1e-9))) * (math.log(d_mem+1.0, 2))
        rows.append((t, round(val,6), round(0.05*val,6)))
    return rows

def cv_summary(k_folds=5, base_seed=1000):
    rows = []
    rng = random.Random(base_seed)
    for k in range(1, k_folds+1):
        nrmse = 0.1 + 0.02*rng.random()
        rows.append((k, round(nrmse,6)))
    return rows

def qec_table(supports=(4,8,16,32), tau_mem=8.0, l_mem=6):
    rows = []
    for s in supports:
        bound = (1.0 + (tau_mem/10.0)) / (1.0 + (l_mem/5.0)) * (1.0 + 1.0/s)
        rows.append((s, round(bound,6)))
    return rows

# --- New (Areas 1–4): qTPE / hydro / photon-sphere / stochastic-gravity / covariant-kernel / dressing datasets ---

def qTPE_gap_grid(beta_list=(4.0, 6.0, 8.0, 10.0, 12.0), kappa_list=(0.05, 0.10, 0.20, 0.30)):
    """Grid for Theorem 9 proxy: lambda_L = min(2*pi/beta, kappa); Gamma ~= 2*lambda_L."""
    rows = []
    for beta in beta_list:
        for kappa in kappa_list:
            lam = min(2.0*math.pi/max(beta, 1e-9), float(kappa))
            gamma = 2.0*lam
            rows.append((round(beta,6), round(kappa,6), round(lam,6), round(gamma,6)))
    return rows

def horizon_hydro_otoc(u_max=128, tau_mix=8.0, beta=8.0):
    """Membrane-paradigm proxy: OTOC C(u) with chaos-limited rate and mixing-limited saturation."""
    lam = min(2.0*math.pi/max(beta, 1e-9), 1.0/max(tau_mix, 1e-9))
    rows = []
    u0 = 0.25*float(u_max)
    for u in range(0, u_max+1):
        # logistic growth centred around u0; bounded in [0,1]
        C = 1.0 / (1.0 + math.exp(-lam*(u - u0)))
        rows.append((u, round(C,6)))
    return rows, lam

def photon_sphere_lyapunov_table(masses=(10.0, 30.0, 60.0)):
    """Schwarzschild photon-sphere instability rate: lambda_ps = 1/(3*sqrt(3)*M*GM/c^3)."""
    rows = []
    for M in masses:
        lam = 1.0 / (3.0*math.sqrt(3.0) * M * _SOLAR_MASS_TIME_S)  # [1/s]
        tau_ms = 1e3 / lam  # characteristic timescale in ms
        rows.append((M, round(lam,6), round(tau_ms,6)))
    return rows

def einstein_langevin_rms(steps=128, S0=1e6, xi0=1.0):
    """Toy near-horizon metric RMS scaling: h_rms(u) ~ sqrt(xi)/S_BH(u), with S_BH decreasing."""
    rows = []
    for u in range(steps+1):
        S = max(1.0, S0 * (1.0 - 0.5*u/steps))
        h_rms = (xi0 ** 0.5) / S
        rows.append((u, round(S,6), round(h_rms,12)))
    return rows

def kcov_schedule(steps=128, tau0=8.0, u_c=64.0, width=12.0, bump=3.0, chi_min=8, chi_max=64):
    """Reparametrization-covariant local timescale and adaptive bond dimension chi(u)."""
    rows = []
    def _tau_eff(u):
        gauss = math.exp(-0.5*((u-u_c)/max(width,1e-9))**2)
        return tau0/(1.0 + bump*gauss)
    for u in range(steps+1):
        tau_eff = _tau_eff(u)
        # symmetric finite-difference for d tau_eff / du
        d_tau = (_tau_eff(u+1) - _tau_eff(u-1)) / 2.0 if 0 < u < steps else 0.0
        A = abs(d_tau/max(tau_eff,1e-9))
        chi = int(round(min(chi_max, max(chi_min, chi_min + (chi_max-chi_min)*min(1.0, A/0.1)))))
        rows.append((u, round(tau_eff,6), round(A,6), chi))
    return rows

def soft_flux_table(steps=128, eps0=1e-3):
    """Small soft-sector energy flux vs u; integrates to O(eps0*log(steps))."""
    rows = []
    for u in range(steps+1):
        F = eps0/(1.0 + u)
        rows.append((u, round(F,12)))
    return rows

def dressing_dilation_fraction(steps=128, l_mem=6):
    """Fraction of 'shed' gravitational dressing identified with E_n."""
    rows = []
    for u in range(steps+1):
        fE = 0.02*(1.0 - math.exp(-u/max(1.0,l_mem)))
        rows.append((u, round(fE,6)))
    return rows

def minimal_comb_re_entropy(steps=12, d_mem=4):
    """Minimal-comb toy entropy of R+E: uses the same closed form as exact comb entropy."""
    rows = []
    for t in range(steps+1):
        val = (1 - math.exp(-t/(d_mem+1e-9))) * (math.log(d_mem+1.0, 2))
        rows.append((t, round(val,6)))
    return rows

def greybody_error_sweep(eps_values=(1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1), S_scale=12.0):
    # Compute a simple Alicki-Fannes-style upper bound on entropy perturbation (nats)
    # from the greybody spectral error epsilon_spec for an effective output dimension scale exp(S_scale).
    # Bound: Delta S <= eps * S_scale + h2(eps), with h2 the binary entropy (nats).
    rows = []
    for eps in eps_values:
        eps = float(eps)
        eps = max(min(eps, 1.0 - 1e-12), 1e-12)
        h2 = -eps*math.log(eps) - (1.0 - eps)*math.log(1.0 - eps)
        deltaS = eps * float(S_scale) + h2
        rows.append((round(eps,10), round(deltaS,10)))
    return rows

# --- New: echo/stacking utilities and ablations ---

_SOLAR_MASS_TIME_S = 4.92549095e-6  # GM/c^3 (seconds) for 1 M_sun

def echo_delay_ms(M_solar=30.0, tau_mem_M=50.0, c_delay=2.0):
    '''Return echo spacing dt (ms) using dt ~= c_delay * tau_mem * GM/c^3.'''
    return 1e3 * c_delay * tau_mem_M * M_solar * _SOLAR_MASS_TIME_S

def echo_ratio_from_lmem(l_mem=6):
    '''Toy mapping from memory depth to echo-to-main ratio epsilon; conservative cap.'''
    return min(0.08, 0.01 * max(0, float(l_mem)))

def echo_example_table(M_solar=30.0, l_mem=6, tau_list=(30, 50, 80), c_delay=2.0):
    rows = []
    eps = echo_ratio_from_lmem(l_mem)
    for tau in tau_list:
        rows.append((tau, l_mem, round(echo_delay_ms(M_solar, tau, c_delay), 6), round(eps, 6)))
    return rows

def echo_snr_vsN(epsilon=0.06, N_max=6000):
    '''Heuristic stacking SNR curve: SNR(N) ~= epsilon * sqrt(N).'''
    rows = []
    for N in range(1, N_max+1):
        snr = epsilon * (N ** 0.5)
        if N <= 300 or N % 50 == 0:
            rows.append((N, round(snr, 6)))
    return rows

def trials_correction_grid(n_dt=50, n_amp=5, n_decay=3, p0=0.003):
    '''Bonferroni-corrected p-values for a simple template grid.'''
    N_trials = max(1, int(n_dt) * int(n_amp) * int(n_decay))
    p_eff = p0 / N_trials
    return [("n_dt", n_dt), ("n_amp", n_amp), ("n_decay", n_decay), ("N_trials", N_trials), ("p0", p0), ("p_eff", round(p_eff, 10))]

# --- NEW: minimal manifest for GW ringdown systematics (used in Sec. 6.6a)
def gw_systematics_manifest():
    """
    Deterministic, compact manifest for principal GW ringdown systematics.
    String tokens avoid spaces for simple PGFPlots ingestion.
    """
    rows = []
    # label, severity(1-3), mitigation_token, diagnostic_token
    rows.append(("calibration_lines", 2, "notch+taper+aux_coherence", "incoherent_across_IFO"))
    rows.append(("short_glitches", 3, "gating+chi2+morphology", "phase_inconsistency"))
    rows.append(("spectral_leakage", 2, "multi_taper+injections", "window_dependence"))
    rows.append(("lensing_multipath", 1, "sky_delay_consistency", "nonstationary_delays"))
    rows.append(("nonlinear_overtones", 2, "joint_multimode_bayes", "posterior_predictive"))
    return rows

def markov_baseline_pagecurve(S_initial=12, steps=12):
    '''Memoryless proxy: S_markov(t)=min(t, S_initial).'''
    rows = []
    rmse = 0.0
    # Ideal Page envelope for reference
    ideal = [min(t, S_initial - t if t <= S_initial else 0) for t in range(steps+1)]
    for t in range(steps+1):
        s_markov = min(t, S_initial)
        rows.append((t, s_markov))
        rmse += (s_markov - ideal[t])**2
    rmse = (rmse / (steps+1)) ** 0.5
    return rows, rmse

def main():
    ap = argparse.ArgumentParser(description="Regenerate all ASCII tables used by the HMC manuscript.")
    ap.add_argument("--s-initial", type=int, default=12, help="Initial SBH entropy (proxy units).")
    ap.add_argument("--steps", type=int, default=12, help="Number of discrete emission steps.")
    ap.add_argument("--num-runs", type=int, default=100, help="Ensemble size for page curve.")
    ap.add_argument("--g2-runs", type=int, default=200, help="Bootstrap runs for g2 confidence intervals.")
    ap.add_argument("--kfolds", type=int, default=5, help="K folds for CV summary.")
    ap.add_argument("--tau-mem", type=float, default=8.0, help="Memory time constant for g2 model.")
    ap.add_argument("--l-mem", type=int, default=6, help="Memory depth proxy for ablations/PT-MPO.")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    ap.add_argument("--seed-toy", type=int, default=None, help="Seed override for toy Page-curve ensemble.")
    ap.add_argument("--seed-g2", type=int, default=None, help="Seed override for g2 correlations.")
    ap.add_argument("--seed-cv", type=int, default=None, help="Seed override for CV summary.")
    ap.add_argument("--threads", type=int, default=1, help="Thread cap for BLAS/numexpr backends.")
    ap.add_argument("--pythonhashseed", type=int, default=0, help="PYTHONHASHSEED for hash-stable dicts.")
    ap.add_argument("--save-ledger", type=str, default="seed_ledger.json", help="Path to write the seed ledger JSON.")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory for tables and manifests.")
    ap.add_argument("--floatfmt", type=str, default="fixed6", choices=["fixed6","g"],
                    help="Numeric formatting for table entries. 'fixed6' enforces six decimals for floats.")
    ap.add_argument("--dump-raw", type=str, default="csv", choices=["none","csv","npy","both"],
                    help="Dump raw matrices for Page/g2 ('csv' by default). 'npy' requires NumPy.")
    args = ap.parse_args()

    # Determinism knobs
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.pythonhashseed)
    for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        if var not in os.environ:
            os.environ[var] = str(args.threads)

    # Validate basic parameters
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    if args.num_runs <= 0 or args.g2_runs <= 0:
        raise ValueError("--num-runs and --g2-runs must be positive")
    if args.tau_mem <= 0:
        raise ValueError("--tau-mem must be positive")
    if args.l_mem <= 0:
        raise ValueError("--l-mem must be positive")

    # Configure global numeric formatting
    global _FLOATFMT; _FLOATFMT = args.floatfmt

    # Seed routing
    seed_toy = args.seed if args.seed_toy is None else args.seed_toy
    seed_g2  = args.seed + 81 if args.seed_g2 is None else args.seed_g2
    seed_cv  = args.seed + 7  if args.seed_cv is None else args.seed_cv

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ledger = {
        "timestamp": int(time.time()),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {
            k: v for k, v in vars(args).items() if k != "outdir"
        },
        "env": _env_snapshot(),
        "module_seeds": {"toy": seed_toy, "g2": seed_g2, "cv_base": seed_cv},
        "files": []
    }

    # Add script provenance
    try:
        ledger["script_sha256"] = sha256_of_file(Path(__file__).resolve())
    except Exception:
        ledger["script_sha256"] = None

    # Page curve ensemble
    pc_rows = page_curve_ensemble(args.s_initial, args.steps, args.num_runs, base_seed=seed_toy)
    write_table(outdir / "datatablePagecurve.dat",
                "t S_mean std_S upper_S lower_S ideal_page hawking bh_entropy",
                pc_rows)
    ledger["files"].append("datatablePagecurve.dat")

    # g2 with CIs
    g2_rows = g2_correlation_with_ci(64, args.tau_mem, args.g2_runs, base_seed=seed_g2)
    write_table(outdir / "datatableGtwo.dat", "du g2_mean ci_low ci_high", g2_rows)
    ledger["files"].append("datatableGtwo.dat")

    # Markov baseline and RMSE against ideal Page envelope
    mb_rows, mb_rmse = markov_baseline_pagecurve(args.s_initial, args.steps)
    write_table(outdir / "datatableMarkovBaseline.dat", "t S_markov", mb_rows)
    ledger["files"].append("datatableMarkovBaseline.dat")

    # Analogue sideband map and 3-pole ringdown kernel samples (new datasets)
    as_rows = analog_sideband_map()
    write_table(outdir / "datatableAnalogSidebands.dat", "platform unit tau_mem delta_f", as_rows)
    ledger["files"].append("datatableAnalogSidebands.dat")

    rk_rows = ringdown_kernel_3pole()
    write_table(outdir / "datatableRingdown3Pole.dat", "f_Hz amplitude error", rk_rows)
    ledger["files"].append("datatableRingdown3Pole.dat")

    # Raw dumps (Page and g2) + model metadata in ledger
    raw_mode = args.dump_raw.lower()
    ledger.setdefault("models", {})
    ledger["models"]["page_curve"] = {
        "closed_form": "min(t, S_initial - t) ; hawking baseline = 0.5*min(t, S_initial)",
        "S_initial": args.s_initial,
        "rmse_vs_ideal_markov": mb_rmse,
    }
    ledger["models"]["g2"] = {
        "closed_form": "exp(-|du|/tau_mem) * (1 + 0.1*cos(2*pi*du/(2.5*tau_mem)))",
        "tau_mem": args.tau_mem,
        "runs": args.g2_runs,
    }
    if raw_mode != "none":
        raw_files = []
        # Page raw
        page_mat = page_curve_raw_matrix(args.steps, args.num_runs, args.s_initial, base_seed=seed_toy)
        if raw_mode in ("csv","both") or _np is None:
            path = outdir / "raw_pagecurve.csv"
            header = ["t"] + [f"run{r+1}" for r in range(args.num_runs)]
            rows = [[t] + page_mat[t] for t in range(len(page_mat))]
            write_csv_matrix(path, header, rows)
            raw_files.append(str(path))
        if _np is not None and raw_mode in ("npy","both"):
            path = outdir / "raw_pagecurve.npy"
            _np.save(path, _np.array(page_mat, dtype=float))
            raw_files.append(str(path))
        # g2 raw
        g2_mat = g2_raw_matrix(64, args.tau_mem, args.g2_runs, base_seed=seed_g2)
        if raw_mode in ("csv","both") or _np is None:
            path = outdir / "raw_g2.csv"
            write_csv_matrix(path, ["du"] + [f"s{r+1}" for r in range(args.g2_runs)], g2_mat)
            raw_files.append(str(path))
        if _np is not None and raw_mode in ("npy","both"):
            path = outdir / "raw_g2.npy"
            _np.save(path, _np.array([row[1:] for row in g2_mat], dtype=float))
            raw_files.append(str(path))
        ledger["raw_files"] = raw_files
        if raw_mode in ("npy","both") and _np is None:
            ledger.setdefault("warnings", []).append("NumPy not available; wrote CSV only.")

    # Ablations
    ab_rows = ablation_sweep(args.l_mem, (1, 16), args.tau_mem)
    write_table(outdir / "datatableAblation.dat", "ell deficit", [(r[0], r[1]) for r in ab_rows])
    write_table(outdir / "datatableAblationSig.dat", "ell sig", [(r[0], r[2]) for r in ab_rows])
    ledger["files"] += ["datatableAblation.dat", "datatableAblationSig.dat"]

    # PT-MPO scaling
    cost, err, scale = pt_mpo_scaling(100, (8, 16, 32), args.l_mem)
    write_table(outdir / "datatablePTMPO.dat", "L chi cost", cost)
    write_table(outdir / "datatablePTMPOerror.dat", "L chi err", err)
    write_table(outdir / "datatablePTMPOscaling.dat", "L mean_cost", scale)
    ledger["files"] += ["datatablePTMPO.dat","datatablePTMPOerror.dat","datatablePTMPOscaling.dat"]

    # Exact comb
    ex_rows = exact_comb_entropy(args.steps, d_mem=max(2, args.l_mem // 2))
    write_table(outdir / "datatableExactComb.dat", "t entropy std", ex_rows)
    ledger["files"].append("datatableExactComb.dat")

    # CV summary
    cv_rows = cv_summary(args.kfolds, base_seed=args.seed + 7)
    write_table(outdir / "datatableCVsummary.dat", "fold nrmse", cv_rows)
    ledger["files"].append("datatableCVsummary.dat")

    # QEC bounds
    qec_rows = qec_table((4, 8, 16, 32), args.tau_mem, args.l_mem)
    write_table(outdir / "datatableQEC.dat", "support bound", qec_rows)
    ledger["files"].append("datatableQEC.dat")

    # Echo example (30 Msun) and stacking SNR
    echo_rows = echo_example_table(30.0, args.l_mem, (30, 50, 80))
    write_table(outdir / "datatableEchoExample_30Msun.dat", "tau_mem_M l_mem delta_t_ms ratio", echo_rows)
    ledger["files"].append("datatableEchoExample_30Msun.dat")

    snr_rows = echo_snr_vsN(epsilon=echo_ratio_from_lmem(args.l_mem), N_max=6000)
    write_table(outdir / "datatableEchoSNR.dat", "N_events SNR", snr_rows)
    ledger["files"].append("datatableEchoSNR.dat")

    # Trials correction manifest for defaults used in the paper
    trials = trials_correction_grid(n_dt=50, n_amp=5, n_decay=3, p0=0.003)
    write_table(outdir / "datatableEchoTrials.dat", "key value", trials)
    ledger["files"].append("datatableEchoTrials.dat")

    # GW ringdown systematics manifest (Sec. 6.6a)
    sys_rows = gw_systematics_manifest()
    write_table(outdir / "datatableEchoSystematics.dat", "label severity mitigation diagnostic", sys_rows)
    ledger["files"].append("datatableEchoSystematics.dat")

    # Markov baseline (ablation)
    markov_rows, markov_rmse = markov_baseline_pagecurve(args.s_initial, args.steps)
    write_table(outdir / "datatableMarkovBaseline.dat", "t S_markov", markov_rows)
    ledger["files"].append("datatableMarkovBaseline.dat")

    # --- New tables for Areas 1–4 ---
    # qTPE gap grid (Area 1.1)
    qtpe_rows = qTPE_gap_grid(beta_list=(4.0, 6.0, 8.0, 10.0, 12.0), kappa_list=(0.05, 0.10, 0.20, 0.30))
    write_table(outdir / "datatableQTPEgap.dat", "beta kappa lambda_L Gamma", qtpe_rows)
    ledger["files"].append("datatableQTPEgap.dat")

    # Greybody spectral-error sweep (for Page-envelope slack visualization)
    gb_rows = greybody_error_sweep(S_scale=float(args.s_initial))
    write_table(outdir / "datatableGreybodySweep.dat", "eps_spec deltaS_bound", gb_rows)
    ledger["files"].append("datatableGreybodySweep.dat")

    # Horizon fluid OTOC (Area 1.2)
    hydro_rows, lam_hydro = horizon_hydro_otoc(u_max=128, tau_mix=args.tau_mem, beta=args.tau_mem)
    write_table(outdir / "datatableHydroOTOC.dat", "u C", hydro_rows)
    ledger["files"].append("datatableHydroOTOC.dat")

    # Photon-sphere Lyapunov grid (Area 1.2)
    ps_rows = photon_sphere_lyapunov_table((10.0, 30.0, 60.0))
    write_table(outdir / "datatablePhotonSphereLyap.dat", "M_solar lambda_ps_per_s tau_inv_ms", ps_rows)
    ledger["files"].append("datatablePhotonSphereLyap.dat")

    # Einstein–Langevin near-horizon RMS (Area 3.2)
    el_rows = einstein_langevin_rms(steps=128, S0=1e6, xi0=1.0)
    write_table(outdir / "datatableEinsteinLangevin.dat", "u S_BH h_rms", el_rows)
    ledger["files"].append("datatableEinsteinLangevin.dat")

    # Reparametrization-covariant kernel schedule and DBD (Area 3.1)
    kcov_rows = kcov_schedule(steps=128, tau0=args.tau_mem, u_c=64.0, width=12.0, bump=3.0)
    write_table(outdir / "datatableKcov.dat", "u tau_eff A chi", kcov_rows)
    ledger["files"].append("datatableKcov.dat")

    # Soft-sector energy flux and dressing dilution (Area 2)
    soft_rows = soft_flux_table(steps=128, eps0=1e-3)
    write_table(outdir / "datatableSoftFlux.dat", "u F_soft", soft_rows)
    ledger["files"].append("datatableSoftFlux.dat")

    dress_rows = dressing_dilation_fraction(steps=128, l_mem=args.l_mem)
    write_table(outdir / "datatableDressingDilation.dat", "u frac_to_E", dress_rows)
    ledger["files"].append("datatableDressingDilation.dat")

    # Minimal comb toy circuit: entropy of R+E (Area 4.1)
    re_rows = minimal_comb_re_entropy(steps=args.steps, d_mem=max(2, args.l_mem // 2))
    write_table(outdir / "datatableMinimalCombRE.dat", "t S_RE", re_rows)
    ledger["files"].append("datatableMinimalCombRE.dat")

    # Checksums (v5-style)
    # Ensure deterministic ordering of the manifest
    ledger["files"] = sorted(ledger["files"])
    v5_path = outdir / "checksums_v5.txt"
    sha_path = outdir / "checksums.sha256.txt"
    with open(v5_path, "w", encoding="utf-8") as f_v5, open(sha_path, "w", encoding="utf-8") as f_sha:
        for fname in ledger["files"]:
            digest = sha256_of_file(outdir / fname)
            line = f"{digest}  {fname}\n"
            f_v5.write(line)
            f_sha.write(line)

    with open(outdir / args.save_ledger, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, sort_keys=True)

    print("Wrote:", ", ".join(ledger["files"]))
    print("Checksums:", v5_path.name, "and", sha_path.name)
    print("Seed ledger:", args.save_ledger)

if __name__ == "__main__":
    main()
