# Auto-generated from LaTeX code blocks; consolidate all simulation here.
# === Begin extracted block 1 ===
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible Monte Carlo used for the results reported in the paper.

Authoritative model (Model 2):
  - Physics-driven mapping from HCF coexistence parameters to asymmetric Pauli
    error probabilities using distributed integrals along the span:
      SpRS ~ c_R * ∫ P(z) dz,  FWM ~ c_F * Δλ * ∫ P(z)^2 dz,
    with P(z)=P0*exp(-alpha*z), alpha = kappa * alpha_dB (nepers/km).
  - Temporal correlation via a two-state Markov-modulated Bernoulli process (MMBP)
    with persistence rho (Gilbert--Elliott style). The "bad" state's error prob.
    is scaled by a user parameter beta (default 2.0), clipped to 1.
  - Block failure under bounded-distance decoding (BDD): fail if wX>t or wZ>t.
  - 95% Wilson intervals; optional report of expected run length 1/(1-rho).

Outputs CSV with point estimates, 95% Wilson confidence intervals, raw counts,
and throughput measurements.

Python: 3.8+ (stdlib only)

Examples:
  python3 simulation.py --model model2 --L 100 --pcl 10 --sep-nm 6.4 --rho 0.6 --trials-bdd 1000000
  python3 simulation.py --model model2 --rho-sweep 0.00,0.30,0.60,0.85,0.95 --trials-bdd 1000000
  python3 simulation.py --model model2 --atten-kappa-alt 0.2302585093 --rho 0.6 --trials-bdd 1000000
  python3 simulation.py --model model2 --length-sweep 50,100,150 --pcl 10 --sep-nm 6.4 --rho 0.6 --trials-bdd 1000000
  python3 simulation.py --model model2 --power-sweep 0,5,10 --L 100 --sep-nm 6.4 --rho 0.6 --trials-bdd 1000000
  python3 simulation.py --model model2 --sep-sweep 3.2,6.4,12.8 --L 100 --pcl 10 --rho 0.6 --trials-bdd 1000000
  python3 simulation.py --model model2 --eta-sweep 0.10,0.30,0.50 --L 100 --pcl 10 --sep-nm 6.4 --rho 0.6 --trials-bdd 1000000
  python3 simulation.py --model model2 --shared-state --rho 0.6 --trials-bdd 1000000
"""

import argparse, math, random, sys, time, platform
from typing import List, Tuple

def set_seed(seed: int) -> None:
    random.seed(seed)

def hcf_noise_model(length_km: float,
                    classical_power_dBm: float,
                    wavelength_separation_nm: float,
                    raman_coeff: float = 0.025,
                    fwm_coeff: float = 1e-4,
                    attenuation_db_per_km: float = 0.25,
                    eta_px: float = 0.3,
                    atten_kappa: float = 0.1) -> Tuple[float,float,float,float,float]:
    """Effective physics-inspired mapping from HCF coexistence to baseline Pauli probs
    using distributed-noise integrals along the span.

    alpha [nepers/km] = attenuation_db_per_km * atten_kappa,
    SpRS ~ raman_coeff * (P0/alpha) * (1 - e^{-alpha L}),
    FWM  ~ fwm_coeff  * Δλ * ∫ P(z)^2 dz = fwm_coeff * Δλ * (P0^2/(2 alpha)) * (1 - e^{-2 alpha L}).

    The exact dB->neper factor is ln(10)/10 ≈ 0.2302585093. We expose atten_kappa
    to support calibration/sensitivity studies.
    Returns: (px, pz, sprs, fwm, alpha)
    """
    P0 = 10 ** ((classical_power_dBm - 30) / 10.0)  # W
    alpha = attenuation_db_per_km * atten_kappa     # nepers/km
    if alpha <= 0:
        alpha = 1e-12
    I1 = (P0/alpha) * (1.0 - math.exp(-alpha * length_km))
    I2 = (P0*P0/(2.0*alpha)) * (1.0 - math.exp(-2.0*alpha * length_km))
    sprs = raman_coeff * I1
    fwm  = fwm_coeff  * wavelength_separation_nm * I2
    tau = sprs + fwm
    pz = 1.0 - math.exp(-tau)
    px = eta_px * pz
    return px, pz, sprs, fwm, alpha

def markov_states(rho: float, n: int) -> List[int]:
    """Generate hidden states for a symmetric two-state MMBP."""
    state = 1 if random.random() < 0.5 else 0
    out = [0]*n
    for i in range(n):
        out[i] = state
        if random.random() > rho:
            state ^= 1
    return out

def markov_bits(p_base: float, rho: float, n: int, beta: float) -> List[int]:
    """Two-state symmetric MMBP: low-noise: p_base; high-noise: min(1, beta*p_base)."""
    states = markov_states(rho, n)
    out = [0]*n
    hi = min(1.0, beta*p_base)
    for i, s in enumerate(states):
        p = p_base if s == 0 else hi
        out[i] = 1 if random.random() < p else 0
    return out

def bdd_block_fail_model2(n: int, t: int, px: float, pz: float, rho: float, trials: int,
                          beta: float = 2.0,
                          shared_state: bool=False) -> Tuple[int,int]:
    fails = 0
    for _ in range(trials):
        if shared_state:
            states = markov_states(rho, n)
            hiX = min(1.0, beta*px)
            hiZ = min(1.0, beta*pz)
            x = [1 if random.random() < (px if s == 0 else hiX) else 0 for s in states]
            z = [1 if random.random() < (pz if s == 0 else hiZ) else 0 for s in states]
        else:
            x = markov_bits(px, rho, n, beta)
            z = markov_bits(pz, rho, n, beta)
        if sum(x) > t or sum(z) > t:
            fails += 1
    return fails, trials

def depolarizing_trial(n: int, t: int, p: float) -> bool:
    """Single codeword trial under depolarizing channel with error prob p."""
    wx = wz = 0
    for _ in range(n):
        r = random.random()
        if r < p/3.0:            # X
            wx += 1
        elif r < 2*p/3.0:        # Y
            wx += 1; wz += 1
        elif r < p:              # Z
            wz += 1
    return (wx > t) or (wz > t)

def bdd_block_fail_depol(n: int, t: int, p: float, trials: int) -> Tuple[int,int]:
    fails = 0
    for _ in range(trials):
        if depolarizing_trial(n, t, p):
            fails += 1
    return fails, trials

def wilson_interval(k: int, n: int) -> Tuple[float,float,float]:
    """Wilson score interval for binomial proportion (95%)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    from math import sqrt
    z = 1.959963984540054 # 95%
    phat = k/n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = (z/denom) * sqrt(phat*(1-phat)/n + z*z/(4*n*n))
    lo = max(0.0, center-half)
    hi = min(1.0, center+half)
    return phat, lo, hi

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["model2","depolarizing"], default="model2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--L", type=float, default=100.0)
    ap.add_argument("--pcl", type=float, default=10.0)
    ap.add_argument("--sep-nm", type=float, default=6.4)
    ap.add_argument("--eta-px", type=float, default=0.3)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=2.0,
                    help="bad-state scaling factor for error probability")
    ap.add_argument("--rho-sweep", type=str, default="")
    ap.add_argument("--length-sweep", type=str, default="")
    ap.add_argument("--power-sweep", type=str, default="")
    ap.add_argument("--sep-sweep", type=str, default="")
    ap.add_argument("--eta-sweep", type=str, default="")
    ap.add_argument("--shared-state", action="store_true",
                    help="Use a shared hidden state for X and Z processes")
    ap.add_argument("--n", type=int, default=255)
    ap.add_argument("--t", type=int, default=10)
    ap.add_argument("--trials-bdd", type=int, default=1000000)
    ap.add_argument("--p-depol", type=float, default=0.02)
    ap.add_argument("--atten-kappa", type=float, default=0.1,
                    help="attenuation conversion constant; exact ln(10)/10 ≈ 0.2302585093")
    ap.add_argument("--atten-kappa-alt", type=float, default=None,
                    help="optional second kappa to print sensitivity (and run BDD if provided)")
    args = ap.parse_args()

    set_seed(args.seed)

    # Echo environment/context (commented lines, do not break CSV)
    print(f"# cmdline: {' '.join(sys.argv)}")
    print(f"# python: {sys.version.splitlines()[0]}")
    print(f"# platform: {platform.platform()}")
    print("quantity,value,lower95,upper95,k,n,comment")

    if args.model == "model2":
        px_base, pz_base, sprs, fwm, alpha = hcf_noise_model(args.L, args.pcl, args.sep_nm,
                                                             eta_px=args.eta_px, atten_kappa=args.atten_kappa)
        p_eff_sum = 0.5*(min(1.0, args.beta*px_base)+px_base) + 0.5*(min(1.0, args.beta*pz_base)+pz_base)
        print(f"pz_base,{pz_base:.8e},,,,,phase baseline (SpRS+FWM); kappa={args.atten_kappa:.9f}")
        print(f"px_base,{px_base:.8e},,,,,amplitude baseline = eta_px*pz; kappa={args.atten_kappa:.9f}")
        print(f"sprs_term,{sprs:.8e},,,,,distributed SpRS integral contribution")
        print(f"fwm_term,{fwm:.8e},,,,,distributed FWM integral contribution")
        print(f"alpha_nepers_per_km,{alpha:.8e},,,,,attenuation in nepers/km")
        print(f"p_eff_sum,{p_eff_sum:.8e},,,,,state-averaged per-qubit error probability (diagnostic)")
        # Single rho
        t0 = time.time()
        k, ntr = bdd_block_fail_model2(args.n, args.t, px_base, pz_base, args.rho, args.trials_bdd,
                                       beta=args.beta, shared_state=args.shared_state)
        dur = time.time() - t0
        ph, lo, hi = wilson_interval(k, ntr)
        tag = f"P_L_BDD_rho={args.rho:.2f}" + ("_shared" if args.shared_state else "")
        print(f"{tag},{ph:.8e},{lo:.8e},{hi:.8e},{k},{ntr},BDD (n={args.n}, t={args.t}); trials={ntr}; seed={args.seed}; kappa={args.atten_kappa:.9f}; beta={args.beta:.2f}")
        if dur > 0:
            print(f"runtime_sec_rho={args.rho:.2f},{dur:.3f},,,,,wall-clock seconds for previous BDD run")
            print(f"throughput_rho={args.rho:.2f},{ntr/dur:.2f},,,,,trials per second for previous BDD run")
        if args.rho_sweep:
            for tok in args.rho_sweep.split(","):
                r = float(tok)
                set_seed(args.seed)  # reset to keep comparability across rho
                t1 = time.time()
                k, ntr = bdd_block_fail_model2(args.n, args.t, px_base, pz_base, r, args.trials_bdd,
                                               beta=args.beta, shared_state=args.shared_state)
                dur = time.time() - t1
                ph, lo, hi = wilson_interval(k, ntr)
                tag = f"P_L_BDD_rho={r:.2f}" + ("_shared" if args.shared_state else "")
                print(f"{tag},{ph:.8e},{lo:.8e},{hi:.8e},{k},{ntr},BDD sweep; trials={ntr}; seed={args.seed}; kappa={args.atten_kappa:.9f}; beta={args.beta:.2f}")
                if dur > 0:
                    print(f"runtime_sec_rho={r:.2f},{dur:.3f},,,,,wall-clock seconds for previous BDD run")
                    print(f"throughput_rho={r:.2f},{ntr/dur:.2f},,,,,trials per second for previous BDD run")
                if r < 1.0:
                    runlen = 1.0/(1.0 - r) if r < 1.0 else float('inf')
                    print(f"expected_run_length_rho={r:.2f},{runlen:.8e},,,,,E[run length]=1/(1-rho)")
        # Optional sensitivity to exact conversion constant
        if args.atten_kappa_alt is not None:
            set_seed(args.seed)
            px2, pz2, sprs2, fwm2, alpha2 = hcf_noise_model(args.L, args.pcl, args.sep_nm,
                                                             eta_px=args.eta_px, atten_kappa=args.atten_kappa_alt)
            p_eff_sum2 = 0.5*(min(1.0, args.beta*px2)+px2) + 0.5*(min(1.0, args.beta*pz2)+pz2)
            print(f"pz_base_kappa_alt,{pz2:.8e},,,,,phase baseline; kappa={args.atten_kappa_alt:.9f}")
            print(f"px_base_kappa_alt,{px2:.8e},,,,,amplitude baseline; kappa={args.atten_kappa_alt:.9f}")
            print(f"sprs_term_kappa_alt,{sprs2:.8e},,,,,distributed SpRS contribution; kappa={args.atten_kappa_alt:.9f}")
            print(f"fwm_term_kappa_alt,{fwm2:.8e},,,,,distributed FWM contribution; kappa={args.atten_kappa_alt:.9f}")
            print(f"alpha_nepers_per_km_kappa_alt,{alpha2:.8e},,,,,attenuation in nepers/km; kappa={args.atten_kappa_alt:.9f}")
            print(f"p_eff_sum_kappa_alt,{p_eff_sum2:.8e},,,,,state-averaged per-qubit error (diagnostic); kappa={args.atten_kappa_alt:.9f}")
            t2 = time.time()
            k2, ntr2 = bdd_block_fail_model2(args.n, args.t, px2, pz2, args.rho, args.trials_bdd,
                                             beta=args.beta, shared_state=args.shared_state)
            dur2 = time.time() - t2
            ph2, lo2, hi2 = wilson_interval(k2, ntr2)
            tag2 = f"P_L_BDD_rho={args.rho:.2f}_kappa_alt" + ("_shared" if args.shared_state else "")
            print(f"{tag2},{ph2:.8e},{lo2:.8e},{hi2:.8e},{k2},{ntr2},BDD with kappa={args.atten_kappa_alt:.9f} (n={args.n}, t={args.t}); trials={ntr2}; seed={args.seed}; beta={args.beta:.2f}")
            if dur2 > 0:
                print(f"runtime_sec_rho={args.rho:.2f}_kappa_alt,{dur2:.3f},,,,,wall-clock seconds for previous BDD run")
                print(f"throughput_rho={args.rho:.2f}_kappa_alt,{ntr2/dur2:.2f},,,,,trials per second for previous BDD run")
        # Length/power/separation/eta sweeps
        def run_point(label: str, L: float=None, pcl: float=None, sep_nm: float=None, eta_px: float=None):
            L = args.L if L is None else L
            pcl = args.pcl if pcl is None else pcl
            sep_nm = args.sep_nm if sep_nm is None else sep_nm
            eta_px = args.eta_px if eta_px is None else eta_px
            set_seed(args.seed)
            px, pz, sprs_p, fwm_p, _ = hcf_noise_model(L, pcl, sep_nm, eta_px=eta_px, atten_kappa=args.atten_kappa)
            print(f"pz_base_{label},{pz:.8e},,,,,baseline for {label} (kappa={args.atten_kappa:.9f})")
            print(f"px_base_{label},{px:.8e},,,,,baseline for {label} (kappa={args.atten_kappa:.9f})")
            print(f"sprs_term_{label},{sprs_p:.8e},,,,,SpRS contribution for {label}")
            print(f"fwm_term_{label},{fwm_p:.8e},,,,,FWM contribution for {label}")
            t0 = time.time()
            k, ntr = bdd_block_fail_model2(args.n, args.t, px, pz, args.rho, args.trials_bdd,
                                           beta=args.beta, shared_state=args.shared_state)
            dur = time.time() - t0
            ph, lo, hi = wilson_interval(k, ntr)
            print(f"P_L_BDD_{label},{ph:.8e},{lo:.8e},{hi:.8e},{k},{ntr},BDD for {label}; trials={ntr}; seed={args.seed}; rho={args.rho:.2f}; beta={args.beta:.2f}")
            if dur > 0:
                print(f"runtime_sec_{label},{dur:.3f},,,,,wall-clock seconds for previous BDD run")
                print(f"throughput_{label},{ntr/dur:.2f},,,,,trials per second for previous BDD run")
        if args.length_sweep:
            for tok in args.length_sweep.split(","):
                Lp = float(tok)
                run_point(f"L={Lp:.0f}", L=Lp)
        if args.power_sweep:
            for tok in args.power_sweep.split(","):
                Pp = float(tok)
                run_point(f"pcl={Pp:.0f}", pcl=Pp)
        if args.sep_sweep:
            for tok in args.sep_sweep.split(","):
                Sp = float(tok)
                run_point(f"sep_nm={Sp:g}", sep_nm=Sp)
        if args.eta_sweep:
            for tok in args.eta_sweep.split(","):
                Ep = float(tok)
                run_point(f"eta={Ep:.2f}", eta_px=Ep)
    else:
        # Depolarizing baseline
        t0 = time.time()
        k, ntr = bdd_block_fail_depol(args.n, args.t, args.p_depol, args.trials_bdd)
        dur = time.time() - t0
        ph, lo, hi = wilson_interval(k, ntr)
        print(f"P_L_BDD_depol_p={args.p_depol:.3f},{ph:.8e},{lo:.8e},{hi:.8e},{k},{ntr},Depolarizing; trials={ntr}; seed={args.seed}")
        if dur > 0:
            print(f"runtime_sec_depol_p={args.p_depol:.3f},{dur:.3f},,,,,wall-clock seconds for previous BDD run")
            print(f"throughput_depol_p={args.p_depol:.3f},{ntr/dur:.2f},,,,,trials per second for previous BDD run")

if __name__ == "__main__":
    getattr(sys.stdout, "reconfigure", lambda **k: None)(encoding="ascii", errors="backslashreplace")
    run()
# === End block 1 ===
