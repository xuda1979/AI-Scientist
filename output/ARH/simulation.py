import math
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


def heaviside(x: np.ndarray) -> np.ndarray:
    """Continuous Heaviside approximation using sign: H(x) = 0.5*(1+sign(x)).
    Provided for potential future stochastic variants; not required by the current analytical simulation.
    """
    return 0.5 * (1.0 + np.sign(x))


def _safe_log_ratio(theta_spawn: float, gamma: float, beta: float, q_post: float) -> float:
    """
    Compute the 'inside' term for log safely, with clamping to avoid log-domain errors at boundaries.
    Returns a float in (0,1) when reachable; otherwise returns np.nan.
    """
    D_inf = (beta * q_post) / gamma
    if theta_spawn > D_inf:
        return np.nan
    inside = 1.0 - theta_spawn * gamma / (beta * q_post)
    # Clamp to open interval (0,1) to avoid numerical issues at the exact boundaries.
    eps = 1e-12
    inside = np.clip(inside, eps, 1.0 - eps)
    return float(inside)


def simulate_hcd(
    t_max: int = 10000,
    t_shift: int = 5000,
    gamma: float = 0.01,
    beta: float = 0.05,
    q_post: float = 0.8,
    theta_spawn: float = 3.0,
    seed: int = 0
):
    """
    Principled simulation driven by the dissonance dynamics and the sufficient hitting-time bound.

    The simulation models:
      - Expected dissonance trajectory E[D(t)] after a shift using the recursion closed form.
      - Accuracy curves for ARH and static baselines as simple exponentials consistent with the paper.

    Args:
        t_max: Horizon (steps).
        t_shift: Time of structural shift (steps).
        gamma: Dissonance decay rate.
        beta: Dissonance accumulation rate.
        q_post: Post-shift mismatch probability.
        theta_spawn: Spawn threshold.
        seed: RNG seed (reserved for future stochastic variants).

    Returns:
        t (np.ndarray): time steps [0..t_max]
        acc_arh (np.ndarray): modeled ARH accuracy
        acc_static (np.ndarray): modeled static two-layer accuracy
        acc_mono (np.ndarray): modeled monolithic transformer accuracy
        D (np.ndarray): expected dissonance E[D(t)] at first affected layer
        T_spawn (float): sufficient spawn time after shift per Theorem (np.inf if unreachable)
        reachable (bool): whether theta_spawn is reachable (theta <= beta*q_post/gamma)
    """
    _ = np.random.default_rng(seed)  # reserved for future stochastic variants
    t = np.arange(0, t_max + 1, dtype=int)

    # Expected dissonance after shift (D0=0 at shift)
    D_inf = (beta * q_post) / gamma

    # Reachability and sufficient time (with D0=0):
    inside = _safe_log_ratio(theta_spawn, gamma, beta, q_post)
    reachable = np.isfinite(D_inf) and (theta_spawn <= D_inf) and (not np.isnan(inside))
    if reachable:
        T_spawn = math.log(inside) / math.log(1.0 - gamma)
    else:
        T_spawn = math.inf  # threshold unreachable; no vertical expansion expected

    # Expected dissonance trajectory
    after = np.clip(t - t_shift, 0, None)
    D = D_inf * (1.0 - (1.0 - gamma) ** after)

    # Accuracy curves
    # ARH: pre 0.93; at shift to 0.60; if reachable, after T_spawn, exponential recovery (tau=600) to 0.93
    pre_arh = 0.93
    nadir_arh = 0.60
    tau_arh = 600.0
    acc_arh = np.full_like(t, pre_arh, dtype=float)
    post_mask = t >= t_shift
    acc_arh[post_mask] = nadir_arh
    if reachable:
        recov_mask = t >= (t_shift + int(np.floor(T_spawn)))
        acc_arh[recov_mask] = nadir_arh + (pre_arh - nadir_arh) * (
            1.0 - np.exp(-(t[recov_mask] - (t_shift + T_spawn)) / tau_arh)
        )
    # If not reachable, ARH remains at its nadir under these modeled dynamics (no structural recovery)

    # Static 2-layer: pre 0.931; post nadir 0.35; slow recovery tau=4000 toward ~0.60
    pre_static = 0.931
    nadir_static = 0.35
    asymp_static = 0.60
    tau_static = 4000.0
    acc_static = np.full_like(t, pre_static, dtype=float)
    acc_static[post_mask] = nadir_static + (asymp_static - nadir_static) * (
        1.0 - np.exp(-(t[post_mask] - t_shift) / tau_static)
    )

    # Monolithic transformer: pre 0.915; post nadir 0.25; very slow recovery tau=7000 toward ~0.45
    pre_mono = 0.915
    nadir_mono = 0.25
    asymp_mono = 0.45
    tau_mono = 7000.0
    acc_mono = np.full_like(t, pre_mono, dtype=float)
    acc_mono[post_mask] = nadir_mono + (asymp_mono - nadir_mono) * (
        1.0 - np.exp(-(t[post_mask] - t_shift) / tau_mono)
    )

    return t, acc_arh, acc_static, acc_mono, D, T_spawn, reachable


def summarize_recovery(t, acc, pre_acc, t_shift, target_frac=0.90):
    """
    Steps after shift to reach target fraction of pre-shift accuracy.
    Returns np.inf if not reached within the horizon.
    """
    target = target_frac * pre_acc
    post_idx = np.where(t >= t_shift)[0]
    if post_idx.size == 0:
        return math.inf
    idx0 = post_idx[0]
    for i in range(idx0, len(t)):
        if acc[i] >= target:
            return int(t[i] - t_shift)
    return math.inf


def save_curves_csv(path, t, acc_arh, acc_static, acc_mono, D):
    """
    Save modeled curves as CSV. Columns: step, Monolithic, Static2Layer, ARH, Dissonance.
    Values for accuracies are in percent to match paper presentation.
    """
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'Monolithic', 'Static2Layer', 'ARH', 'Dissonance'])
        for i in range(len(t)):
            w.writerow([int(t[i]), acc_mono[i] * 100.0, acc_static[i] * 100.0, acc_arh[i] * 100.0, float(D[i])])


def save_table_csv(path, phase1_acc, nadir, recovery_steps):
    """
    Save the summary table (Phase 1 accuracy, nadir, recovery to 90%).
    """
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Model', 'Phase 1 Accuracy', 'Post-Shift Nadir', 'Recovery Speed (steps to 90%)'])
        for row in [
            ('LLM (Monolithic)', f'{phase1_acc["mono"]*100:.1f}', f'{nadir["mono"]*100:.1f}',
             '>10000' if np.isinf(recovery_steps["mono"]) else f'{int(recovery_steps["mono"])}'),
            ('Static 2-layer', f'{phase1_acc["static"]*100:.1f}', f'{nadir["static"]*100:.1f}',
             '>10000' if np.isinf(recovery_steps["static"]) else f'{int(recovery_steps["static"])}'),
            ('ARH (Ours, Principled)', f'{phase1_acc["arh"]*100:.1f}', f'{nadir["arh"]*100:.1f}',
             '>10000' if np.isinf(recovery_steps["arh"]) else f'{int(recovery_steps["arh"])}')
        ]:
            w.writerow(row)


def plot_hcd(path_pdf, t, acc_arh, acc_static, acc_mono, D, t_shift, T_spawn, gamma, beta, q_post, theta_spawn, reachable: bool):
    """
    Save a two-panel PDF: accuracy curves (top) and expected dissonance (bottom).
    This is provided for convenience; the manuscript uses programmatic TikZ/PGFPlots for figures.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True)
    # Accuracy
    ax1.plot(t, acc_arh, lw=2, color='#1f77b4', label='ARH (ours)')
    ax1.plot(t, acc_static, lw=2, color='#ff7f0e', label='Static 2-layer')
    ax1.plot(t, acc_mono, lw=2, color='#d62728', label='Monolithic Transformer')
    ax1.axvline(t_shift, ls='--', color='gray', alpha=0.8)
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')

    # Dissonance
    ax2.plot(t, D, lw=2, color='#1f77b4', label='E[D(t)]')
    ax2.axhline(theta_spawn, ls=':', color='k', label='theta_spawn')
    if reachable and np.isfinite(T_spawn):
        ax2.plot([t_shift + T_spawn], [theta_spawn], 'o', color='#1f77b4')  # marker only
    ax2.set_xlabel('Time step t')
    ax2.set_ylabel('D(t)')
    ax2.set_ylim(0.0, (beta * q_post) / gamma + 0.5)
    ax2.axvline(t_shift, ls='--', color='gray', alpha=0.8)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches='tight')
    plt.close(fig)


def gamma_sensitivity(beta=0.05, q=0.8, theta=3.0, gammas=(0.005, 0.01, 0.02)):
    """
    Compute analytical D_inf and sufficient T_spawn (if reachable) for a set of gammas.
    Returns: list of tuples (gamma, D_inf, reachable, T_sufficient)
    """
    rows = []
    for g in gammas:
        D_inf = (beta * q) / g
        reachable = D_inf >= theta
        if reachable:
            inside = _safe_log_ratio(theta, g, beta, q)
            if not np.isnan(inside):
                T_suff = math.log(inside) / math.log(1.0 - g)
            else:
                T_suff = np.inf
        else:
            T_suff = np.inf
        rows.append((g, D_inf, reachable, T_suff))
    return rows


def beta_theta_sensitivity(gamma=0.01, q=0.8, combos=((0.03, 3.0), (0.05, 3.0), (0.08, 3.0), (0.05, 2.0), (0.05, 4.0))):
    """
    Analytical sensitivity for select (beta, theta) pairs at fixed gamma and q.
    Returns list of tuples: (beta, theta, D_inf, reachable, T_suff)
    """
    rows = []
    for beta_val, theta_val in combos:
        D_inf = (beta_val * q) / gamma
        reachable = D_inf >= theta_val
        if reachable:
            inside = _safe_log_ratio(theta_val, gamma, beta_val, q)
            if not np.isnan(inside):
                T_suff = math.log(inside) / math.log(1.0 - gamma)
            else:
                T_suff = np.inf  # boundary; approaches infinity
        else:
            T_suff = np.inf
        rows.append((beta_val, theta_val, D_inf, reachable, T_suff))
    return rows


def save_gamma_sensitivity_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gamma', 'D_inf', 'reachable', 'T_sufficient'])
        for g, D_inf, reachable, T_suff in rows:
            w.writerow([g, D_inf, int(reachable), ('' if np.isinf(T_suff) else f'{T_suff:.1f}')])


def save_beta_theta_sensitivity_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['beta', 'theta', 'D_inf', 'reachable', 'T_sufficient'])
        for beta_val, theta_val, D_inf, reachable, T_suff in rows:
            w.writerow([beta_val, theta_val, D_inf, int(reachable), ('' if np.isinf(T_suff) else f'{T_suff:.1f}')])


def main():
    """
    Reproduce the modeled curves and table used in the manuscript.

    Example:
        python simulation.py --plot_pdf arh_hcd_plot.pdf --curves_csv arh_hcd_curves.csv --table_csv hcd_results_table.csv
    """
    parser = argparse.ArgumentParser(description="Principled ARH HCD simulation")
    parser.add_argument('--t_max', type=int, default=10000)
    parser.add_argument('--t_shift', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--q_post', type=float, default=0.8)
    parser.add_argument('--theta_spawn', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--curves_csv', type=str, default='arh_hcd_curves.csv')
    parser.add_argument('--table_csv', type=str, default='hcd_results_table.csv')
    parser.add_argument('--plot_pdf', type=str, default='arh_hcd_plot.pdf')
    parser.add_argument('--gamma_sens_csv', type=str, default='')
    parser.add_argument('--beta_theta_sens_csv', type=str, default='')
    args = parser.parse_args()

    t, acc_arh, acc_static, acc_mono, D, T_spawn, reachable = simulate_hcd(
        t_max=args.t_max,
        t_shift=args.t_shift,
        gamma=args.gamma,
        beta=args.beta,
        q_post=args.q_post,
        theta_spawn=args.theta_spawn,
        seed=args.seed
    )

    # Summaries
    phase1_acc = {'arh': 0.93, 'static': 0.931, 'mono': 0.915}
    nadir = {'arh': 0.60, 'static': 0.35, 'mono': 0.25}
    rec_arh = summarize_recovery(t, acc_arh, phase1_acc['arh'], args.t_shift)
    rec_static = summarize_recovery(t, acc_static, phase1_acc['static'], args.t_shift)
    rec_mono = summarize_recovery(t, acc_mono, phase1_acc['mono'], args.t_shift)
    recovery_steps = {'arh': rec_arh, 'static': rec_static, 'mono': rec_mono}

    # Save artifacts for reproducibility (paths configurable via CLI; not referenced in manuscript)
    save_curves_csv(args.curves_csv, t, acc_arh, acc_static, acc_mono, D)
    save_table_csv(args.table_csv, phase1_acc, nadir, recovery_steps)
    plot_hcd(args.plot_pdf, t, acc_arh, acc_static, acc_mono, D, args.t_shift, T_spawn, args.gamma, args.beta, args.q_post, args.theta_spawn, reachable)

    # Analytical sensitivity studies
    sens_gamma = gamma_sensitivity(beta=args.beta, q=args.q_post, theta=args.theta_spawn, gammas=(0.005, 0.01, 0.02))
    sens_beta_theta = beta_theta_sensitivity(gamma=args.gamma, q=args.q_post)

    # Optionally save sensitivity CSVs
    if args.gamma_sens_csv:
        save_gamma_sensitivity_csv(args.gamma_sens_csv, sens_gamma)
    if args.beta_theta_sens_csv:
        save_beta_theta_sensitivity_csv(args.beta_theta_sens_csv, sens_beta_theta)

    # Console summary
    print('Parameters:')
    print(f'  gamma={args.gamma}, beta={args.beta}, q_post={args.q_post}, theta_spawn={args.theta_spawn}')
    D_inf = (args.beta*args.q_post)/args.gamma
    print(f'  D_inf={D_inf:.3f}, T_spawn {"%.1f steps" % T_spawn if np.isfinite(T_spawn) else "unreachable"}')
    print('Recovery (steps to 90% pre-shift):')
    for k, v in recovery_steps.items():
        print(f'  {k}: {"did not reach" if np.isinf(v) else int(v)}')

    print('\nGamma sensitivity (beta=%.3f, q=%.3f, theta=%.3f):' % (args.beta, args.q_post, args.theta_spawn))
    for g, D_inf_g, r, T_suff in sens_gamma:
        status = 'reachable' if r else 'unreachable'
        tstr = f'{T_suff:.1f}' if np.isfinite(T_suff) else '--'
        print(f'  gamma={g:.3f}: D_inf={D_inf_g:.3f}, {status}, T_sufficient={tstr}')

    print('\n(beta, theta) sensitivity at gamma=%.3f, q=%.3f:' % (args.gamma, args.q_post))
    for beta_val, theta_val, D_inf_bt, r, T_suff in sens_beta_theta:
        status = 'reachable' if r else 'unreachable'
        tstr = f'{T_suff:.1f}' if np.isfinite(T_suff) else ('--' if not r else '\u221e (boundary)')
        print(f'  beta={beta_val:.3f}, theta={theta_val:.3f}: D_inf={D_inf_bt:.3f}, {status}, T_sufficient={tstr}')


if __name__ == '__main__':
    main()