#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation for: Traffic-Aware Massive MIMO Parameter Optimization (QUBO + SA)
Generates results for tables/figures in the paper and optionally produces results_plot.pdf.

Dependencies: numpy (matplotlib only required if --plot is used)
"""
import argparse, math, random
import numpy as np

# ------------------ Utilities ------------------

def db2lin(x_db: float) -> float:
    return 10**(x_db/10.0)

def lin2db(x_lin: np.ndarray) -> np.ndarray:
    return 10*np.log10(np.maximum(x_lin, 1e-15))

def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)

def norm_cdf(z: float) -> float:
    # Standard normal CDF via error function
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# ------------------ Geometry and Users ------------------

def _hex_axial_to_xy(q: int, r: int, spacing_km: float):
    # Pointy-topped hex axial coordinate to 2D Cartesian
    x = spacing_km * (np.sqrt(3) * (q + r/2))
    y = spacing_km * (1.5 * r)
    return x, y

def hex_grid(Bside=1, spacing_km=0.5):
    """
    Returns coordinates for a hexagonal layout with given side length Bside.
    Number of cells = 1 + 3*Bside*(Bside+1). Center at (0,0).
    """
    coords = [(0.0, 0.0)]
    if Bside <= 0:
        return np.array(coords, dtype=float)
    # Axial neighbor directions (pointy-top)
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    for r in range(1, Bside+1):
        q, s = r, 0  # start at (r,0)
        # walk the ring
        for d in range(6):
            dq, ds = dirs[d]
            for _ in range(r):
                x, y = _hex_axial_to_xy(q, s, spacing_km)
                coords.append((x, y))
                q += dq
                s += ds
    return np.array(coords, dtype=float)

def min_Bside_for_B(B: int) -> int:
    """Find minimal hex side length Bside to cover at least B sites."""
    if B <= 1:
        return 0
    s = 0
    while True:
        n = 1 + 3*s*(s+1)
        if n >= B:
            return s
        s += 1

def user_points(N=400, extent=1.0, traffic_type='uniform',
                hotspot_center=(0.25, 0.25), hotspot_radius=0.3, hotspot_ratio=0.6):
    if traffic_type == 'hotspot':
        Nh = int(N * hotspot_ratio)
        Nu = N - Nh
        theta = np.random.uniform(0, 2*np.pi, Nh)
        r = hotspot_radius * np.sqrt(np.random.uniform(0, 1, Nh))
        hx = hotspot_center[0] + r*np.cos(theta)
        hy = hotspot_center[1] + r*np.sin(theta)
        hu = np.stack([hx, hy], axis=1)
        uu = np.random.uniform(-extent, extent, size=(Nu, 2))
        return np.vstack([hu, uu])
    else:
        return np.random.uniform(-extent, extent, size=(N, 2))

# ------------------ Channel and Gains ------------------

def vertical_pattern(delta_deg, th3dB=8.0, Amax=25.0):
    return -np.minimum(12.0*(delta_deg/th3dB)**2, Amax)

def get_link_gains(bs_xy, ue_xy, M_sel, T_sel, h_bs=30.0, h_ue=1.5):
    """
    3GPP-like UMa large-scale pathloss and vertical antenna pattern; returns linear gains.
    """
    B, K = bs_xy.shape[0], ue_xy.shape[0]
    G = np.zeros((B, K), dtype=float)
    for b in range(B):
        for k in range(K):
            d2 = np.linalg.norm(bs_xy[b] - ue_xy[k])*1000.0 + 1.0
            d3 = np.sqrt(d2**2 + (h_bs - h_ue)**2)
            pl_db = 128.1 + 37.6*np.log10(d3/1000.0 + 1e-3)  # 3GPP UMa
            elev_deg = np.rad2deg(np.arctan2(h_bs - h_ue, d2))
            av_db = vertical_pattern(elev_deg - T_sel[b])
            G[b, k] = M_sel[b] * db2lin(-pl_db + av_db)
    return G

# ------------------ QUBO Construction and SA ------------------

def build_qubo(B, M_choices, T_choices, R_choices, bs_xy, ue_xy,
               lam_onehot=10.0, alpha=1.0, beta=0.05, gamma=1.0, cell_independent=False):
    """
    Construct QUBO matrix using baseline-delta inclusion-exclusion.
    Diagonal: local utility; Off-diagonal: inter-cell couplings (skipped if cell_independent=True).
    """
    M_list, T_list, R_list = list(M_choices), list(T_choices), list(R_choices)
    n_m, n_t, n_r = len(M_list), len(T_list), len(R_list)
    n_vars_per_b = n_m + n_t + n_r
    n_total_vars = B * n_vars_per_b

    def idx_map(b, p_type, p_idx):
        if p_type == 'M':
            offset = 0
        elif p_type == 'T':
            offset = n_m
        else:
            offset = n_m + n_t
        return b * n_vars_per_b + offset + p_idx

    Q = np.zeros((n_total_vars, n_total_vars), dtype=float)

    # Precompute large-scale gain cache for all (M, T) pairs for utility approximation
    gain_cache = {}
    for mi, mval in enumerate(M_list):
        for ti, tval in enumerate(T_list):
            M_tmp = np.full(B, mval, dtype=float)
            T_tmp = np.full(B, tval, dtype=float)
            gain_cache[(mi, ti)] = get_link_gains(bs_xy, ue_xy, M_tmp, T_tmp)

    # Baseline mid-range
    base_mi = min(1, n_m-1)
    base_ti = min(1, n_t-1)
    noise_W = db2lin(-174 + 10*np.log10(20e6))
    sinr_thresh = db2lin(5.0)

    # Assign users to serving cell based on baseline (mid) config
    G_base_full = gain_cache[(base_mi, base_ti)]
    serving_bs = np.argmax(G_base_full, axis=0)

    # Diagonal terms: local contributions for M and T (R treated as overhead proxy)
    for b in range(B):
        ue_mask = (serving_bs == b)
        if not np.any(ue_mask):
            continue

        # M diagonal: energy and capacity deltas
        for mi, mval in enumerate(M_list):
            im = idx_map(b, 'M', mi)
            G_new = gain_cache[(mi, base_ti)][b, ue_mask]
            G_basel = gain_cache[(base_mi, base_ti)][b, ue_mask]
            cap_delta = np.sum(np.log2(1 + G_new / noise_W) - np.log2(1 + G_basel / noise_W))
            energy_delta = (mval - M_list[base_mi]) * 3.0  # 3 W per RF chain proxy
            util = alpha * cap_delta - beta * energy_delta
            Q[im, im] -= util

        # T diagonal: capacity and coverage deltas
        for ti, tval in enumerate(T_list):
            it = idx_map(b, 'T', ti)
            G_new = gain_cache[(base_mi, ti)][b, ue_mask]
            G_basel = gain_cache[(base_mi, base_ti)][b, ue_mask]
            cap_delta = np.sum(np.log2(1 + G_new / noise_W) - np.log2(1 + G_basel / noise_W))
            cov_delta = np.sum((G_new / noise_W) < sinr_thresh) - np.sum((G_basel / noise_W) < sinr_thresh)
            util = alpha * cap_delta - gamma * cov_delta
            Q[it, it] -= util

        # R diagonal: small overhead/capacity proxy (optional, simplified)
        for ri, rval in enumerate(R_list):
            ir = idx_map(b, 'R', ri)
            overhead_penalty = -alpha * 0.05 * (rval - 1)  # simple pilot overhead model
            Q[ir, ir] += overhead_penalty

    # Off-diagonal: inter-cell couplings (only if coordinated)
    if not cell_independent:
        for b in range(B):
            for bp in range(b + 1, B):
                ue_mask_victim = (serving_bs == bp)
                if not np.any(ue_mask_victim):
                    continue
                for mi, mval in enumerate(M_list):
                    im_interferer = idx_map(b, 'M', mi)
                    for ti, tval in enumerate(T_list):
                        it_victim = idx_map(bp, 'T', ti)
                        # baseline interferer contribution
                        interf_base = gain_cache[(base_mi, base_ti)][b, ue_mask_victim]
                        interf_new = gain_cache[(mi, base_ti)][b, ue_mask_victim]
                        sig_victim = gain_cache[(base_mi, ti)][bp, ue_mask_victim]
                        sinr_base = sig_victim / (noise_W + interf_base)
                        sinr_new = sig_victim / (noise_W + interf_new)
                        delta_util = alpha * np.sum(np.log2(1 + sinr_new) - np.log2(1 + sinr_base))
                        # negative of utility change because we minimize x^T Q x
                        Q[im_interferer, it_victim] -= delta_util

    # One-hot penalties per parameter per cell
    for b in range(B):
        for ptype, plist in [('M', M_list), ('T', T_list), ('R', R_list)]:
            indices = [idx_map(b, ptype, i) for i in range(len(plist))]
            for i in indices:
                Q[i, i] -= lam_onehot
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    Q[indices[i], indices[j]] += 2 * lam_onehot

    return Q

def simulated_annealing_qubo(Q, MTR_lists, B, steps=30000, T0=1.0, T1=1e-4):
    """
    Simulated annealing on QUBO with one-hot neighborhood moves.
    """
    M_list, T_list, R_list = MTR_lists
    n_m, n_t, n_r = len(M_list), len(T_list), len(R_list)
    n_per_b = n_m + n_t + n_r

    def idxs(b, mi, ti, ri):
        return b*n_per_b + mi, b*n_per_b + n_m + ti, b*n_per_b + n_m + n_t + ri

    # Random valid initial solution
    x = np.zeros(Q.shape[0], dtype=int)
    m_sels = np.random.randint(0, n_m, B)
    t_sels = np.random.randint(0, n_t, B)
    r_sels = np.random.randint(0, n_r, B)
    for b in range(B):
        im, it, ir = idxs(b, m_sels[b], t_sels[b], r_sels[b])
        x[im] = x[it] = x[ir] = 1

    def energy(xvec):
        return float(xvec.T @ Q @ xvec)

    E = energy(x)
    best_x = x.copy()
    best_E = E

    for step in range(steps):
        T = T0 * (T1/T0)**(step/max(1, steps-1))
        b = np.random.randint(0, B)
        ptype = np.random.choice(['M', 'T', 'R'])
        x_new = x.copy()

        if ptype == 'M':
            old = m_sels[b]
            cand = (old + np.random.randint(1, n_m)) % n_m
            im_old, _, _ = idxs(b, old, t_sels[b], r_sels[b])
            im_new, _, _ = idxs(b, cand, t_sels[b], r_sels[b])
            x_new[im_old] = 0
            x_new[im_new] = 1
        elif ptype == 'T':
            old = t_sels[b]
            cand = (old + np.random.randint(1, n_t)) % n_t
            _, it_old, _ = idxs(b, m_sels[b], old, r_sels[b])
            _, it_new, _ = idxs(b, m_sels[b], cand, r_sels[b])
            x_new[it_old] = 0
            x_new[it_new] = 1
        else:
            old = r_sels[b]
            cand = (old + np.random.randint(1, n_r)) % n_r
            _, _, ir_old = idxs(b, m_sels[b], t_sels[b], old)
            _, _, ir_new = idxs(b, m_sels[b], t_sels[b], cand)
            x_new[ir_old] = 0
            x_new[ir_new] = 1

        E_new = energy(x_new)
        dE = E_new - E
        if dE < 0 or np.random.rand() < math.exp(-dE / max(T, 1e-12)):
            x = x_new
            E = E_new
            if ptype == 'M':
                m_sels[b] = cand
            elif ptype == 'T':
                t_sels[b] = cand
            else:
                r_sels[b] = cand
            if E < best_E:
                best_E = E
                best_x = x.copy()

    return best_x

def decode_solution(x, MTR_lists, B):
    M_list, T_list, R_list = MTR_lists
    n_m, n_t, n_r = len(M_list), len(T_list), len(R_list)
    n_per_b = n_m + n_t + n_r
    M_sel = np.zeros(B, dtype=int)
    T_sel = np.zeros(B, dtype=int)
    R_sel = np.zeros(B, dtype=int)
    for b in range(B):
        seg = x[b*n_per_b:(b+1)*n_per_b]
        mvec = seg[:n_m]
        tvec = seg[n_m:n_m+n_t]
        rvec = seg[n_m+n_t:]
        M_sel[b] = M_list[np.argmax(mvec)] if mvec.sum() > 0 else M_list[0]
        T_sel[b] = T_list[np.argmax(tvec)] if tvec.sum() > 0 else T_list[0]
        R_sel[b] = R_list[np.argmax(rvec)] if rvec.sum() > 0 else R_list[0]
    return M_sel, T_sel, R_sel

# ------------------ Evaluation and Baselines ------------------

def evaluate_solution(M_sel, T_sel, R_sel, bs_xy, ue_xy):
    """
    Returns (sum_rate_gbps, p5_mbps, power_W, energy_eff_Mbps_per_W).
    """
    B = len(M_sel)
    G = get_link_gains(bs_xy, ue_xy, M_sel, T_sel)
    noise_W = db2lin(-174 + 10*np.log10(20e6))
    serving_bs = np.argmax(G, axis=0)

    user_sinr = np.zeros(ue_xy.shape[0], dtype=float)
    for k in range(ue_xy.shape[0]):
        b = serving_bs[k]
        sig = G[b, k]
        interf = np.sum(G[:, k]) - sig
        # Pilot contamination proxy: add interference from cells in same reuse class as serving
        pilot_interf = 0.0
        if R_sel[b] > 1:
            for bp in range(B):
                if bp != b and (bp % R_sel[b]) == (b % R_sel[b]):
                    pilot_interf += G[bp, k]
        user_sinr[k] = sig / (noise_W + interf + pilot_interf)

    user_rates_Mbps = 20e6 * np.log2(1 + user_sinr) / 1e6
    sum_rate_gbps = float(np.sum(user_rates_Mbps) / 1000.0)
    p5_mbps = float(np.percentile(user_rates_Mbps, 5))
    power_W = float(np.sum(500.0 + 3.0*np.array(M_sel)))  # 500W static + 3W per RF chain
    energy_eff = float(np.sum(user_rates_Mbps) / power_W)
    return sum_rate_gbps, p5_mbps, power_W, energy_eff

def run_greedy_solver(bs_xy, ue_xy, M_choices, T_choices, R_choices):
    B = bs_xy.shape[0]
    M = np.random.choice(M_choices, B)
    T = np.random.choice(T_choices, B)
    R = np.random.choice(R_choices, B)

    for _ in range(3):  # 3 passes
        for b in range(B):
            best = (None, -1e9)
            for m in M_choices:
                for t in T_choices:
                    for r in R_choices:
                        M[b], T[b], R[b] = m, t, r
                        sr, p5, pw, ee = evaluate_solution(M, T, R, bs_xy, ue_xy)
                        util = sr - 0.005*pw  # proxy utility
                        if util > best[1]:
                            best = ((m, t, r), util)
            M[b], T[b], R[b] = best[0]
    return M, T, R

# ------------------ Plotting ------------------

def plot_config(bs_xy, ue_xy, M_sel, T_sel, traffic_type, filename='results_plot.pdf'):
    # Lazy import matplotlib to avoid dependency when not plotting
    import matplotlib as mpl
    mpl.use('Agg')  # headless-safe backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.figure(figsize=(15, 5))
    # Traffic map
    ax1 = plt.subplot(1, 3, 1)
    if traffic_type == 'hotspot':
        hb = ax1.hexbin(ue_xy[:,0], ue_xy[:,1], gridsize=30, cmap='inferno', mincnt=1)
        cbar = plt.colorbar(hb, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('User density (rel.)')
        circle = plt.Circle((0.25,0.25), 0.3, color='red', fill=False, linestyle='--', linewidth=2)
        ax1.add_artist(circle)
    else:
        ax1.plot(ue_xy[:,0], ue_xy[:,1], 'k.', markersize=1, alpha=0.5)
    ax1.scatter(bs_xy[:,0], bs_xy[:,1], marker='^', s=100, c='cyan', edgecolors='k', zorder=5)
    ax1.set_title('(a) Traffic Density')
    ax1.set_aspect('equal'); ax1.set_xlim(-1.2,1.2); ax1.set_ylim(-1.2,1.2)

    # Antenna count
    ax2 = plt.subplot(1, 3, 2)
    normM = Normalize(vmin=min(M_sel), vmax=max(M_sel))
    colorsM = plt.cm.Blues(normM(M_sel))
    ax2.scatter(bs_xy[:,0], bs_xy[:,1], marker='^', s=500, c=colorsM, edgecolors='k')
    for i,(x,y) in enumerate(bs_xy):
        ax2.text(x, y, f'{M_sel[i]}', ha='center', va='center', color='white', weight='bold')
    ax2.set_title('(b) Optimized Antenna Count ($M_b$)')
    ax2.set_aspect('equal'); ax2.set_xlim(-1.2,1.2); ax2.set_ylim(-1.2,1.2)

    # Tilt
    ax3 = plt.subplot(1, 3, 3)
    normT = Normalize(vmin=min(T_sel), vmax=max(T_sel))
    colorsT = plt.cm.Greens(normT(T_sel))
    ax3.scatter(bs_xy[:,0], bs_xy[:,1], marker='^', s=500, c=colorsT, edgecolors='k')
    for i,(x,y) in enumerate(bs_xy):
        ax3.text(x, y, f'{T_sel[i]}°', ha='center', va='center', color='white', weight='bold')
    ax3.set_title('(c) Optimized Downtilt ($\\theta_b$)')
    ax3.set_aspect('equal'); ax3.set_xlim(-1.2,1.2); ax3.set_ylim(-1.2,1.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved '{filename}'")

# ------------------ Significance and Reporting ------------------

def paired_t_approx(x, y):
    """
    Approximate two-sided paired t-test against zero mean difference using normal approximation for p-value.
    Returns (t_stat, p_approx, cohen_d).
    """
    d = np.array(x) - np.array(y)
    n = len(d)
    mean = float(np.mean(d))
    std = float(np.std(d, ddof=1)) if n > 1 else 0.0
    t = mean / (std / math.sqrt(n)) if std > 0 and n > 1 else float('inf') if mean != 0 else 0.0
    # normal approximation for p-value
    p = 2.0 * (1.0 - norm_cdf(abs(t)))
    # Cohen's d for paired: mean(diff)/std(diff)
    d_eff = mean / std if std > 0 else float('inf') if mean != 0 else 0.0
    return t, p, d_eff

# ------------------ Main Experiment ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=7)
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--traffic', type=str, default='hotspot', choices=['uniform','hotspot'])
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--users', type=int, default=400)
    ap.add_argument('--plot', action='store_true', help='Generate results_plot.pdf from coordinated QUBO-SA solution (first run)')
    # Be tolerant to unknown args (e.g., environments that inject flags)
    args, _ = ap.parse_known_args()

    M_choices = (32, 64, 128)
    T_choices = (6, 10, 14)
    R_choices = (1, 3)
    MTR_lists = (M_choices, T_choices, R_choices)

    methods = ['Static-Max', 'Greedy', 'QUBO-Indep', 'QUBO-Coordinated']
    results = {m: [] for m in methods}

    print(f'=== Simulation: B={args.B}, runs={args.runs}, traffic={args.traffic} ===')
    for run in range(args.runs):
        print(f'Run {run+1}/{args.runs}')
        seed_all(run)
        # Generate enough BS locations for requested B
        Bside_needed = min_Bside_for_B(args.B)
        bs_xy = hex_grid(Bside=Bside_needed, spacing_km=0.5)[:args.B]
        ue_xy = user_points(N=args.users, extent=1.0, traffic_type=args.traffic)

        # Static-Max
        M_sm = np.full(args.B, 128)
        T_sm = np.full(args.B, 10)
        R_sm = np.full(args.B, 1)
        results['Static-Max'].append(evaluate_solution(M_sm, T_sm, R_sm, bs_xy, ue_xy))

        # Greedy
        M_g, T_g, R_g = run_greedy_solver(bs_xy, ue_xy, M_choices, T_choices, R_choices)
        results['Greedy'].append(evaluate_solution(M_g, T_g, R_g, bs_xy, ue_xy))

        # QUBO Independent
        Q_indep = build_qubo(args.B, M_choices, T_choices, R_choices, bs_xy, ue_xy, cell_independent=True)
        x_indep = simulated_annealing_qubo(Q_indep, MTR_lists, args.B, steps=args.steps)
        M_i, T_i, R_i = decode_solution(x_indep, MTR_lists, args.B)
        results['QUBO-Indep'].append(evaluate_solution(M_i, T_i, R_i, bs_xy, ue_xy))

        # QUBO Coordinated
        Q_coord = build_qubo(args.B, M_choices, T_choices, R_choices, bs_xy, ue_xy, cell_independent=False)
        x_coord = simulated_annealing_qubo(Q_coord, MTR_lists, args.B, steps=args.steps)
        M_c, T_c, R_c = decode_solution(x_coord, MTR_lists, args.B)
        results['QUBO-Coordinated'].append(evaluate_solution(M_c, T_c, R_c, bs_xy, ue_xy))

        if args.plot and run == 0:
            try:
                plot_config(bs_xy, ue_xy, M_c, T_c, args.traffic)
            except Exception as e:
                print(f"Plotting failed: {e}")

    # Print summary
    print('\n--- Results Summary (mean +/- std) ---')
    for m in methods:
        arr = np.array(results[m])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        print(f'{m:>16}:  SR={mean[0]:.2f}+/-{std[0]:.2f}  P5={mean[1]:.2f}+/-{std[1]:.2f}  P={mean[2]:.1f}+/-{std[2]:.1f}  EE={mean[3]:.2f}+/-{std[3]:.2f}')

    # Approximate significance: QUBO-Coordinated vs QUBO-Indep and vs Greedy for sum-rate
    print('\n--- Approximate Paired Significance (normal-based) ---')
    for baseline in ['QUBO-Indep', 'Greedy']:
        x = [r[0] for r in results['QUBO-Coordinated']]
        y = [r[0] for r in results[baseline]]
        t, p, d_eff = paired_t_approx(x, y)
        print(f'QUBO-Coordinated vs {baseline}: t={t:.2f}, p~={p:.3e}, Cohen d={d_eff:.2f} (Sum-Rate)')

if __name__ == '__main__':
    main()