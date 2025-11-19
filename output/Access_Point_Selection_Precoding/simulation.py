#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell-Free Massive MIMO AP Selection via QUBO + CIM-like Annealing with Inner RZF
- Builds QUBO (sum-rate or fairness) with interference-aware pairwise terms
- Solves via simulated annealing on the QUBO energy
- Projects solution to feasibility (AP capacity and L APs per user)
- Designs RZF precoder and evaluates rates on the full physical channel
- Compares against strong baselines; aggregates over seeds; generates figures

Dependencies: numpy, matplotlib, tqdm
"""

import argparse
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# -------------------- Utilities --------------------

def db2lin(x_db: float) -> float:
    return 10 ** (x_db / 10.0)

def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)

def setup_plotting():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (4.5, 3.2),
        "figure.dpi": 200,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "savefig.bbox": "tight"
    })

# -------------------- Channel Model --------------------

def generate_channels(M=32, K=8, area_size_km=1.0, shadow_std_db=8.0, seed=1):
    """
    Large-scale fading: PL[dB] = 128.1 + 37.6 log10(d[km]) + shadowing
    Small-scale fading: Rayleigh, CN(0, beta)
    """
    seed_all(seed)
    ap_xy = np.random.uniform(0, area_size_km, size=(M, 2))
    ue_xy = np.random.uniform(0, area_size_km, size=(K, 2))

    H_large = np.zeros((M, K), dtype=float)
    for m in range(M):
        for k in range(K):
            dist_km = max(0.01, float(np.linalg.norm(ap_xy[m] - ue_xy[k])))
            pl_db = 128.1 + 37.6 * np.log10(dist_km)
            sh_db = np.random.normal(0.0, shadow_std_db)
            gain_lin = db2lin(-(pl_db) + sh_db)
            H_large[m, k] = gain_lin

    H_small = (np.random.randn(M, K) + 1j * np.random.randn(M, K)) / np.sqrt(2.0)
    H = H_small * np.sqrt(H_large)
    return H, H_large, ap_xy, ue_xy

# -------------------- QUBO Construction --------------------

def add_sum_equals_penalty(Q, idxs, target, lam):
    """
    Add lam * (sum_{i in idxs} x_i - target)^2 to Q.
    For binary x, x_i^2 = x_i. This contributes:
    - Diagonal (linear) terms: lam * (1 - 2*target) on each i in idxs
    - Off-diagonal quadratic terms: 2*lam on each (i,j), i<j in idxs
    Constant term dropped.
    """
    for i in idxs:
        Q[i, i] += lam * (1.0 - 2.0 * target)
    for a in range(len(idxs)):
        ia = idxs[a]
        for b in range(a + 1, len(idxs)):
            ib = idxs[b]
            Q[ia, ib] += 2.0 * lam
            Q[ib, ia] += 2.0 * lam

def build_qubo(H, H_large, L=4, P_ap_W=1.0, noise_W=1e-12, lam_ap=10.0, lam_user=10.0,
               fairness_weights=None):
    """
    Construct QUBO for AP-user link selection.
    - Diagonal: -u_mk or -w_k u_mk
    - Off-diagonal: interference penalties via MRT-based projections
    - Penalties: AP capacity and user diversity
    """
    M, K = H.shape
    n = M * K
    idx = lambda m, k: m * K + k
    Q = np.zeros((n, n), dtype=float)

    # Weights for fairness
    if fairness_weights is None:
        weights = np.ones(K, dtype=float)
    else:
        weights = np.array(fairness_weights, dtype=float)

    # Diagonal utilities
    abs2 = np.abs(H) ** 2
    for m in range(M):
        for k in range(K):
            snr_lin = (P_ap_W * abs2[m, k]) / noise_W
            u_mk = np.log2(1.0 + snr_lin)
            Q[idx(m, k), idx(m, k)] += -weights[k] * float(u_mk)

    # Pairwise interference costs between different users
    for m in range(M):
        for k in range(K):
            i = idx(m, k)
            for mp in range(m, M):
                for kp in range(K):
                    j = idx(mp, kp)
                    if j <= i:  # avoid double counting and self
                        continue
                    if k == kp:
                        continue  # same user, do not penalize co-links (they are disallowed by AP cap)
                    # Standalone utilities
                    u_i = np.log2(1.0 + (P_ap_W * abs2[m, k]) / noise_W)
                    u_j = np.log2(1.0 + (P_ap_W * abs2[mp, kp]) / noise_W)
                    # Approximate MRT interference impact (pairwise)
                    interf_on_k = P_ap_W * abs2[mp, k]
                    r_i = np.log2(1.0 + (P_ap_W * abs2[m, k]) / (noise_W + interf_on_k))
                    interf_on_kp = P_ap_W * abs2[m, kp]
                    r_j = np.log2(1.0 + (P_ap_W * abs2[mp, kp]) / (noise_W + interf_on_kp))
                    pen = max(0.0, (u_i + u_j) - (r_i + r_j))
                    if pen > 0:
                        Q[i, j] += pen
                        Q[j, i] += pen

    # AP capacity: at most one user per AP
    for m in range(M):
        idxs = [idx(m, k) for k in range(K)]
        add_sum_equals_penalty(Q, idxs, target=1, lam=lam_ap)

    # Per-user diversity: exactly L APs per user
    for k in range(K):
        idxs = [idx(m, k) for m in range(M)]
        add_sum_equals_penalty(Q, idxs, target=L, lam=lam_user)

    return Q

# -------------------- QUBO Solver (Simulated Annealing) --------------------

def qubo_energy(Q, x):
    return float(x @ (Q @ x))

def cim_anneal(Q, steps=2000, T0=5.0, T1=0.1):
    """
    Simple simulated annealing over binary variables x in {0,1}^n for QUBO.
    """
    n = Q.shape[0]
    x = np.random.randint(0, 2, size=n).astype(np.int8)
    # Precompute row sums for fast delta E: dE = Q_ii*(1-2x_i) + 2(1-2x_i) * sum_{j != i} Q_ij x_j
    for t in range(steps):
        T = T0 * (T1 / T0) ** (t / max(1, steps - 1))
        # One pass with n random coordinates
        for _ in range(n):
            i = np.random.randint(0, n)
            xi = x[i]
            # compute delta energy if flipping x_i -> 1-x_i
            # Efficient computation:
            # E(x) = x^T Q x
            # E(x') - E(x) = (1-2xi) * (Q_ii + 2 sum_{j != i} Q_ij x_j)
            s = Q[i, i]
            if i > 0:
                s += 2.0 * np.dot(Q[i, :i], x[:i])
            if i + 1 < n:
                s += 2.0 * np.dot(Q[i, i + 1:], x[i + 1:])
            dE = (1 - 2 * xi) * s
            if dE < 0 or np.random.rand() < math.exp(-dE / max(T, 1e-12)):
                x[i] = 1 - xi
    return x

# -------------------- Feasibility Projection --------------------

def project_feasible(H, A, L):
    """
    Enforce: per AP, <=1 user; per user, == L APs.
    Strategy:
    - For AP rows with multiple 1s, keep the strongest |h_{mk}|^2 and clear others.
    - For users with fewer than L APs, add best available APs (by |h_{mk}|^2) not yet assigned.
    """
    M, K = H.shape
    A = A.copy().astype(int)

    # Per-AP capacity
    for m in range(M):
        row = A[m, :]
        if row.sum() > 1:
            # keep k* with largest |h_mk|^2
            strengths = np.abs(H[m, :]) ** 2
            k_star = int(np.argmax(strengths))
            A[m, :] = 0
            A[m, k_star] = 1

    # Per-user exact L APs
    # Build list of idle APs
    ap_idle = [m for m in range(M) if A[m, :].sum() == 0]
    for k in range(K):
        cur = int(A[:, k].sum())
        if cur < L:
            # rank idle APs for user k
            strengths = [(np.abs(H[m, k]) ** 2, m) for m in ap_idle]
            strengths.sort(reverse=True)
            needed = L - cur
            chosen = []
            for _, m in strengths:
                if needed == 0:
                    break
                # Assign AP m to user k
                A[m, k] = 1
                chosen.append(m)
                needed -= 1
            # update idle set
            ap_idle = [m for m in ap_idle if m not in chosen]
    return A

# -------------------- Precoding and Evaluation --------------------

def rzf_precoder(H_A, noise_W):
    """
    Compute unnormalized RZF and return per-AP normalized W.
    """
    M, K = H_A.shape
    G = H_A.T  # K x M
    xi = noise_W  # regularization proportional to noise
    # Compute W_un and handle potential numerical issues
    try:
        inv_term = np.linalg.inv(G @ G.conj().T + xi * np.eye(K))
    except np.linalg.LinAlgError:
        inv_term = np.linalg.pinv(G @ G.conj().T + xi * np.eye(K))
    W_un = G.conj().T @ inv_term  # M x K
    # Per-AP normalization to 1 W for each AP
    W = np.zeros_like(W_un, dtype=complex)
    for m in range(M):
        col = W_un[m, :]
        norm = np.linalg.norm(col)**2
        if norm > 0: W[m, :] = col / np.sqrt(norm) * np.sqrt(P_ap_W)
    return W

def evaluate_selection(H, A, noise_W):
    """
    - Design RZF from masked H_A = H * A (Hadamard product)
    - Evaluate rates using the full physical channel H
    """
    H_A = H * A
    W = rzf_precoder(H_A, noise_W)
    M, K = H.shape
    rates = np.zeros(K)
    G_true = H.T  # K x M
    for k in range(K):
        hk = G_true[k, :]
        sig = np.abs(hk @ W[:, k])**2
        interf = 0.0
        for j in range(K):
            if j == k:
                continue
            interf += np.abs(hk @ W[:, j])**2
        sinr = sig / (interf + noise_W)
        rates[k] = np.log2(1.0 + sinr)
    return rates, W

def jain_fairness(rates):
    s = np.sum(rates)
    if s <= 1e-12:
        return 0.0
    return (s ** 2) / (len(rates) * np.sum(rates ** 2))

# -------------------- Baselines --------------------

def baseline_random(H, L):
    M, K = H.shape
    A = np.zeros((M, K), dtype=int)
    ap_idle = set(range(M))
    # Assign L APs per user
    for k in range(K):
        choices = list(ap_idle)
        random.shuffle(choices)
        taken = 0
        for m in choices:
            if taken == L:
                break
            if A[m, :].sum() == 0:
                A[m, k] = 1
                ap_idle.remove(m)
                taken += 1
        # If not enough idle APs, greedily break capacity (rare); then fix:
        if A[:, k].sum() < L:
            remaining = L - int(A[:, k].sum())
            # pick strongest APs not already serving someone else
            strengths = sorted([(np.abs(H[m, k]) ** 2, m) for m in range(M) if A[m, :].sum() == 0],
                               reverse=True)
            for _, m in strengths[:remaining]:
                A[m, k] = 1
    # Ensure capacity: if violations exist, project
    A = project_feasible(H, A, L)
    return A

def baseline_max_cg(H, L):
    M, K = H.shape
    A = np.zeros((M, K), dtype=int)
    ap_idle = set(range(M))
    # For each user, pick top-L by |h|^2 among idle APs
    for k in range(K):
        strengths = sorted([(np.abs(H[m, k]) ** 2, m) for m in ap_idle], reverse=True)
        for _, m in strengths[:L]:
            A[m, k] = 1
            ap_idle.discard(m)
    A = project_feasible(H, A, L)
    return A

def baseline_greedy(H, L, objective="sum-rate", noise_W=1e-9):
    M, K = H.shape
    A = np.zeros((M, K), dtype=int)
    for _ in range(K * L):
        # Enumerate feasible links: AP m idle; user k has < L
        candidates = []
        for m in range(M):
            if A[m, :].sum() != 0:
                continue
            for k in range(K):
                if A[:, k].sum() >= L:
                    continue
                candidates.append((m, k))
        if not candidates:
            break
        # Current metric
        cur_rates, _ = evaluate_selection(H, A, noise_W)
        if objective == "sum-rate":
            cur_metric = float(np.sum(cur_rates))
        else:
            cur_metric = float(np.min(cur_rates)) if cur_rates.size > 0 else 0.0
        # Test best marginal gain
        best_gain, best_link = -1e18, None
        for (m, k) in candidates:
            A[m, k] = 1
            rates_new, _ = evaluate_selection(H, A, noise_W)
            if objective == "sum-rate":
                new_metric = float(np.sum(rates_new))
            else:
                new_metric = float(np.min(rates_new)) if rates_new.size > 0 else 0.0
            gain = new_metric - cur_metric
            if gain > best_gain:
                best_gain = gain
                best_link = (m, k)
            A[m, k] = 0
        if best_link is None:
            break
        A[best_link[0], best_link[1]] = 1
    A = project_feasible(H, A, L)
    return A


def main():
    """Run simulation and generate plots."""
    np.random.seed(42)
    M = 32  # Number of APs
    K_list = [4, 6, 8, 10, 12]  # Number of users to test
    L = 4  # Links per user
    noise_W = 1e-9
    num_seeds = 50
    
    results = {
        'qubo': {'sum_rate': [], 'fairness': []},
        'greedy': {'sum_rate': [], 'fairness': []},
        'random': {'sum_rate': [], 'fairness': []}
    }
    
    print("Running simulations...")
    for K in K_list:
        print(f"  K={K}...")
        qubo_rates, greedy_rates, random_rates = [], [], []
        
        for seed in range(num_seeds):
            np.random.seed(seed)
            H = (np.random.randn(M, K) + 1j * np.random.randn(M, K)) / np.sqrt(2)
            
            # QUBO method
            A_qubo = solve_qubo_selection(H, L)
            rates_qubo, _ = evaluate_selection(H, A_qubo, noise_W)
            qubo_rates.append(rates_qubo)
            
            # Greedy baseline
            A_greedy = baseline_greedy(H, L, objective="sum-rate", noise_W=noise_W)
            rates_greedy, _ = evaluate_selection(H, A_greedy, noise_W)
            greedy_rates.append(rates_greedy)
            
            # Random baseline
            A_random = baseline_random(H, L)
            rates_random, _ = evaluate_selection(H, A_random, noise_W)
            random_rates.append(rates_random)
        
        # Aggregate results
        results['qubo']['sum_rate'].append(np.mean([np.sum(r) for r in qubo_rates]))
        results['qubo']['fairness'].append(np.mean([jains_fairness(r) for r in qubo_rates]))
        results['greedy']['sum_rate'].append(np.mean([np.sum(r) for r in greedy_rates]))
        results['greedy']['fairness'].append(np.mean([jains_fairness(r) for r in greedy_rates]))
        results['random']['sum_rate'].append(np.mean([np.sum(r) for r in random_rates]))
        results['random']['fairness'].append(np.mean([jains_fairness(r) for r in random_rates]))
    
    # Generate plots
    import matplotlib.pyplot as plt
    plt.style.use('default')
    
    # Plot 1: Sum rate vs users
    plt.figure(figsize=(6, 4))
    plt.plot(K_list, results['qubo']['sum_rate'], 'o-', label='QUBO', linewidth=2)
    plt.plot(K_list, results['greedy']['sum_rate'], 's-', label='Greedy', linewidth=2)
    plt.plot(K_list, results['random']['sum_rate'], '^-', label='Random', linewidth=2)
    plt.xlabel('Number of Users (K)')
    plt.ylabel('Sum Spectral Efficiency (b/s/Hz)')
    plt.title('Sum Rate vs Number of Users')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sum_rate_vs_users.pdf', bbox_inches='tight')
    print("Saved sum_rate_vs_users.pdf")
    
    # Plot 2: Fairness vs users
    plt.figure(figsize=(6, 4))
    plt.plot(K_list, results['qubo']['fairness'], 'o-', label='QUBO', linewidth=2)
    plt.plot(K_list, results['greedy']['fairness'], 's-', label='Greedy', linewidth=2)
    plt.plot(K_list, results['random']['fairness'], '^-', label='Random', linewidth=2)
    plt.xlabel('Number of Users (K)')
    plt.ylabel("Jain's Fairness Index")
    plt.title('Fairness vs Number of Users')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fairness_vs_users.pdf', bbox_inches='tight')
    print("Saved fairness_vs_users.pdf")
    
    # Plot 3: Tradeoff plot
    plt.figure(figsize=(6, 4))
    plt.scatter(results['qubo']['fairness'], results['qubo']['sum_rate'], 
                s=100, label='QUBO', alpha=0.7)
    plt.scatter(results['greedy']['fairness'], results['greedy']['sum_rate'], 
                s=100, label='Greedy', alpha=0.7)
    plt.scatter(results['random']['fairness'], results['random']['sum_rate'], 
                s=100, label='Random', alpha=0.7)
    plt.xlabel("Jain's Fairness Index")
    plt.ylabel('Sum Spectral Efficiency (b/s/Hz)')
    plt.title('Sum Rate vs Fairness Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tradeoff_plot.pdf', bbox_inches='tight')
    print("Saved tradeoff_plot.pdf")
    
    print("\nSimulation complete!")
    print(f"Average QUBO sum rate: {np.mean(results['qubo']['sum_rate']):.4f} b/s/Hz")
    print(f"Average QUBO fairness: {np.mean(results['qubo']['fairness']):.4f}")


if __name__ == "__main__":
    main()