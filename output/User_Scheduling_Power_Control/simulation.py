#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation for: Interference-Aware Joint User Scheduling & Discrete Power Control via QUBO + CIM
Author: Anonymous
Reproducible, single-file. No external dependencies beyond numpy.
"""
import argparse, math, random, sys, time
import numpy as np

def db2lin(x_db): return 10**(x_db/10.0)
def lin2db(x): return 10*np.log10(x + 1e-30)

def seed_all(s):
    random.seed(s); np.random.seed(s)

def generate_layout(B=3, users_per_cell=6, R=5, area_km=2.0, shadow_std_db=6.0, fc_GHz=3.5):
    cell_xy = np.random.uniform(0, area_km, size=(B,2))
    user_xy = []
    gains = {}
    for b in range(B):
        Kb = users_per_cell
        u = np.random.uniform(0, area_km, size=(Kb,2))
        user_xy.append(u)
    c = 3e8; fc = fc_GHz * 1e9
    for b in range(B):
        gains[b] = {"to": {}}
        for bb in range(B):
            Kb = users_per_cell
            for k in range(Kb):
                d_km = np.linalg.norm(cell_xy[b]-user_xy[bb][k])
                d = max(d_km*1000.0, 10.0)
                pl_db = 32.4 + 20*np.log10(fc/1e9) + 31.9*np.log10(d/1.0)
                sh = np.random.normal(0.0, shadow_std_db)
                g_lin = db2lin(- (pl_db + sh))
                gains[b]["to"][(bb,k)] = g_lin
    return cell_xy, user_xy, gains

def add_onehot_penalty(Q, idxs, lam):
    for i in idxs: Q[i,i] += -lam
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            Q[idxs[i], idxs[j]] += 2*lam

def add_power_budget_penalty(Q, idxs, powers, Pmax, lam):
    for a, i in enumerate(idxs):
        p_i = powers[a]
        Q[i,i] += lam * (p_i**2 - 2*Pmax*p_i)
    for a in range(len(idxs)):
        for b in range(a+1, len(idxs)):
            Q[idxs[a], idxs[b]] += lam * (2 * powers[a]*powers[b])

def qubo_energy(Q, x):
    xx = x.astype(np.float64)
    return float(xx @ Q @ xx)

def qubo_to_ising(Q):
    n = Q.shape[0]
    Qsym = np.triu(Q) + np.triu(Q,1).T
    J = -0.5 * Qsym
    h = -0.5 * (Qsym @ np.ones(n))
    return J, h

def ising_energy(J, h, s):
    return float(-0.5 * s @ (J @ s) - h @ s)

def cim_anneal(J, h, steps=2000, T0=5.0, T1=0.1, sweep=1):
    n = J.shape[0]
    s = np.random.choice([-1,1], size=n)
    for t in range(steps):
        T = T0 * (T1/T0)**(t/max(1,steps-1))
        for _ in range(sweep*n):
            i = np.random.randint(0, n)
            dE = 2.0 * s[i] * ( (J[i,:] @ s) + h[i] )
            if dE <= 0.0 or np.random.rand() < math.exp(-dE/max(T,1e-9)):
                s[i] *= -1
    E = ising_energy(J,h,s)
    return s, E

def build_qubo_user_sched(B=3, K=6, R=5, L=3, Pmax_W=10.0, W_r=180e3, noise_figure_db=7.0, seed=1):
    seed_all(seed)
    cell_xy, user_xy, gains = generate_layout(B=B, users_per_cell=K, R=R)
    P_levels_W = np.array([0.0, 0.1, 0.3, 1.0])[:L+1]
    var_index = {}; rev_index = {}; idx = 0
    for b in range(B):
        for r in range(R):
            for k in range(K):
                for l in range(L+1):
                    var_index[(b,k,r,l)] = idx; rev_index[idx]=(b,k,r,l); idx+=1
    n = idx
    Q = np.zeros((n,n), dtype=np.float64)
    N0_dbm_perHz = -174.0; NF_db = noise_figure_db
    N0_W = 10**((N0_dbm_perHz - 30.0)/10.0)
    N_r_W = N0_W * W_r * 10**(NF_db/10.0)

    for b in range(B):
        for r in range(R):
            for k in range(K):
                g_sig = gains[b]["to"][(b,k)]
                for l in range(L+1):
                    idx_i = var_index[(b,k,r,l)]
                    P = P_levels_W[l]
                    util = 0.0 if P<=0 else W_r * math.log2(1.0 + g_sig*P/N_r_W)
                    Q[idx_i, idx_i] += -util

    for r in range(R):
        for b in range(B):
            for k in range(K):
                for l in range(1, L+1):
                    i = var_index[(b,k,r,l)]
                    g_sig_i = gains[b]["to"][(b,k)]; P_i = P_levels_W[l]
                    util_i = W_r * math.log2(1.0 + g_sig_i*P_i/N_r_W)
                    for bp in range(b+1, B):
                        for kp in range(K):
                            for lp in range(1, L+1):
                                j = var_index[(bp,kp,r,lp)]
                                g_sig_j = gains[bp]["to"][(bp,kp)]; P_j = P_levels_W[lp]
                                util_j = W_r * math.log2(1.0 + g_sig_j*P_j/N_r_W)
                                g_j_to_i = gains[bp]["to"][(b,k)]
                                g_i_to_j = gains[b]["to"][(bp,kp)]
                                r_i = W_r * math.log2(1.0 + g_sig_i*P_i/(N_r_W + g_j_to_i*P_j))
                                r_j = W_r * math.log2(1.0 + g_sig_j*P_j/(N_r_W + g_i_to_j*P_i))
                                penalty = max(0.0, (util_i + util_j - (r_i + r_j)))
                                if penalty>0: Q[i,j] += penalty

    lam_onehot = 10.0 * W_r
    for b in range(B):
        for r in range(R):
            idxs = [var_index[(b,k,r,l)] for k in range(K) for l in range(L+1)]
            for i in idxs: Q[i,i] += -lam_onehot
            for a in range(len(idxs)):
                for c in range(a+1, len(idxs)):
                    Q[idxs[a], idxs[c]] += 2*lam_onehot

    lam_pow = 1.0 * W_r
    for b in range(B):
        idxs = []; pows = []
        for r in range(R):
            for k in range(K):
                for l in range(L+1):
                    idxs.append(var_index[(b,k,r,l)]); pows.append(P_levels_W[l])
        for a,i in enumerate(idxs):
            p_i = pows[a]; Q[i,i] += lam_pow * (p_i**2 - 2*Pmax_W*p_i)
        for a in range(len(idxs)):
            for c in range(a+1, len(idxs)):
                Q[idxs[a], idxs[c]] += lam_pow * (2 * pows[a]*pows[c])

    return Q, var_index, rev_index, (cell_xy, user_xy, gains, P_levels_W, N_r_W, W_r, Pmax_W, (B,K,R,L))

def solve_and_evaluate(Q, var_index, rev_index, context, steps=1500, seed=1):
    (cell_xy, user_xy, gains, P_levels_W, N_r_W, W_r, Pmax_W, dims) = context
    B,K,R,L = dims
    J,h = qubo_to_ising(Q)
    s, E = cim_anneal(J,h,steps=steps,T0=5.0,T1=0.05,sweep=1)
    x = (s > 0).astype(np.int8)

    sum_rate = 0.0
    per_cell_power = np.zeros(B)
    for b in range(B):
        for r in range(R):
            chosen = [(k,l) for k in range(K) for l in range(L+1) if x[var_index[(b,k,r,l)]]==1]
            if len(chosen)==0: continue
            if len(chosen)>1:
                chosen.sort(key=lambda t: P_levels_W[t[1]], reverse=True)
            k,l = chosen[0]
            per_cell_power[b] += P_levels_W[l]

    for r in range(R):
        act = []
        for b in range(B):
            ks = [(k,l) for k in range(K) for l in range(L+1) if x[var_index[(b,k,r,l)]]==1]
            if not ks: continue
            k,l = ks[0]
            if P_levels_W[l] <= 0: continue
            act.append((b,k,l))
        for (b,k,l) in act:
            num = gains[b]["to"][(b,k)] * P_levels_W[l]
            den = N_r_W
            for (bp,kp,lp) in act:
                if bp==b: continue
                den += gains[bp]["to"][(b,k)] * P_levels_W[lp]
            sinr = num / max(den,1e-15)
            sum_rate += W_r * math.log2(1.0 + sinr)

    feasible = np.all(per_cell_power <= (Pmax_W+1e-6))
    return x, float(sum_rate), feasible, per_cell_power

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=3)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--Pmax", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    Q, var_index, rev_index, context = build_qubo_user_sched(
        B=args.B, K=args.K, R=args.R, L=args.L, Pmax_W=args.Pmax, seed=args.seed
    )
    x, rate, feasible, pcell = solve_and_evaluate(Q, var_index, rev_index, context, steps=args.steps, seed=args.seed)
    B,K,R,L = context[-1]
    print("=== Joint Scheduling & Power Control (QUBO + CIM) ===")
    print(f"Cells={B}, Users/Cell={K}, RBs={R}, Power levels={L}+1(with 0)")
    print(f"Total sum-rate: {rate/1e6:.3f} Mb/s")
    print(f"Per-cell power usage: {pcell} W (budget {context[-2]} W) -> feasible={feasible}")
    for b in range(B):
        print(f"Cell {b}:")
        for r in range(R):
            cand = [(k,l) for k in range(K) for l in range(L+1) if x[var_index[(b,k,r,l)]]==1]
            if not cand: print(f"  RB{r}: idle")
            else:
                k,l = cand[0]
                print(f"  RB{r}: user={k}, P={context[3][l]:.2f} W")
    return 0

if __name__ == "__main__":
    sys.exit(main())
