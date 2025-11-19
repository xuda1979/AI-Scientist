import numpy as np
import json
import os
import time
import argparse
import matplotlib.pyplot as plt

"""
Hardware-matched VQC for Max-Cut on 5 qubits with baselines and ablations.

Features:
- Two-layer rotation–entangler–rotation ansatz with ring/all-to-all/star entanglers.
- Parameter-shift gradients and gradient descent optimizer.
- Exact expectation evaluation from state vector (n=5), with optional shot-based estimates.
- Classical baselines: random, greedy local search, simulated annealing.
- QAOA p=1 comparator (expectation from state vector).
- Multi-seed runs with summary statistics and aggregate convergence plots.
- Deterministic seeding and CLI configurability.
"""

# -----------------------------
# Graph definition (five-node)
# -----------------------------
n = 5
edges = [(0, 1), (0, 2), (0, 4), (1, 2), (2, 3), (3, 4)]
weights = {e: 1.0 for e in edges}

# -----------------------------
# Utilities
# -----------------------------
def basis_bits(nbits: int) -> np.ndarray:
    """Return an array of shape (2^n, n) of bit representations of basis states (LSB = qubit 0)."""
    N = 1 << nbits
    bits = np.zeros((N, nbits), dtype=np.int8)
    for i in range(N):
        for q in range(nbits):
            bits[i, q] = (i >> q) & 1
    return bits


BITS = basis_bits(n)


def Rx(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def Ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2.0), 0], [0, np.exp(1j * theta / 2.0)]], dtype=complex)


CZ = np.diag([1, 1, 1, -1]).astype(complex)


def apply_single(state: np.ndarray, gate: np.ndarray, q: int, nbits: int) -> np.ndarray:
    """Apply a 1-qubit gate on qubit q to the state vector."""
    tensor = state.reshape([2] * nbits)
    axes = list(range(nbits))
    axes_without = [a for a in axes if a != q]
    perm = axes_without + [q]
    inv_perm = np.argsort(perm)
    t = np.transpose(tensor, axes=perm)
    t2 = t.reshape((-1, 2))
    t2 = (t2 @ gate.T)
    t_new = t2.reshape(t.shape)
    t_back = np.transpose(t_new, axes=inv_perm)
    return t_back.reshape(-1)


def apply_two_qubit(state: np.ndarray, gate4: np.ndarray, q: int, r: int, nbits: int) -> np.ndarray:
    """Apply a 2-qubit gate on qubits (q, r) to the state vector."""
    tensor = state.reshape([2] * nbits)
    axes = list(range(nbits))
    axes_without = [a for a in axes if a not in (q, r)]
    perm = axes_without + [q, r]
    inv_perm = np.argsort(perm)
    t = np.transpose(tensor, axes=perm)
    m = int(np.prod(t.shape[:-2])) if t.ndim > 2 else 1
    t2 = t.reshape((m, 4))
    t2 = (t2 @ gate4.T)
    t_new = t2.reshape(t.shape)
    t_back = np.transpose(t_new, axes=inv_perm)
    return t_back.reshape(-1)


# -----------------------------
# Entanglers
# -----------------------------
def entangle_ring(state: np.ndarray, nbits: int) -> np.ndarray:
    for i in range(nbits):
        state = apply_two_qubit(state, CZ, i, (i + 1) % nbits, nbits)
    return state


def entangle_all_to_all(state: np.ndarray, nbits: int) -> np.ndarray:
    for i in range(nbits):
        for j in range(i + 1, nbits):
            state = apply_two_qubit(state, CZ, i, j, nbits)
    return state


def entangle_star(state: np.ndarray, nbits: int, center: int = 0) -> np.ndarray:
    for j in range(nbits):
        if j != center:
            state = apply_two_qubit(state, CZ, center, j, nbits)
    return state


def apply_entangler(state: np.ndarray, kind: str = "ring", nbits: int = 5) -> np.ndarray:
    if kind == "ring":
        return entangle_ring(state, nbits)
    elif kind == "all_to_all":
        return entangle_all_to_all(state, nbits)
    elif kind == "star":
        return entangle_star(state, nbits, center=0)
    else:
        raise ValueError(f"Unknown entangler kind: {kind}")


# -----------------------------
# State preparation
# params shape: (L, n, 2) for L rotation layers; each qubit has [Ry, Rz]
# -----------------------------
def prepare_state(params: np.ndarray, entangler_kind: str = "ring") -> np.ndarray:
    """Prepare the state vector with L rotation layers and a single entangler between them."""
    L = params.shape[0]
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0
    for ell in range(L):
        for q in range(n):
            state = apply_single(state, Ry(params[ell, q, 0]), q, n)
            state = apply_single(state, Rz(params[ell, q, 1]), q, n)
        # Entangle between rotation layers. If L==1 we still add an entangler once.
        if ell < L - 1 or (L == 1):
            state = apply_entangler(state, entangler_kind, n)
    return state


# -----------------------------
# Cost and gradients
# -----------------------------
def maxcut_cost_from_probs(probs: np.ndarray) -> float:
    """Compute Max-Cut cost from basis-state probabilities for the 5-node graph."""
    e = 0.0
    for (i, j), w in weights.items():
        zzij = np.sum(probs * ((-1.0) ** (BITS[:, i] ^ BITS[:, j])))
        e += w * (1.0 - zzij) / 2.0
    return float(np.real(e))


def maxcut_cost(params: np.ndarray, entangler_kind: str = "ring") -> float:
    st = prepare_state(params, entangler_kind=entangler_kind)
    probs = np.abs(st) ** 2
    return maxcut_cost_from_probs(probs)


def loss(params: np.ndarray, entangler_kind: str = "ring") -> float:
    return -maxcut_cost(params, entangler_kind=entangler_kind)


def param_shift_grad(params: np.ndarray, entangler_kind: str = "ring", shift: float = np.pi / 2) -> np.ndarray:
    """Parameter-shift gradient for all parameters in params."""
    grad = np.zeros_like(params)
    for idx, _ in np.ndenumerate(params):
        params[idx] += shift
        eplus = loss(params, entangler_kind=entangler_kind)
        params[idx] -= 2 * shift
        eminus = loss(params, entangler_kind=entangler_kind)
        params[idx] += shift
        grad[idx] = 0.5 * (eplus - eminus)
    return grad


# -----------------------------
# Optional shot-based energy estimator
# -----------------------------
def sample_bitstrings(probs: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Sample bitstrings according to probs and return empirical probabilities."""
    idxs = rng.choice(len(probs), size=shots, p=probs, replace=True)
    counts = np.bincount(idxs, minlength=len(probs)).astype(float)
    return counts / shots


def estimate_energy_from_shots(probs: np.ndarray, shots: int, rng: np.random.Generator) -> float:
    """Estimate Max-Cut energy using shot sampling."""
    emp = sample_bitstrings(probs, shots, rng)
    return maxcut_cost_from_probs(emp)


# -----------------------------
# Baselines
# -----------------------------
def cut_value(bitstr: int) -> float:
    val = 0.0
    for (i, j), w in weights.items():
        if ((bitstr >> i) & 1) ^ ((bitstr >> j) & 1):
            val += w
    return val


def random_cut(rng: np.random.Generator) -> int:
    x = 0
    for q in range(n):
        if rng.random() < 0.5:
            x |= (1 << q)
    return x


def greedy_local_search(rng: np.random.Generator, restarts: int = 10) -> int:
    best = -1
    best_val = -1.0
    for _ in range(restarts):
        x = random_cut(rng)
        improved = True
        while improved:
            improved = False
            for q in range(n):
                y = x ^ (1 << q)
                if cut_value(y) > cut_value(x):
                    x = y
                    improved = True
        v = cut_value(x)
        if v > best_val:
            best_val = v
            best = x
    return best


def simulated_annealing(
    rng: np.random.Generator, iters: int = 2000, T0: float = 1.0, alpha: float = 0.995
) -> int:
    x = random_cut(rng)
    v = cut_value(x)
    T = T0
    for _ in range(iters):
        q = rng.integers(0, n)
        y = x ^ (1 << q)
        vy = cut_value(y)
        if vy >= v or rng.random() < np.exp((vy - v) / max(1e-9, T)):
            x, v = y, vy
        T *= alpha
    return x


def qaoa_p1_expectation(gamma: float, beta: float) -> float:
    """Compute QAOA p=1 expected Max-Cut value via state-vector."""
    st = np.ones(1 << n, dtype=complex) / np.sqrt(1 << n)  # |+>^n
    # Apply cost unitary (diagonal phases)
    phases = np.zeros(1 << n, dtype=complex)
    for x in range(1 << n):
        z = 1 - 2 * BITS[x]  # z_i in {+1,-1}
        phase_sum = 0.0
        for (i, j), w in weights.items():
            phase_sum += w * (1 - z[i] * z[j]) / 2.0
        phases[x] = np.exp(-1j * gamma * phase_sum)
    st = st * phases
    # Mixer U_B(beta) = \bigotimes_i R_x(2 beta)
    for q in range(n):
        st = apply_single(st, Rx(2 * beta), q, n)
    probs = np.abs(st) ** 2
    return maxcut_cost_from_probs(probs)


# -----------------------------
# Statistics helpers
# -----------------------------
def summarize_runs(results_list, key):
    arr = np.array([r[key] for r in results_list], dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def paired_t_test(xs, ys):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    if len(xs) != len(ys):
        raise ValueError("Paired samples must have equal length")
    d = xs - ys
    n_s = len(d)
    if n_s < 2:
        return np.nan, np.nan
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return np.inf, 0.0
    t = mean / (sd / np.sqrt(n_s))
    # Two-sided normal-tail approximation (no SciPy dependency)
    p_approx = float(2.0 * np.exp(-0.5 * t * t))
    return t, p_approx


def cohen_d(xs, ys):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    d = xs - ys
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return np.inf
    return mean / sd


def holm_bonferroni(p_values):
    """Return Holm–Bonferroni adjusted boolean rejections given p-values list and alpha=0.05."""
    alpha = 0.05
    m = len(p_values)
    order = np.argsort(p_values)
    adj = [False] * m
    for k, idx in enumerate(order, start=1):
        thresh = alpha / (m - k + 1)
        if p_values[idx] <= thresh:
            adj[idx] = True
        else:
            break
    return adj


def ci95(arr):
    arr = np.array(arr, dtype=float)
    n_s = len(arr)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    halfwidth = 1.96 * (sd / np.sqrt(n_s)) if n_s > 1 else 0.0
    return (mean - halfwidth, mean + halfwidth)


# -----------------------------
# Training and evaluation
# -----------------------------
def bitstring(i: int, nbits: int) -> str:
    return format(i, "0{}b".format(nbits))


def run_vqc(
    seed: int = 42,
    iters: int = 180,
    lr: float = 0.15,
    entangler_kind: str = "ring",
    layers: int = 2,
    out_dir: str = ".",
    record_plots: bool = True,
    use_shots: bool = False,
    shots: int = 0,
):
    rng = np.random.default_rng(seed)
    params = rng.normal(0, 0.3, size=(layers, n, 2))

    history = []
    best_loss = 1e9
    best_params = params.copy()

    start = time.time()
    for _ in range(iters):
        # Exact or shot-based energy for current params
        st = prepare_state(params, entangler_kind=entangler_kind)
        probs = np.abs(st) ** 2
        if use_shots and shots > 0:
            E = estimate_energy_from_shots(probs, shots, rng)
        else:
            E = maxcut_cost_from_probs(probs)
        L = -E
        history.append(E)
        if L < best_loss:
            best_loss = L
            best_params = params.copy()

        # Gradient step (uses exact shifts; for hardware, use_shots would be applied per shift)
        g = param_shift_grad(params, entangler_kind=entangler_kind)
        params = params - lr * g
        # Angle wrapping to (-pi, pi]
        params = (params + np.pi) % (2 * np.pi) - np.pi
    elapsed = time.time() - start

    # Final evaluation at best params
    st = prepare_state(best_params, entangler_kind=entangler_kind)
    probs = np.abs(st) ** 2
    maxcut_value = maxcut_cost_from_probs(probs)

    # Four optimal bitstrings in q4..q0 convention
    target_strings = ["10100", "10110", "01001", "01011"]
    target_mass = float(sum(probs[int(s, 2)] for s in target_strings))

    if record_plots:
        os.makedirs(out_dir, exist_ok=True)
        # Convergence plot
        plt.figure()
        plt.plot(history)
        plt.xlabel("Iteration")
        plt.ylabel("Max-Cut cost")
        plt.title(f"Convergence ({entangler_kind}, seed={seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"energy_{entangler_kind}_seed{seed}.png"), dpi=150)
        plt.close()

        # Top probabilities
        idx_sorted = np.argsort(-probs)[:8]
        labels = [bitstring(i, n) for i in idx_sorted]
        values = probs[idx_sorted]
        plt.figure()
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), labels, rotation=45, ha="right")
        plt.ylabel("Probability")
        plt.title("Top basis states (|q4 q3 q2 q1 q0⟩)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"state_probs_{entangler_kind}_seed{seed}.png"), dpi=150)
        plt.close()

    result = {
        "seed": seed,
        "iterations": iters,
        "learning_rate": lr,
        "best_loss": float(best_loss),
        "best_cost": float(-best_loss),
        "final_cost_from_probs": float(maxcut_value),
        "target_strings": target_strings,
        "target_mass": float(target_mass),
        "edges": edges,
        "entangler_kind": entangler_kind,
        "layers": layers,
        "elapsed_sec": elapsed,
        "use_shots": use_shots,
        "shots": shots,
    }
    return result, history, probs


def aggregate_energy_plot(energies, out_path):
    # energies: list of lists (per-seed trace)
    L = min(len(e) for e in energies)
    A = np.array([e[:L] for e in energies], dtype=float)
    mean = A.mean(axis=0)
    std = A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(mean)
    iters = np.arange(L)
    plt.figure()
    plt.plot(iters, mean, label="mean")
    plt.fill_between(iters, mean - std, mean + std, color="C0", alpha=0.2, label="±1 std")
    plt.xlabel("Iteration")
    plt.ylabel("Max-Cut cost")
    plt.title("Multi-seed convergence (mean ± std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_experiment(seed=42, iters=180, lr=0.15, out_dir=".", entangler_kind="ring"):
    res, history, probs = run_vqc(
        seed=seed, iters=iters, lr=lr, out_dir=out_dir, entangler_kind=entangler_kind, layers=2, record_plots=True
    )

    print("=== Variational Max-Cut on 5 qubits ===")
    print(f"Entangler: {entangler_kind}, Iterations: {iters}, Learning rate: {lr}, Seed: {seed}")
    print(f"Best Max-Cut cost reached: {res['best_cost']:.6f}")
    print(f"Probability mass on the four optimal strings: {res['target_mass']:.4f}")
    print("Edges:", edges)

    idx_sorted = np.argsort(-probs)[:8]
    print("Top 8 states:")
    for i in idx_sorted:
        print(f"  {bitstring(i, n)} : {probs[i]:.6f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)

    with open(os.path.join(out_dir, "energy_trace.txt"), "w") as f:
        for v in history:
            f.write(f"{v:.10f}\n")

    return res


def run_multi_seed(
    seeds,
    method="vqc",
    entangler_kind="ring",
    qaoa_angles=(0.8, 0.6),
    out_dir=".",
    iters=180,
    lr=0.15,
    use_shots=False,
    shots=0,
):
    results = []
    energies = []
    rng_for_classical = np.random.default_rng(123)

    os.makedirs(out_dir, exist_ok=True)

    for s in seeds:
        if method == "vqc":
            res, hist, _ = run_vqc(
                seed=s,
                iters=iters,
                lr=lr,
                entangler_kind=entangler_kind,
                layers=2,
                out_dir=out_dir,
                record_plots=False,
                use_shots=use_shots,
                shots=shots,
            )
            results.append(res)
            energies.append(hist)
        elif method == "qaoa_p1":
            gamma, beta = qaoa_angles
            val = qaoa_p1_expectation(gamma=gamma, beta=beta)
            res = {
                "seed": s,
                "best_cost": float(val),
                "final_cost_from_probs": float(val),
                "target_mass": 0.0,
                "method": "qaoa_p1",
            }
            results.append(res)
        elif method == "greedy":
            x = greedy_local_search(rng_for_classical)
            v = cut_value(x)
            res = {
                "seed": s,
                "best_cost": float(v),
                "final_cost_from_probs": float(v),
                "target_mass": float(1.0 if v == 5.0 else 0.0),
                "method": "greedy",
            }
            results.append(res)
        elif method == "sa":
            x = simulated_annealing(rng_for_classical)
            v = cut_value(x)
            res = {
                "seed": s,
                "best_cost": float(v),
                "final_cost_from_probs": float(v),
                "target_mass": float(1.0 if v == 5.0 else 0.0),
                "method": "sa",
            }
            results.append(res)
        elif method == "random":
            x = random_cut(rng_for_classical)
            v = cut_value(x)
            res = {
                "seed": s,
                "best_cost": float(v),
                "final_cost_from_probs": float(v),
                "target_mass": float(1.0 if v == 5.0 else 0.0),
                "method": "random",
            }
            results.append(res)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Save raw results
    with open(os.path.join(out_dir, f"{method}_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Aggregate plot for VQC
    if method == "vqc" and len(energies) > 0:
        aggregate_energy_plot(energies, os.path.join(out_dir, f"energy_multi_seed_{entangler_kind}.png"))

    # Print summary with CI
    vals = [r.get("final_cost_from_probs", r.get("best_cost", np.nan)) for r in results]
    lo, hi = ci95(vals)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    print(f"[{method}] over {len(results)} seeds: Max-Cut value = {mean:.4f} ± {std:.4f} (95% CI [{lo:.4f},{hi:.4f}])")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hardware-matched VQC for Max-Cut on 5 qubits with baselines and ablations."
    )
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multi"], help="single run or multi-seed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(1, 21)))
    parser.add_argument("--iters", type=int, default=180)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--entangler", type=str, default="ring", choices=["ring", "all_to_all", "star"])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--method", type=str, default="vqc", choices=["vqc", "qaoa_p1", "greedy", "sa", "random"])
    parser.add_argument("--qaoa_gamma", type=float, default=0.8)
    parser.add_argument("--qaoa_beta", type=float, default=0.6)
    parser.add_argument("--use_shots", action="store_true", help="Use shot-based energy estimates in VQC training")
    parser.add_argument("--shots", type=int, default=0, help="Number of shots per energy estimate if use_shots is set")
    args = parser.parse_args()

    if args.mode == "single":
        run_experiment(seed=args.seed, iters=args.iters, lr=args.lr, out_dir=args.out_dir, entangler_kind=args.entangler)
    else:
        res = run_multi_seed(
            seeds=args.seeds,
            method=args.method,
            entangler_kind=args.entangler,
            out_dir=args.out_dir,
            iters=args.iters,
            lr=args.lr,
            qaoa_angles=(args.qaoa_gamma, args.qaoa_beta),
            use_shots=args.use_shots,
            shots=args.shots,
        )
        vals = [r.get("final_cost_from_probs", r.get("best_cost", np.nan)) for r in res]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        print(f"[{args.method}] over {len(res)} seeds: Max-Cut value = {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()