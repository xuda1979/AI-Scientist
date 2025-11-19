#!/usr/bin/env python3
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def hilbert_transform_periodic(y):
    import numpy as np
    N = len(y)
    Y = np.fft.fft(y)
    k = np.fft.fftfreq(N)
    H = -1j * np.sign(k)
    H[0] = 0.0
    yH = np.fft.ifft(Y * H).real
    return yH

def simulate_kernel_and_kk(T=200.0, dt=0.01, seed=7):
    rng = np.random.default_rng(seed)
    A = np.array([1.0, 0.6, 0.35])
    gamma = np.array([0.08, 0.04, 0.02])
    omega0 = np.array([0.5, 1.2, 2.0])
    t = np.arange(0, T, dt)
    K = np.zeros_like(t)
    for Aj, gj, wj in zip(A, gamma, omega0):
        K += Aj * np.exp(-gj * t) * np.cos(wj * t)
    pad = 4
    Nt = len(t) * pad
    dt_eff = dt
    K_padded = np.zeros(Nt)
    K_padded[:len(K)] = K
    Xi = dt_eff * np.fft.fft(K_padded)
    freq = 2 * np.pi * np.fft.fftfreq(Nt, d=dt_eff)
    Xi = np.fft.fftshift(Xi)
    freq = np.fft.fftshift(freq)
    Re = Xi.real
    Im = Xi.imag
    Re_from_H = hilbert_transform_periodic(Im)
    Im_from_H = -hilbert_transform_periodic(Re)
    residual_Re = Re - Re_from_H
    residual_Im = Im - Im_from_H
    band = (np.abs(freq) < 4.0)
    def metrics(arr):
        sel = arr[band]
        return {"l2": float(np.linalg.norm(sel)),
                "linf": float(np.max(np.abs(sel))),
                "mean_abs": float(np.mean(np.abs(sel)))}
    m_re = metrics(residual_Re)
    m_im = metrics(residual_Im)
    results = {
        "omega": freq.tolist(),
        "Re": Re.tolist(),
        "Im": Im.tolist(),
        "Re_from_hilbert": Re_from_H.tolist(),
        "Im_from_hilbert": Im_from_H.tolist(),
        "residual_Re": residual_Re.tolist(),
        "residual_Im": residual_Im.tolist(),
        "metrics": {"band_abs_omega_lt_4": {"residual_Re": m_re, "residual_Im": m_im}},
        "kernel_timegrid": t.tolist(),
        "kernel_values": K.tolist(),
        "params": {"A": A.tolist(), "gamma": gamma.tolist(), "omega0": omega0.tolist(), "T": T, "dt": dt, "pad": pad}
    }
    return results

def plot_kk_results(res, out_dir):
    import numpy as np, os, matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    omega = np.array(res["omega"])
    Re = np.array(res["Re"])
    ReH = np.array(res["Re_from_hilbert"])
    plt.figure()
    plt.plot(omega, Re, label="Re")
    plt.plot(omega, ReH, label="Hilbert(Im)")
    plt.xlabel(r"$\omega$"); plt.ylabel(r"$\Re\,\Xi^R(\omega)$"); plt.legend()
    plt.title("KK check: Re vs Hilbert(Im)")
    plt.savefig(os.path.join(out_dir, "kk_re_vs_hilbert.pdf"), bbox_inches="tight")
    plt.close()
    residual_Re = np.array(res["residual_Re"])
    plt.figure()
    plt.plot(omega, residual_Re, label="Residual Re - H(Im)")
    plt.xlabel(r"$\omega$"); plt.ylabel("Residual"); plt.legend()
    plt.title("KK residuals (central band is most reliable)")
    plt.savefig(os.path.join(out_dir, "kk_residuals.pdf"), bbox_inches="tight")
    plt.close()

def simulate_identifiability(N=4000, r=40, lam=1e-3, seed=11):
    import numpy as np
    rng = np.random.default_rng(seed)
    k_idx = np.arange(r)
    kappa_true = (0.9**k_idx) * np.cos(0.3 * k_idx)
    x = rng.standard_normal(N + r)
    y = np.zeros(N)
    for t in range(N):
        y[t] = np.dot(kappa_true, x[t:t+r][::-1])
    y += 0.05 * rng.standard_normal(N)
    X = np.zeros((N - r, r))
    y_eff = y[r:]
    for t in range(N - r):
        X[t, :] = x[t:t+r][::-1]
    XtX = X.T @ X
    beta_hat = np.linalg.solve(XtX + lam * np.eye(r), X.T @ y_eff)
    mse = float(np.mean((y_eff - X @ beta_hat)**2))
    nmse = float(mse / np.var(y_eff))
    Xf = np.fft.fft(x[:N])
    Sxx = (Xf * np.conj(Xf)).real + 1e-12
    cond_XtX = float(np.linalg.cond(XtX))
    cond_spec = float(np.max(Sxx) / np.min(Sxx))
    return {"r": int(r), "lambda": float(lam), "N": int(N),
            "cond_XtX": cond_XtX, "cond_spectrum": cond_spec,
            "mse": mse, "nmse": nmse,
            "kappa_true": kappa_true.tolist(), "kappa_est": beta_hat.tolist()}

def plot_identifiability(report, out_dir):
    import numpy as np, os, matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    k_true = np.array(report["kappa_true"])
    k_est = np.array(report["kappa_est"])
    k_idx = np.arange(len(k_true))
    plt.figure()
    plt.plot(k_idx, k_true, label="true")
    plt.plot(k_idx, k_est, label="est")
    plt.xlabel("lag index"); plt.ylabel("kernel coeff."); plt.legend()
    plt.title("Finite-memory kernel: true vs estimated")
    plt.savefig(os.path.join(out_dir, "identifiability_true_vs_est_kernel.pdf"), bbox_inches="tight")
    plt.close()

def main(out_base=".", fig_subdir="fig"):
    figdir = os.path.join(out_base, fig_subdir)
    os.makedirs(figdir, exist_ok=True)
    # KK
    res = simulate_kernel_and_kk()
    with open(os.path.join(out_base, "kk_checks.json"), "w") as f:
        json.dump(res, f)
    plot_kk_results(res, figdir)
    # Identifiability
    rep = simulate_identifiability()
    with open(os.path.join(out_base, "identifiability_report.json"), "w") as f:
        json.dump(rep, f)
    plot_identifiability(rep, figdir)

if __name__ == "__main__":
    main()
