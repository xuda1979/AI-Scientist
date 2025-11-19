
from __future__ import annotations
import math
from typing import List, Tuple

def ci95_from_mean_sd_n(mean: float, sd: float, n: int) -> Tuple[float, float]:
    if n <= 1:
        return (mean, mean)
    # Normal approx for simplicity (n is typically 50 in synthetic)
    err = 1.96 * (sd / math.sqrt(n))
    return (mean - err, mean + err)

def mvc_from_series(budgets: List[int], means: List[float], variances: List[float]) -> Tuple[List[float], List[Tuple[float,float]]]:
    """Finite differences for MVC and delta-method CIs.
    budgets must be strictly increasing.
    variances should be unbiased var estimates of the means.
    """
    mvc = [float('nan')]
    ci = [(float('nan'), float('nan'))]
    for i in range(1, len(budgets)):
        b0, b1 = budgets[i-1], budgets[i]
        m0, m1 = means[i-1], means[i]
        v0, v1 = variances[i-1], variances[i]
        slope = (m1 - m0) / (b1 - b0)
        var_slope = (v0 + v1) / ((b1 - b0) ** 2)
        se = math.sqrt(max(var_slope, 1e-12))
        lo, hi = slope - 1.96 * se, slope + 1.96 * se
        mvc.append(slope)
        ci.append((lo, hi))
    return mvc, ci

def hedges_g(m1: float, sd1: float, n1: int, m2: float, sd2: float, n2: int) -> float:
    """Hedges' g with small-sample correction."""
    if n1 < 2 or n2 < 2:
        return float('nan')
    sp2 = ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
    sp = math.sqrt(max(sp2, 1e-12))
    g = (m1 - m2) / sp if sp > 0 else float('nan')
    df = n1 + n2 - 2
    J = 1 - (3 / (4*df - 1)) if df > 1 else 1.0
    return g * J

def welch_t(m1: float, sd1: float, n1: int, m2: float, sd2: float, n2: int) -> Tuple[float, float]:
    """Welch t-statistic and (approx) two-sided p-value using normal fallback.
    If SciPy is available, you can replace p-value with an exact computation.
    """
    import importlib
    se2 = sd1**2 / max(n1,1) + sd2**2 / max(n2,1)
    if se2 <= 0:
        return float('nan'), float('nan')
    t = (m1 - m2) / math.sqrt(se2)
    # Welch–Satterthwaite df
    df_num = se2**2
    df_den = (sd1**2 / max(n1,1))**2 / max(n1-1,1) + (sd2**2 / max(n2,1))**2 / max(n2-1,1)
    df = df_num / max(df_den, 1e-12)
    # Try SciPy for p-value; else normal approx
    try:
        sp = importlib.import_module("scipy.stats")
        p = 2 * sp.t.sf(abs(t), df)
    except Exception:
        # Normal approx
        import math
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p

def bh_fdr(pvals: List[float], alpha: float=0.05) -> List[bool]:
    """Benjamini–Hochberg FDR control: returns a list of rejections."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    rejections = [False]*n
    threshold = 0.0
    for k, i in enumerate(order, start=1):
        if pvals[i] <= (k / n) * alpha:
            threshold = pvals[i]
    for i, p in enumerate(pvals):
        if p <= threshold:
            rejections[i] = True
    return rejections
