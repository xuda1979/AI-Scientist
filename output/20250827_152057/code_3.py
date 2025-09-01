import math, torch
def uniform_shots_for_tolerance(n, eps_K, delta):
    # Solve sqrt(2 n/N ln(2n/delta)) <= eps_K for N (ignoring the log term from R)
    return math.ceil((2.0 * n * math.log(2*n/delta)) / (eps_K**2))

def adaptive_allocation(sigmas, total_shots):
    # Allocate shots proportional to estimated std dev: N_ij ∝ sigma_ij
    s = sigmas / (sigmas.sum() + 1e-12)
    alloc = (s * total_shots).round().long()
    alloc[alloc == 0] = 1
    return alloc