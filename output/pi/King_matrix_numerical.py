# king_matrix_pi.py
# Verify the π-identity with rational King-walk Laplacian

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math, sys, time


def build_king_laplacian(R: int) -> sp.csr_matrix:
    """Return sparse Dirichlet Laplacian 𝕂_R for the 8-neighbour walk."""
    N = R * R
    diag_data   = np.ones(N, dtype=np.float64)
    row, col, data = [], [], []

    def idx(x, y):           # (1-based) lattice → 0-based index
        return (y - 1) * R + (x - 1)

    # offsets for the 8 neighbours
    nbrs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    w = -0.125                # −1/8

    for y in range(1, R + 1):
        for x in range(1, R + 1):
            i = idx(x, y)
            for dx, dy in nbrs:
                xx, yy = x + dx, y + dy
                if 1 <= xx <= R and 1 <= yy <= R:
                    j = idx(xx, yy)
                    row.append(i)
                    col.append(j)
                    data.append(w)

    off_diag = sp.csr_matrix((data, (row, col)), shape=(N, N))
    return sp.csr_matrix(sp.eye(N)) + off_diag   # I − (1/8)A


def trace_inverse(lap: sp.csr_matrix) -> float:
    """Trace(𝕂^{-1}) via full eigendecomposition (OK up to R≈180)."""
    eigvals = spla.eigsh(lap, k=lap.shape[0]-2, return_eigenvectors=False)
    # For Dirichlet Laplacian all eigenvalues > 0, so inverse is safe
    return np.sum(1.0 / eigvals)


def king_pi(R: int) -> float:
    lap = build_king_laplacian(R)
    tr_inv = trace_inverse(lap)
    return (2.0 / 3.0) * (R**2 * math.log(R**2)) / tr_inv


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python king_matrix_pi.py <R>")
        sys.exit(1)

    R = int(sys.argv[1])
    t0 = time.time()
    pi_R = king_pi(R)
    t1 = time.time()

    print(f"π_R  (R={R}) = {pi_R}")
    print(f"actual π     = {math.pi}")
    print(f"abs. error   = {abs(pi_R - math.pi)}")
    print(f"time         = {t1 - t0:.2f} s")
