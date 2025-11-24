# verify_triangular_pi.py
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, math, sys, time

# --- build triangular Laplacian ----------------------------------------------
def build_tri_lap(R:int) -> sp.csr_matrix:
    N = R*R
    main  = np.ones(N)
    row, col, data = [], [], []
    idx = lambda x,y: (y-1)*R + (x-1)

    nbrs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for y in range(1,R+1):
        for x in range(1,R+1):
            i = idx(x,y)
            for dx,dy in nbrs:
                xx,yy = x+dx, y+dy
                if 1<=xx<=R and 1<=yy<=R:
                    row.append(i); col.append(idx(xx,yy)); data.append(-1/6)
    return sp.csr_matrix((data,(row,col)), shape=(N,N)) + sp.eye(N)

# --- trace inverse via eigsh --------------------------------------------------
def trace_inv(lap):   # full spectrum for small R (≤250)
    vals = spla.eigsh(lap, k=lap.shape[0]-2, return_eigenvectors=False)
    return np.sum(1/vals)

def tri_pi(R:int):
    lap = build_tri_lap(R)
    tr  = trace_inv(lap)
    return (2/np.sqrt(3)) * (R**2 * math.log(R**2)) / tr

# --- main ---------------------------------------------------------------------
if __name__=="__main__":
    R=int(sys.argv[1])
    t0=time.time(); val=tri_pi(R); t1=time.time()
    print(f"Tri-π (R={R}): {val}  error={abs(val-math.pi):.3e}  time={t1-t0:.2f}s")
