#!/usr/bin/env python3
"""Generate ablation datasets (deficit and significance) consistent with simulation.py.
"""
from simulation import ablation_sweep, write_table

def main():
	rows = ablation_sweep(l_mem_true=6, l_range=(1,16), tau_mem=8.0)
	write_table("datatableAblation.dat", "ell deficit", [(r[0], r[1]) for r in rows])
	write_table("datatableAblationSig.dat", "ell sig", [(r[0], r[2]) for r in rows])
	print("Wrote datatableAblation.dat, datatableAblationSig.dat")

if __name__ == "__main__":
	main()
