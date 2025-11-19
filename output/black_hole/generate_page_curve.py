#!/usr/bin/env python3
"""Legacy helper updated to call page_curve_ensemble for consistency.
Generates the page curve dataset with the same columns used in the manuscript.
"""
from simulation import page_curve_ensemble, write_table

def main():
	rows = page_curve_ensemble(S_initial=12, steps=12, num_runs=100, base_seed=42)
	write_table("datatablePagecurve.dat", "t S_mean std_S upper_S lower_S ideal_page hawking bh_entropy", rows)
	print("Wrote datatablePagecurve.dat")

if __name__ == "__main__":
	main()
