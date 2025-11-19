#!/usr/bin/env python3
"""Legacy helper updated to call g2_correlation_with_ci for consistency.
Generates the g2 correlation dataset with confidence intervals used in the manuscript.
"""
from simulation import g2_correlation_with_ci, write_table

def main():
	rows = g2_correlation_with_ci(max_lag=64, tau_mem=8.0, runs=200, base_seed=42+81)
	write_table("datatableGtwo.dat", "du g2_mean ci_low ci_high", rows)
	print("Wrote datatableGtwo.dat")

if __name__ == "__main__":
	main()
