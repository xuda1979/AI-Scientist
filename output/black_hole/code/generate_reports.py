#!/usr/bin/env python3
# Convenience driver: builds KK and identifiability reports in this folder.
import os, json
from simulate_kk_identifiability import simulate_kernel_and_kk, plot_kk_results, simulate_identifiability, plot_identifiability
from make_comb_figure import make_comb_code_embedding_pdf

def main(out_base=".", fig_subdir="fig"):
    figdir = os.path.join(out_base, fig_subdir)
    os.makedirs(figdir, exist_ok=True)
    # schematic
    make_comb_code_embedding_pdf(os.path.join(figdir, "comb_code_embedding.pdf"))
    # KK
    kk = simulate_kernel_and_kk()
    with open(os.path.join(out_base, "kk_checks.json"), "w") as f:
        json.dump(kk, f)
    plot_kk_results(kk, figdir)
    # Identifiability
    rep = simulate_identifiability()
    with open(os.path.join(out_base, "identifiability_report.json"), "w") as f:
        json.dump(rep, f)
    plot_identifiability(rep, figdir)

if __name__ == "__main__":
    main()
