#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, Circle

def make_comb_code_embedding_pdf(path_pdf: str):
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')
    mem_heights = [1.2, 0.9, 0.6]
    mem_xs = [1.0, 4.5, 7.5]; mem_y = 2.5
    for i, (x, h) in enumerate(zip(mem_xs, mem_heights)):
        rect = FancyBboxPatch((x-0.8, mem_y-0.5*h), 1.6, h, boxstyle="round,pad=0.02,rounding_size=0.06")
        ax.add_patch(rect)
        ax.text(x, mem_y, f"$M_{i}$", ha="center", va="center", fontsize=12)
    V_center = (0.8, 0.9)
    circ = Circle(V_center, 0.25, fill=False); ax.add_patch(circ)
    ax.text(V_center[0], V_center[1], "$V_n$", ha="center", va="center", fontsize=12)
    R_positions = [(3.8, 0.9), (6.8, 0.9), (9.2, 0.9)]
    for j, (rx, ry) in enumerate(R_positions, 1):
        rbox = Rectangle((rx-0.35, ry-0.2), 0.7, 0.4, fill=False)
        ax.add_patch(rbox); ax.text(rx, ry, f"$R_{j}$", ha="center", va="center", fontsize=10)
    a1 = FancyArrowPatch((1.8, 2.5), (3.6, 2.5), arrowstyle="->", mutation_scale=12); ax.add_patch(a1)
    ax.text(3.0, 2.8, "$W_1$", ha="center")
    s1 = FancyArrowPatch((3.6, 2.1), (3.8, 1.2), arrowstyle="->", mutation_scale=12); ax.add_patch(s1)
    ax.text(3.9, 1.8, "swap", ha="left")
    a2 = FancyArrowPatch((5.4, 2.5), (6.6, 2.5), arrowstyle="->", mutation_scale=12); ax.add_patch(a2)
    ax.text(6.0, 2.8, "$W_2$", ha="center")
    s2 = FancyArrowPatch((6.6, 2.0), (6.8, 1.2), arrowstyle="->", mutation_scale=12); ax.add_patch(s2)
    ax.text(6.9, 1.8, "swap", ha="left")
    a3 = FancyArrowPatch((8.4, 2.5), (9.6, 2.5), arrowstyle="->", mutation_scale=12); ax.add_patch(a3)
    ax.text(8.7, 2.8, "$W_3$", ha="center")
    s3 = FancyArrowPatch((9.0, 2.0), (9.2, 1.2), arrowstyle="->", mutation_scale=12); ax.add_patch(s3)
    ax.text(9.3, 1.8, "swap", ha="left")
    ax.text(4.5, 3.4, "code embedding", ha="center", fontsize=11)
    ax.text(7.5, 3.4, "code embedding", ha="center", fontsize=11)
    fig.savefig(path_pdf, bbox_inches="tight"); plt.close(fig)

def main():
    out = "fig/comb_code_embedding.pdf"
    os.makedirs("fig", exist_ok=True)
    make_comb_code_embedding_pdf(out)

if __name__ == "__main__":
    main()
