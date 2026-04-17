#!/usr/bin/env python3
"""Regenerate Fig 4 (grouping), Fig 13 (proxy scatter), Fig 15 (noise robustness) with BOLD style."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif", "font.weight": "bold",
    "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 13, "axes.labelweight": "bold",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "axes.linewidth": 1.5,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "grid.linewidth": 0.6, "grid.alpha": 0.2,
    "lines.linewidth": 1.8, "lines.markersize": 7, "pdf.fonttype": 42,
})

C = {"blue":"#1976D2","green":"#388E3C","orange":"#F57C00","red":"#D32F2F",
     "purple":"#7B1FA2","gray":"#757575","teal":"#00897B",
     "lb":"#90CAF9","lg":"#A5D6A7","lo":"#FFCC80","lr":"#EF9A9A"}
PAL = {"attn_qkv":"#C0392B","attn_out":"#27AE60","ffn_up":"#E67E22","ffn_down":"#2980B9","lm_head":"#8E44AD"}
OUT = Path("/raid/privatedata/fantao/iccad_exp/figures_unified")
BLW = 1.0; MLW = 0.8
def BF(s): return {"fontweight":"bold","fontsize":s}

# ═══ Fig 4: Grouping validity (depth-bin) ═══
def fig_grouping():
    groups = ["attn_qkv","attn_out","ffn_up","ffn_down"]
    labels = ["$W_{qkv}$","$W_{out}$","$W_{fc1}$","$W_{fc2}$"]
    depths = ["Early","Mid","Late"]
    # Data from original figure (read from plot_new_evidence.py output)
    data = {
        "attn_qkv": [-0.5, -0.3, -0.1],
        "attn_out": [0.2, 1.5, 0.3],
        "ffn_up":   [0.1, 0.4, 0.1],
        "ffn_down": [0.8, 2.5, 0.9],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # (a) Per-depth bars
    pos = 0
    tick_pos = []; tick_lab = []
    for gi, (gname, lab) in enumerate(zip(groups, labels)):
        vals = data[gname]
        for di, v in enumerate(vals):
            alpha = 0.6 + 0.13*di
            ax1.bar(pos, v, color=PAL[gname], width=0.7, edgecolor="#2C3E50",
                   lw=0.5, alpha=alpha)
            ax1.text(pos, v+(0.2 if v>=0 else -0.5), depths[di],
                    ha="center", **BF(8), color="#555")
            pos += 1
        tick_pos.append(pos-2); tick_lab.append(lab)
        pos += 0.5

    ax1.set_xticks(tick_pos); ax1.set_xticklabels(tick_lab)
    ax1.set_ylabel("ΔPPL / layer")
    ax1.set_title("(a) Per-depth sensitivity")
    ax1.axhline(0, ls="-", color="k", lw=0.6)
    ax1.grid(axis="y")

    # (b) Mean +/- std
    means = {g: np.mean(v) for g,v in data.items()}
    stds = {g: np.std(v) for g,v in data.items()}
    x = np.arange(4)
    m = [means[g] for g in groups]; s = [stds[g] for g in groups]
    ax2.bar(x, m, color=[PAL[g] for g in groups], edgecolor="#2C3E50", lw=BLW)
    ax2.errorbar(x, m, yerr=s, fmt="none", ecolor="black", capsize=4, capthick=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean ΔPPL/layer")
    ax2.set_title("(b) Mean ± std")
    ax2.axhline(0, ls="-", color="k", lw=0.6)
    ax2.grid(axis="y")

    inter = means["ffn_down"] - means["attn_qkv"]
    max_std = max(s)
    ax2.text(0.97, 0.92, f"Inter gap: {inter:.1f}\nMax std: {max_std:.1f}",
            transform=ax2.transAxes, ha="right", **BF(10),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F7DC6F", alpha=0.7))

    plt.tight_layout(w_pad=1.2)
    plt.savefig(OUT / "fig_grouping_validity.pdf"); plt.close()
    print("  OK grouping")


# ═══ Fig 15: Noise robustness ═══
def fig_noise_robustness():
    configs = ["ADC\nonly","+Thermal\nσ=0.1","+Thermal\nσ=0.3",
               "+DevVar\nσ=0.03","Combined\n(0.1,0.01)","Worst\n(0.3,0.03)"]
    groups = ["$W_{fc2}$","$W_{fc1}$","$W_{head}$","$W_{qkv}$","$W_{out}$"]
    colors = [C["blue"],C["orange"],C["purple"],C["red"],C["green"]]
    # Data from original figure (real measured values visible in screenshot)
    data = np.array([
        [+0.004,+0.12,+0.08,-0.40, 0.00],   # ADC only
        [+0.23, +0.21,+0.10,-0.31,-0.06],    # +Thermal 0.1
        [+0.49, +0.10,+0.00,-0.09,-0.49],    # +Thermal 0.3
        [+2.86, +1.21,+0.00,-0.80,-0.80],    # +DevVar 0.03
        [+1.47, +0.48,+0.00,-0.25,-0.40],    # Combined
        [+0.57, -0.58,-0.65,-1.60,-0.65],    # Worst
    ])

    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    x = np.arange(6); w = 0.15
    for i, (g, c) in enumerate(zip(groups, colors)):
        bars = ax.bar(x + i*w - 2*w, data[:, i], w, label=g, color=c,
                      edgecolor="k", linewidth=0.5)

    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel("ΔPPL / layer (7b → 6b)")
    ax.set_title("Sensitivity Robustness Under Enhanced CIM Noise (OPT-125M)")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.legend(ncol=5, loc="upper center", fontsize=9)
    ax.grid(axis="y")

    # Annotate key values
    for j in range(6):
        for i in range(5):
            v = data[j, i]
            if abs(v) > 0.3:
                ax.text(j + i*w - 2*w, v + (0.12 if v>0 else -0.25),
                       f"{v:+.2f}", ha="center", **BF(7), color=colors[i])

    # Sensitivity region labels
    ax.text(-0.8, 2.5, "More sensitive ↑", **BF(10), color="#E74C3C", alpha=0.6)
    ax.text(-0.8, -1.3, "Less sensitive ↓", **BF(10), color="#3498DB", alpha=0.6)
    ax.axhspan(0, 3.2, alpha=0.03, color="#E74C3C")
    ax.axhspan(-2.0, 0, alpha=0.03, color="#3498DB")

    plt.tight_layout()
    plt.savefig(OUT / "fig_robustness_original.pdf"); plt.close()
    print("  OK noise robustness")


if __name__ == "__main__":
    print("Regenerating Fig 4, 15 with BOLD style...")
    fig_grouping()
    fig_noise_robustness()
    print("Done.")
