#!/usr/bin/env python3
import json, csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif", "font.weight": "bold",
    "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 13, "axes.labelweight": "bold",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "axes.linewidth": 1.5,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "grid.linewidth": 0.6, "pdf.fonttype": 42,
    "lines.linewidth": 1.8, "lines.markersize": 7,
})

R = Path("/raid/privatedata/fantao/iccad_exp/results")

pareto = json.load(open(R / "sensitivity/opt125m/pareto_frontier.json"))

ppa = {}
with open(R / "ppa/opt125m/ppa_sweep_opt125m.csv") as f:
    for r in csv.DictReader(f):
        ppa[int(r["adc_bits"])] = float(r["adc_area_um2"]) / 1e6

sweep = {}
with open(R / "opt125m/sweep_adc_facebook_opt-125m.csv") as f:
    for r in csv.DictReader(f):
        sweep[int(r["adc_bits"])] = float(r.get("ppl_baseline", 0))

fig, ax = plt.subplots(figsize=(7.16, 3.4))

# 1. Uniform sweep 6b/7b/8b
uni_bits = [6, 7, 8]
uni_x = [ppa[b] for b in uni_bits]
uni_y = [sweep[b] for b in uni_bits]
ax.plot(uni_x, uni_y, "s--", color="#BDC3C7", lw=1.0, ms=7,
        markeredgecolor="#7F8C8D", markeredgewidth=0.6, zorder=2, label="Uniform sweep")
label_pos = {6: (8, -14), 7: (8, 6), 8: (-10, 6)}
for b, x, y in zip(uni_bits, uni_x, uni_y):
    dx, dy = label_pos[b]
    ax.annotate(f"{b}b", (x, y), xytext=(dx, dy), textcoords="offset points",
               fontsize=10, color="#7F8C8D", fontweight="bold", ha="center")

# 2. Mixed-ILP Pareto
areas_p = [p["adc_area_mm2"] for p in pareto]
ppls_p = [p["ppl"] for p in pareto]
ax.plot(areas_p, ppls_p, "o-", color="#C0392B", lw=1.6, ms=5,
        label="Mixed-ILP Pareto", zorder=3)
pct_pos = {5: (0, 7), 14: (12, -5), 23: (0, 7), 32: (-14, 2), 41: (0, 7), 50: (0, 7)}
for p in pareto:
    s = round(p["actual_savings"])
    dx, dy = pct_pos.get(s, (0, 7))
    ax.annotate(f"{s}%", (p["adc_area_mm2"], p["ppl"]),
               xytext=(dx, dy), textcoords="offset points",
               fontsize=8, color="#C0392B", ha="center")

# 3. Signal comparison — stacked vertically at 20% area
ax.scatter([186], [308.6], s=80, c="#2980B9", edgecolors="#2C3E50",
          linewidths=0.6, marker="D", label="Measured-ILP", zorder=6)
ax.scatter([181], [309.2], s=70, c="#F39C12", edgecolors="#2C3E50",
          linewidths=0.6, marker="v", label="Sat-ILP", zorder=6)
ax.scatter([176], [309.7], s=70, c="#E74C3C", edgecolors="#2C3E50",
          linewidths=0.6, marker="^", label="Hessian-ILP", zorder=6)

# Bracket annotation for signal comparison group
ax.annotate("", xy=(173, 310.2), xytext=(189, 310.2),
           arrowprops=dict(arrowstyle="|-|", color="#555555", lw=0.8))
ax.text(181, 311.5, "20% budget\nsignal comparison", fontsize=7,
       ha="center", color="#555555", fontstyle="italic")

# SQ+6b
ax.scatter([114.2], [305.0], s=150, c="#8E44AD", edgecolors="#2C3E50",
          linewidths=0.6, marker="*", label="SQ+6b", zorder=6)
ax.annotate("SQ+6b\n305.0", (114.2, 305.0), xytext=(-30, -16),
           textcoords="offset points", fontsize=8, color="#8E44AD", ha="center")

# 4. 7b baseline
ax.axhline(306.4, ls="--", color="#7F8C8D", lw=0.7, alpha=0.5)
ax.text(460, 306.9, "7b baseline", fontsize=8, color="#7F8C8D", va="bottom")

# 5. Improvement region
ax.axhspan(293, 306.4, alpha=0.05, color="#27AE60")
ax.text(460, 296, "PPL improvement\nregion", fontsize=8,
       fontstyle="italic", color="#27AE60", va="bottom")

ax.set_xlabel("ADC area (mm\u00b2)")
ax.set_ylabel("Perplexity (WikiText-2)")
ax.legend(fontsize=8, framealpha=0.9, edgecolor="none",
         loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
ax.invert_xaxis()
ax.set_xlim(470, 80)
ax.set_ylim(293, 352)
ax.grid(True, alpha=0.12)

plt.tight_layout()
plt.savefig("/raid/privatedata/fantao/iccad_exp/figures_unified/fig6_pareto_frontier.pdf")
print("Saved")
