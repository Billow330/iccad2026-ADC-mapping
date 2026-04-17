#!/usr/bin/env python3
"""Fix Fig 10 (Pareto), Fig 11 (Interaction), Fig 12 (Proxy scatter)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'legend.framealpha': 0.9, 'legend.edgecolor': '0.8',
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
    'axes.linewidth': 0.6, 'grid.linewidth': 0.4, 'grid.alpha': 0.25,
    'lines.linewidth': 1.2, 'lines.markersize': 5,
}
plt.rcParams.update(STYLE)

C = {'blue':'#1976D2','green':'#388E3C','orange':'#F57C00','red':'#D32F2F',
     'purple':'#7B1FA2','gray':'#757575','teal':'#00897B',
     'lb':'#90CAF9','lg':'#A5D6A7','lo':'#FFCC80','lr':'#EF9A9A'}

R = Path("/raid/privatedata/fantao/iccad_exp/results_p0")
OUT = Path("/raid/privatedata/fantao/iccad_exp/figures_unified")
OUT.mkdir(exist_ok=True)

def P(msg): print(msg, flush=True)


def fig_pareto():
    """Fig 10: Full-range Pareto with all key configurations annotated."""
    d4 = json.load(open(R / "p0_4_baselines.json"))
    base7 = d4["baseline_7b"]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    # Uniform sweep (background reference)
    uni_sav = [0, 50]
    uni_ppl = [base7, 315.3]
    ax.plot(uni_sav, uni_ppl, 'o--', color=C['gray'], markersize=4, linewidth=0.8,
            alpha=0.5, label='Uniform (7b, 6b)', zorder=2)
    ax.annotate('Uniform 7b\n306.4', xy=(0, base7), fontsize=6, ha='left',
               xytext=(2, base7+1.5), color=C['gray'])
    ax.annotate('Uniform 6b\n315.3', xy=(50, 315.3), fontsize=6, ha='right',
               xytext=(48, 316.8), color=C['gray'])

    # SQ+6b
    ax.scatter([50], [305.0], marker='*', s=120, c=C['teal'], edgecolors='k',
              linewidths=0.4, zorder=6, label='SQ+6b (305.0)')

    # Signals at 20% and 30%
    signals = [
        ('Measured-ILP', C['green'], 's', [
            (20.5, d4["methods"]["measured_ilp_20pct"]["ppl"]),
            (30, d4["methods"]["measured_ilp_30pct"]["ppl"])]),
        ('Sat-ILP', C['orange'], '^', [
            (20.5, d4["methods"]["saturation_ilp_20pct"]["ppl"]),
            (30, d4["methods"]["saturation_ilp_30pct"]["ppl"])]),
        ('Hessian-ILP', C['purple'], 'D', [
            (20.5, d4["methods"]["hessian_ilp_20pct"]["ppl"]),
            (30, d4["methods"]["hessian_ilp_30pct"]["ppl"])]),
    ]
    for label, color, marker, pts in signals:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=color, marker=marker, s=70, edgecolors='k',
                  linewidths=0.4, label=label, zorder=5)
        ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1.0)

    # Random distribution at 20%
    r = d4["methods"]["random_20pct"]
    ax.scatter([20.5], [r["mean"]], c=C['gray'], marker='x', s=60,
              label=f'Random (mean {r["mean"]:.0f})', zorder=4)
    ax.errorbar([20.5], [r["mean"]],
               yerr=[[r["mean"]-r["best"]], [r["worst"]-r["mean"]]],
               fmt='none', color=C['gray'], alpha=0.6, capsize=3, capthick=0.5)

    # 10b regime result (annotate in upper region)
    ax.annotate('10b regime:\nILP saves 20% ADC\nat −1.5% PPL improvement',
               xy=(15, 320), fontsize=6.5, fontstyle='italic', color=C['green'],
               bbox=dict(boxstyle='round,pad=0.4', facecolor=C['lg'], alpha=0.5))

    # Oracle
    ax.scatter([20.5], [308.6], c='none', marker='o', s=100, edgecolors=C['red'],
              linewidths=1.2, label='Oracle (brute-force)', zorder=7)

    ax.set_xlabel('ADC Area Savings (%)')
    ax.set_ylabel('PPL (WikiText-2)')
    ax.set_title('Allocation Performance: Measured Signal vs. Proxy Baselines')
    ax.set_xlim(-2, 55)
    ax.set_ylim(303, 323)
    ax.legend(loc='upper left', fontsize=6, ncol=2)
    ax.grid(True)

    # Highlight 20% budget region
    ax.axvspan(18, 23, alpha=0.05, color=C['green'])

    plt.tight_layout()
    plt.savefig(OUT / "fig_pareto_multisignal.pdf")
    plt.close()


def fig_interaction():
    """Fig 11: Interaction scatter with pair labels and error regions."""
    d3 = json.load(open(R / "p0_3_surrogate_fixed.json"))
    pairs = d3["pairs"]

    short = {'attn_qkv':'qkv','attn_out':'out','ffn_up':'fc1','ffn_down':'fc2','lm_head':'hd'}

    fig, ax = plt.subplots(figsize=(3.3, 3.3))

    pred = [p["delta_predicted"] for p in pairs]
    meas = [p["delta_measured"] for p in pairs]

    lo = min(min(pred), min(meas)) - 2
    hi = max(max(pred), max(meas)) + 2

    # Shaded regions
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.04, color=C['green'],
                   label='Overestimate (conservative)')
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], alpha=0.04, color=C['red'],
                   label='Underestimate')

    # Identity line
    ax.plot([lo, hi], [lo, hi], 'k-', alpha=0.3, linewidth=0.8)

    # Points with labels
    for p in pairs:
        g1, g2 = p["pair"]
        label = f'{short[g1]}/{short[g2]}'
        err = p["rel_error_pct"]
        color = C['green'] if err < 20 else C['blue'] if err < 50 else C['orange'] if err < 100 else C['red']
        ax.scatter(p["delta_predicted"], p["delta_measured"], c=color, s=50,
                  edgecolors='k', linewidths=0.3, zorder=5)
        ax.annotate(label, (p["delta_predicted"], p["delta_measured"]),
                   textcoords="offset points", xytext=(4, -7), fontsize=5.5,
                   color='0.3')

    ax.set_xlabel('Predicted ΔPPL (sum of individuals)')
    ax.set_ylabel('Measured ΔPPL (pairwise)')
    ax.set_title('Surrogate Additivity Validation')
    ax.legend(fontsize=5.5, loc='upper left')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.grid(True)

    # Summary annotation
    mean_err = np.mean([abs(p["delta_predicted"]-p["delta_measured"]) for p in pairs])
    ax.annotate(f'Mean |error| = {mean_err:.1f} PPL\nILP = oracle (regret 0.0)',
               xy=(0.97, 0.03), xycoords='axes fraction', ha='right', va='bottom',
               fontsize=6, fontstyle='italic', color=C['green'],
               bbox=dict(boxstyle='round,pad=0.3', facecolor=C['lg'], alpha=0.6))

    plt.tight_layout()
    plt.savefig(OUT / "fig_interaction_scatter.pdf")
    plt.close()


def fig_proxy():
    """Fig 12: Proxy failure with FFN/Attention grouping and rank annotations."""
    groups = ['$W_{qkv}$', '$W_{out}$', '$W_{fc1}$', '$W_{fc2}$', '$W_{head}$']
    types  = ['Attn', 'Attn', 'FFN', 'FFN', 'Other']
    sat = [1.0, 0.04, 0.94, 0.19, 1.0]
    hess = [0.15, 0.02, 0.21, 0.57, 1.00]
    meas = [0.13, 0.87, 0.20, 1.41, 0.70]

    type_colors = {'Attn': C['blue'], 'FFN': C['red'], 'Other': C['gray']}
    type_markers = {'Attn': 'o', 'FFN': 's', 'Other': 'D'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)

    for ax, proxy, xlabel, rho_val, rho_sign in [
        (ax1, sat, 'Saturation Rate', 0.80, '−'),
        (ax2, hess, 'Hessian Trace (normalized)', 0.20, '+')
    ]:
        # Plot by type
        for t in ['Attn', 'FFN', 'Other']:
            idxs = [i for i in range(5) if types[i] == t]
            if idxs:
                ax.scatter([proxy[i] for i in idxs], [meas[i] for i in idxs],
                          c=type_colors[t], marker=type_markers[t], s=80,
                          edgecolors='k', linewidths=0.4, label=t, zorder=5)

        # Labels with sensitivity rank
        meas_rank = {i: r+1 for r, i in enumerate(sorted(range(5), key=lambda k: meas[k], reverse=True))}
        sat_rank = {i: r+1 for r, i in enumerate(sorted(range(5), key=lambda k: proxy[k], reverse=True))}
        for i in range(5):
            offset_x = 6 if proxy[i] < 0.5 else -6
            ha = 'left' if proxy[i] < 0.5 else 'right'
            ax.annotate(f'{groups[i]}\n(rank {meas_rank[i]})',
                       (proxy[i], meas[i]),
                       textcoords="offset points", xytext=(offset_x, 6),
                       fontsize=6, ha=ha, color='0.2')

        ax.set_xlabel(xlabel)
        ax.grid(True)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

        # Correlation box
        ax.text(0.95, 0.95, f'ρ = {rho_sign}{rho_val}',
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor=C['lr'] if rho_val > 0.5 else C['lo'],
                        alpha=0.7))

    ax1.set_ylabel('Measured Sensitivity (ΔPPL/layer)')
    ax1.legend(fontsize=6.5, loc='center left')

    # "INVERTED" arrow on saturation panel
    ax1.annotate('', xy=(0.95, 1.41), xytext=(0.95, 0.13),
                arrowprops=dict(arrowstyle='<->', color=C['red'], lw=1.5))
    ax1.text(1.02, 0.77, '11×\ninversion', fontsize=7, color=C['red'],
            fontweight='bold', va='center')

    fig.suptitle('Proxy Failure: Saturation and Hessian Do Not Predict CIM ADC Sensitivity',
                fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig_proxy_scatter.pdf")
    plt.close()


if __name__ == "__main__":
    P("Fixing Fig 10, 11, 12...")
    for name, fn in [
        ("Fig 10: Pareto", fig_pareto),
        ("Fig 11: Interaction", fig_interaction),
        ("Fig 12: Proxy Scatter", fig_proxy),
    ]:
        try:
            fn()
            P(f"  ✓ {name}")
        except Exception as e:
            P(f"  ✗ {name}: {e}")
            import traceback; traceback.print_exc()
    P("Done.")
