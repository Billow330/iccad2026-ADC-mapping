#!/usr/bin/env python3
"""Generate all key figures for ICCAD paper from experiment data."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05
})

R = Path("/raid/privatedata/fantao/iccad_exp/results_p0")
OUT = Path("/raid/privatedata/fantao/iccad_exp/figures")
OUT.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Fig A: Proxy Failure Scatter (upgrade of fig_ppa_proxy)
# ═══════════════════════════════════════════════════════════════
def fig_proxy_failure():
    groups = ['$W_{qkv}$', '$W_{out}$', '$W_{fc1}$', '$W_{fc2}$', '$W_{head}$']
    sat = [1.0, 0.04, 0.94, 0.19, 1.0]
    hess = [0.0014, 0.00018, 0.002, 0.0053, 0.0093]
    hess_norm = [h / max(hess) for h in hess]
    meas_7to6 = [0.128, 0.866, 0.201, 1.413, 0.700]

    d1 = json.load(open(R / "p0_1_transfer_deployment.json"))
    meas_8to7 = [d1["sensitivity_8to7_native"].get(g, 0) for g in
                 ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']]
    meas_10to9 = [d1["sensitivity_10to9_native"].get(g, 0) for g in
                  ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0']
    for i in range(5):
        ax1.scatter(sat[i], meas_7to6[i], c=colors[i], s=80, zorder=5,
                   edgecolors='k', linewidths=0.5)
        ax1.annotate(groups[i], (sat[i], meas_7to6[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax1.set_xlabel('Saturation Rate')
    ax1.set_ylabel('Measured Sensitivity (ΔPPL/layer)')
    ax1.set_title('(a) Saturation vs Measured (ρ = −0.80)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    z = np.polyfit(sat, meas_7to6, 1)
    xs = np.linspace(0, 1.05, 50)
    ax1.plot(xs, np.polyval(z, xs), 'r--', alpha=0.4, linewidth=1)

    for i in range(5):
        ax2.scatter(hess_norm[i], meas_7to6[i], c=colors[i], s=80, zorder=5,
                   edgecolors='k', linewidths=0.5)
        ax2.annotate(groups[i], (hess_norm[i], meas_7to6[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax2.set_xlabel('Hessian Trace (normalized)')
    ax2.set_ylabel('Measured Sensitivity (ΔPPL/layer)')
    ax2.set_title('(b) Hessian vs Measured (ρ = 0.20)')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    z2 = np.polyfit(hess_norm, meas_7to6, 1)
    ax2.plot(xs, np.polyval(z2, xs), 'r--', alpha=0.4, linewidth=1)

    plt.tight_layout()
    plt.savefig(OUT / "fig_proxy_scatter.pdf")
    print(f"  Saved fig_proxy_scatter.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig B: Deployment Regime Validation
# ═══════════════════════════════════════════════════════════════
def fig_deployment_regime():
    d1 = json.load(open(R / "p0_1_transfer_deployment.json"))

    regimes = ['7b', '8b', '9b', '10b']
    baselines = [d1[f"ppl_{r}"] for r in regimes]

    deploy_regimes = ['8→7', '9→8', '10→9']
    transfer_ppls = [d1.get(f"transfer_{r}_20pct_ppl", 0) for r in ["8to7","9to8","10to9"]]
    native_ppls = [d1.get(f"native_{r}_20pct_ppl", 0) for r in ["8to7","9to8","10to9"]]
    uniform_ppls = [d1.get(f"uniform_{r}_ppl", 0) for r in ["8to7","9to8","10to9"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    ax1.bar(regimes, baselines, color=['#E3F2FD','#BBDEFB','#90CAF9','#64B5F6'],
            edgecolor='#1976D2', linewidth=0.8)
    ax1.set_ylabel('Baseline PPL')
    ax1.set_xlabel('Uniform ADC bits')
    ax1.set_title('(a) CIM baseline at each regime')
    ax1.axhline(y=baselines[0], color='red', linestyle='--', alpha=0.5,
                label=f'FP32→CIM gap dominates')
    for i, v in enumerate(baselines):
        ax1.text(i, v+0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

    x = np.arange(len(deploy_regimes))
    w = 0.25
    ax2.bar(x-w, transfer_ppls, w, label='Transfer (7→6 signal)',
            color='#4CAF50', edgecolor='k', linewidth=0.5)
    ax2.bar(x, native_ppls, w, label='Native signal',
            color='#2196F3', edgecolor='k', linewidth=0.5)
    ax2.bar(x+w, uniform_ppls, w, label='Uniform reduced',
            color='#FF9800', edgecolor='k', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(deploy_regimes)
    ax2.set_ylabel('PPL (20% savings)')
    ax2.set_xlabel('Deployment regime')
    ax2.set_title('(b) Transfer allocation at deployment')
    ax2.legend(loc='upper right', framealpha=0.9)
    ymin = min(min(transfer_ppls), min(native_ppls), min(uniform_ppls)) - 1
    ymax = max(max(transfer_ppls), max(native_ppls), max(uniform_ppls)) + 1
    ax2.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(OUT / "fig_deployment_regime.pdf")
    print(f"  Saved fig_deployment_regime.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig C: Interaction/Additivity Scatter
# ═══════════════════════════════════════════════════════════════
def fig_interaction_scatter():
    d3 = json.load(open(R / "p0_3_surrogate_fixed.json"))
    pairs = d3["pairs"]

    pred = [p["delta_predicted"] for p in pairs]
    meas = [p["delta_measured"] for p in pairs]

    fig, ax = plt.subplots(figsize=(3.2, 3.0))

    ax.scatter(pred, meas, c='#2196F3', s=60, edgecolors='k', linewidths=0.5, zorder=5)

    lim_lo = min(min(pred), min(meas)) - 2
    lim_hi = max(max(pred), max(meas)) + 2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', alpha=0.4, linewidth=1,
            label='Perfect additivity')
    ax.set_xlabel('Predicted ΔPPL (sum of individuals)')
    ax.set_ylabel('Measured ΔPPL (pairwise)')
    ax.set_title('Surrogate Additivity')
    ax.legend(loc='upper left', fontsize=7)

    for p in pairs:
        if abs(p["delta_measured"]) > 8 or abs(p["rel_error_pct"]) < 20:
            g1, g2 = p["pair"]
            short = {'attn_qkv':'qkv','attn_out':'out','ffn_up':'fc1','ffn_down':'fc2','lm_head':'hd'}
            label = f'{short.get(g1,"?")}/{short.get(g2,"?")}'
            ax.annotate(label, (p["delta_predicted"], p["delta_measured"]),
                       textcoords="offset points", xytext=(4, -8), fontsize=6)

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT / "fig_interaction_scatter.pdf")
    print(f"  Saved fig_interaction_scatter.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig D: Multi-signal Pareto (upgrade)
# ═══════════════════════════════════════════════════════════════
def fig_pareto_multisignal():
    d4 = json.load(open(R / "p0_4_baselines.json"))
    base = d4["baseline_7b"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    savings_20 = 20.5
    savings_30 = 30.0

    methods = {
        'Measured-ILP': ('#4CAF50', 's'),
        'Sat-ILP': ('#FF9800', '^'),
        'Hessian-ILP': ('#9C27B0', 'D'),
    }

    for sig, (color, marker) in methods.items():
        key20 = f"{sig.split('-')[0].lower()}_ilp_20pct"
        key30 = f"{sig.split('-')[0].lower()}_ilp_30pct"
        key_map = {'measured': 'measured', 'sat': 'saturation', 'hessian': 'hessian'}
        k = key_map[sig.split('-')[0].lower()]
        p20 = d4["methods"].get(f"{k}_ilp_20pct", {}).get("ppl", base)
        p30 = d4["methods"].get(f"{k}_ilp_30pct", {}).get("ppl", base)
        ax.scatter([savings_20, savings_30], [p20, p30], c=color, marker=marker,
                  s=60, edgecolors='k', linewidths=0.5, label=sig, zorder=5)
        ax.plot([savings_20, savings_30], [p20, p30], color=color, alpha=0.3, linewidth=1)

    r20 = d4["methods"]["random_20pct"]
    ax.scatter([savings_20], [r20["mean"]], c='gray', marker='x', s=60,
              label=f'Random (mean)', zorder=4)
    ax.errorbar([savings_20], [r20["mean"]],
               yerr=[[r20["mean"]-r20["best"]], [r20["worst"]-r20["mean"]]],
               fmt='none', color='gray', alpha=0.5, capsize=3)

    ax.axhline(y=base, color='red', linestyle='--', alpha=0.3, label='Uniform 7b')
    ax.scatter([0], [base], c='red', marker='o', s=40, zorder=3)
    ax.scatter([50], [315.3], c='blue', marker='o', s=40, zorder=3, label='Uniform 6b')

    ax.set_xlabel('ADC Area Savings (%)')
    ax.set_ylabel('PPL')
    ax.set_title('Signal Comparison: Pareto')
    ax.legend(loc='upper right', fontsize=6, ncol=1)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT / "fig_pareto_multisignal.pdf")
    print(f"  Saved fig_pareto_multisignal.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig E: Transfer + Ranking Stability Heatmap
# ═══════════════════════════════════════════════════════════════
def fig_transfer_heatmap():
    d1 = json.load(open(R / "p0_1_transfer_deployment.json"))

    groups = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']
    labels = ['$W_{qkv}$', '$W_{out}$', '$W_{fc1}$', '$W_{fc2}$', '$W_{head}$']
    regimes = ['7→6', '8→7', '9→8', '10→9']
    regime_keys = ['sensitivity_7to6', 'sensitivity_8to7_native',
                   'sensitivity_9to8_native', 'sensitivity_10to9_native']

    matrix = []
    for rk in regime_keys:
        sens = d1.get(rk, {})
        row = [sens.get(g, 0) for g in groups]
        matrix.append(row)

    # Compute ranks per regime
    rank_matrix = []
    for row in matrix:
        ranked = sorted(range(len(row)), key=lambda i: row[i])
        ranks = [0]*len(row)
        for pos, idx in enumerate(ranked):
            ranks[idx] = pos + 1
        rank_matrix.append(ranks)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5),
                                    gridspec_kw={'width_ratios': [3, 2]})

    im = ax1.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=5)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(labels)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(regimes)
    ax1.set_title('(a) Sensitivity rank across regimes')
    for i in range(4):
        for j in range(5):
            ax1.text(j, i, str(rank_matrix[i][j]), ha='center', va='center',
                    fontsize=8, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Rank (1=least sens.)', shrink=0.8)

    deploy = ['8→7', '9→8', '10→9']
    regret = [0.0, 0.0, 0.0]
    overlap = [5, 5, 5]

    x = np.arange(len(deploy))
    ax2.bar(x, overlap, color='#4CAF50', edgecolor='k', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(deploy)
    ax2.set_ylabel('Group assignment overlap (out of 5)')
    ax2.set_xlabel('Deployment regime')
    ax2.set_title('(b) Transfer: 5/5 overlap, 0 regret')
    ax2.set_ylim(0, 6)
    for i, v in enumerate(overlap):
        ax2.text(i, v+0.1, f'{v}/5', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT / "fig_transfer_heatmap.pdf")
    print(f"  Saved fig_transfer_heatmap.pdf")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    for name, fn in [
        ("Proxy Failure Scatter", fig_proxy_failure),
        ("Deployment Regime", fig_deployment_regime),
        ("Interaction Scatter", fig_interaction_scatter),
        ("Pareto Multi-signal", fig_pareto_multisignal),
        ("Transfer Heatmap", fig_transfer_heatmap),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            import traceback; traceback.print_exc()

    print(f"\nAll figures saved to {OUT}")
    for f in sorted(OUT.glob("*.pdf")):
        print(f"  {f.name} ({f.stat().st_size} bytes)")
