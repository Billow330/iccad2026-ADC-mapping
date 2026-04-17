"""
plot_new_evidence.py — Generate new figures for ICCAD rebuttal
=============================================================
Uses existing Phase2 + sensitivity data to produce:
  Fig A: Transfer study (profile@low → deploy@high)
  Fig B: Group-level validity (depth-bin box plot)  
  Fig C: Robustness (5 seeds, error bars)
  Fig D: Relative degradation vs budget
"""

import os, json, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).parent
RES = ROOT / 'results'
PHASE2 = Path('/tmp/fantaog_iccad/results/phase2')
OUT = RES / 'figures_new_evidence'
OUT.mkdir(parents=True, exist_ok=True)
PAPER = ROOT.parent / 'paper'

COL_W = 3.5
FULL_W = 7.16

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'axes.grid': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

PALETTE = {
    'attn_qkv': '#C0392B', 'attn_out': '#27AE60',
    'ffn_up': '#E67E22', 'ffn_down': '#2980B9', 'lm_head': '#8E44AD',
}
LABELS = {
    'attn_qkv': r'$W_{qkv}$', 'attn_out': r'$W_{out}$',
    'ffn_up': r'$W_{fc1}$', 'ffn_down': r'$W_{fc2}$', 'lm_head': r'$W_{head}$',
}

def savefig(name):
    for fmt in ('pdf', 'png'):
        plt.savefig(OUT / f'{name}.{fmt}', bbox_inches='tight', dpi=300)
    import shutil
    shutil.copy2(str(OUT / f'{name}.pdf'), str(PAPER / f'{name}.pdf'))
    plt.close()
    print(f"  -> {name}")


# ═══════════════════════════════════════════════════════════════════
# Fig A: Transfer Study — Sensitivity ordering across operating points
# Shows: rank consistency enables cross-regime allocation transfer
# ═══════════════════════════════════════════════════════════════════
def fig_transfer():
    print("[Transfer] Profiling transferability across operating points")
    rc = json.load(open(PHASE2 / 'rank_consistency.json'))

    groups = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']
    probes = ['10to9', '9to8', '8to7', '7to6']
    probe_labels = ['10b→9b', '9b→8b', '8b→7b', '7b→6b']

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))

    # (a) Sensitivity per group across operating points (normalized)
    ax = axes[0]
    x = np.arange(len(groups))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']

    for i, (probe, label) in enumerate(zip(probes, probe_labels)):
        sens = rc['probe_results'][probe]['sens']
        baseline = rc['probe_results'][probe]['baseline']
        vals = [sens[g] / baseline * 100 for g in groups]
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=colors[i],
                      edgecolor='#2C3E50', lw=0.3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[g] for g in groups])
    ax.set_ylabel('Normalized sensitivity\n($\\Delta$PPL / baseline PPL, %)')
    ax.set_title('(a) Sensitivity across operating points', fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='upper left', framealpha=0.85, edgecolor='none')
    ax.axhline(0, ls='-', color='k', lw=0.4)

    # (b) Rank correlation matrix
    ax = axes[1]
    from itertools import combinations

    ranks = {}
    for probe in probes:
        sens = rc['probe_results'][probe]['sens']
        vals = [sens[g] for g in groups]
        order = np.argsort(vals)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        ranks[probe] = rank

    n = len(probes)
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            r1, r2 = ranks[probes[i]], ranks[probes[j]]
            d = r1 - r2
            rho = 1 - 6 * np.sum(d**2) / (len(d) * (len(d)**2 - 1))
            corr_matrix[i, j] = rho

    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_xticklabels(probe_labels, fontsize=6.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(probe_labels, fontsize=6.5)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                    fontsize=7, color='k' if abs(corr_matrix[i,j]) < 0.7 else 'w')
    ax.set_title('(b) Spearman rank correlation', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='$\\rho$')

    plt.tight_layout(w_pad=1.5)
    savefig('fig_transfer_study')


# ═══════════════════════════════════════════════════════════════════
# Fig B: Group-level Validity — Depth-bin sensitivity
# Shows: inter-group gap >> intra-group variance
# ═══════════════════════════════════════════════════════════════════
def fig_grouping_validity():
    print("[Grouping] Depth-bin sensitivity analysis")
    db = json.load(open(PHASE2 / 'depth_bin_sensitivity.json'))

    type_map = {
        'attn_qkv': ['attn_qkv_shallow', 'attn_qkv_middle', 'attn_qkv_deep'],
        'attn_out': ['attn_out_shallow', 'attn_out_middle', 'attn_out_deep'],
        'ffn_up': ['ffn_up_shallow', 'ffn_up_middle', 'ffn_up_deep'],
        'ffn_down': ['ffn_down_shallow', 'ffn_down_middle', 'ffn_down_deep'],
    }
    depth_labels = ['Early', 'Mid', 'Late']

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))

    # (a) Per-depth sensitivity with inter/intra group comparison
    ax = axes[0]
    positions = []
    all_vals = []
    all_colors = []
    tick_positions = []
    tick_labels = []
    group_names = list(type_map.keys())

    pos = 0
    for gi, gname in enumerate(group_names):
        bins = type_map[gname]
        vals = [db['groups'][b]['dppl_per_layer'] for b in bins]
        for di, v in enumerate(vals):
            bar = ax.bar(pos, v, color=PALETTE[gname], width=0.7,
                        edgecolor='#2C3E50', lw=0.3, alpha=0.7 + 0.1*di)
            ax.text(pos, v + (0.3 if v >= 0 else -1.2), depth_labels[di],
                    ha='center', fontsize=5.5, color='#555')
            pos += 1
        tick_positions.append(pos - 2)
        tick_labels.append(LABELS[gname])
        pos += 0.5

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('$\\Delta$PPL / layer')
    ax.set_title('(a) Per-depth sensitivity', fontweight='bold')
    ax.axhline(0, ls='-', color='k', lw=0.4)

    # (b) Inter-group vs intra-group gap summary
    ax = axes[1]
    stats = {}
    for gname in group_names:
        bins = type_map[gname]
        vals = [db['groups'][b]['dppl_per_layer'] for b in bins]
        stats[gname] = {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals),
        }

    x = np.arange(len(group_names))
    means = [stats[g]['mean'] for g in group_names]
    stds = [stats[g]['std'] for g in group_names]
    mins = [stats[g]['min'] for g in group_names]
    maxs = [stats[g]['max'] for g in group_names]

    colors = [PALETTE[g] for g in group_names]
    bars = ax.bar(x, means, color=colors, edgecolor='#2C3E50', lw=0.4,
                  width=0.6, alpha=0.85, zorder=3)
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='#2C3E50',
                capsize=3, lw=1.0, zorder=4)

    for i, (mn, mx) in enumerate(zip(mins, maxs)):
        ax.plot([i, i], [mn, mx], color='#7F8C8D', lw=1.5, zorder=2)
        ax.plot(i, mn, 'v', color='#7F8C8D', ms=4, zorder=5)
        ax.plot(i, mx, '^', color='#7F8C8D', ms=4, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[g] for g in group_names])
    ax.set_ylabel('$\\Delta$PPL / layer (mean $\\pm$ std)')
    ax.set_title('(b) Inter-group gap vs. intra-group range', fontweight='bold')
    ax.axhline(0, ls='-', color='k', lw=0.4)

    inter_gap = stats['ffn_down']['mean'] - stats['attn_qkv']['mean']
    max_intra = max(s['std'] for s in stats.values())
    ax.text(0.97, 0.92, f'Inter-group gap: {inter_gap:.1f}\nMax intra-std: {max_intra:.1f}',
            transform=ax.transAxes, ha='right', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7DC6F', alpha=0.7))

    plt.tight_layout(w_pad=1.2)
    savefig('fig_grouping_validity')
    return stats


# ═══════════════════════════════════════════════════════════════════
# Fig C: Statistical Robustness (5 seeds)
# ═══════════════════════════════════════════════════════════════════
def fig_robustness():
    print("[Robustness] 5-seed statistical analysis")
    ci = json.load(open(PHASE2 / 'statistical_ci.json'))

    configs = ['uniform_7b', 'ilp_20pct', 'greedy_sens']
    labels_cfg = ['Uniform 7b', 'ILP 20%', 'Greedy (sens.)']
    colors_cfg = ['#BDC3C7', '#C0392B', '#2980B9']

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4))

    # (a) PPL distributions
    ax = axes[0]
    for i, (cfg, label, color) in enumerate(zip(configs, labels_cfg, colors_cfg)):
        vals = ci[cfg]['values']
        mean = ci[cfg]['mean']
        std = ci[cfg]['std']
        ax.bar(i, mean, color=color, edgecolor='#2C3E50', lw=0.4,
               width=0.55, alpha=0.85, zorder=3)
        ax.errorbar(i, mean, yerr=std, fmt='none', ecolor='#2C3E50',
                    capsize=4, lw=1.2, zorder=4)
        ax.text(i, mean + std + 5, f'{mean:.1f}\n$\\pm${std:.1f}',
                ha='center', fontsize=6, color='#2C3E50')

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels_cfg, fontsize=7)
    ax.set_ylabel('PPL (5 seeds)')
    ax.set_title('(a) PPL stability across seeds', fontweight='bold')

    # (b) Relative degradation with error bars
    ax = axes[1]
    baseline_vals = np.array(ci['uniform_7b']['values'])
    ilp_vals = np.array(ci['ilp_20pct']['values'])
    greedy_vals = np.array(ci['greedy_sens']['values'])

    ilp_rel = (ilp_vals - baseline_vals) / baseline_vals * 100
    greedy_rel = (greedy_vals - baseline_vals) / baseline_vals * 100

    data = [ilp_rel, greedy_rel]
    bp = ax.boxplot(data, labels=['ILP 20%', 'Greedy (sens.)'],
                    patch_artist=True, widths=0.5,
                    boxprops=dict(lw=0.6),
                    medianprops=dict(color='#2C3E50', lw=1.2),
                    whiskerprops=dict(lw=0.6),
                    capprops=dict(lw=0.6))
    bp['boxes'][0].set_facecolor('#C0392B')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('#2980B9')
    bp['boxes'][1].set_alpha(0.5)

    ax.axhline(0, ls='--', color='#7F8C8D', lw=0.6)
    ax.set_ylabel('Relative degradation (%)')
    ax.set_title('(b) Paired relative degradation', fontweight='bold')

    ilp_mean_rel = np.mean(ilp_rel)
    greedy_mean_rel = np.mean(greedy_rel)
    ax.text(0.97, 0.92, f'ILP mean: {ilp_mean_rel:+.1f}%\nGreedy mean: {greedy_mean_rel:+.1f}%',
            transform=ax.transAxes, ha='right', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', alpha=0.7))

    plt.tight_layout(w_pad=1.2)
    savefig('fig_robustness')


# ═══════════════════════════════════════════════════════════════════
# Fig D: Transfer allocation comparison (7→6 vs 8→7 profiling)
# ═══════════════════════════════════════════════════════════════════
def fig_allocation_transfer():
    print("[Allocation Transfer] 7→6 vs 8→7 profiling comparison")
    rc = json.load(open(PHASE2 / 'rank_consistency.json'))

    groups = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']
    n_layers = {'attn_qkv': 36, 'attn_out': 12, 'ffn_up': 12, 'ffn_down': 12, 'lm_head': 1}

    ppa = {}
    with open(RES / 'ppa/opt125m/ppa_sweep_opt125m.csv') as f:
        for r in csv.DictReader(f):
            ppa[int(r['adc_bits'])] = float(r['adc_area_um2'])

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4))

    probes_to_compare = ['7to6', '8to7', '9to8', '10to9']
    probe_labels = ['7b→6b', '8b→7b', '9b→8b', '10b→9b']
    colors_p = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB']

    # (a) Sensitivity ranking comparison
    ax = axes[0]
    for pi, (probe, plabel) in enumerate(zip(probes_to_compare, probe_labels)):
        sens = rc['probe_results'][probe]['sens']
        vals = [sens[g] for g in groups]
        order = np.argsort(vals)[::-1]
        rank_labels = [groups[i] for i in order]
        for ri, g in enumerate(rank_labels):
            ax.scatter(ri, pi, s=80, c=PALETTE[g], edgecolors='#2C3E50',
                       lw=0.5, zorder=3)
            if pi == 0:
                ax.text(ri, -0.6, LABELS[g], ha='center', fontsize=6.5,
                        color=PALETTE[g], fontweight='bold')

    ax.set_yticks(range(len(probes_to_compare)))
    ax.set_yticklabels(probe_labels)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(['Rank 1\n(most)', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5\n(least)'],
                       fontsize=6)
    ax.set_title('(a) Sensitivity rank per operating point', fontweight='bold')
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.0, 3.5)
    ax.invert_yaxis()

    # (b) ILP allocation comparison: what bits would ILP assign at 20% budget
    # using sensitivity from different probes
    ax = axes[1]
    N_total = sum(n_layers.values())
    area_per_layer = {b: ppa[b] / N_total for b in [4, 5, 6, 7]}
    baseline_area = ppa[7]
    target_savings = 0.20
    budget = baseline_area * (1 - target_savings)

    alloc_results = {}
    for probe in probes_to_compare:
        sens = rc['probe_results'][probe]['sens']
        sorted_groups = sorted(groups, key=lambda g: sens[g])

        alloc = {g: 7 for g in groups}
        current_area = sum(n_layers[g] * area_per_layer[7] for g in groups)

        for g in sorted_groups:
            if current_area <= budget:
                break
            area_saved = n_layers[g] * (area_per_layer[7] - area_per_layer[6])
            if current_area - area_saved >= budget:
                alloc[g] = 6
                current_area -= area_saved

        alloc_results[probe] = alloc

    x = np.arange(len(groups))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    for pi, (probe, plabel) in enumerate(zip(probes_to_compare, probe_labels)):
        bits = [alloc_results[probe][g] for g in groups]
        ax.bar(x + offsets[pi], bits, width, label=plabel, color=colors_p[pi],
               edgecolor='#2C3E50', lw=0.3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[g] for g in groups])
    ax.set_ylabel('Assigned ADC bits')
    ax.set_title('(b) Greedy allocation @ 20% savings', fontweight='bold')
    ax.set_ylim(5.5, 7.5)
    ax.set_yticks([6, 7])
    ax.axhline(7, ls='--', color='#7F8C8D', lw=0.5)
    ax.legend(fontsize=6, ncol=2, loc='lower left', framealpha=0.85, edgecolor='none')

    overlap = sum(1 for g in groups
                  if alloc_results['7to6'][g] == alloc_results['8to7'][g])
    ax.text(0.97, 0.92, f'7→6 vs 8→7\noverlap: {overlap}/{len(groups)}',
            transform=ax.transAxes, ha='right', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', alpha=0.7))

    plt.tight_layout(w_pad=1.2)
    savefig('fig_allocation_transfer')
    return alloc_results


if __name__ == '__main__':
    print(f"Output: {OUT}")
    fig_transfer()
    stats = fig_grouping_validity()
    fig_robustness()
    alloc_results = fig_allocation_transfer()

    # Print summary statistics for paper
    print("\n=== Paper Statistics ===")
    print(f"Inter-group gap (fc2 - qkv mean): {stats['ffn_down']['mean'] - stats['attn_qkv']['mean']:.1f}")
    print(f"Max intra-group std: {max(s['std'] for s in stats.values()):.1f}")
    print(f"Min inter-group gap / Max intra-std ratio: "
          f"{(stats['ffn_down']['mean'] - stats['attn_qkv']['mean']) / max(s['std'] for s in stats.values()):.1f}x")

    groups = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']
    print(f"\nAllocation overlap (7→6 vs 8→7): "
          f"{sum(1 for g in groups if alloc_results['7to6'][g] == alloc_results['8to7'][g])}/{len(groups)}")
    print(f"Allocation overlap (7→6 vs 10→9): "
          f"{sum(1 for g in groups if alloc_results['7to6'][g] == alloc_results['10to9'][g])}/{len(groups)}")

    print("\nAll figures generated.")
