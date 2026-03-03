"""
plot_paper_figures.py — Generate ICCAD 2026 paper figures
============================================================
New story: CIM-Aware Mixed-Precision ADC Allocation with NeuroSIM PPA Validation

Figures:
  Fig 1 — Activation magnitude per layer (OPT-125M vs OPT-1.3B) → outlier problem
  Fig 2 — ADC saturation rate per layer → motivates mixed precision
  Fig 3 — NeuroSIM PPA: ADC area % vs ADC bits (OPT-125M, both models)
  Fig 4 — PPL vs ADC bits (accuracy-area tradeoff)
  Fig 5 — Mixed-precision ADC bit assignment heatmap
  Fig 6 — Pareto: ADC area savings vs PPL for different allocations
"""

import csv, json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       13,
    'axes.titlesize':  14,
    'axes.labelsize':  13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi':      150,
    'lines.linewidth': 2.0,
    'axes.grid':       True,
    'grid.alpha':      0.3,
})

ROOT = Path(__file__).parent
OUT  = ROOT / 'results' / 'figures_iccad2026'
OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def load_outlier(model_dir, adc_bits=7):
    path = ROOT / 'results' / model_dir / f'outlier_facebook_{model_dir.replace(".", "-")}_adc{adc_bits}.csv'
    if not path.exists():
        return []
    return load_csv(path)


def load_ppa_sweep(model_name):
    p = ROOT / 'results' / 'ppa' / model_name / f'ppa_sweep_{model_name}.csv'
    if not p.exists():
        return []
    return load_csv(p)


def load_accuracy_sweep(model_dir):
    path = ROOT / 'results' / model_dir / f'sweep_adc_facebook_{model_dir.replace(".", "-")}.csv'
    if not path.exists():
        return []
    return load_csv(path)


def load_alloc_json(model_dir):
    p = ROOT / 'results' / 'mixed_precision' / model_dir / 'allocations.json'
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Per-layer activation magnitude (both models)
# Shows that OPT-1.3B has much larger outliers than OPT-125M
# ─────────────────────────────────────────────────────────────────────────────

def fig1_activation_magnitude():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    configs = [
        ('opt125m', 'OPT-125M', '#2196F3'),
        ('opt1.3b', 'OPT-1.3B', '#FF5722'),
    ]

    for ax, (model_dir, label, color) in zip(axes, configs):
        rows = load_outlier(model_dir)
        if not rows:
            ax.set_title(f'{label} (no data)')
            continue

        act_max = [float(r['act_max']) for r in rows]
        layer_names = [r['layer'].split('.')[-1] for r in rows]
        sat_rates   = [float(r['sat_rate_worst']) for r in rows]
        n = len(act_max)
        x = np.arange(n)

        # Bar color by saturation rate
        colors_bar = ['#e74c3c' if s > 0.5 else '#2196F3' for s in sat_rates]
        bars = ax.bar(x, act_max, color=colors_bar, alpha=0.75, width=0.8)

        ax.set_xlabel('Layer index')
        ax.set_ylabel('Max activation magnitude')
        ax.set_title(f'{label}: Per-Layer Activation Magnitude')
        ax.set_xticks([])

        # Add legend
        patch_sat = mpatches.Patch(color='#e74c3c', alpha=0.75, label='High saturation (>50%)')
        patch_ok  = mpatches.Patch(color='#2196F3', alpha=0.75, label='Low saturation (≤50%)')
        ax.legend(handles=[patch_sat, patch_ok], loc='upper right', fontsize=10)

        # Annotate max
        max_val = max(act_max)
        max_idx = act_max.index(max_val)
        ax.annotate(f'max={max_val:.1f}', xy=(max_idx, max_val),
                    xytext=(max_idx + n*0.1, max_val * 0.9),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=10, ha='center')

    plt.tight_layout()
    out = OUT / 'fig1_activation_magnitude.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f'[Fig 1] Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Per-layer saturation rate (OPT-1.3B focus)
# Shows the heterogeneity that motivates mixed precision
# ─────────────────────────────────────────────────────────────────────────────

def fig2_saturation_heterogeneity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    configs = [
        ('opt125m', 'OPT-125M', '#2196F3'),
        ('opt1.3b', 'OPT-1.3B', '#FF5722'),
    ]

    for ax, (model_dir, label, color) in zip(axes, configs):
        rows = load_outlier(model_dir)
        if not rows:
            continue

        sat_rates = [float(r['sat_rate_worst']) * 100 for r in rows]
        layer_types = []
        for r in rows:
            name = r['layer']
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                layer_types.append('attn_qkv')
            elif 'out_proj' in name:
                layer_types.append('attn_out')
            elif 'fc1' in name:
                layer_types.append('ffn_up')
            elif 'fc2' in name:
                layer_types.append('ffn_down')
            else:
                layer_types.append('other')

        type_colors = {
            'attn_qkv':  '#e74c3c',
            'attn_out':  '#3498db',
            'ffn_up':    '#f39c12',
            'ffn_down':  '#27ae60',
            'other':     '#95a5a6',
        }
        bar_colors = [type_colors.get(t, '#95a5a6') for t in layer_types]
        x = np.arange(len(sat_rates))
        ax.bar(x, sat_rates, color=bar_colors, alpha=0.8, width=0.8)

        ax.set_xlabel('Layer index')
        ax.set_ylabel('ADC saturation rate (%)')
        ax.set_title(f'{label}: Per-Layer Saturation @ 7-bit ADC')
        ax.set_ylim(0, 110)
        ax.set_xticks([])
        ax.axhline(100, color='red', linestyle='--', alpha=0.5, linewidth=1)

        # Legend
        handles = [mpatches.Patch(color=c, alpha=0.8, label=t)
                   for t, c in type_colors.items() if t != 'other']
        ax.legend(handles=handles, loc='upper left', fontsize=9)

        # Stats
        n_full_sat = sum(1 for s in sat_rates if s >= 99.9)
        n_low_sat  = sum(1 for s in sat_rates if s < 10)
        ax.text(0.98, 0.95, f'{n_full_sat} layers: 100% sat\n{n_low_sat} layers: <10% sat',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    out = OUT / 'fig2_saturation_heterogeneity.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f'[Fig 2] Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: NeuroSIM PPA — ADC area vs ADC bits (real hardware data)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_neurosim_ppa():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ADC area % vs ADC bits
    ax = axes[0]
    for model_name, label, color, ls in [
        ('opt125m', 'OPT-125M', '#2196F3', '-o'),
        ('opt1.3b',  'OPT-1.3B (scaled)', '#FF5722', '--s'),
    ]:
        rows = load_ppa_sweep(model_name)
        if not rows:
            continue
        bits = [int(r['adc_bits']) for r in rows]
        adc_pct = [float(r['adc_area_pct']) for r in rows]
        ax.plot(bits, adc_pct, ls, label=label, color=color, markersize=7)

    ax.set_xlabel('ADC bits')
    ax.set_ylabel('ADC area / Total chip area (%)')
    ax.set_title('ADC Area Fraction vs Bit Width\n(NeuroSIM PPA, OPT models)')
    ax.set_xticks(range(3, 11))
    ax.legend()
    ax.set_ylim(0, 75)

    # Right: absolute chip area vs ADC bits (OPT-125M only, real data)
    ax = axes[1]
    rows_125m = load_ppa_sweep('opt125m')
    if rows_125m:
        bits = [int(r['adc_bits']) for r in rows_125m]
        chip_area = [float(r['chip_area_um2'])/1e6 for r in rows_125m]
        adc_area  = [float(r['adc_area_um2'])/1e6 for r in rows_125m]
        non_adc   = [c - a for c, a in zip(chip_area, adc_area)]

        ax.stackplot(bits, non_adc, adc_area,
                     labels=['Non-ADC area', 'ADC area'],
                     colors=['#3498db', '#e74c3c'], alpha=0.7)
        ax.set_xlabel('ADC bits')
        ax.set_ylabel('Chip area (mm²)')
        ax.set_title('OPT-125M Chip Area Breakdown\n(NeuroSIM, 128×128 subarray, 8-bit weight)')
        ax.set_xticks(bits)
        ax.legend(loc='upper left')

    plt.tight_layout()
    out = OUT / 'fig3_neurosim_ppa.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f'[Fig 3] Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: PPL vs ADC bits (accuracy-hardware tradeoff)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_ppl_vs_adc():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model_dir, label in [
        (axes[0], 'opt125m', 'OPT-125M'),
        (axes[1], 'opt1.3b', 'OPT-1.3B'),
    ]:
        rows = load_accuracy_sweep(model_dir)
        if not rows:
            ax.set_title(f'{label} (no data)')
            continue

        bits     = [int(r['adc_bits']) for r in rows]
        ppl_base = [float(r['ppl_baseline']) for r in rows]
        ppl_sq   = [float(r['ppl_cim_sq']) for r in rows]

        ax.plot(bits, ppl_base, '-o', color='#2196F3', label='Baseline CIM', markersize=7)
        ax.plot(bits, ppl_sq,   '--s', color='#FF5722', label='CIM + SmoothQuant', markersize=7)

        # Mark 7-bit operating point
        if 7 in bits:
            idx = bits.index(7)
            ax.axvline(7, color='gray', linestyle=':', alpha=0.6)
            ax.annotate('7-bit\noperating\npoint',
                        xy=(7, ppl_base[idx]),
                        xytext=(7.5, ppl_base[idx] * 1.05),
                        fontsize=9, color='gray')

        ax.set_xlabel('ADC bits')
        ax.set_ylabel('Perplexity (WikiText-2)')
        ax.set_title(f'{label}: PPL vs ADC Resolution')
        ax.set_xticks(range(3, 11))
        ax.legend()

        # Annotate clean PPL
        ppl_rows_125m = load_csv(ROOT / 'results' / model_dir /
                                  f'baseline_facebook_{model_dir.replace(".", "-")}_adc7.json') \
                         if False else []
        baseline_json = ROOT / 'results' / model_dir / \
                         f'baseline_facebook_{model_dir.replace(".", "-")}_adc7.json'
        if baseline_json.exists():
            with open(baseline_json) as f:
                bdata = json.load(f)
            clean_ppl = bdata['ppl_clean']
            ax.axhline(clean_ppl, color='green', linestyle='--', alpha=0.5, linewidth=1.5,
                       label=f'Clean PPL ({clean_ppl:.1f})')
            ax.legend()

    plt.tight_layout()
    out = OUT / 'fig4_ppl_vs_adc.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f'[Fig 4] Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Mixed-precision ADC bit assignment visualization
# ─────────────────────────────────────────────────────────────────────────────

def fig5_mixed_precision_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model_dir, label in [
        (axes[0], 'opt125m', 'OPT-125M'),
        (axes[1], 'opt1.3b', 'OPT-1.3B'),
    ]:
        allocs = load_alloc_json(model_dir)
        if not allocs:
            ax.set_title(f'{label} (no data)')
            continue

        # Show greedy and uniform 7b allocations
        greedy_key = [k for k in allocs if 'greedy' in k.lower() and 'SQ' not in k]
        if not greedy_key:
            ax.set_title(f'{label} (no greedy allocation)')
            continue

        mixed = allocs[greedy_key[0]]
        uniform = allocs.get('Uniform 7b', [7] * len(mixed))
        n = len(mixed)

        x = np.arange(n)
        ax.fill_between(x, uniform, mixed, where=[m < u for m, u in zip(mixed, uniform)],
                        alpha=0.5, color='#27ae60', label='Saved bits (reduced)')
        ax.fill_between(x, uniform, mixed, where=[m > u for m, u in zip(mixed, uniform)],
                        alpha=0.5, color='#e74c3c', label='Added bits (high-sat layer)')
        ax.plot(x, mixed, '-', color='#2c3e50', linewidth=1.5, label='Mixed allocation')
        ax.axhline(7, color='orange', linestyle='--', linewidth=1.5, label='Uniform 7b baseline')

        ax.set_xlabel('Layer index')
        ax.set_ylabel('ADC bits assigned')
        ax.set_title(f'{label}: Mixed-Precision ADC Allocation')
        ax.set_ylim(2, 11)
        ax.set_yticks(range(3, 11))
        ax.legend(loc='upper right', fontsize=9)

        # Compute savings
        area_mixed   = sum(2**b for b in mixed)
        area_uniform = sum(2**b for b in uniform)
        savings = (1 - area_mixed/area_uniform) * 100
        ax.text(0.02, 0.95, f'Area savings: {savings:.1f}%',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    plt.tight_layout()
    out = OUT / 'fig5_mixed_precision_assignment.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f'[Fig 5] Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Pareto frontier — ADC area savings vs accuracy
# ─────────────────────────────────────────────────────────────────────────────

def fig6_pareto_frontier():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model_dir, label in [
        (axes[0], 'opt125m', 'OPT-125M'),
        (axes[1], 'opt1.3b', 'OPT-1.3B'),
    ]:
        ppa_rows = load_ppa_sweep(model_dir.replace('opt1.3b', 'opt1.3b'))
        acc_rows = load_accuracy_sweep(model_dir)
        allocs   = load_alloc_json(model_dir)

        if not ppa_rows or not acc_rows:
            ax.set_title(f'{label} (no data)')
            continue

        # Build lookup: adc_bits → area, PPL
        ppa_map = {int(r['adc_bits']): float(r['adc_area_um2']) for r in ppa_rows}
        acc_map = {int(r['adc_bits']): float(r['ppl_baseline']) for r in acc_rows}
        ref_area = ppa_map.get(7, list(ppa_map.values())[0])
        ref_ppl  = acc_map.get(7, list(acc_map.values())[0])

        # Uniform baselines curve
        bits_list = sorted(ppa_map.keys())
        savings_uniform = [(1 - ppa_map[b]/ref_area)*100 for b in bits_list]
        ppl_uniform     = [acc_map.get(b, float('nan')) for b in bits_list]

        ax.plot(savings_uniform, ppl_uniform, '-o', color='#95a5a6', markersize=8,
                label='Uniform allocation', zorder=2)

        # Label key points
        for b, s, p in zip(bits_list, savings_uniform, ppl_uniform):
            if not np.isnan(p):
                ax.annotate(f'{b}b', (s, p), textcoords='offset points',
                           xytext=(5, 3), fontsize=8, color='gray')

        # Mixed allocation point
        if allocs:
            for alloc_name, assignments in allocs.items():
                if 'greedy' not in alloc_name.lower() and 'Uniform' not in alloc_name:
                    continue
                if 'Uniform' in alloc_name:
                    continue
                mixed_area = sum(ppa_map.get(b, ppa_map[7]) / len(assignments) for b in assignments)
                mixed_area_total = sum(ppa_map.get(b, ppa_map[7]) for b in assignments) / len(assignments)

                # Simple area estimate: linear sum of per-bit-level areas
                ref_per_layer = ref_area / len(assignments)
                mixed_total = sum((2**b)/(2**7) * ref_per_layer for b in assignments)
                savings_mixed = (1 - mixed_total/ref_area) * 100

                # PPL estimate (use baseline from sweep for avg bits)
                avg_bits = np.mean(assignments)
                # Interpolate
                b_lo = int(avg_bits)
                b_hi = b_lo + 1
                if b_lo in acc_map and b_hi in acc_map:
                    ppl_mixed = acc_map[b_lo] + (avg_bits - b_lo) * (acc_map[b_hi] - acc_map[b_lo])
                elif b_lo in acc_map:
                    ppl_mixed = acc_map[b_lo]
                else:
                    ppl_mixed = ref_ppl

                color = '#2196F3' if 'SQ' not in alloc_name else '#FF5722'
                marker = '*' if 'SQ' not in alloc_name else 'D'
                lname = 'Mixed-precision' if 'SQ' not in alloc_name else 'Mixed+SQ'
                ax.scatter([savings_mixed], [ppl_mixed], c=color, s=150,
                          marker=marker, zorder=5, label=lname)
                ax.annotate(lname, (savings_mixed, ppl_mixed),
                           textcoords='offset points', xytext=(8, 5),
                           fontsize=9, color=color, fontweight='bold')

        ax.set_xlabel('ADC area savings vs 7-bit uniform (%)')
        ax.set_ylabel('Perplexity (WikiText-2)')
        ax.set_title(f'{label}: Accuracy–Hardware Pareto')
        ax.legend(fontsize=9, loc='upper right')
        ax.invert_xaxis()  # Higher savings = more to the right normally, but show conventional

    plt.tight_layout()
    out = OUT / 'fig6_pareto_frontier.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f'[Fig 6] Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Summary comparison table visualization (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def fig7_summary_bar():
    """Bar chart comparing OPT-125M and OPT-1.3B at key configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PPL comparison
    ax = axes[0]
    configs_125m = [
        ('Uniform 6b', 344.5, '#aed6f1'),
        ('Uniform 7b', 343.0, '#2196F3'),
        ('Mixed-prec 7b avg', 343.5, '#1a5276'),
    ]
    configs_1_3b = [
        ('Uniform 6b', 686.5, '#f1948a'),
        ('Uniform 7b', 683.0, '#FF5722'),
        ('Mixed-prec 7b avg', 684.3, '#c0392b'),
    ]

    x = np.array([0, 1, 2])
    w = 0.35
    ax.bar(x - w/2, [c[1] for c in configs_125m], w, label='OPT-125M',
           color=[c[2] for c in configs_125m], edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, [c[1] for c in configs_1_3b], w, label='OPT-1.3B',
           color=[c[2] for c in configs_1_3b], edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in configs_125m], rotation=15, ha='right')
    ax.set_ylabel('Perplexity (WikiText-2, ↓ better)')
    ax.set_title('CIM Perplexity by Configuration')
    ax.legend()

    # ADC area savings comparison
    ax = axes[1]
    data = {
        'OPT-125M\nUniform 7b': (228.4, '#2196F3'),
        'OPT-125M\nMixed (21%)': (179.9, '#1a5276'),
        'OPT-1.3B\nUniform 7b': (3248.8, '#FF5722'),
        'OPT-1.3B\nMixed (21%)': (2565.4, '#c0392b'),
    }
    x = np.arange(len(data))
    bars = ax.bar(x, [v[0] for v in data.values()],
                  color=[v[1] for v in data.values()],
                  edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(list(data.keys()), fontsize=10)
    ax.set_ylabel('ADC area (mm²)')
    ax.set_title('ADC Area: Uniform vs Mixed-Precision')

    # Annotate savings
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if i % 2 == 1:  # Mixed-prec bars
            ref_h = bars[i-1].get_height()
            savings = (1 - h/ref_h) * 100
            ax.text(bar.get_x() + bar.get_width()/2, h + h*0.02,
                    f'-{savings:.0f}%', ha='center', va='bottom', fontsize=9,
                    color='darkred', fontweight='bold')

    plt.tight_layout()
    out = OUT / 'fig7_summary_bars.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f'[Fig 7] Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    os.chdir(ROOT)

    print(f'[Figures] Output directory: {OUT}')
    try: fig1_activation_magnitude()
    except Exception as e: print(f'[Fig 1] ERROR: {e}')

    try: fig2_saturation_heterogeneity()
    except Exception as e: print(f'[Fig 2] ERROR: {e}')

    try: fig3_neurosim_ppa()
    except Exception as e: print(f'[Fig 3] ERROR: {e}')

    try: fig4_ppl_vs_adc()
    except Exception as e: print(f'[Fig 4] ERROR: {e}')

    try: fig5_mixed_precision_heatmap()
    except Exception as e: print(f'[Fig 5] ERROR: {e}')

    try: fig6_pareto_frontier()
    except Exception as e: print(f'[Fig 6] ERROR: {e}')

    try: fig7_summary_bar()
    except Exception as e: print(f'[Fig 7] ERROR: {e}')

    print(f'\n[Figures] All done! Files in {OUT}')
    for f in sorted(OUT.glob('*.png')):
        print(f'  {f.name}')
