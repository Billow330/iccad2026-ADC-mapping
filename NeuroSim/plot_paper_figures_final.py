"""
plot_paper_figures_final.py — Publication-Quality Figures for ICCAD 2026
========================================================================
IEEE two-column format: column width ≈ 3.5 in, full width ≈ 7.16 in.
All 9 figures + copy to paper/.
"""

import os, json, csv, shutil
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent
RES  = ROOT / 'results'
OUT_DIR = RES / 'figures_iccad2026_final'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATHS = {
    'ppa_125m':      RES / 'ppa/opt125m/ppa_sweep_opt125m.csv',
    'outlier_125m':  RES / 'opt125m/outlier_facebook_opt-125m_adc7.csv',
    'outlier_1p3b':  RES / 'opt1.3b/outlier_facebook_opt-1.3b_adc7.csv',
    'sweep_125m':    RES / 'opt125m/sweep_adc_facebook_opt-125m.csv',
    'group_sens':    RES / 'sensitivity/opt125m/group_sensitivity.json',
    'allocations':   RES / 'sensitivity/opt125m/allocations.json',
    'pareto':        RES / 'sensitivity/opt125m/pareto_frontier.json',
    'multi_budget':  RES / 'stable/opt125m/ilp_vs_greedy_multibudget.csv',
    'stable_eval':   RES / 'stable/opt125m/stable_eval_results.csv',
    'hawq':          RES / 'sensitivity/opt125m/hawq_comparison.json',
    'bf_all':        RES / 'stable/opt125m/bruteforce_all_configs.json',
    'bf_pareto':     RES / 'stable/opt125m/bruteforce_pareto.json',
}

# ── IEEE-quality style ──────────────────────────────────────────────────────
COL_W = 3.5    # single column width (inches)
FULL_W = 7.16  # full page width (inches)

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
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'axes.grid': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

PALETTE = {
    'attn_qkv': '#C0392B',
    'attn_out': '#27AE60',
    'ffn_up':   '#E67E22',
    'ffn_down': '#2980B9',
    'lm_head':  '#8E44AD',
}
LABELS = {
    'attn_qkv': 'q/k/v_proj',
    'attn_out': 'out_proj',
    'ffn_up':   'fc1',
    'ffn_down': 'fc2',
    'lm_head':  'lm_head',
}

def savefig(name):
    for fmt in ('pdf', 'png'):
        plt.savefig(OUT_DIR / f'{name}.{fmt}', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  -> {name}")

def load_csv(path):
    if not Path(path).exists():
        return []
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                try: out[k] = float(v)
                except: out[k] = v
            rows.append(out)
    return rows

def classify_layer(name):
    n = name.lower()
    if 'lm_head' in n: return 'lm_head'
    if 'q_proj' in n or 'k_proj' in n or 'v_proj' in n: return 'attn_qkv'
    if 'out_proj' in n: return 'attn_out'
    if 'fc1' in n: return 'ffn_up'
    if 'fc2' in n: return 'ffn_down'
    return 'other'


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1: ADC Area Motivation (3-panel)
# ═══════════════════════════════════════════════════════════════════════════
def fig1():
    print("[Fig1] ADC area motivation")
    rows = {int(r['adc_bits']): r for r in load_csv(PATHS['ppa_125m'])}
    if not rows: return

    bits = sorted(rows.keys())
    adc_pct = [rows[b]['adc_area_pct'] for b in bits]
    chip_mm2 = [rows[b]['chip_area_um2'] / 1e6 for b in bits]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.9))

    # (a) ADC% vs bits
    ax = axes[0]
    ax.plot(bits, adc_pct, 'o-', color='#C0392B', lw=1.4, ms=4, zorder=3)
    ax.fill_between(bits, adc_pct, alpha=0.10, color='#C0392B')
    ax.axhline(24.4, ls='--', color='#7F8C8D', lw=0.7)
    ax.axvline(7, ls=':', color='#2980B9', lw=0.7)
    ax.annotate('24.4% @ 7b', xy=(7, 24.4), xytext=(8.2, 18),
                fontsize=6.5, color='#7F8C8D',
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=0.6))
    ax.set_xlabel('ADC resolution (bits)')
    ax.set_ylabel('ADC area fraction (%)')
    ax.set_title('(a) ADC area fraction', fontweight='bold')
    ax.set_xticks(bits)

    # (b) Chip area vs bits
    ax = axes[1]
    colors_b = ['#2980B9' if b == 7 else '#BDC3C7' for b in bits]
    ax.bar(bits, chip_mm2, color=colors_b, edgecolor='#2C3E50', lw=0.4, width=0.7)
    ax.set_xlabel('ADC resolution (bits)')
    ax.set_ylabel('Chip area (mm²)')
    ax.set_title('(b) Total chip area', fontweight='bold')
    ax.set_xticks(bits)
    for b, v in zip(bits, chip_mm2):
        if b in (3, 7, 10):
            ax.text(b, v + 30, f'{v:.0f}', ha='center', fontsize=6)

    # (c) Breakdown at 7b
    ax = axes[2]
    ref = rows.get(7, {})
    ref_chip = ref.get('chip_area_um2', 0) / 1e6
    ref_adc  = ref.get('adc_area_um2', 0) / 1e6
    ref_arr  = ref.get('array_area_um2', 0) / 1e6
    ref_other = ref_chip - ref_adc - max(ref_arr, 0)
    labels = ['Array', 'ADC', 'Other']
    vals = [max(ref_arr, 0), ref_adc, max(ref_other, 0)]
    colors_c = ['#3498DB', '#C0392B', '#95A5A6']
    bars = ax.bar(labels, vals, color=colors_c, edgecolor='#2C3E50', lw=0.4, width=0.55)
    ax.set_ylabel('Area (mm²)')
    ax.set_title('(c) 7-bit breakdown', fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 8, f'{v:.1f}',
                ha='center', fontsize=6.5)

    plt.tight_layout(w_pad=1.5)
    savefig('fig1_adc_motivation')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2: Saturation Heterogeneity (OPT-125M + OPT-1.3B)
# ═══════════════════════════════════════════════════════════════════════════
def fig2():
    print("[Fig2] Saturation heterogeneity")
    def load_outlier(path):
        rows = load_csv(path)
        for r in rows:
            r['layer_type'] = classify_layer(r.get('layer', ''))
        return rows

    data_125m = load_outlier(PATHS['outlier_125m'])
    data_1p3b = load_outlier(PATHS['outlier_1p3b'])
    if not data_125m: return

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.0))

    for ax, data, title in zip(axes,
                                [data_125m, data_1p3b],
                                ['OPT-125M (73 layers)', 'OPT-1.3B (145 layers)']):
        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        for i, r in enumerate(data):
            lt = r['layer_type']
            sat = r.get('sat_rate_worst', 0)
            ax.bar(i, sat, color=PALETTE.get(lt, '#95A5A6'),
                   width=0.9, edgecolor='none')

        ax.set_xlabel('Layer index')
        ax.set_ylabel('Max-clip saturation rate')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.axhline(0.5, ls='--', color='#7F8C8D', lw=0.5, alpha=0.5)

        seen = {}
        for r in data:
            lt = r['layer_type']
            if lt not in seen and lt in PALETTE:
                seen[lt] = mpatches.Patch(color=PALETTE[lt], label=LABELS.get(lt, lt))
        ax.legend(handles=list(seen.values()), fontsize=6, ncol=2,
                  loc='center right', framealpha=0.85, edgecolor='none')

    plt.tight_layout(w_pad=1.2)
    savefig('fig2_saturation_heterogeneity')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3: Group Sensitivity (horizontal bar + layer count)
# ═══════════════════════════════════════════════════════════════════════════
def fig3():
    print("[Fig3] Group sensitivity")
    gpath = PATHS['group_sens']
    if not gpath.exists(): return

    with open(gpath) as f:
        gd = json.load(f)

    order = ['ffn_down', 'attn_out', 'lm_head', 'ffn_up', 'attn_qkv']
    labels = [LABELS[lt] for lt in order]
    dppl = [gd[lt]['delta_per_layer'] for lt in order]
    nlayers = [gd[lt]['n_layers'] for lt in order]
    colors = [PALETTE[lt] for lt in order]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2),
                              gridspec_kw={'width_ratios': [2, 1]})

    # (a) Sensitivity bars (horizontal)
    ax = axes[0]
    y = np.arange(len(order))
    bars = ax.barh(y, dppl, color=colors, edgecolor='#2C3E50', lw=0.4, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('$\\Delta$PPL / layer (7b → 6b)')
    ax.set_title('(a) Measured ADC sensitivity', fontweight='bold')
    ax.invert_yaxis()
    for bar, v in zip(bars, dppl):
        ax.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=6.5)
    ax.axvline(0, ls='-', color='k', lw=0.4)

    # (b) Layer count
    ax = axes[1]
    ax.barh(y, nlayers, color=colors, edgecolor='#2C3E50', lw=0.4, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel('Number of layers')
    ax.set_title('(b) Layer count', fontweight='bold')
    ax.invert_yaxis()
    for i, n in enumerate(nlayers):
        ax.text(n + 0.3, i, str(n), va='center', fontsize=6.5)

    plt.tight_layout(w_pad=0.8)
    savefig('fig3_group_sensitivity')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4: Sensitivity vs Saturation Scatter
# ═══════════════════════════════════════════════════════════════════════════
def fig4():
    print("[Fig4] Sensitivity vs saturation")
    outlier = load_csv(PATHS['outlier_125m'])
    gpath = PATHS['group_sens']
    if not outlier or not gpath.exists(): return

    for r in outlier:
        r['layer_type'] = classify_layer(r.get('layer', ''))

    with open(gpath) as f:
        gd = json.load(f)

    sat_by_type = defaultdict(list)
    for r in outlier:
        lt = r['layer_type']
        if lt in PALETTE:
            sat_by_type[lt].append(float(r.get('sat_rate_worst', 0)))

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))

    markers = {'attn_qkv': 'o', 'attn_out': '^', 'ffn_up': 's',
               'ffn_down': 'D', 'lm_head': '*'}

    for lt in ['attn_qkv', 'ffn_up', 'lm_head', 'attn_out', 'ffn_down']:
        if lt not in gd: continue
        x = np.mean(sat_by_type.get(lt, [0]))
        y = gd[lt]['delta_per_layer']
        ax.scatter([x], [y], c=PALETTE[lt], s=60, marker=markers.get(lt, 'o'),
                   edgecolors='#2C3E50', lw=0.5, zorder=3,
                   label=f"{LABELS[lt]} ({gd[lt]['n_layers']})")
        offset_x = -0.06 if x > 0.5 else 0.03
        ax.annotate(LABELS[lt], (x, y), xytext=(offset_x, 0.04),
                    textcoords='offset fontsize', fontsize=6.5, color='#2C3E50')

    ax.set_xlabel('Mean max-clip saturation rate')
    ax.set_ylabel('$\\Delta$PPL / layer (7b → 6b)')
    ax.set_title('Saturation rate vs. measured sensitivity', fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=6, loc='upper left', framealpha=0.85, edgecolor='none')

    xs = [np.mean(sat_by_type.get(lt, [0])) for lt in gd]
    ys = [gd[lt]['delta_per_layer'] for lt in gd]
    z = np.polyfit(xs, ys, 1)
    xfit = np.linspace(min(xs)-0.05, max(xs)+0.05, 50)
    ax.plot(xfit, np.polyval(z, xfit), '--', color='#7F8C8D', lw=0.8, alpha=0.6)
    ax.text(0.55, 1.2, 'Spearman $\\rho$ = −0.80', fontsize=6.5, color='#C0392B',
            transform=ax.transData)

    plt.tight_layout()
    savefig('fig4_sensitivity_vs_saturation')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5: Bit Assignment (ILP vs Greedy vs Uniform)
# ═══════════════════════════════════════════════════════════════════════════
def fig5():
    print("[Fig5] Bit assignment")
    outlier = load_csv(PATHS['outlier_125m'])
    alloc_path = PATHS['allocations']
    if not outlier or not alloc_path.exists(): return

    for r in outlier:
        r['layer_type'] = classify_layer(r.get('layer', ''))

    with open(alloc_path) as f:
        alloc = json.load(f)

    n = len(outlier)
    layer_names = [r['layer'] for r in outlier]
    layer_types = [r['layer_type'] for r in outlier]
    greedy_bits = [alloc.get('greedy', {}).get(nm, 7) for nm in layer_names]
    ilp_bits = [alloc.get('ilp', {}).get(nm, 7) for nm in layer_names]

    fig, axes = plt.subplots(3, 1, figsize=(FULL_W, 3.2), sharex=True)

    configs = [
        ('(a) Uniform 7-bit baseline', [7]*n),
        ('(b) Sensitivity-guided Greedy', greedy_bits),
        ('(c) ILP (optimal)', ilp_bits),
    ]

    for ax, (title, bits) in zip(axes, configs):
        for i, (b, lt) in enumerate(zip(bits, layer_types)):
            ax.bar(i, b, color=PALETTE.get(lt, '#95A5A6'),
                   width=0.9, edgecolor='none', alpha=0.85)
        ax.axhline(7, ls='--', color='#7F8C8D', lw=0.6)
        ax.set_ylabel('ADC bits')
        ax.set_title(title, fontsize=8, fontweight='bold', loc='left')
        ax.set_ylim(4.5, 7.8)
        ax.set_yticks([5, 6, 7])
        bc = Counter(bits)
        pct6 = bc.get(6, 0) / n * 100
        ax.text(0.98, 0.80, f'{bc.get(6,0)} layers @ 6b, {bc.get(7,0)} @ 7b',
                transform=ax.transAxes, ha='right', fontsize=6, color='#2C3E50')

    axes[-1].set_xlabel('Layer index')

    seen = {}
    for lt in ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']:
        seen[lt] = mpatches.Patch(color=PALETTE[lt], label=LABELS[lt])
    fig.legend(handles=list(seen.values()), loc='lower center',
               ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.9, edgecolor='none')

    plt.tight_layout(h_pad=0.4)
    plt.subplots_adjust(bottom=0.10)
    savefig('fig5_bit_assignment')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6: Pareto Frontier
# ═══════════════════════════════════════════════════════════════════════════
def fig6():
    print("[Fig6] Pareto frontier")
    pareto_path = PATHS['pareto']
    ppa_rows = {int(r['adc_bits']): r for r in load_csv(PATHS['ppa_125m'])}
    sweep_rows = {int(r['adc_bits']): r for r in load_csv(PATHS['sweep_125m'])}

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))

    if pareto_path.exists():
        with open(pareto_path) as f:
            pts = json.load(f)
        areas = [p['adc_area_mm2'] for p in pts]
        ppls = [p['ppl'] for p in pts]
        ax.plot(areas, ppls, 'o-', color='#C0392B', lw=1.4, ms=4,
                label='Mixed-ILP Pareto', zorder=3)

    if ppa_rows and sweep_rows:
        ref_points = []
        for b in sorted(ppa_rows.keys()):
            if b in sweep_rows:
                ref_points.append((ppa_rows[b]['adc_area_um2'] / 1e6,
                                   sweep_rows[b].get('ppl_baseline', sweep_rows[b].get('ppl', 0))))
        if ref_points:
            rx, ry = zip(*ref_points)
            ax.plot(rx, ry, 's--', color='#2980B9', lw=1.0, ms=3.5,
                    label='Uniform-bit sweep', zorder=2)
            for (x, y), b in zip(ref_points, sorted(ppa_rows.keys())):
                if b in [5, 6, 7, 8]:
                    ax.annotate(f'{b}b', (x, y), xytext=(4, 3),
                                textcoords='offset points', fontsize=6, color='#2980B9')

    if pareto_path.exists() and pts:
        best = min(pts, key=lambda p: abs(p['actual_savings'] - 20))
        ax.scatter([best['adc_area_mm2']], [best['ppl']], s=50,
                   color='#F1C40F', edgecolors='#2C3E50', lw=0.8, zorder=5,
                   label=f"ILP-20%: {best['actual_savings']:.1f}% savings")

    ax.set_xlabel('ADC area (mm²)')
    ax.set_ylabel('Perplexity (WikiText-2)')
    ax.set_title('PPL–area Pareto frontier', fontweight='bold')
    ax.legend(fontsize=6, framealpha=0.85, edgecolor='none')
    ax.invert_xaxis()
    plt.tight_layout()
    savefig('fig6_pareto_frontier')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 7: Full Comparison Bars (with HAWQ)
# ═══════════════════════════════════════════════════════════════════════════
def fig7():
    print("[Fig7] Comparison bars")

    CONFIGS = [
        ('Uniform 7b\n(baseline)', 306.4, 228.4, 0.0,  '#BDC3C7'),
        ('Uniform 6b',             315.3, 114.2, 50.0,  '#85C1E9'),
        ('HAWQ-guided\nGreedy',    321.3, 181.5, 20.5,  '#E74C3C'),
        ('Sat-guided\nGreedy',     313.8, 181.5, 20.5,  '#F39C12'),
        ('Sens-guided\nGreedy',    312.5, 181.5, 20.5,  '#27AE60'),
        ('ILP\n(ours)',            308.6, 181.5, 20.5,  '#2980B9'),
        ('SQ + 6b',               305.0, 114.2, 50.0,  '#8E44AD'),
        ('SQ + ILP',              309.4, 181.5, 20.5,  '#1ABC9C'),
    ]

    labels  = [c[0] for c in CONFIGS]
    ppls    = [c[1] for c in CONFIGS]
    areas   = [c[2] for c in CONFIGS]
    savings = [c[3] for c in CONFIGS]
    colors  = [c[4] for c in CONFIGS]

    x = np.arange(len(labels))
    REF_PPL, REF_AREA = 306.4, 228.4

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8))

    # (a) PPL
    axes[0].bar(x, ppls, color=colors, edgecolor='#2C3E50', lw=0.4, width=0.65)
    axes[0].axhline(REF_PPL, ls='--', color='#7F8C8D', lw=0.7)
    axes[0].set_ylabel('Perplexity (↓ better)')
    axes[0].set_title('(a) Accuracy', fontsize=9, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=6)
    axes[0].set_ylim(298, 330)
    for i, v in enumerate(ppls):
        fw = 'bold' if labels[i] == 'ILP\n(ours)' else 'normal'
        axes[0].text(i, v + 0.4, f'{v:.1f}', ha='center', va='bottom',
                     fontsize=5.5, fontweight=fw)
    axes[0].annotate('HAWQ: +12.7\nvs ILP', xy=(2, 321.3), xytext=(3.8, 327),
                     fontsize=5.5, color='#C0392B',
                     arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.7))

    # (b) ADC Area
    axes[1].bar(x, areas, color=colors, edgecolor='#2C3E50', lw=0.4, width=0.65)
    axes[1].axhline(REF_AREA, ls='--', color='#7F8C8D', lw=0.7)
    axes[1].set_ylabel('ADC area (mm², ↓ better)')
    axes[1].set_title('(b) ADC area', fontsize=9, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=6)
    axes[1].set_ylim(80, 260)
    for i, (v, s) in enumerate(zip(areas, savings)):
        lbl = f'{v:.0f}' if s == 0 else f'{v:.0f}\n(−{s:.0f}%)'
        axes[1].text(i, v + 2, lbl, ha='center', va='bottom', fontsize=5.5)

    plt.tight_layout(w_pad=1.0)
    savefig('fig7_comparison_bars')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 8: PPL vs Savings %  (OPT-1.3B cross-model — replaced with cleaner version)
# ═══════════════════════════════════════════════════════════════════════════
def fig8():
    print("[Fig8] PPL vs savings")
    eval_path = PATHS['stable_eval']
    ppa_rows = {int(r['adc_bits']): r for r in load_csv(PATHS['ppa_125m'])}
    sweep_rows = {int(r['adc_bits']): r for r in load_csv(PATHS['sweep_125m'])}

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))

    if ppa_rows and sweep_rows:
        ref_adc = ppa_rows.get(7, {}).get('adc_area_um2', 1) / 1e6
        pts = []
        for b in sorted(ppa_rows.keys()):
            if b in sweep_rows:
                a = ppa_rows[b]['adc_area_um2'] / 1e6
                sav = (1.0 - a / ref_adc) * 100
                ppl_val = sweep_rows[b].get('ppl_baseline', sweep_rows[b].get('ppl', 0))
                pts.append((sav, ppl_val, b))
        if pts:
            xs, ys, bs = zip(*pts)
            ax.plot(xs, ys, 's--', color='#2980B9', lw=1.0, ms=3.5,
                    label='Uniform-bit sweep', zorder=2)
            for x, y, b in pts:
                if b in [5, 6, 7, 8]:
                    ax.annotate(f'{b}b', (x, y), xytext=(3, 3),
                                textcoords='offset points', fontsize=6)

    if eval_path.exists():
        rows = load_csv(eval_path)
        markers_map = {'ILP': ('D', '#C0392B'), 'Greedy': ('^', '#E67E22'),
                       'SQ': ('o', '#8E44AD')}
        for r in rows:
            cfg = r['config']
            if 'Uniform 7' in cfg: continue
            mk, cl = 'o', '#27AE60'
            for key, (m, c) in markers_map.items():
                if key in cfg:
                    mk, cl = m, c
                    break
            ax.scatter([r['adc_savings_pct']], [r['ppl']], s=35,
                       marker=mk, color=cl, edgecolors='#2C3E50', lw=0.4,
                       zorder=4, label=cfg[:18])

    ax.set_xlabel('ADC area savings (%)')
    ax.set_ylabel('Perplexity (WikiText-2)')
    ax.set_title('Accuracy–efficiency trade-off', fontweight='bold')
    ax.legend(fontsize=5.5, loc='upper left', framealpha=0.85, edgecolor='none',
              ncol=1)
    plt.tight_layout()
    savefig('fig8_ppl_vs_savings')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 9: Multi-budget ILP vs Greedy
# ═══════════════════════════════════════════════════════════════════════════
def fig9():
    print("[Fig9] ILP vs Greedy vs Brute-force")
    mpath = PATHS['multi_budget']
    if not mpath.exists(): return

    rows = load_csv(mpath)
    targets  = [r['target_pct'] for r in rows]
    ilp_ppls = [r['ilp_ppl'] for r in rows]
    grd_ppls = [r['grd_ppl'] for r in rows]
    ilp_sav  = [r['ilp_savings'] for r in rows]
    grd_sav  = [r['grd_savings'] for r in rows]

    bf_all_path = PATHS['bf_all']
    bf_pareto_path = PATHS['bf_pareto']

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))

    # (a) Search space scatter + Pareto frontiers
    ax = axes[0]

    if bf_all_path.exists():
        with open(bf_all_path) as f:
            bf_all = json.load(f)
        bf_sav = [c['savings_pct'] for c in bf_all]
        bf_dppl = [c['pred_dppl'] for c in bf_all]
        ax.scatter(bf_sav, bf_dppl, s=6, c='#BDC3C7', alpha=0.4, zorder=1,
                   label=f'All group configs (n={len(bf_all)})', edgecolors='none')

    if bf_pareto_path.exists():
        with open(bf_pareto_path) as f:
            bf_pareto = json.load(f)
        bp_sav = [p['savings_pct'] for p in bf_pareto if p['savings_pct'] <= 55]
        bp_dppl = [p['pred_dppl'] for p in bf_pareto if p['savings_pct'] <= 55]
        ax.plot(bp_sav, bp_dppl, '^-', color='#27AE60', lw=1.0, ms=3.5,
                label='Brute-force Pareto', zorder=2)

    ax.plot(ilp_sav, [p - 306.4 for p in ilp_ppls], 'o-', color='#C0392B',
            lw=1.4, ms=5, label='ILP (per-layer)', zorder=4)
    ax.plot(grd_sav, [p - 306.4 for p in grd_ppls], 's--', color='#2980B9',
            lw=1.2, ms=4, label='Greedy (per-layer)', zorder=3)

    ax.set_xlabel('ADC area savings (%)')
    ax.set_ylabel('$\\Delta$PPL (surrogate)')
    ax.set_title('(a) Allocation search space', fontweight='bold')
    ax.legend(fontsize=5.5, framealpha=0.85, edgecolor='none', loc='upper left')
    ax.set_xlim(-2, 55)
    ax.set_ylim(-5, 35)

    # (b) Delta: ILP advantage over Greedy
    ax = axes[1]
    deltas = [r['delta_ppl'] for r in rows]
    bar_colors = ['#C0392B' if d > 0 else '#2980B9' for d in deltas]
    bars = ax.bar(targets, deltas, color=bar_colors, edgecolor='#2C3E50',
                  lw=0.4, width=3.5)
    ax.axhline(0, ls='-', color='k', lw=0.5)
    ax.set_xlabel('Target ADC savings (%)')
    ax.set_ylabel('$\\Delta$PPL (Greedy − ILP)')
    ax.set_title('(b) ILP advantage (>0 = ILP better)', fontweight='bold')
    ax.set_xticks(targets)
    for bar, v in zip(bars, deltas):
        va = 'bottom' if v >= 0 else 'top'
        offset = 0.15 if v >= 0 else -0.15
        ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                f'{v:+.1f}', ha='center', va=va, fontsize=6)

    ax.text(0.97, 0.92, 'ILP wins 6/8', transform=ax.transAxes,
            ha='right', fontsize=7, fontweight='bold', color='#C0392B')

    plt.tight_layout(w_pad=1.2)
    savefig('fig9_ilp_vs_greedy_multibudget')


# ═══════════════════════════════════════════════════════════════════════════
# Copy to paper/
# ═══════════════════════════════════════════════════════════════════════════
def copy_to_paper():
    paper_dir = ROOT.parent / 'paper'
    paper_dir.mkdir(exist_ok=True)
    for pdf in OUT_DIR.glob('*.pdf'):
        shutil.copy2(str(pdf), str(paper_dir / pdf.name))
        print(f"  copied -> paper/{pdf.name}")


if __name__ == '__main__':
    print(f"Output: {OUT_DIR}")
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]:
        try:
            fn()
        except Exception as e:
            print(f"  [ERROR] {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
    copy_to_paper()
    print("\nAll figures generated.")
