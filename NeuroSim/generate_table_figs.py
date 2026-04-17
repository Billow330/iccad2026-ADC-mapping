#!/usr/bin/env python3
"""Generate figures to replace tables: sensitivity heatmap, downstream CI bars, PPA curve."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    'font.family': 'serif',
})

OUT = Path("/raid/privatedata/fantao/iccad_exp/figures")
OUT.mkdir(exist_ok=True)

# ═══ Fig 1: Sensitivity heatmap + rank inversion bar ═══════════
def fig_sensitivity_heatmap():
    groups = ['$W_{qkv}$\n(36)', '$W_{out}$\n(12)', '$W_{fc1}$\n(12)',
              '$W_{fc2}$\n(12)', '$W_{head}$\n(1)']
    regimes = ['10→9', '9→8', '8→7', '7→6']

    data = np.array([
        [-0.10, -0.12, -0.03,  0.13],
        [-0.07,  0.45, -0.78,  0.87],
        [ 0.13,  1.12, 15.89,  0.20],
        [24.81, 16.89,  3.35,  1.41],
        [-0.01,  0.05,  3.65,  0.70],
    ])

    sat  = [100, 4, 94, 19, 100]
    meas = [0.13, 0.87, 0.20, 1.41, 0.70]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.2),
                                    gridspec_kw={'width_ratios': [3, 2]})

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=25)
    im = ax1.imshow(data, cmap='RdBu_r', norm=norm, aspect='auto')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(regimes)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(groups)
    ax1.set_title('(a) Measured ΔPPL/layer across operating points')
    for i in range(5):
        for j in range(4):
            v = data[i, j]
            color = 'white' if abs(v) > 8 else 'black'
            ax1.text(j, i, f'{v:+.1f}' if abs(v)<10 else f'{v:+.0f}',
                    ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')
    cb = plt.colorbar(im, ax=ax1, shrink=0.85, pad=0.02)
    cb.set_label('ΔPPL/layer', fontsize=7)

    x = np.arange(5)
    w = 0.35
    bars_sat = ax2.bar(x - w/2, [s/100 for s in sat], w, label='Saturation rate',
                       color='#FFAB91', edgecolor='#BF360C', linewidth=0.6)
    bars_meas = ax2.bar(x + w/2, [m/max(meas) for m in meas], w, label='Measured sens.\n(normalized)',
                        color='#90CAF9', edgecolor='#1565C0', linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['$W_{qkv}$', '$W_{out}$', '$W_{fc1}$', '$W_{fc2}$', '$W_{head}$'], fontsize=7)
    ax2.set_ylabel('Normalized score')
    ax2.set_title('(b) Proxy vs. measured rank')
    ax2.legend(loc='upper left', fontsize=6)
    ax2.set_ylim(0, 1.15)

    for i in range(5):
        sat_rank = sorted(range(5), key=lambda k: sat[k], reverse=True).index(i) + 1
        meas_rank = sorted(range(5), key=lambda k: meas[k], reverse=True).index(i) + 1
        if sat_rank != meas_rank:
            ax2.annotate('', xy=(i+w/2, meas[i]/max(meas)+0.05),
                        xytext=(i-w/2, sat[i]/100+0.05),
                        arrowprops=dict(arrowstyle='->', color='red', lw=0.8, ls='--'))

    plt.tight_layout(w_pad=1.5)
    plt.savefig(OUT / "fig_sensitivity_heatmap.pdf")
    print("  Saved fig_sensitivity_heatmap.pdf")


# ═══ Fig 2: Downstream CI bars ════════════════════════════════
def fig_downstream_ci():
    tasks = ['PIQA\n(1838)', 'HSwag\n(3000)', 'WGrde\n(1267)', 'BoolQ\n(3000)', 'ARC-E\n(570)']
    configs = ['FP32', 'CIM-7b', 'ILP-20%', 'Uni-6b']
    colors = ['#E0E0E0', '#2196F3', '#4CAF50', '#FF9800']

    acc = {
        'FP32':    [48.2, 29.7, 52.3, 38.3, 38.1],
        'CIM-7b':  [48.4, 27.0, 49.9, 37.9, 36.1],
        'ILP-20%': [48.4, 26.9, 49.7, 37.9, 36.8],
        'Uni-6b':  [48.4, 27.3, 49.8, 37.9, 36.1],
    }
    ci_half = {
        'CIM-7b':  [2.3, 1.6, 2.8, 1.7, 4.0],
        'ILP-20%': [2.3, 1.6, 2.8, 1.7, 4.1],
        'Uni-6b':  [2.3, 1.6, 2.7, 1.7, 4.0],
    }

    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    x = np.arange(len(tasks))
    w = 0.2
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

    for idx, (cfg, color) in enumerate(zip(configs, colors)):
        vals = acc[cfg]
        yerr = ci_half.get(cfg, [0]*5)
        bars = ax.bar(x + offsets[idx], vals, w, label=cfg, color=color,
                     edgecolor='k', linewidth=0.4, zorder=3)
        if cfg != 'FP32':
            ax.errorbar(x + offsets[idx], vals, yerr=yerr, fmt='none',
                       ecolor='black', capsize=2, capthick=0.5, linewidth=0.5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Zero-shot task accuracy with 95% bootstrap CI')
    ax.legend(loc='upper right', ncol=4, fontsize=6.5)
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylim(20, 58)

    ax.annotate('ILP-20% within\nCIM-7b CI on all tasks',
               xy=(2.5, 26), fontsize=6.5, fontstyle='italic', color='#2E7D32',
               ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUT / "fig_downstream_ci.pdf")
    print("  Saved fig_downstream_ci.pdf")


# ═══ Fig 3: PPA exponential scaling curve ═══════════════════════
def fig_ppa_curve():
    bits = [3, 4, 5, 6, 7, 8, 9, 10]
    area_125m = [668, 695, 731, 806, 937, 1147, 1672, 2822]
    area_13b  = [9496, 9878, 10397, 11469, 13321, 16311, 23773, 40131]
    adc_pct   = [3.9, 5.8, 8.7, 15.5, 24.4, 35.5, 49.1, 63.9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    ax1.semilogy(bits, area_125m, 'o-', color='#1976D2', markersize=5, linewidth=1.5,
                label='OPT-125M', zorder=5)
    ax1.semilogy(bits, area_13b, 's-', color='#F57C00', markersize=5, linewidth=1.5,
                label='OPT-1.3B (14.2×)', zorder=5)
    ax1.axvline(x=7, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.annotate('7b profiling\nregime', xy=(7, 600), fontsize=6, color='red',
                ha='center', va='top')
    ax1.axvspan(8, 10, alpha=0.08, color='green')
    ax1.annotate('deployment\nregime', xy=(9, 600), fontsize=6, color='#2E7D32',
                ha='center', va='top')
    ax1.set_xlabel('ADC bits')
    ax1.set_ylabel('Chip area (mm²)')
    ax1.set_title('(a) Chip area vs. ADC resolution')
    ax1.legend(fontsize=6.5)
    ax1.grid(True, alpha=0.2)
    ax1.set_xticks(bits)

    ax2.bar(bits, adc_pct, color=['#E3F2FD']*4 + ['#1976D2'] + ['#BBDEFB']*3,
           edgecolor='#1565C0', linewidth=0.6)
    ax2.set_xlabel('ADC bits')
    ax2.set_ylabel('ADC area fraction (%)')
    ax2.set_title('(b) ADC share of total chip area')
    ax2.set_xticks(bits)
    for i, (b, p) in enumerate(zip(bits, adc_pct)):
        ax2.text(b, p+1.5, f'{p:.0f}%', ha='center', fontsize=6)
    ax2.axhline(y=50, color='red', linestyle=':', alpha=0.4)
    ax2.annotate('ADC dominates\n(>50%)', xy=(9.5, 52), fontsize=6, color='red', ha='center')

    plt.tight_layout()
    plt.savefig(OUT / "fig_ppa_curve.pdf")
    print("  Saved fig_ppa_curve.pdf")


if __name__ == "__main__":
    print("Generating table-replacement figures...", flush=True)
    for name, fn in [
        ("Sensitivity Heatmap", fig_sensitivity_heatmap),
        ("Downstream CI", fig_downstream_ci),
        ("PPA Curve", fig_ppa_curve),
    ]:
        try:
            fn()
            print(f"  ✓ {name}", flush=True)
        except Exception as e:
            print(f"  ✗ {name}: {e}", flush=True)
            import traceback; traceback.print_exc()

    print(f"\nAll saved to {OUT}", flush=True)
