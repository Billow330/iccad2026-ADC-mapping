"""
plot_results.py  —  Generate paper figures from experiment output
=================================================================
Reads JSON/CSV files produced by llm_inference.py and generates
publication-quality plots for the ICCAD 2026 paper.

Output files from llm_inference.py (inside results/<model_dir>/):
  outlier_<model>_adc<N>.csv      — per-layer outlier stats
  baseline_<model>_adc<N>.json   — baseline perplexity
  smooth_<model>_adc<N>.json     — SmoothQuant comparison
  sweep_adc_<model>.csv           — ADC bits sweep

Figures produced:
  Fig 1 — Per-layer outlier channel fraction + ADC saturation
  Fig 2 — ADC saturation rate vs ADC bits (Pareto curve)
  Fig 3 — CIMSmoothQuant vs baseline perplexity (grouped bar)
  Fig 4 — CIMSmoothQuant per-layer alpha distribution (violin)
  Fig 5 — Per-layer activation magnitude bar chart

Usage:
    python plot_results.py --results_dir ./results --output_dir ./results/figures
"""

import argparse, json, os, csv, glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family'    : 'DejaVu Sans',
    'font.size'      : 14,
    'axes.titlesize' : 15,
    'axes.labelsize' : 14,
    'legend.fontsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi'     : 150,
    'lines.linewidth': 2.2,
    'axes.grid'      : True,
    'grid.alpha'     : 0.3,
})

MODEL_LABELS = {
    'gpt2'              : 'GPT-2',
    'gpt2-medium'       : 'GPT-2-Medium',
    'gpt2_medium'       : 'GPT-2-Medium',
    'facebook_opt-125m' : 'OPT-125M',
    'facebook_opt-350m' : 'OPT-350M',
}
MODEL_COLORS = {
    'gpt2'              : '#2196F3',
    'gpt2-medium'       : '#4CAF50',
    'gpt2_medium'       : '#4CAF50',
    'facebook_opt-125m' : '#FF5722',
    'facebook_opt-350m' : '#9C27B0',
}


def model_label(mdir):
    return MODEL_LABELS.get(mdir, mdir)

def model_color(mdir):
    return MODEL_COLORS.get(mdir, '#607D8B')


def glob_first(directory, pattern):
    """Return the first file matching glob pattern in directory, or None."""
    matches = sorted(Path(directory).glob(pattern))
    return matches[0] if matches else None


def read_csv_dicts(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Per-layer outlier channel fraction + ADC saturation
# ─────────────────────────────────────────────────────────────────────────────

def fig_outlier_channels(results_dir, output_dir, model_dirs):
    """Bar: outlier channel % per layer; step-line: ADC saturation %."""
    n = len(model_dirs)
    fig, axes = plt.subplots(n, 1, figsize=(13, 5.0 * n))
    if n == 1:
        axes = [axes]

    any_data = False
    for ax, mdir in zip(axes, model_dirs):
        mpath = Path(results_dir) / mdir
        f = glob_first(mpath, 'outlier_*.csv')
        if f is None:
            ax.text(0.5, 0.5, f'No outlier data for {mdir}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{model_label(mdir)}')
            continue

        rows = read_csv_dicts(f)
        if not rows:
            continue
        any_data = True

        # Shorten layer names for display
        def shorten(name):
            parts = name.split('.')
            return '.'.join(parts[-3:]) if len(parts) >= 3 else name

        layer_names = [shorten(r['layer']) for r in rows]
        frac = [float(r['outlier_channel_fraction']) * 100 for r in rows]
        sat  = [float(r['sat_rate_worst']) * 100 for r in rows]

        x = np.arange(len(rows))
        step = max(1, len(x) // 20)
        ax.bar(x, frac, color=model_color(mdir), alpha=0.75, label='Outlier channel %')
        ax.step(x, sat, color='#F44336', where='mid', linewidth=2.2, label='ADC sat. rate %')
        ax.set_xticks(x[::step])
        ax.set_xticklabels([layer_names[i] for i in x[::step]],
                           rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{model_label(mdir)} — Per-layer Outlier & ADC Saturation')
        # Legend below the plot area to avoid covering the saturation line
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.38),
                  ncol=2, framealpha=0.9)
        ax.set_ylim(0, 115)

    plt.tight_layout()
    out = Path(output_dir) / 'fig1_outlier_channels.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    if any_data:
        print(f'[plot] Saved {out}')
    else:
        print(f'[plot] Fig1: no data found, placeholder saved to {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — ADC sweep Pareto: perplexity vs ADC bits
# ─────────────────────────────────────────────────────────────────────────────

def fig_adc_pareto(results_dir, output_dir, model_dirs):
    """Line: perplexity vs ADC bits — baseline vs CIM-SmoothQuant."""
    n = len(model_dirs)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    any_data = False
    for ax, mdir in zip(axes, model_dirs):
        mpath = Path(results_dir) / mdir
        f = glob_first(mpath, 'sweep_adc_*.csv')
        if f is None:
            ax.text(0.5, 0.5, 'No sweep data\nRun --task sweep_adc first',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{model_label(mdir)}')
            ax.set_xlabel('ADC Bits')
            ax.set_ylabel('Perplexity')
            continue

        rows = read_csv_dicts(f)
        bits      = [int(r['adc_bits'])       for r in rows]
        ppl_base  = [float(r['ppl_baseline']) for r in rows]
        ppl_sq    = [float(r['ppl_cim_sq'])   for r in rows]
        any_data  = True

        ax.plot(bits, ppl_base, 'o--', color='#F44336', label='Baseline (no smooth)', markersize=8)
        ax.plot(bits, ppl_sq,   's-',  color='#4CAF50', label='CIM-SmoothQuant', markersize=8)
        ax.fill_between(bits, ppl_base, ppl_sq,
                        where=[sq < base for sq, base in zip(ppl_sq, ppl_base)],
                        alpha=0.15, color='#4CAF50', label='SQ improvement')
        ax.set_yscale('log')
        ax.set_xlabel('ADC Bits')
        ax.set_ylabel('Perplexity (log scale)')
        ax.set_title(f'{model_label(mdir)}')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_xticks(bits)

    fig.suptitle('Pareto: Perplexity vs ADC Bits Under CIM Noise', fontsize=15, y=1.02)
    plt.tight_layout()
    out = Path(output_dir) / 'fig2_adc_pareto.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    if any_data:
        print(f'[plot] Saved {out}')
    else:
        print(f'[plot] Fig2: no sweep data found')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Perplexity comparison grouped bar
# ─────────────────────────────────────────────────────────────────────────────

def fig_ppl_comparison(results_dir, output_dir, model_dirs):
    """Grouped bar: FP32 / CIM baseline / Std-SQ / CIM-SQ perplexity."""
    methods = ['FP32', 'CIM Baseline', 'Std SmoothQuant', 'CIM-SmoothQuant']
    keys    = ['ppl_clean', 'ppl_cim_base', 'ppl_smooth_std', 'ppl_smooth_cim']
    colors  = ['#78909C', '#F44336', '#FF9800', '#4CAF50']

    vals_by_key = {k: [] for k in keys}
    valid_models = []

    for mdir in model_dirs:
        mpath = Path(results_dir) / mdir
        f = glob_first(mpath, 'smooth_*.json')
        if f is None:
            continue
        data = read_json(f)
        valid_models.append(mdir)
        for key in keys:
            vals_by_key[key].append(data.get(key, float('nan')))

    if not valid_models:
        print('[plot] Fig3: no smooth_*.json data found')
        return

    n = len(valid_models)
    m_count = len(methods)
    x = np.arange(n)
    w = 0.18

    fig, ax = plt.subplots(figsize=(max(5, 2.5 * n), 4))
    for i, (key, label, color) in enumerate(zip(keys, methods, colors)):
        offset = (i - m_count / 2 + 0.5) * w
        vals = vals_by_key[key]
        ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85)
        for xi, val in zip(x + offset, vals):
            if not np.isnan(val):
                ax.text(xi, val * 1.02, f'{val:.0f}',
                        ha='center', va='bottom', fontsize=6, rotation=90)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([model_label(md) for md in valid_models])
    ax.set_ylabel('Perplexity (log scale, lower is better)')
    ax.set_title('Perplexity: FP32 / CIM Baseline / SmoothQuant Variants')
    ax.legend(loc='upper right')
    plt.tight_layout()

    out = Path(output_dir) / 'fig3_ppl_comparison.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'[plot] Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Per-layer alpha distribution violin
# ─────────────────────────────────────────────────────────────────────────────

def fig_alpha_dist(results_dir, output_dir, model_dirs):
    """Violin: per-layer optimal alpha for CIM-SmoothQuant per model."""
    n = len(model_dirs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    any_data = False
    for ax, mdir in zip(axes, model_dirs):
        mpath = Path(results_dir) / mdir
        f = glob_first(mpath, 'smooth_*.json')
        if f is None:
            ax.set_title(f'{model_label(mdir)}\n(no data)')
            continue
        data = read_json(f)

        # alpha_per_layer is a dict {layer_name: alpha_float}
        alpha_dict = data.get('alpha_per_layer', {})
        if not alpha_dict:
            continue

        alphas = list(alpha_dict.values())
        any_data = True

        vp = ax.violinplot([alphas], positions=[1], showmedians=True,
                           showextrema=True)
        for body in vp['bodies']:
            body.set_facecolor(model_color(mdir))
            body.set_alpha(0.7)
        vp['cmedians'].set_color('black')
        vp['cmedians'].set_linewidth(2)

        ax.set_xticks([1])
        ax.set_xticklabels(['CIM-SQ'])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Optimal α per layer' if ax == axes[0] else '')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='α=0.5')
        ax.set_title(f'{model_label(mdir)}\n'
                     f'mean={np.mean(alphas):.2f} ± {np.std(alphas):.2f}')
        ax.legend(loc='lower right', fontsize=8)

    if any_data:
        fig.suptitle('Per-layer Optimal α Distribution (CIM-SmoothQuant)', fontsize=12)
        plt.tight_layout()
        out = Path(output_dir) / 'fig4_alpha_distribution.pdf'
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        print(f'[plot] Saved {out}')
    else:
        plt.close()
        print('[plot] Fig4: no alpha data found')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Activation magnitude + outlier fraction per layer
# ─────────────────────────────────────────────────────────────────────────────

def fig_act_magnitude(results_dir, output_dir, model_dirs):
    """Stacked sub-plots: act_max and outlier fraction per layer."""
    for mdir in model_dirs:
        mpath = Path(results_dir) / mdir
        f = glob_first(mpath, 'outlier_*.csv')
        if f is None:
            continue

        rows = read_csv_dicts(f)
        if not rows:
            continue

        def shorten(name):
            parts = name.split('.')
            return '.'.join(parts[-2:]) if len(parts) >= 2 else name

        layer_names = [shorten(r['layer']) for r in rows]
        act_max     = np.array([float(r['act_max'])                  for r in rows])
        outlier_f   = np.array([float(r['outlier_channel_fraction'])  for r in rows]) * 100
        sat_worst   = np.array([float(r['sat_rate_worst'])            for r in rows]) * 100

        x = np.arange(len(rows))
        step = max(1, len(x) // 15)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

        ax1.bar(x, act_max, color='#E91E63', alpha=0.8)
        ax1.set_ylabel('Max |activation|')
        ax1.set_title(f'{model_label(mdir)} — Per-layer Activation Statistics')

        ax2.bar(x, outlier_f, color='#FF5722', alpha=0.8)
        ax2.set_ylabel('Outlier Channel %')
        ax2.set_ylim(0, 105)

        ax3.bar(x, sat_worst, color='#F44336', alpha=0.8)
        ax3.set_ylabel('ADC Sat. Rate % (worst)')
        ax3.set_ylim(0, 105)
        ax3.set_xticks(x[::step])
        ax3.set_xticklabels([layer_names[i] for i in x[::step]],
                            rotation=45, ha='right', fontsize=7)

        plt.tight_layout()
        out = Path(output_dir) / f'fig5_act_stats_{mdir}.pdf'
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        print(f'[plot] Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — ADC overhead bits per layer
# ─────────────────────────────────────────────────────────────────────────────

def fig_adc_overhead(results_dir, output_dir, model_dirs):
    """Bar: how many extra ADC bits each layer needs to avoid saturation."""
    for mdir in model_dirs:
        mpath = Path(results_dir) / mdir
        f = glob_first(mpath, 'outlier_*.csv')
        if f is None:
            continue

        rows = read_csv_dicts(f)
        if not rows:
            continue

        overhead = [int(r['adc_overhead_bits']) for r in rows]
        if max(overhead) == 0:
            continue  # no overhead needed, skip figure

        def shorten(name):
            parts = name.split('.')
            return '.'.join(parts[-2:]) if len(parts) >= 2 else name

        layer_names = [shorten(r['layer']) for r in rows]
        x    = np.arange(len(rows))
        step = max(1, len(x) // 15)

        colors = ['#F44336' if v > 0 else '#4CAF50' for v in overhead]
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.bar(x, overhead, color=colors, alpha=0.85)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([layer_names[i] for i in x[::step]],
                           rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Extra ADC bits needed')
        ax.set_title(f'{model_label(mdir)} — ADC Overhead Bits per Layer')
        ax.axhline(0, color='black', linewidth=0.8)

        plt.tight_layout()
        out = Path(output_dir) / f'fig6_adc_overhead_{mdir}.pdf'
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        print(f'[plot] Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--output_dir',  default='./results/figures')
    p.add_argument('--models', nargs='+', default=None,
                   help='Specific model subdirectory names to include (default: all)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results_path = Path(args.results_dir)
    all_dirs = sorted([
        d.name for d in results_path.iterdir()
        if d.is_dir() and not d.name.startswith('figures')
    ])
    if args.models:
        model_dirs = [d for d in all_dirs if d in args.models]
    else:
        model_dirs = all_dirs

    if not model_dirs:
        print('[plot] No model result directories found. Run experiments first.')
        return

    print(f'[plot] Found model directories: {model_dirs}')
    print(f'[plot] Output directory: {args.output_dir}\n')

    fig_outlier_channels(args.results_dir, args.output_dir, model_dirs)
    fig_adc_pareto       (args.results_dir, args.output_dir, model_dirs)
    fig_ppl_comparison   (args.results_dir, args.output_dir, model_dirs)
    fig_alpha_dist       (args.results_dir, args.output_dir, model_dirs)
    fig_act_magnitude    (args.results_dir, args.output_dir, model_dirs)
    fig_adc_overhead     (args.results_dir, args.output_dir, model_dirs)

    print(f'\n[plot] All figures saved to: {args.output_dir}/')


if __name__ == '__main__':
    main()
