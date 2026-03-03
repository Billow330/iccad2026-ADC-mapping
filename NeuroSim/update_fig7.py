"""
update_fig7.py — Regenerate fig7_comparison_bars with HAWQ row added
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

OUT_DIR  = Path('results/figures_iccad2026_v2')
PAPER_DIR = Path('../paper')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name):
    for fmt in ('pdf', 'png'):
        p = OUT_DIR / f'{name}.{fmt}'
        plt.savefig(p, dpi=150, bbox_inches='tight')
    # Copy to paper dir
    for fmt in ('pdf',):
        src = OUT_DIR / f'{name}.{fmt}'
        dst = PAPER_DIR / f'{name}.{fmt}'
        shutil.copy(src, dst)
    print(f"  Saved {name}")

# ── Complete data (stable_eval_results.csv + HAWQ from hawq_comparison.json)
# Note: HAWQ PPL (321.3) uses same calib as stable_eval (canonical [0-3])
# so comparison is valid; it appears higher than Uniform 7b because the
# Hessian-guided allocation hurts (wrong sensitivity ordering).

CONFIGS = [
    # label,              PPL,    adc_mm2, savings%,  color
    ('Uniform 7b\n(base)', 306.4, 228.4,   0.0,  '#BDC3C7'),
    ('Uniform 6b',         315.3, 114.2,  50.0,  '#85C1E9'),
    ('HAWQ-guided\nGreedy',321.3, 181.5,  20.5,  '#E74C3C'),
    ('Sat-guided\nGreedy', 313.8, 181.5,  20.5,  '#F39C12'),
    ('Sens-guided\nGreedy',312.5, 181.5,  20.5,  '#27AE60'),
    ('ILP\n(ours)',         308.6, 181.5,  20.5,  '#2980B9'),
    ('SQ +\nUnif 6b',      305.0, 114.2,  50.0,  '#8E44AD'),
    ('SQ +\nILP 20%',      309.4, 181.5,  20.5,  '#1ABC9C'),
]

labels  = [c[0] for c in CONFIGS]
ppls    = [c[1] for c in CONFIGS]
areas   = [c[2] for c in CONFIGS]
savings = [c[3] for c in CONFIGS]
colors  = [c[4] for c in CONFIGS]

x = np.arange(len(labels))
REF_PPL  = 306.4
REF_AREA = 228.4

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
plt.rcParams.update({'font.size': 8})

# ── Left: PPL bars ──────────────────────────────────────────────────────────
bars0 = axes[0].bar(x, ppls, color=colors, edgecolor='k', lw=0.5, width=0.65)
axes[0].axhline(REF_PPL, ls='--', color='#7F8C8D', lw=1.0, label=f'7b baseline ({REF_PPL})')
axes[0].set_ylabel('Perplexity (WikiText-2, ↓ better)')
axes[0].set_title('(a) Accuracy', fontsize=9, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=7)
axes[0].set_ylim(295, 335)
# Annotate
for i, v in enumerate(ppls):
    dy = 0.5 if v <= REF_PPL else 0.5
    axes[0].text(i, v + dy, f'{v:.1f}', ha='center', va='bottom', fontsize=6.5,
                 fontweight='bold' if labels[i] == 'ILP\n(ours)' else 'normal')
# Highlight HAWQ as "bad"
axes[0].annotate('HAWQ:\n+12.7↑\nvs ILP',
                 xy=(2, 321.3), xytext=(3.5, 328),
                 fontsize=6, color='#C0392B',
                 arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))
axes[0].legend(fontsize=7, loc='upper right')

# ── Right: ADC Area bars ─────────────────────────────────────────────────────
bars1 = axes[1].bar(x, areas, color=colors, edgecolor='k', lw=0.5, width=0.65)
axes[1].axhline(REF_AREA, ls='--', color='#7F8C8D', lw=1.0, label=f'7b ref ({REF_AREA:.0f} mm²)')
axes[1].set_ylabel('ADC area (mm², ↓ better, NeuroSIM)')
axes[1].set_title('(b) ADC Area', fontsize=9, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=7)
axes[1].set_ylim(80, 260)
for i, (v, s) in enumerate(zip(areas, savings)):
    label = f'{v:.0f}' if s == 0 else f'{v:.0f}\n(−{s:.0f}%)'
    axes[1].text(i, v + 2, label, ha='center', va='bottom', fontsize=6,
                 fontweight='bold' if labels[i] == 'ILP\n(ours)' else 'normal')
axes[1].legend(fontsize=7, loc='upper right')

# Shared legend patch for "ours"
import matplotlib.patches as mpatches
ours_patch = mpatches.Patch(color='#2980B9', label='ILP (ours) — best accuracy\nat 20.5% ADC savings')
hawq_patch = mpatches.Patch(color='#E74C3C', label='HAWQ-guided — 12.7 PPL worse\nthan ILP (wrong noise model)')
fig.legend(handles=[ours_patch, hawq_patch], loc='lower center',
           ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.05), framealpha=0.9)

plt.suptitle('OPT-125M Mixed-Precision ADC Allocation: All Configurations (Measured)', fontsize=9)
plt.tight_layout(rect=[0, 0.08, 1, 1])
savefig('fig7_comparison_bars')
plt.close()
print("Done.")
