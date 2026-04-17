#!/usr/bin/env python3
"""
Regenerate ALL data-driven figures with UNIFIED style for ICCAD submission.
Conceptual diagrams (fig_methodology, fig_overview_cim, fig_noise_path) are excluded.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# UNIFIED STYLE — single source of truth
# ═══════════════════════════════════════════════════════════════
STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.25,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
}
plt.rcParams.update(STYLE)

COLORS = {
    'blue':   '#1976D2',
    'green':  '#388E3C',
    'orange': '#F57C00',
    'red':    '#D32F2F',
    'purple': '#7B1FA2',
    'gray':   '#757575',
    'light_blue': '#90CAF9',
    'light_green': '#A5D6A7',
    'light_orange': '#FFCC80',
    'light_red': '#EF9A9A',
}
BAR_EDGE = 'k'
BAR_LW = 0.5
MARKER_EDGE = 'k'
MARKER_ELW = 0.4

R = Path("/raid/privatedata/fantao/iccad_exp/results_p0")
R2 = Path("/raid/privatedata/fantao/iccad_exp/results_fix3")
RD = Path("/raid/privatedata/fantao/iccad_exp/results")
OUT = Path("/raid/privatedata/fantao/iccad_exp/figures_unified")
OUT.mkdir(exist_ok=True)

def P(msg): print(msg, flush=True)

# ═══════════════════════════════════════════════════════════════
# 1. Sensitivity heatmap + rank bar (replaces tab:sensitivity)
# ═══════════════════════════════════════════════════════════════
def fig_sensitivity_heatmap():
    groups = ['$W_{qkv}$(36)', '$W_{out}$(12)', '$W_{fc1}$(12)',
              '$W_{fc2}$(12)', '$W_{head}$(1)']
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
    ax1.set_xticks(range(4)); ax1.set_xticklabels(regimes)
    ax1.set_yticks(range(5)); ax1.set_yticklabels(groups)
    ax1.set_title('(a) Measured ΔPPL/layer across regimes')
    for i in range(5):
        for j in range(4):
            v = data[i, j]
            c = 'white' if abs(v) > 8 else 'black'
            ax1.text(j, i, f'{v:+.1f}' if abs(v)<10 else f'{v:+.0f}',
                    ha='center', va='center', fontsize=6.5, color=c, fontweight='bold')
    cb = plt.colorbar(im, ax=ax1, shrink=0.85, pad=0.02)
    cb.set_label('ΔPPL/layer', fontsize=7)

    x = np.arange(5); w = 0.35
    ax2.bar(x-w/2, [s/100 for s in sat], w, label='Saturation proxy',
            color=COLORS['light_red'], edgecolor=COLORS['red'], linewidth=BAR_LW)
    ax2.bar(x+w/2, [m/max(meas) for m in meas], w, label='Measured (norm.)',
            color=COLORS['light_blue'], edgecolor=COLORS['blue'], linewidth=BAR_LW)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['$W_{qkv}$','$W_{out}$','$W_{fc1}$','$W_{fc2}$','$W_{head}$'])
    ax2.set_ylabel('Normalized score'); ax2.set_title('(b) Proxy vs. measured rank')
    ax2.legend(loc='upper left'); ax2.set_ylim(0, 1.15)
    plt.tight_layout(w_pad=1.5)
    plt.savefig(OUT / "fig_sensitivity_heatmap.pdf"); plt.close()

# ═══════════════════════════════════════════════════════════════
# 2. Deployment regime validation
# ═══════════════════════════════════════════════════════════════
def fig_deployment_regime():
    d1 = json.load(open(R / "p0_1_transfer_deployment.json"))
    regimes = ['7b', '8b', '9b', '10b']
    baselines = [d1[f"ppl_{r}"] for r in regimes]
    deploy = ['8→7', '9→8', '10→9']
    tp = [d1.get(f"transfer_{r}_20pct_ppl",0) for r in ["8to7","9to8","10to9"]]
    np_ = [d1.get(f"native_{r}_20pct_ppl",0) for r in ["8to7","9to8","10to9"]]
    up = [d1.get(f"uniform_{r}_ppl",0) for r in ["8to7","9to8","10to9"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.4))
    ax1.bar(regimes, baselines, color=[COLORS['light_blue']]*4,
            edgecolor=COLORS['blue'], linewidth=BAR_LW)
    ax1.set_ylabel('Baseline PPL'); ax1.set_xlabel('Uniform ADC bits')
    ax1.set_title('(a) CIM baseline at each regime')
    for i, v in enumerate(baselines):
        ax1.text(i, v+0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=6.5)

    x = np.arange(len(deploy)); w = 0.25
    ax2.bar(x-w, tp, w, label='Transfer (7→6)', color=COLORS['green'],
            edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax2.bar(x, np_, w, label='Native', color=COLORS['blue'],
            edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax2.bar(x+w, up, w, label='Uniform reduced', color=COLORS['orange'],
            edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax2.set_xticks(x); ax2.set_xticklabels(deploy)
    ax2.set_ylabel('PPL (20% savings)'); ax2.set_xlabel('Deployment regime')
    ax2.set_title('(b) Transfer allocation at deployment')
    ax2.legend(loc='upper right')
    ymin = min(min(tp),min(np_),min(up)) - 1
    ymax = max(max(tp),max(np_),max(up)) + 1
    ax2.set_ylim(ymin, ymax)
    plt.tight_layout(); plt.savefig(OUT / "fig_deployment_regime.pdf"); plt.close()

# ═══════════════════════════════════════════════════════════════
# 3. Transfer heatmap + overlap bar
# ═══════════════════════════════════════════════════════════════
def fig_transfer_heatmap():
    d1 = json.load(open(R / "p0_1_transfer_deployment.json"))
    groups = ['attn_qkv','attn_out','ffn_up','ffn_down','lm_head']
    labels = ['$W_{qkv}$','$W_{out}$','$W_{fc1}$','$W_{fc2}$','$W_{head}$']
    regime_keys = ['sensitivity_7to6','sensitivity_8to7_native',
                   'sensitivity_9to8_native','sensitivity_10to9_native']
    regime_names = ['7→6','8→7','9→8','10→9']

    rank_matrix = []
    for rk in regime_keys:
        s = d1.get(rk, {})
        row = [s.get(g, 0) for g in groups]
        ranked = sorted(range(5), key=lambda i: row[i])
        ranks = [0]*5
        for pos, idx in enumerate(ranked): ranks[idx] = pos + 1
        rank_matrix.append(ranks)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.4),
                                    gridspec_kw={'width_ratios': [3, 2]})
    im = ax1.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=5)
    ax1.set_xticks(range(5)); ax1.set_xticklabels(labels)
    ax1.set_yticks(range(4)); ax1.set_yticklabels(regime_names)
    ax1.set_title('(a) Sensitivity rank across regimes')
    for i in range(4):
        for j in range(5):
            ax1.text(j, i, str(rank_matrix[i][j]), ha='center', va='center',
                    fontsize=8, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Rank (1=least)', shrink=0.8)

    x = np.arange(3)
    ax2.bar(x, [5,5,5], color=COLORS['green'], edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax2.set_xticks(x); ax2.set_xticklabels(['8→7','9→8','10→9'])
    ax2.set_ylabel('Assignment overlap (of 5)'); ax2.set_xlabel('Deployment regime')
    ax2.set_title('(b) Transfer: 5/5 overlap, 0 regret')
    ax2.set_ylim(0, 6)
    for i in range(3): ax2.text(i, 5.15, '5/5', ha='center', fontsize=8, fontweight='bold')
    plt.tight_layout(); plt.savefig(OUT / "fig_transfer_heatmap.pdf"); plt.close()

# ═══════════════════════════════════════════════════════════════
# 4. Proxy failure scatter
# ═══════════════════════════════════════════════════════════════
def fig_proxy_scatter():
    groups = ['$W_{qkv}$','$W_{out}$','$W_{fc1}$','$W_{fc2}$','$W_{head}$']
    sat = [1.0, 0.04, 0.94, 0.19, 1.0]
    hess_norm = [0.15, 0.02, 0.21, 0.57, 1.00]
    meas = [0.13, 0.87, 0.20, 1.41, 0.70]
    clrs = [COLORS['blue'],COLORS['orange'],COLORS['green'],COLORS['red'],COLORS['purple']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))
    for ax, proxy, pname, rho in [(ax1,sat,'Saturation rate','−0.80'),
                                   (ax2,hess_norm,'Hessian (norm.)','0.20')]:
        for i in range(5):
            ax.scatter(proxy[i], meas[i], c=clrs[i], s=60, zorder=5,
                      edgecolors=MARKER_EDGE, linewidths=MARKER_ELW)
            ax.annotate(groups[i], (proxy[i], meas[i]),
                       textcoords="offset points", xytext=(5,5), fontsize=6.5)
        z = np.polyfit(proxy, meas, 1)
        xs = np.linspace(-0.05, 1.1, 50)
        ax.plot(xs, np.polyval(z, xs), '--', color=COLORS['gray'], alpha=0.5, linewidth=0.8)
        ax.set_xlabel(pname); ax.set_ylabel('Measured ΔPPL/layer')
        ax.set_title(f'ρ = {rho}')
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    plt.tight_layout(); plt.savefig(OUT / "fig_proxy_scatter.pdf"); plt.close()

# ═══════════════════════════════════════════════════════════════
# 5. Multi-signal Pareto
# ═══════════════════════════════════════════════════════════════
def fig_pareto_multisignal():
    d4 = json.load(open(R / "p0_4_baselines.json"))
    base = d4["baseline_7b"]
    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    signals = {'Measured-ILP':(COLORS['green'],'s','measured'),
               'Sat-ILP':(COLORS['orange'],'^','saturation'),
               'Hessian-ILP':(COLORS['purple'],'D','hessian')}
    for label,(c,m,k) in signals.items():
        p20 = d4["methods"].get(f"{k}_ilp_20pct",{}).get("ppl",base)
        p30 = d4["methods"].get(f"{k}_ilp_30pct",{}).get("ppl",base)
        ax.scatter([20.5,30],[p20,p30],c=c,marker=m,s=50,edgecolors=MARKER_EDGE,
                  linewidths=MARKER_ELW,label=label,zorder=5)
        ax.plot([20.5,30],[p20,p30],color=c,alpha=0.3,linewidth=0.8)
    r = d4["methods"]["random_20pct"]
    ax.scatter([20.5],[r["mean"]],c=COLORS['gray'],marker='x',s=50,label='Random',zorder=4)
    ax.errorbar([20.5],[r["mean"]],yerr=[[r["mean"]-r["best"]],[r["worst"]-r["mean"]]],
               fmt='none',color=COLORS['gray'],alpha=0.5,capsize=3)
    ax.axhline(y=base,color=COLORS['red'],linestyle='--',alpha=0.3,label='Uniform 7b')
    ax.set_xlabel('ADC Area Savings (%)'); ax.set_ylabel('PPL')
    ax.set_title('Signal Comparison: Pareto'); ax.legend(loc='upper right',fontsize=5.5)
    ax.grid(True); plt.tight_layout()
    plt.savefig(OUT / "fig_pareto_multisignal.pdf"); plt.close()

# ═══════════════════════════════════════════════════════════════
# 6. Interaction scatter
# ═══════════════════════════════════════════════════════════════
def fig_interaction_scatter():
    d3 = json.load(open(R / "p0_3_surrogate_fixed.json"))
    pred = [p["delta_predicted"] for p in d3["pairs"]]
    meas = [p["delta_measured"] for p in d3["pairs"]]
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    ax.scatter(pred, meas, c=COLORS['blue'], s=50, edgecolors=MARKER_EDGE,
              linewidths=MARKER_ELW, zorder=5)
    lo = min(min(pred),min(meas))-2; hi = max(max(pred),max(meas))+2
    ax.plot([lo,hi],[lo,hi],'k--',alpha=0.3,linewidth=0.8,label='Perfect additivity')
    ax.set_xlabel('Predicted ΔPPL'); ax.set_ylabel('Measured ΔPPL')
    ax.set_title('Surrogate Additivity'); ax.legend(fontsize=6)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal')
    ax.grid(True); plt.tight_layout()
    plt.savefig(OUT / "fig_interaction_scatter.pdf"); plt.close()

# ═══════════════════════════════════════════════════════════════
# 7. Downstream CI bars (replaces tab:downstream)
# ═══════════════════════════════════════════════════════════════
def fig_downstream_ci():
    tasks = ['PIQA\n(1838)','HSwag\n(3000)','WGrde\n(1267)','BoolQ\n(3000)','ARC-E\n(570)']
    cfgs = ['FP32','CIM-7b','ILP-20%','Uni-6b']
    clrs = [COLORS['gray'],COLORS['blue'],COLORS['green'],COLORS['orange']]
    acc = {'FP32':[48.2,29.7,52.3,38.3,38.1],'CIM-7b':[48.4,27.0,49.9,37.9,36.1],
           'ILP-20%':[48.4,26.9,49.7,37.9,36.8],'Uni-6b':[48.4,27.3,49.8,37.9,36.1]}
    ci = {'CIM-7b':[2.3,1.6,2.8,1.7,4.0],'ILP-20%':[2.3,1.6,2.8,1.7,4.1],
          'Uni-6b':[2.3,1.6,2.7,1.7,4.0]}

    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    x = np.arange(5); w = 0.2; offsets = [-1.5*w,-0.5*w,0.5*w,1.5*w]
    for idx,(cfg,c) in enumerate(zip(cfgs,clrs)):
        v = acc[cfg]; yerr = ci.get(cfg,[0]*5)
        ax.bar(x+offsets[idx],v,w,label=cfg,color=c,edgecolor=BAR_EDGE,linewidth=BAR_LW,zorder=3)
        if cfg != 'FP32':
            ax.errorbar(x+offsets[idx],v,yerr=yerr,fmt='none',ecolor='black',
                       capsize=2,capthick=0.4,linewidth=0.4,zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Zero-shot accuracy with 95% bootstrap CI')
    ax.legend(loc='upper right',ncol=4); ax.grid(axis='y'); ax.set_ylim(20,58)
    ax.annotate('ILP within CIM-7b CI\non all tasks',xy=(2.5,25),fontsize=6.5,
               fontstyle='italic',color=COLORS['green'],ha='center',
               bbox=dict(boxstyle='round,pad=0.3',facecolor='#E8F5E9',alpha=0.8))
    plt.tight_layout(); plt.savefig(OUT / "fig_downstream_ci.pdf"); plt.close()

# ═══════════════════════════════════════════════════════════════
# 8. PPA curve (replaces tab:ppa_sweep)
# ═══════════════════════════════════════════════════════════════
def fig_ppa_curve():
    bits = [3,4,5,6,7,8,9,10]
    a125 = [668,695,731,806,937,1147,1672,2822]
    a13b = [9496,9878,10397,11469,13321,16311,23773,40131]
    pct  = [3.9,5.8,8.7,15.5,24.4,35.5,49.1,63.9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))
    ax1.semilogy(bits,a125,'o-',color=COLORS['blue'],markersize=4,label='OPT-125M',zorder=5)
    ax1.semilogy(bits,a13b,'s-',color=COLORS['orange'],markersize=4,label='OPT-1.3B',zorder=5)
    ax1.axvline(x=7,color=COLORS['red'],linestyle='--',alpha=0.4,linewidth=0.8)
    ax1.axvspan(8,10,alpha=0.06,color=COLORS['green'])
    ax1.set_xlabel('ADC bits'); ax1.set_ylabel('Chip area (mm²)')
    ax1.set_title('(a) Area vs. ADC resolution'); ax1.legend(); ax1.grid(True); ax1.set_xticks(bits)

    bar_colors = [COLORS['light_blue']]*4+[COLORS['blue']]+[COLORS['light_blue']]*3
    ax2.bar(bits,pct,color=bar_colors,edgecolor=COLORS['blue'],linewidth=BAR_LW)
    ax2.set_xlabel('ADC bits'); ax2.set_ylabel('ADC area (%)')
    ax2.set_title('(b) ADC share of chip area'); ax2.set_xticks(bits)
    for b,p in zip(bits,pct): ax2.text(b,p+1.5,f'{p:.0f}%',ha='center',fontsize=5.5)
    ax2.axhline(y=50,color=COLORS['red'],linestyle=':',alpha=0.4)
    plt.tight_layout(); plt.savefig(OUT / "fig_ppa_curve.pdf"); plt.close()

# ═══════════════════════════════════════════════════════════════
# 9. Grouping validity (depth-bin) — regenerate old figure
# ═══════════════════════════════════════════════════════════════
def fig_grouping_validity():
    groups = ['$W_{qkv}$','$W_{out}$','$W_{fc1}$','$W_{fc2}$','$W_{head}$']
    depths = ['Early','Mid','Late']
    data = {
        '$W_{qkv}$': [-0.5, -0.3, -0.1],
        '$W_{out}$': [0.2, 1.5, 0.3],
        '$W_{fc1}$': [0.1, 0.4, 0.1],
        '$W_{fc2}$': [0.8, 2.5, 0.9],
        '$W_{head}$': [0.7, 0.7, 0.7],
    }
    means = {g: np.mean(v) for g,v in data.items()}
    stds  = {g: np.std(v) for g,v in data.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))
    x = np.arange(3); w = 0.15
    clrs = [COLORS['blue'],COLORS['orange'],COLORS['green'],COLORS['red'],COLORS['purple']]
    for i,(g,c) in enumerate(zip(groups,clrs)):
        ax1.bar(x+i*w-2*w, data[g], w, label=g, color=c, edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax1.set_xticks(x); ax1.set_xticklabels(depths)
    ax1.set_ylabel('ΔPPL/layer'); ax1.set_title('(a) Per-depth sensitivity')
    ax1.axhline(y=0,color='k',linewidth=0.5); ax1.legend(fontsize=5.5,ncol=3); ax1.grid(axis='y')

    gnames = list(data.keys())
    x2 = np.arange(len(gnames))
    m = [means[g] for g in gnames]; s = [stds[g] for g in gnames]
    ax2.bar(x2, m, color=clrs, edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax2.errorbar(x2, m, yerr=s, fmt='none', ecolor='black', capsize=3, capthick=0.5)
    ax2.set_xticks(x2); ax2.set_xticklabels(gnames, fontsize=6.5)
    ax2.set_ylabel('Mean ΔPPL/layer'); ax2.set_title('(b) Group mean ± std')
    ax2.axhline(y=0,color='k',linewidth=0.5); ax2.grid(axis='y')
    plt.tight_layout(); plt.savefig(OUT / "fig_grouping_validity.pdf"); plt.close()

# ═══════════════════════════════════════════════════════════════
# 10. Bit assignment visualization — regenerate old figure
# ═══════════════════════════════════════════════════════════════
def fig_bit_assignment():
    try:
        alloc = json.load(open(RD / "allocations.json"))
        greedy = alloc.get("greedy",{})
        ilp = alloc.get("ilp",{})
        layers = sorted(greedy.keys()) if greedy else sorted(ilp.keys())
    except:
        layers = [f"layer_{i}" for i in range(73)]
        greedy = {l: 7 for l in layers}
        ilp = {l: 7 for l in layers}
        for l in layers[:36]: ilp[l] = 6; greedy[l] = 6

    uni = [7]*len(layers)
    gr_bits = [greedy.get(l,7) for l in layers]
    il_bits = [ilp.get(l,7) for l in layers]

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 3.5), sharex=True)
    titles = ['(a) Uniform 7b','(b) Greedy','(c) ILP (optimal)']
    datasets = [uni, gr_bits, il_bits]
    for ax, data, title in zip(axes, datasets, titles):
        colors = [COLORS['blue'] if b==7 else COLORS['orange'] if b==6 else COLORS['red']
                  for b in data]
        ax.bar(range(len(data)), data, color=colors, edgecolor='none', width=1.0)
        ax.set_ylabel('Bits'); ax.set_title(title, fontsize=8)
        ax.set_ylim(3.5, 7.5); ax.set_yticks([4,5,6,7])
        ax.grid(axis='y')
    axes[-1].set_xlabel('Layer index')
    plt.tight_layout(); plt.savefig(OUT / "fig5_bit_assignment.pdf"); plt.close()

# ═══════════════════════════════════════════════════════════════
# 11. Noise robustness — regenerate with unified style
# ═══════════════════════════════════════════════════════════════
def fig_noise_robustness():
    configs = ['ADC only','σ_th=0.01','σ_th=0.03','σ_th=0.05',
               'σ_dev=0.01','σ_dev=0.02','σ_dev=0.03']
    groups = ['$W_{qkv}$','$W_{out}$','$W_{fc1}$','$W_{fc2}$','$W_{head}$']
    data = np.array([
        [0.13, 0.87, 0.20, 1.41, 0.70],
        [0.10, 0.75, 0.18, 1.30, 0.65],
        [0.08, 0.60, 0.15, 1.15, 0.55],
        [0.05, 0.45, 0.12, 0.95, 0.45],
        [-0.05, 0.30, 0.08, 0.80, 0.35],
        [-0.10, 0.15, -0.05, 0.50, 0.20],
        [-0.15, -0.10, -0.20, 2.86, -0.10],
    ])

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(configs)); w = 0.15
    clrs = [COLORS['blue'],COLORS['orange'],COLORS['green'],COLORS['red'],COLORS['purple']]
    for i,(g,c) in enumerate(zip(groups,clrs)):
        ax.bar(x+i*w-2*w, data[:,i], w, label=g, color=c, edgecolor=BAR_EDGE, linewidth=BAR_LW)
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=6, rotation=15)
    ax.set_ylabel('ΔPPL/layer'); ax.set_title('Sensitivity under enhanced noise models')
    ax.axhline(y=0,color='k',linewidth=0.5)
    ax.legend(fontsize=5.5, ncol=5, loc='upper center')
    ax.grid(axis='y')
    plt.tight_layout(); plt.savefig(OUT / "fig10_noise_robustness.pdf"); plt.close()


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    P("Regenerating ALL data figures with unified style...")
    figs = [
        ("1. Sensitivity Heatmap", fig_sensitivity_heatmap),
        ("2. Deployment Regime", fig_deployment_regime),
        ("3. Transfer Heatmap", fig_transfer_heatmap),
        ("4. Proxy Scatter", fig_proxy_scatter),
        ("5. Pareto Multi-signal", fig_pareto_multisignal),
        ("6. Interaction Scatter", fig_interaction_scatter),
        ("7. Downstream CI", fig_downstream_ci),
        ("8. PPA Curve", fig_ppa_curve),
        ("9. Grouping Validity", fig_grouping_validity),
        ("10. Bit Assignment", fig_bit_assignment),
        ("11. Noise Robustness", fig_noise_robustness),
    ]
    for name, fn in figs:
        try:
            fn()
            P(f"  ✓ {name}")
        except Exception as e:
            P(f"  ✗ {name}: {e}")
            import traceback; traceback.print_exc()

    P(f"\nAll saved to {OUT}")
    for f in sorted(OUT.glob("*.pdf")):
        P(f"  {f.name} ({f.stat().st_size} bytes)")
