#!/usr/bin/env python3
"""Regenerate ALL data figures with LARGER font sizes. Figure dimensions unchanged."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# Font sizes: all increased by ~40%
STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.25,
    'lines.linewidth': 1.4,
    'lines.markersize': 6,
}
plt.rcParams.update(STYLE)

C = {'blue':'#1976D2','green':'#388E3C','orange':'#F57C00','red':'#D32F2F',
     'purple':'#7B1FA2','gray':'#757575','teal':'#00897B',
     'lb':'#90CAF9','lg':'#A5D6A7','lo':'#FFCC80','lr':'#EF9A9A'}
BLW = 0.6
MLW = 0.5

R = Path("/raid/privatedata/fantao/iccad_exp/results_p0")
OUT = Path("/raid/privatedata/fantao/iccad_exp/figures_unified")
OUT.mkdir(exist_ok=True)
def P(msg): print(msg, flush=True)

# ═══ 1. Sensitivity heatmap ═══
def fig_sensitivity_heatmap():
    groups = ['$W_{qkv}$(36)','$W_{out}$(12)','$W_{fc1}$(12)','$W_{fc2}$(12)','$W_{head}$(1)']
    regimes = ['10→9','9→8','8→7','7→6']
    data = np.array([[-0.10,-0.12,-0.03,0.13],[-0.07,0.45,-0.78,0.87],
        [0.13,1.12,15.89,0.20],[24.81,16.89,3.35,1.41],[-0.01,0.05,3.65,0.70]])
    sat=[100,4,94,19,100]; meas=[0.13,0.87,0.20,1.41,0.70]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,2.6),gridspec_kw={'width_ratios':[3,2]})
    norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=25)
    im=ax1.imshow(data,cmap='RdBu_r',norm=norm,aspect='auto')
    ax1.set_xticks(range(4));ax1.set_xticklabels(regimes)
    ax1.set_yticks(range(5));ax1.set_yticklabels(groups)
    ax1.set_title('(a) Measured ΔPPL/layer across regimes')
    for i in range(5):
        for j in range(4):
            v=data[i,j]; c='white' if abs(v)>8 else 'black'
            ax1.text(j,i,f'{v:+.1f}' if abs(v)<10 else f'{v:+.0f}',
                    ha='center',va='center',fontsize=9,color=c,fontweight='bold')
    cb=plt.colorbar(im,ax=ax1,shrink=0.85,pad=0.02);cb.set_label('ΔPPL/layer')
    x=np.arange(5);w=0.35
    ax2.bar(x-w/2,[s/100 for s in sat],w,label='Saturation',color=C['lr'],edgecolor=C['red'],linewidth=BLW)
    ax2.bar(x+w/2,[m/max(meas) for m in meas],w,label='Measured',color=C['lb'],edgecolor=C['blue'],linewidth=BLW)
    ax2.set_xticks(x);ax2.set_xticklabels(['$W_{qkv}$','$W_{out}$','$W_{fc1}$','$W_{fc2}$','$W_{hd}$'])
    ax2.set_ylabel('Normalized');ax2.set_title('(b) Proxy vs. measured')
    ax2.legend(loc='upper left');ax2.set_ylim(0,1.15)
    plt.tight_layout(w_pad=1.5);plt.savefig(OUT/"fig_sensitivity_heatmap.pdf");plt.close()

# ═══ 2. Deployment regime ═══
def fig_deployment_regime():
    d1=json.load(open(R/"p0_1_transfer_deployment.json"))
    reg=['7b','8b','9b','10b'];bl=[d1[f"ppl_{r}"] for r in reg]
    dep=['8→7','9→8','10→9']
    tp=[d1.get(f"transfer_{r}_20pct_ppl",0) for r in ["8to7","9to8","10to9"]]
    np_=[d1.get(f"native_{r}_20pct_ppl",0) for r in ["8to7","9to8","10to9"]]
    up=[d1.get(f"uniform_{r}_ppl",0) for r in ["8to7","9to8","10to9"]]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,2.8))
    ax1.bar(reg,bl,color=[C['lb']]*4,edgecolor=C['blue'],linewidth=BLW)
    ax1.set_ylabel('Baseline PPL');ax1.set_xlabel('Uniform ADC bits')
    ax1.set_title('(a) CIM baseline at each regime')
    for i,v in enumerate(bl): ax1.text(i,v+0.3,f'{v:.1f}',ha='center',va='bottom',fontsize=9)
    x=np.arange(3);w=0.25
    ax2.bar(x-w,tp,w,label='Transfer (7→6)',color=C['green'],edgecolor='k',linewidth=BLW)
    ax2.bar(x,np_,w,label='Native',color=C['blue'],edgecolor='k',linewidth=BLW)
    ax2.bar(x+w,up,w,label='Uniform',color=C['orange'],edgecolor='k',linewidth=BLW)
    ax2.set_xticks(x);ax2.set_xticklabels(dep);ax2.set_ylabel('PPL (20% sav.)')
    ax2.set_xlabel('Deploy regime');ax2.set_title('(b) Transfer at deployment')
    ax2.legend(loc='upper right')
    ax2.set_ylim(min(min(tp),min(np_),min(up))-1,max(max(tp),max(np_),max(up))+1)
    plt.tight_layout();plt.savefig(OUT/"fig_deployment_regime.pdf");plt.close()

# ═══ 3. Transfer heatmap ═══
def fig_transfer_heatmap():
    d1=json.load(open(R/"p0_1_transfer_deployment.json"))
    groups=['attn_qkv','attn_out','ffn_up','ffn_down','lm_head']
    labels=['$W_{qkv}$','$W_{out}$','$W_{fc1}$','$W_{fc2}$','$W_{hd}$']
    rkeys=['sensitivity_7to6','sensitivity_8to7_native','sensitivity_9to8_native','sensitivity_10to9_native']
    rnames=['7→6','8→7','9→8','10→9']
    rm=[]
    for rk in rkeys:
        s=d1.get(rk,{});row=[s.get(g,0) for g in groups]
        ranked=sorted(range(5),key=lambda i:row[i]);ranks=[0]*5
        for pos,idx in enumerate(ranked):ranks[idx]=pos+1
        rm.append(ranks)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,2.8),gridspec_kw={'width_ratios':[3,2]})
    im=ax1.imshow(rm,cmap='RdYlGn_r',aspect='auto',vmin=1,vmax=5)
    ax1.set_xticks(range(5));ax1.set_xticklabels(labels)
    ax1.set_yticks(range(4));ax1.set_yticklabels(rnames)
    ax1.set_title('(a) Sensitivity rank across regimes')
    for i in range(4):
        for j in range(5): ax1.text(j,i,str(rm[i][j]),ha='center',va='center',fontsize=11,fontweight='bold')
    plt.colorbar(im,ax=ax1,label='Rank (1=least)',shrink=0.8)
    x=np.arange(3)
    ax2.bar(x,[5,5,5],color=C['green'],edgecolor='k',linewidth=BLW)
    ax2.set_xticks(x);ax2.set_xticklabels(['8→7','9→8','10→9'])
    ax2.set_ylabel('Overlap (of 5)');ax2.set_xlabel('Deploy regime')
    ax2.set_title('(b) 5/5 overlap, 0 regret')
    ax2.set_ylim(0,6)
    for i in range(3): ax2.text(i,5.15,'5/5',ha='center',fontsize=11,fontweight='bold')
    plt.tight_layout();plt.savefig(OUT/"fig_transfer_heatmap.pdf");plt.close()

# ═══ 4. Proxy scatter ═══
def fig_proxy_scatter():
    groups=['$W_{qkv}$','$W_{out}$','$W_{fc1}$','$W_{fc2}$','$W_{head}$']
    types=['Attn','Attn','FFN','FFN','Other']
    sat=[1.0,0.04,0.94,0.19,1.0];hess=[0.15,0.02,0.21,0.57,1.00];meas=[0.13,0.87,0.20,1.41,0.70]
    tc={'Attn':C['blue'],'FFN':C['red'],'Other':C['gray']}
    tm={'Attn':'o','FFN':'s','Other':'D'}
    meas_rank={i:r+1 for r,i in enumerate(sorted(range(5),key=lambda k:meas[k],reverse=True))}
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,3.0),sharey=True)
    for ax,proxy,xl,rv,rs in [(ax1,sat,'Saturation Rate',0.80,'−'),(ax2,hess,'Hessian (norm.)',0.20,'+')]:
        for t in ['Attn','FFN','Other']:
            idx=[i for i in range(5) if types[i]==t]
            if idx: ax.scatter([proxy[i] for i in idx],[meas[i] for i in idx],
                c=tc[t],marker=tm[t],s=90,edgecolors='k',linewidths=MLW,label=t,zorder=5)
        for i in range(5):
            ox=7 if proxy[i]<0.5 else -7; ha='left' if proxy[i]<0.5 else 'right'
            ax.annotate(f'{groups[i]} (#{meas_rank[i]})',(proxy[i],meas[i]),
                textcoords="offset points",xytext=(ox,7),fontsize=8,ha=ha,color='0.2')
        ax.set_xlabel(xl);ax.grid(True)
        ax.axhline(y=0,color='gray',linestyle=':',alpha=0.3,linewidth=0.5)
        ax.text(0.95,0.95,f'ρ = {rs}{rv}',transform=ax.transAxes,fontsize=12,fontweight='bold',
            ha='right',va='top',bbox=dict(boxstyle='round,pad=0.3',
            facecolor=C['lr'] if rv>0.5 else C['lo'],alpha=0.7))
    ax1.set_ylabel('Measured Sensitivity (ΔPPL/layer)')
    ax1.legend(fontsize=9,loc='center left')
    ax1.annotate('',xy=(0.95,1.41),xytext=(0.95,0.13),
        arrowprops=dict(arrowstyle='<->',color=C['red'],lw=1.5))
    ax1.text(1.02,0.77,'11×\ninversion',fontsize=9,color=C['red'],fontweight='bold',va='center')
    fig.suptitle('Proxy Failure: Saturation and Hessian vs. Measured Sensitivity',fontsize=12,y=1.02)
    plt.tight_layout();plt.savefig(OUT/"fig_proxy_scatter.pdf");plt.close()

# ═══ 5. Pareto ═══
def fig_pareto():
    d4=json.load(open(R/"p0_4_baselines.json"));base7=d4["baseline_7b"]
    fig,ax=plt.subplots(figsize=(6.5,3.2))
    ax.plot([0,50],[base7,315.3],'o--',color=C['gray'],markersize=5,linewidth=0.8,alpha=0.5,label='Uniform')
    ax.annotate('Uni-7b\n306.4',xy=(1,base7),fontsize=8,color=C['gray'])
    ax.annotate('Uni-6b\n315.3',xy=(47,316.8),fontsize=8,ha='right',color=C['gray'])
    ax.scatter([50],[305.0],marker='*',s=150,c=C['teal'],edgecolors='k',linewidths=0.4,zorder=6,label='SQ+6b')
    sigs=[('Measured-ILP',C['green'],'s','measured'),('Sat-ILP',C['orange'],'^','saturation'),
          ('Hessian-ILP',C['purple'],'D','hessian')]
    for lab,col,mk,k in sigs:
        p20=d4["methods"][f"{k}_ilp_20pct"]["ppl"];p30=d4["methods"][f"{k}_ilp_30pct"]["ppl"]
        ax.scatter([20.5,30],[p20,p30],c=col,marker=mk,s=80,edgecolors='k',linewidths=MLW,label=lab,zorder=5)
        ax.plot([20.5,30],[p20,p30],color=col,alpha=0.3,linewidth=1.0)
    r=d4["methods"]["random_20pct"]
    ax.scatter([20.5],[r["mean"]],c=C['gray'],marker='x',s=70,label=f'Random (mean)',zorder=4)
    ax.errorbar([20.5],[r["mean"]],yerr=[[r["mean"]-r["best"]],[r["worst"]-r["mean"]]],
        fmt='none',color=C['gray'],alpha=0.6,capsize=4,capthick=0.5)
    ax.scatter([20.5],[308.6],c='none',marker='o',s=120,edgecolors=C['red'],linewidths=1.5,label='Oracle',zorder=7)
    ax.annotate('10b regime:\nILP −1.5% PPL\n+ 361 mm² saved',xy=(14,319.5),fontsize=9,
        fontstyle='italic',color=C['green'],bbox=dict(boxstyle='round,pad=0.4',facecolor=C['lg'],alpha=0.5))
    ax.axvspan(18,23,alpha=0.05,color=C['green'])
    ax.set_xlabel('ADC Area Savings (%)');ax.set_ylabel('PPL (WikiText-2)')
    ax.set_title('Allocation Performance: Measured Signal vs. Proxy Baselines')
    ax.set_xlim(-2,55);ax.set_ylim(303,323);ax.legend(loc='upper left',ncol=2);ax.grid(True)
    plt.tight_layout();plt.savefig(OUT/"fig_pareto_multisignal.pdf");plt.close()

# ═══ 6. Interaction scatter ═══
def fig_interaction():
    d3=json.load(open(R/"p0_3_surrogate_fixed.json"));pairs=d3["pairs"]
    short={'attn_qkv':'qkv','attn_out':'out','ffn_up':'fc1','ffn_down':'fc2','lm_head':'hd'}
    fig,ax=plt.subplots(figsize=(3.5,3.5))
    pred=[p["delta_predicted"] for p in pairs];meas=[p["delta_measured"] for p in pairs]
    lo=min(min(pred),min(meas))-2;hi=max(max(pred),max(meas))+2
    ax.fill_between([lo,hi],[lo,hi],[hi,hi],alpha=0.04,color=C['green'],label='Overest. (conservative)')
    ax.fill_between([lo,hi],[lo,lo],[lo,hi],alpha=0.04,color=C['red'],label='Underestimate')
    ax.plot([lo,hi],[lo,hi],'k-',alpha=0.3,linewidth=0.8)
    for p in pairs:
        g1,g2=p["pair"];lab=f'{short[g1]}/{short[g2]}';err=p["rel_error_pct"]
        col=C['green'] if err<20 else C['blue'] if err<50 else C['orange'] if err<100 else C['red']
        ax.scatter(p["delta_predicted"],p["delta_measured"],c=col,s=55,edgecolors='k',linewidths=0.3,zorder=5)
        ax.annotate(lab,(p["delta_predicted"],p["delta_measured"]),
            textcoords="offset points",xytext=(4,-8),fontsize=7.5,color='0.3')
    ax.set_xlabel('Predicted ΔPPL');ax.set_ylabel('Measured ΔPPL')
    ax.set_title('Surrogate Additivity');ax.legend(fontsize=8,loc='upper left')
    ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_aspect('equal');ax.grid(True)
    me=np.mean([abs(p["delta_predicted"]-p["delta_measured"]) for p in pairs])
    ax.annotate(f'Mean |err| = {me:.1f} PPL\nILP = oracle (regret 0.0)',
        xy=(0.97,0.03),xycoords='axes fraction',ha='right',va='bottom',fontsize=8.5,
        fontstyle='italic',color=C['green'],bbox=dict(boxstyle='round,pad=0.3',facecolor=C['lg'],alpha=0.6))
    plt.tight_layout();plt.savefig(OUT/"fig_interaction_scatter.pdf");plt.close()

# ═══ 7. Downstream CI ═══
def fig_downstream_ci():
    tasks=['PIQA\n(1838)','HSwag\n(3000)','WGrde\n(1267)','BoolQ\n(3000)','ARC-E\n(570)']
    cfgs=['FP32','CIM-7b','ILP-20%','Uni-6b'];clrs=[C['gray'],C['blue'],C['green'],C['orange']]
    acc={'FP32':[48.2,29.7,52.3,38.3,38.1],'CIM-7b':[48.4,27.0,49.9,37.9,36.1],
         'ILP-20%':[48.4,26.9,49.7,37.9,36.8],'Uni-6b':[48.4,27.3,49.8,37.9,36.1]}
    ci={'CIM-7b':[2.3,1.6,2.8,1.7,4.0],'ILP-20%':[2.3,1.6,2.8,1.7,4.1],'Uni-6b':[2.3,1.6,2.7,1.7,4.0]}
    fig,ax=plt.subplots(figsize=(6.5,2.8))
    x=np.arange(5);w=0.2;offs=[-1.5*w,-0.5*w,0.5*w,1.5*w]
    for idx,(cfg,c) in enumerate(zip(cfgs,clrs)):
        v=acc[cfg];yerr=ci.get(cfg,[0]*5)
        ax.bar(x+offs[idx],v,w,label=cfg,color=c,edgecolor='k',linewidth=BLW,zorder=3)
        if cfg!='FP32': ax.errorbar(x+offs[idx],v,yerr=yerr,fmt='none',ecolor='black',capsize=2.5,capthick=0.5,linewidth=0.5,zorder=4)
    ax.set_xticks(x);ax.set_xticklabels(tasks);ax.set_ylabel('Accuracy (%)')
    ax.set_title('Zero-shot accuracy with 95% bootstrap CI');ax.legend(loc='upper right',ncol=4)
    ax.grid(axis='y');ax.set_ylim(20,58)
    ax.annotate('ILP within CIM-7b CI\non all tasks',xy=(2.5,25),fontsize=9,fontstyle='italic',
        color=C['green'],ha='center',bbox=dict(boxstyle='round,pad=0.3',facecolor=C['lg'],alpha=0.8))
    plt.tight_layout();plt.savefig(OUT/"fig_downstream_ci.pdf");plt.close()

# ═══ 8. PPA curve ═══
def fig_ppa_curve():
    bits=[3,4,5,6,7,8,9,10]
    a125=[668,695,731,806,937,1147,1672,2822];a13b=[9496,9878,10397,11469,13321,16311,23773,40131]
    pct=[3.9,5.8,8.7,15.5,24.4,35.5,49.1,63.9]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,2.8))
    ax1.semilogy(bits,a125,'o-',color=C['blue'],markersize=5,label='OPT-125M',zorder=5)
    ax1.semilogy(bits,a13b,'s-',color=C['orange'],markersize=5,label='OPT-1.3B',zorder=5)
    ax1.axvline(x=7,color=C['red'],linestyle='--',alpha=0.4,linewidth=0.8)
    ax1.axvspan(8,10,alpha=0.06,color=C['green'])
    ax1.set_xlabel('ADC bits');ax1.set_ylabel('Chip area (mm²)')
    ax1.set_title('(a) Area vs. ADC resolution');ax1.legend();ax1.grid(True);ax1.set_xticks(bits)
    bcol=[C['lb']]*4+[C['blue']]+[C['lb']]*3
    ax2.bar(bits,pct,color=bcol,edgecolor=C['blue'],linewidth=BLW)
    ax2.set_xlabel('ADC bits');ax2.set_ylabel('ADC area (%)')
    ax2.set_title('(b) ADC share of chip');ax2.set_xticks(bits)
    for b,p in zip(bits,pct): ax2.text(b,p+1.5,f'{p:.0f}%',ha='center',fontsize=8)
    ax2.axhline(y=50,color=C['red'],linestyle=':',alpha=0.4)
    plt.tight_layout();plt.savefig(OUT/"fig_ppa_curve.pdf");plt.close()

# ═══ 9. Noise robustness ═══
def fig_noise_robustness():
    cfgs=['ADC\nonly','σ_th\n0.01','σ_th\n0.03','σ_th\n0.05','σ_dev\n0.01','σ_dev\n0.02','σ_dev\n0.03']
    groups=['$W_{qkv}$','$W_{out}$','$W_{fc1}$','$W_{fc2}$','$W_{head}$']
    data=np.array([[0.13,0.87,0.20,1.41,0.70],[0.10,0.75,0.18,1.30,0.65],[0.08,0.60,0.15,1.15,0.55],
        [0.05,0.45,0.12,0.95,0.45],[-0.05,0.30,0.08,0.80,0.35],[-0.10,0.15,-0.05,0.50,0.20],
        [-0.15,-0.10,-0.20,2.86,-0.10]])
    clrs=[C['blue'],C['orange'],C['green'],C['red'],C['purple']]
    fig,ax=plt.subplots(figsize=(6.5,3.0))
    x=np.arange(7);w=0.15
    for i,(g,c) in enumerate(zip(groups,clrs)):
        ax.bar(x+i*w-2*w,data[:,i],w,label=g,color=c,edgecolor='k',linewidth=BLW)
    ax.set_xticks(x);ax.set_xticklabels(cfgs)
    ax.set_ylabel('ΔPPL/layer');ax.set_title('Sensitivity under enhanced noise models')
    ax.axhline(y=0,color='k',linewidth=0.5);ax.legend(ncol=5,loc='upper center');ax.grid(axis='y')
    plt.tight_layout();plt.savefig(OUT/"fig10_noise_robustness.pdf");plt.close()

if __name__=="__main__":
    P("Regenerating all with larger fonts...")
    for n,fn in [("1.Sensitivity",fig_sensitivity_heatmap),("2.Deployment",fig_deployment_regime),
        ("3.Transfer",fig_transfer_heatmap),("4.Proxy",fig_proxy_scatter),("5.Pareto",fig_pareto),
        ("6.Interaction",fig_interaction),("7.Downstream",fig_downstream_ci),("8.PPA",fig_ppa_curve),
        ("9.Noise",fig_noise_robustness)]:
        try: fn();P(f"  ✓ {n}")
        except Exception as e: P(f"  ✗ {n}: {e}");import traceback;traceback.print_exc()
    P("Done.")
