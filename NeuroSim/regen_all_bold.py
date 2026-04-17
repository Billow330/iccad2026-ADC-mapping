#!/usr/bin/env python3
"""Regenerate ALL matplotlib figures with BOLD + LARGE text + thick borders."""
import json, csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from collections import Counter
from pathlib import Path

# ══ BOLD LARGE STYLE ══
plt.rcParams.update({
    "font.family": "serif", "font.weight": "bold",
    "font.size": 13,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 13, "axes.labelweight": "bold",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "grid.linewidth": 0.6, "grid.alpha": 0.2,
    "lines.linewidth": 1.8, "lines.markersize": 7,
    "pdf.fonttype": 42,
})
BLW = 1.0  # bar edge linewidth
MLW = 0.8  # marker edge linewidth

C = {"blue":"#1976D2","green":"#388E3C","orange":"#F57C00","red":"#D32F2F",
     "purple":"#7B1FA2","gray":"#757575","teal":"#00897B",
     "lb":"#90CAF9","lg":"#A5D6A7","lo":"#FFCC80","lr":"#EF9A9A"}

R = Path("/raid/privatedata/fantao/iccad_exp/results_p0")
RD = Path("/raid/privatedata/fantao/iccad_exp/results")
OUT = Path("/raid/privatedata/fantao/iccad_exp/figures_unified")
OUT.mkdir(exist_ok=True)
def P(msg): print(msg, flush=True)
def BF(s): return {"fontweight": "bold", "fontsize": s}

# ═══ 1. Sensitivity heatmap ═══
def fig_sensitivity():
    groups = ['$W_{qkv}$(36)','$W_{out}$(12)','$W_{fc1}$(12)','$W_{fc2}$(12)','$W_{head}$(1)']
    regimes = ['10→9','9→8','8→7','7→6']
    data = np.array([[-0.10,-0.12,-0.03,0.13],[-0.07,0.45,-0.78,0.87],
        [0.13,1.12,15.89,0.20],[24.81,16.89,3.35,1.41],[-0.01,0.05,3.65,0.70]])
    sat=[100,4,94,19,100]; meas=[0.13,0.87,0.20,1.41,0.70]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,2.8),gridspec_kw={'width_ratios':[3,2]})
    norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=25)
    im=ax1.imshow(data,cmap='RdBu_r',norm=norm,aspect='auto')
    ax1.set_xticks(range(4));ax1.set_xticklabels(regimes)
    ax1.set_yticks(range(5));ax1.set_yticklabels(groups)
    ax1.set_title('(a) ΔPPL/layer across regimes')
    for i in range(5):
        for j in range(4):
            v=data[i,j]; c='white' if abs(v)>8 else 'black'
            ax1.text(j,i,f'{v:+.1f}' if abs(v)<10 else f'{v:+.0f}',
                    ha='center',va='center',fontsize=10,color=c,fontweight='bold')
    cb=plt.colorbar(im,ax=ax1,shrink=0.85,pad=0.02);cb.set_label('ΔPPL/layer',**BF(12))
    x=np.arange(5);w=0.35
    ax2.bar(x-w/2,[s/100 for s in sat],w,label='Saturation',color=C['lr'],edgecolor=C['red'],linewidth=BLW)
    ax2.bar(x+w/2,[m/max(meas) for m in meas],w,label='Measured',color=C['lb'],edgecolor=C['blue'],linewidth=BLW)
    ax2.set_xticks(x);ax2.set_xticklabels(['$W_{qkv}$','$W_{out}$','$W_{fc1}$','$W_{fc2}$','$W_{hd}$'])
    ax2.set_ylabel('Normalized');ax2.set_title('(b) Proxy vs. measured')
    ax2.legend(loc='upper left');ax2.set_ylim(0,1.15)
    plt.tight_layout(w_pad=1.5);plt.savefig(OUT/"fig_sensitivity_heatmap.pdf");plt.close()

# ═══ 2. Deployment regime ═══
def fig_deployment():
    d1=json.load(open(R/"p0_1_transfer_deployment.json"))
    reg=['7b','8b','9b','10b'];bl=[d1[f"ppl_{r}"] for r in reg]
    dep=['8→7','9→8','10→9']
    tp=[d1.get(f"transfer_{r}_20pct_ppl",0) for r in ["8to7","9to8","10to9"]]
    np_=[d1.get(f"native_{r}_20pct_ppl",0) for r in ["8to7","9to8","10to9"]]
    up=[d1.get(f"uniform_{r}_ppl",0) for r in ["8to7","9to8","10to9"]]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,3.0))
    ax1.bar(reg,bl,color=[C['lb']]*4,edgecolor=C['blue'],linewidth=BLW)
    ax1.set_ylabel('Baseline PPL');ax1.set_xlabel('Uniform ADC bits')
    ax1.set_title('(a) CIM baseline')
    for i,v in enumerate(bl): ax1.text(i,v+0.3,f'{v:.1f}',ha='center',va='bottom',**BF(10))
    x=np.arange(3);w=0.25
    ax2.bar(x-w,tp,w,label='Transfer',color=C['green'],edgecolor='k',linewidth=BLW)
    ax2.bar(x,np_,w,label='Native',color=C['blue'],edgecolor='k',linewidth=BLW)
    ax2.bar(x+w,up,w,label='Uniform',color=C['orange'],edgecolor='k',linewidth=BLW)
    ax2.set_xticks(x);ax2.set_xticklabels(dep);ax2.set_ylabel('PPL (20% sav.)')
    ax2.set_xlabel('Deploy regime');ax2.set_title('(b) Transfer at deployment')
    ax2.legend(loc='upper right')
    ax2.set_ylim(min(min(tp),min(np_),min(up))-1,max(max(tp),max(np_),max(up))+1)
    plt.tight_layout();plt.savefig(OUT/"fig_deployment_regime.pdf");plt.close()

# ═══ 3. Transfer heatmap ═══
def fig_transfer():
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
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,3.0),gridspec_kw={'width_ratios':[3,2]})
    im=ax1.imshow(rm,cmap='RdYlGn_r',aspect='auto',vmin=1,vmax=5)
    ax1.set_xticks(range(5));ax1.set_xticklabels(labels)
    ax1.set_yticks(range(4));ax1.set_yticklabels(rnames)
    ax1.set_title('(a) Rank across regimes')
    for i in range(4):
        for j in range(5): ax1.text(j,i,str(rm[i][j]),ha='center',va='center',**BF(13))
    plt.colorbar(im,ax=ax1,label='Rank',shrink=0.8)
    x=np.arange(3)
    ax2.bar(x,[5,5,5],color=C['green'],edgecolor='k',linewidth=BLW)
    ax2.set_xticks(x);ax2.set_xticklabels(['8→7','9→8','10→9'])
    ax2.set_ylabel('Overlap (of 5)');ax2.set_xlabel('Deploy regime')
    ax2.set_title('(b) 5/5, regret=0')
    ax2.set_ylim(0,6)
    for i in range(3): ax2.text(i,5.15,'5/5',ha='center',**BF(13))
    plt.tight_layout();plt.savefig(OUT/"fig_transfer_heatmap.pdf");plt.close()

# ═══ 4. Proxy scatter ═══
def fig_proxy():
    groups=['$W_{qkv}$','$W_{out}$','$W_{fc1}$','$W_{fc2}$','$W_{head}$']
    types=['Attn','Attn','FFN','FFN','Other']
    sat=[1.0,0.04,0.94,0.19,1.0];hess=[0.15,0.02,0.21,0.57,1.00];meas=[0.13,0.87,0.20,1.41,0.70]
    tc={'Attn':C['blue'],'FFN':C['red'],'Other':C['gray']}
    tm={'Attn':'o','FFN':'s','Other':'D'}
    meas_rank={i:r+1 for r,i in enumerate(sorted(range(5),key=lambda k:meas[k],reverse=True))}
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,3.2),sharey=True)
    for ax,proxy,xl,rv,rs in [(ax1,sat,'Saturation Rate',0.80,'\u2212'),(ax2,hess,'Hessian (norm.)',0.20,'+')]:
        for t in ['Attn','FFN','Other']:
            idx=[i for i in range(5) if types[i]==t]
            if idx: ax.scatter([proxy[i] for i in idx],[meas[i] for i in idx],
                c=tc[t],marker=tm[t],s=100,edgecolors='k',linewidths=MLW,label=t,zorder=5)
        for i in range(5):
            ox=8 if proxy[i]<0.5 else -8; ha='left' if proxy[i]<0.5 else 'right'
            ax.annotate(f'{groups[i]} (#{meas_rank[i]})',(proxy[i],meas[i]),
                textcoords="offset points",xytext=(ox,7),fontsize=9,ha=ha,color='0.2',fontweight='bold')
        ax.set_xlabel(xl);ax.grid(True)
        ax.axhline(y=0,color='gray',linestyle=':',alpha=0.3,linewidth=0.5)
        ax.text(0.95,0.95,f'\u03c1 = {rs}{rv}',transform=ax.transAxes,**BF(14),
            ha='right',va='top',bbox=dict(boxstyle='round,pad=0.3',
            facecolor=C['lr'] if rv>0.5 else C['lo'],alpha=0.7))
    ax1.set_ylabel('Measured (ΔPPL/layer)')
    ax1.legend(fontsize=10,loc='center left')
    ax1.annotate('',xy=(0.95,1.41),xytext=(0.95,0.13),
        arrowprops=dict(arrowstyle='<->',color=C['red'],lw=2.0))
    ax1.text(1.03,0.77,'11\u00d7',fontsize=12,color=C['red'],fontweight='bold',va='center')
    fig.suptitle('Proxy Failure: Saturation and Hessian vs. Measured Sensitivity',**BF(13),y=1.02)
    plt.tight_layout();plt.savefig(OUT/"fig_proxy_scatter.pdf");plt.close()

# ═══ 5. Interaction scatter ═══
def fig_interaction():
    d3=json.load(open(R/"p0_3_surrogate_fixed.json"));pairs=d3["pairs"]
    short={'attn_qkv':'qkv','attn_out':'out','ffn_up':'fc1','ffn_down':'fc2','lm_head':'hd'}
    fig,ax=plt.subplots(figsize=(3.5,3.5))
    pred=[p["delta_predicted"] for p in pairs];meas=[p["delta_measured"] for p in pairs]
    lo=min(min(pred),min(meas))-2;hi=max(max(pred),max(meas))+2
    ax.fill_between([lo,hi],[lo,hi],[hi,hi],alpha=0.04,color=C['green'],label='Overest.')
    ax.fill_between([lo,hi],[lo,lo],[lo,hi],alpha=0.04,color=C['red'],label='Underest.')
    ax.plot([lo,hi],[lo,hi],'k-',alpha=0.3,linewidth=1.0)
    for p in pairs:
        g1,g2=p["pair"];lab=f'{short[g1]}/{short[g2]}';err=p["rel_error_pct"]
        col=C['green'] if err<20 else C['blue'] if err<50 else C['orange'] if err<100 else C['red']
        ax.scatter(p["delta_predicted"],p["delta_measured"],c=col,s=60,edgecolors='k',linewidths=0.4,zorder=5)
        ax.annotate(lab,(p["delta_predicted"],p["delta_measured"]),
            textcoords="offset points",xytext=(4,-9),fontsize=8,color='0.3',fontweight='bold')
    ax.set_xlabel('Predicted ΔPPL');ax.set_ylabel('Measured ΔPPL')
    ax.set_title('Surrogate Additivity')
    ax.legend(fontsize=9,loc='upper left')
    ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_aspect('equal');ax.grid(True)
    me=np.mean([abs(p["delta_predicted"]-p["delta_measured"]) for p in pairs])
    ax.annotate(f'Mean |err|={me:.1f}\nILP=oracle',
        xy=(0.97,0.03),xycoords='axes fraction',ha='right',va='bottom',**BF(10),
        fontstyle='italic',color=C['green'],bbox=dict(boxstyle='round,pad=0.3',facecolor=C['lg'],alpha=0.6))
    plt.tight_layout();plt.savefig(OUT/"fig_interaction_scatter.pdf");plt.close()

# ═══ 6. Downstream CI ═══
def fig_downstream():
    tasks=['PIQA\n(1838)','HSwag\n(3000)','WGrde\n(1267)','BoolQ\n(3000)','ARC-E\n(570)']
    cfgs=['FP32','CIM-7b','ILP-20%','Uni-6b'];clrs=[C['gray'],C['blue'],C['green'],C['orange']]
    acc={'FP32':[48.2,29.7,52.3,38.3,38.1],'CIM-7b':[48.4,27.0,49.9,37.9,36.1],
         'ILP-20%':[48.4,26.9,49.7,37.9,36.8],'Uni-6b':[48.4,27.3,49.8,37.9,36.1]}
    ci={'CIM-7b':[2.3,1.6,2.8,1.7,4.0],'ILP-20%':[2.3,1.6,2.8,1.7,4.1],'Uni-6b':[2.3,1.6,2.7,1.7,4.0]}
    fig,ax=plt.subplots(figsize=(6.5,3.0))
    x=np.arange(5);w=0.2;offs=[-1.5*w,-0.5*w,0.5*w,1.5*w]
    for idx,(cfg,c) in enumerate(zip(cfgs,clrs)):
        v=acc[cfg];yerr=ci.get(cfg,[0]*5)
        ax.bar(x+offs[idx],v,w,label=cfg,color=c,edgecolor='k',linewidth=BLW,zorder=3)
        if cfg!='FP32': ax.errorbar(x+offs[idx],v,yerr=yerr,fmt='none',ecolor='black',capsize=3,capthick=0.6,linewidth=0.6,zorder=4)
    ax.set_xticks(x);ax.set_xticklabels(tasks);ax.set_ylabel('Accuracy (%)')
    ax.set_title('Zero-shot accuracy with 95% bootstrap CI')
    ax.legend(loc='upper right',ncol=4);ax.grid(axis='y');ax.set_ylim(20,58)
    ax.annotate('ILP within CIM-7b CI\non all tasks',xy=(2.5,25),**BF(10),fontstyle='italic',
        color=C['green'],ha='center',bbox=dict(boxstyle='round,pad=0.3',facecolor=C['lg'],alpha=0.8))
    plt.tight_layout();plt.savefig(OUT/"fig_downstream_ci.pdf");plt.close()

# ═══ 7. PPA curve ═══
def fig_ppa():
    bits=[3,4,5,6,7,8,9,10]
    a125=[668,695,731,806,937,1147,1672,2822];a13b=[9496,9878,10397,11469,13321,16311,23773,40131]
    pct=[3.9,5.8,8.7,15.5,24.4,35.5,49.1,63.9]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,3.0))
    ax1.semilogy(bits,a125,'o-',color=C['blue'],markersize=6,label='OPT-125M',zorder=5)
    ax1.semilogy(bits,a13b,'s-',color=C['orange'],markersize=6,label='OPT-1.3B',zorder=5)
    ax1.axvline(x=7,color=C['red'],linestyle='--',alpha=0.4,linewidth=1.0)
    ax1.axvspan(8,10,alpha=0.06,color=C['green'])
    ax1.set_xlabel('ADC bits');ax1.set_ylabel('Chip area (mm\u00b2)')
    ax1.set_title('(a) Area vs. ADC resolution');ax1.legend();ax1.grid(True);ax1.set_xticks(bits)
    bcol=[C['lb']]*4+[C['blue']]+[C['lb']]*3
    ax2.bar(bits,pct,color=bcol,edgecolor=C['blue'],linewidth=BLW)
    ax2.set_xlabel('ADC bits');ax2.set_ylabel('ADC area (%)')
    ax2.set_title('(b) ADC share of chip');ax2.set_xticks(bits)
    for b,p in zip(bits,pct): ax2.text(b,p+1.5,f'{p:.0f}%',ha='center',**BF(9))
    ax2.axhline(y=50,color=C['red'],linestyle=':',alpha=0.4)
    plt.tight_layout();plt.savefig(OUT/"fig_ppa_curve.pdf");plt.close()

# ═══ 8. Bit assignment ═══
def fig_bit_assign():
    PALETTE={"attn_qkv":"#C0392B","attn_out":"#27AE60","ffn_up":"#E67E22","ffn_down":"#2980B9","lm_head":"#8E44AD"}
    LABELS={"attn_qkv":"$W_{qkv}$","attn_out":"$W_{out}$","ffn_up":"$W_{fc1}$","ffn_down":"$W_{fc2}$","lm_head":"$W_{head}$"}
    def classify(n):
        nl=n.lower()
        if "q_proj" in nl or "k_proj" in nl or "v_proj" in nl: return "attn_qkv"
        if "out_proj" in nl: return "attn_out"
        if "fc1" in nl: return "ffn_up"
        if "fc2" in nl: return "ffn_down"
        if "lm_head" in nl: return "lm_head"
        return "other"
    alloc=json.load(open(RD/"sensitivity/opt125m/allocations.json"))
    greedy=alloc["greedy"];ilp=alloc["ilp"];layers=sorted(greedy.keys())
    n=len(layers);types=[classify(l) for l in layers]
    fig,axes=plt.subplots(3,1,figsize=(7.16,4.2),sharex=True)
    configs=[("(a) Uniform 7-bit",[7]*n),("(b) Sens-Greedy",[greedy[l] for l in layers]),("(c) ILP (optimal)",[ilp[l] for l in layers])]
    for ax,(title,bits) in zip(axes,configs):
        for i,(b,lt) in enumerate(zip(bits,types)):
            ax.bar(i,b,color=PALETTE.get(lt,"#95A5A6"),width=0.9,edgecolor="none",alpha=0.85)
        ax.axhline(7,ls="--",color="#7F8C8D",lw=0.8)
        ax.set_ylabel("Bits");ax.set_title(title,fontsize=12,fontweight="bold",loc="left")
        ax.set_ylim(4.5,7.8);ax.set_yticks([5,6,7])
        bc=Counter(bits)
        ax.text(0.98,0.78,f"{bc.get(6,0)}@6b, {bc.get(7,0)}@7b",transform=ax.transAxes,ha="right",**BF(10),color="#2C3E50")
    axes[-1].set_xlabel("Layer index")
    seen={}
    for lt in ["attn_qkv","attn_out","ffn_up","ffn_down","lm_head"]:
        seen[lt]=mpatches.Patch(color=PALETTE[lt],label=LABELS[lt])
    fig.legend(handles=list(seen.values()),loc="lower center",ncol=5,fontsize=10,
              bbox_to_anchor=(0.5,-0.02),framealpha=0.9,edgecolor="none")
    plt.tight_layout(h_pad=0.5);plt.subplots_adjust(bottom=0.12)
    plt.savefig(OUT/"fig5_bit_assignment.pdf");plt.close()

if __name__=="__main__":
    P("Regenerating all with BOLD + LARGE style...")
    for n,fn in [("1.Sensitivity",fig_sensitivity),("2.Deployment",fig_deployment),
        ("3.Transfer",fig_transfer),("4.Proxy",fig_proxy),("5.Interaction",fig_interaction),
        ("6.Downstream",fig_downstream),("7.PPA",fig_ppa),("8.BitAssign",fig_bit_assign)]:
        try: fn();P(f"  OK {n}")
        except Exception as e: P(f"  FAIL {n}: {e}");import traceback;traceback.print_exc()
    # Also regenerate Pareto via separate script
    import subprocess
    subprocess.run(["python3","/raid/privatedata/fantao/iccad_exp/regen_pareto.py"])
    P("Done.")
