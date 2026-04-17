#!/usr/bin/env python3
"""
Unified experiment: ALL operating points on same D_test (100 batch).
Also: PPL decomposition + clamped-sensitivity ILP.
"""
import sys, json, os, torch, numpy as np, random
from pathlib import Path
from collections import defaultdict

ROOT = Path("/raid/privatedata/fantao/iccad_exp")
sys.path.insert(0, str(ROOT))

from llm_inference import load_wikitext2, make_loader
from sensitivity_analysis import eval_with_assignment, ilp_allocation
from smooth_quant import compute_perplexity
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT = ROOT / "results_unified"; OUT.mkdir(exist_ok=True)
MODEL = "facebook/opt-125m"
CACHE = str(ROOT / "model_cache")
DEVICE = "cuda"

def P(msg): print(msg, flush=True)
def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def classify_layer(name):
    nl = name.lower()
    if "q_proj" in nl or "k_proj" in nl or "v_proj" in nl: return "attn_qkv"
    if "out_proj" in nl: return "attn_out"
    if "fc1" in nl: return "ffn_up"
    if "fc2" in nl: return "ffn_down"
    if "lm_head" in nl: return "lm_head"
    return "other"

def ev(model, assign, cld, eld, db, n_c=4, n_e=100, seed=0):
    set_seed(seed)
    r = eval_with_assignment(model, assign, cld, eld, default_bits=db,
                             device=DEVICE, num_calib=n_c, num_eval=n_e, clip_pct=99.0)
    return r[0] if isinstance(r, tuple) else r

def run_ilp(per_layer, nominal, target):
    r = ilp_allocation(per_layer, {b: 2**b for b in range(3,11)},
                       nominal_bits=nominal, bit_choices=tuple(range(max(4,nominal-3), nominal+1)),
                       target_area_savings=target)
    if isinstance(r, list) and r and isinstance(r[0], int):
        return {per_layer[i]['layer']: r[i] for i in range(len(r))}
    return r if isinstance(r, dict) else {}

def measure_groups(model, layer_names, cld, eld, baseline_ppl, nominal=7, probe=6, n_c=4, n_e=10):
    groups = defaultdict(list)
    for n in layer_names: groups[classify_layer(n)].append(n)
    group_sens = {}
    per_layer = []
    for ltype, names in groups.items():
        assign = {n_: nominal for n_ in layer_names}
        for n_ in names: assign[n_] = probe
        ppl = ev(model, assign, cld, eld, nominal, n_c=n_c, n_e=n_e)
        delta = ppl - baseline_ppl
        dpl = delta / len(names) if names else 0
        group_sens[ltype] = dpl
        P(f"  {ltype:15s} ({len(names):2d}): PPL={ppl:.2f} delta/layer={dpl:+.4f}")
        for n_ in names:
            per_layer.append({'layer': n_, 'layer_type': ltype,
                              'sensitivity': max(dpl, 0), 'delta_ppl': dpl,
                              'n_layers': len(names), 'nominal_bits': nominal, 'probe_bits': probe})
    return per_layer, group_sens

def main():
    P("="*60)
    P("UNIFIED EXPERIMENT")
    P("="*60)

    P("\n[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=CACHE, torch_dtype=torch.float32)
    model.to(DEVICE).eval()

    P("[2] Data splits: D_cal(4) / D_prof(10) / D_test(100)...")
    data = load_wikitext2(tokenizer, seq_len=512)
    P(f"  Total sequences: {len(data)}")
    D_cal  = data[0:4]
    D_prof = data[4:14]
    D_test = data[14:114]
    cld = make_loader(D_cal, batch_size=1)
    pld = make_loader(D_prof, batch_size=1)
    tld = make_loader(D_test, batch_size=1)

    layer_names = [n for n, m in model.named_modules()
                   if hasattr(m, 'weight') and m.weight.dim() == 2]
    P(f"  Layers: {len(layer_names)}")

    R = {}

    # ── PPL Decomposition ──
    P("\n[3] PPL Decomposition on D_test (100 batch)...")
    R["fp32"] = round(compute_perplexity(model, tld, device=DEVICE, max_batches=100), 1)
    P(f"  FP32:  {R['fp32']}")

    for bits in [7, 8, 9, 10]:
        uni = {n: bits for n in layer_names}
        R[f"cim_{bits}b"] = round(ev(model, uni, cld, tld, bits, n_e=100), 1)
        P(f"  CIM-{bits}b: {R[f'cim_{bits}b']}")

    # INT8-only (very high ADC = no ADC noise)
    try:
        uni12 = {n: 12 for n in layer_names}
        R["int8_only_12b"] = round(ev(model, uni12, cld, tld, 12, n_e=100), 1)
        P(f"  INT8-only(12b ADC): {R['int8_only_12b']}")
    except Exception as e:
        P(f"  INT8-only failed: {e}")

    uni6 = {n: 6 for n in layer_names}
    R["cim_6b"] = round(ev(model, uni6, cld, tld, 6, n_e=100), 1)
    P(f"  CIM-6b: {R['cim_6b']}")

    # ── Sensitivity profiling on D_prof ──
    P("\n[4] Sensitivity on D_prof (10 batch)...")
    ppl_7b_prof = ev(model, {n:7 for n in layer_names}, cld, pld, 7, n_e=10)
    P(f"  Baseline 7b on D_prof: {ppl_7b_prof:.2f}")
    per_layer_7, groups_7 = measure_groups(model, layer_names, cld, pld, ppl_7b_prof, 7, 6, n_c=4, n_e=10)
    R["sensitivity_7to6"] = {g: round(v, 4) for g, v in groups_7.items()}

    ppl_10b_prof = ev(model, {n:10 for n in layer_names}, cld, pld, 10, n_e=10)
    P(f"  Baseline 10b on D_prof: {ppl_10b_prof:.2f}")
    per_layer_10, groups_10 = measure_groups(model, layer_names, cld, pld, ppl_10b_prof, 10, 9, n_c=4, n_e=10)
    R["sensitivity_10to9"] = {g: round(v, 4) for g, v in groups_10.items()}

    # ── ILP allocations (clamped sensitivity) evaluated on D_test ──
    P("\n[5] ILP allocations on D_test...")

    # 7b regime
    for pct in [0.205, 0.30]:
        label = f"7b_{int(pct*100)}pct"
        alloc = run_ilp(per_layer_7, 7, pct)
        if alloc:
            ppl = ev(model, alloc, cld, tld, 7, n_e=100)
            R[f"measured_ilp_{label}"] = round(ppl, 1)
            P(f"  Measured-ILP {label}: {ppl:.1f}")

    # 10b regime
    alloc_10 = run_ilp(per_layer_10, 10, 0.20)
    if alloc_10:
        ppl_10 = ev(model, alloc_10, cld, tld, 10, n_e=100)
        R["measured_ilp_10b_20pct"] = round(ppl_10, 1)
        P(f"  Measured-ILP 10b 20%: {ppl_10:.1f}")

    # ── Proxy baselines (7b, 20.5%) ──
    P("\n[6] Proxy baselines on D_test...")
    # Saturation-guided: protect high-sat layers, reduce low-sat
    sat_order = ['attn_out', 'ffn_down', 'ffn_up', 'attn_qkv', 'lm_head']
    hess_order = ['attn_out', 'attn_qkv', 'ffn_up', 'ffn_down', 'lm_head']

    for proxy_name, order in [("sat", sat_order), ("hess", hess_order)]:
        proxy_per_layer = []
        for pl in per_layer_7:
            rank = order.index(pl['layer_type']) if pl['layer_type'] in order else len(order)
            proxy_per_layer.append({**pl, 'sensitivity': rank})
        alloc_proxy = run_ilp(proxy_per_layer, 7, 0.205)
        if alloc_proxy:
            ppl_proxy = ev(model, alloc_proxy, cld, tld, 7, n_e=100)
            R[f"{proxy_name}_ilp_20pct"] = round(ppl_proxy, 1)
            P(f"  {proxy_name}-ILP 20%: {ppl_proxy:.1f}")

    # Random (mean of 5 draws)
    P("  Random baselines (5 draws)...")
    rand_ppls = []
    for seed in range(5):
        set_seed(seed + 100)
        rand_assign = {}
        for pl in per_layer_7:
            rand_assign[pl['layer']] = random.choice([6, 7])
        ppl_rand = ev(model, rand_assign, cld, tld, 7, n_e=100, seed=seed+100)
        rand_ppls.append(ppl_rand)
        P(f"    seed {seed}: {ppl_rand:.1f}")
    R["random_mean"] = round(np.mean(rand_ppls), 1)
    R["random_std"] = round(np.std(rand_ppls), 1)
    P(f"  Random mean: {R['random_mean']} +/- {R['random_std']}")

    # ── Save ──
    P("\n[7] Saving...")
    with open(OUT / "unified_results.json", "w") as f:
        json.dump(R, f, indent=2)

    P("\n" + "="*60)
    P("RESULTS")
    P("="*60)
    for k, v in sorted(R.items()):
        P(f"  {k}: {v}")

if __name__ == "__main__":
    main()
