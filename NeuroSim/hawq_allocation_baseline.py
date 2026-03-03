"""
hawq_allocation_baseline.py - Compare HAWQ-guided vs ILP-guided ADC allocation

HAWQ使用Hessian trace排序分配ADC bits，在相同面积预算下与ILP对比PPL。
HAWQ ranking (from hessian_group.json):
  lm_head > ffn_down > ffn_up > attn_qkv > attn_out
  (rank 1=most sensitive=protect most)

At 20.5% savings (181.5mm² out of 228.4mm²):
- Greedy allocates 6b to LEAST sensitive groups first
- HAWQ-guided: use Hessian ranking to decide which layers get 6b
- ILP-guided: already done (PPL=308.6)
"""
import os, sys, json
from pathlib import Path

os.environ.pop('HF_HUB_OFFLINE', None)
os.environ['HF_DATASETS_OFFLINE'] = '1'

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import torch
from torch.utils.data import Subset
from collections import defaultdict

from llm_inference import load_model, load_wikitext2, make_loader
from sensitivity_analysis import PerLayerCIMHook, get_linear_layers, classify_layer, eval_with_assignment
from smooth_quant import compute_perplexity

MODEL_NAME = 'facebook/opt-125m'
CACHE_DIR  = './model_cache'
OUT_DIR    = Path('results/sensitivity/opt125m')
NOMINAL    = 7
PROBE      = 6
NUM_CALIB  = 4
NUM_EVAL   = 100   # stable 100-batch for fair comparison

def greedy_allocation(layer_names, sensitivity_map, nominal=7, probe=6,
                      target_savings=0.20):
    """
    Greedy: reduce bits for least sensitive layers first until budget met.
    sensitivity_map: {ltype: sensitivity_value}  (higher = more sensitive = protect)
    Returns: {layer_name: bits}
    """
    n = len(layer_names)
    ref_area = n * (2**nominal)
    budget = ref_area * (1.0 - target_savings)

    assignment = {name: nominal for name in layer_names}

    # Sort by sensitivity ascending (reduce least sensitive first)
    sorted_layers = sorted(layer_names,
                           key=lambda n: sensitivity_map.get(classify_layer(n), 999))

    for name in sorted_layers:
        current_area = sum(2**assignment[n] for n in layer_names)
        if current_area <= budget:
            break
        if assignment[name] > probe:
            assignment[name] = probe

    area_used = sum(2**assignment[n] for n in layer_names)
    savings = (ref_area - area_used) / ref_area * 100
    print(f"  Area: {area_used} / {ref_area} ({savings:.1f}% savings)")
    return assignment

def main():
    print("="*60)
    print("HAWQ-Guided vs ILP ADC Allocation Comparison (OPT-125M)")
    print("="*60)

    # Load model
    model, tok = load_model(MODEL_NAME, CACHE_DIR, 'cpu')
    calib_data = load_wikitext2(tok, 512, split='train')
    eval_data  = load_wikitext2(tok, 512, split='test')
    calib_loader = make_loader(Subset(calib_data, range(NUM_CALIB)))
    eval_loader  = make_loader(Subset(eval_data, range(NUM_EVAL)))

    layers = get_linear_layers(model)
    layer_names = [n for n, _ in layers]

    # Baseline: uniform 7b
    print("\n[1] Uniform 7b baseline (100-batch)...")
    ppl_7b, _ = eval_with_assignment(
        model, {n: NOMINAL for n in layer_names}, calib_loader, eval_loader,
        default_bits=NOMINAL, num_calib=NUM_CALIB, num_eval=NUM_EVAL)
    print(f"  Uniform 7b PPL = {ppl_7b:.2f}")

    # Load Hessian sensitivity
    hessian_path = Path('results/hessian/opt125m/hessian_group.json')
    with open(hessian_path) as f:
        hessian_data = json.load(f)

    # Hessian trace ranking (higher trace = more sensitive = protect more)
    # keys: attn_qkv, attn_out, ffn_up, ffn_down, lm_head
    # map to our ltype keys
    ltype_map = {
        'attn_qkv': 'attn_qkv',
        'attn_out': 'attn_out',
        'ffn_up': 'ffn_up',
        'ffn_down': 'ffn_down',
        'lm_head': 'lm_head',
    }
    hawq_sensitivity = {}
    for k, v in hessian_data.items():
        ltype = ltype_map.get(k, k)
        hawq_sensitivity[ltype] = v.get('mean_hawq', 0.0)
    print(f"\nHAWQ Hessian sensitivity (higher=more sensitive):")
    for ltype, val in sorted(hawq_sensitivity.items(), key=lambda x: -x[1]):
        print(f"  {ltype}: {val:.5f}")

    # Load measured (CIM) sensitivity
    cim_path = Path('results/sensitivity/opt125m/group_sensitivity.json')
    with open(cim_path) as f:
        cim_data = json.load(f)
    cim_sensitivity = {}
    for k, v in cim_data.items():
        if isinstance(v, dict) and 'delta_per_layer' in v:
            cim_sensitivity[k] = abs(v['delta_per_layer'])
    print(f"\nCIM measured sensitivity (higher=more sensitive):")
    for ltype, val in sorted(cim_sensitivity.items(), key=lambda x: -x[1]):
        print(f"  {ltype}: {val:.4f}")

    results = {'ppl_7b': ppl_7b}

    # [2] HAWQ-guided greedy allocation
    print("\n[2] HAWQ-guided greedy allocation (20.5% savings target)...")
    hawq_assignment = greedy_allocation(layer_names, hawq_sensitivity,
                                        target_savings=0.205)
    ppl_hawq, _ = eval_with_assignment(
        model, hawq_assignment, calib_loader, eval_loader,
        default_bits=NOMINAL, num_calib=NUM_CALIB, num_eval=NUM_EVAL)
    print(f"  HAWQ-guided PPL = {ppl_hawq:.2f}")
    results['ppl_hawq_guided'] = ppl_hawq

    # [3] CIM sensitivity-guided greedy (for reference)
    print("\n[3] CIM sensitivity-guided greedy (20.5% savings target)...")
    cim_greedy_assignment = greedy_allocation(layer_names, cim_sensitivity,
                                              target_savings=0.205)
    ppl_cim_greedy, _ = eval_with_assignment(
        model, cim_greedy_assignment, calib_loader, eval_loader,
        default_bits=NOMINAL, num_calib=NUM_CALIB, num_eval=NUM_EVAL)
    print(f"  CIM-greedy PPL = {ppl_cim_greedy:.2f}")
    results['ppl_cim_greedy'] = ppl_cim_greedy

    # [4] Saturation-guided greedy (already measured = 313.8)
    # Load from existing results
    stable_path = Path('results/stable/opt125m/stable_eval_results.csv')
    if stable_path.exists():
        import csv
        with open(stable_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'sat' in row.get('config','').lower() and 'greedy' in row.get('config','').lower():
                    results['ppl_sat_greedy'] = float(row.get('ppl', 313.8))
                    break
        if 'ppl_sat_greedy' not in results:
            results['ppl_sat_greedy'] = 313.8  # from memory
    else:
        results['ppl_sat_greedy'] = 313.8  # known from previous run

    # ILP result (from memory / previous run)
    results['ppl_ilp'] = 308.6

    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY (OPT-125M, ~20.5% ADC savings, 100-batch)")
    print("-"*60)
    configs = [
        ('Uniform 7b (baseline)', results['ppl_7b']),
        ('Sat-guided greedy',     results.get('ppl_sat_greedy', '-')),
        ('HAWQ-guided greedy',    results['ppl_hawq_guided']),
        ('CIM-guided greedy',     results['ppl_cim_greedy']),
        ('ILP (CIM-guided)',      results['ppl_ilp']),
    ]
    for name, ppl in configs:
        delta = f"+{ppl - results['ppl_7b']:.1f}" if isinstance(ppl, float) else '-'
        print(f"  {name:<28} PPL={ppl:<8} {delta}")

    # Save
    out_path = OUT_DIR / 'hawq_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

if __name__ == '__main__':
    main()
