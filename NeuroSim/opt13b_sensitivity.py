"""
opt13b_sensitivity.py - OPT-1.3B group sensitivity measurement

Strategy:
1. Find acceptable baseline ADC bits (scan 11,12,13 until PPL < 2x clean)
2. Measure group sensitivity at baseline_bits -> (baseline_bits - 1)
3. Compare inversion pattern with OPT-125M
"""
import os, sys, json
from pathlib import Path
from collections import defaultdict

os.environ.pop('HF_HUB_OFFLINE', None)
os.environ['HF_DATASETS_OFFLINE'] = '1'

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import torch
from torch.utils.data import Subset

from llm_inference import load_model, load_wikitext2, make_loader
from sensitivity_analysis import (
    PerLayerCIMHook, get_linear_layers, classify_layer,
    measure_group_sensitivity, eval_with_assignment
)
from smooth_quant import compute_perplexity

MODEL_NAME = 'facebook/opt-1.3b'
CACHE_DIR  = './model_cache'
OUT_DIR    = Path('results/sensitivity/opt1.3b')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_baseline_bits(model, layer_names, calib_loader, eval_loader,
                       clean_ppl, bits_range=(9,10,11,12,13)):
    """Scan ADC bits until PPL < 2x clean_ppl (acceptable)."""
    threshold = clean_ppl * 2.0
    print(f"\n[Baseline scan] clean PPL={clean_ppl:.2f}, threshold={threshold:.2f}")
    for bits in bits_range:
        assignment = {n: bits for n in layer_names}
        ppl, sat = eval_with_assignment(
            model, assignment, calib_loader, eval_loader,
            default_bits=bits, device='cpu',
            num_calib=4, num_eval=10, clip_pct=99.0
        )
        print(f"  {bits}b ADC: PPL={ppl:.2f} (sat={sat:.3f})")
        if ppl < threshold:
            print(f"  -> Acceptable baseline: {bits}b (PPL={ppl:.2f})")
            return bits, ppl
    print(f"  -> No acceptable bits found in {bits_range}")
    return None, None

def main():
    print("="*60)
    print("OPT-1.3B Group Sensitivity Measurement")
    print("="*60)

    # Load model
    print("\n[1] Loading OPT-1.3B...")
    model, tok = load_model(MODEL_NAME, CACHE_DIR, 'cpu')

    # Data loaders
    print("[2] Loading data...")
    calib_data = load_wikitext2(tok, 512, split='train')
    eval_data  = load_wikitext2(tok, 512, split='test')
    calib_loader     = make_loader(Subset(calib_data, range(4)))
    eval_loader_sens = make_loader(Subset(eval_data, range(10)))    # fast: sensitivity pass
    eval_loader_base = make_loader(Subset(eval_data, range(30)))    # 30-batch baseline

    # Clean PPL
    print("[3] Clean FP32 PPL...")
    clean_ppl = compute_perplexity(model, eval_loader_base)
    print(f"  Clean PPL = {clean_ppl:.2f}")

    # Layer list
    layers = get_linear_layers(model)
    layer_names = [n for n, _ in layers]
    print(f"  Layers: {len(layer_names)} total")
    groups = defaultdict(list)
    for n in layer_names:
        groups[classify_layer(n)].append(n)
    for g, ns in groups.items():
        print(f"    {g}: {len(ns)} layers")

    # Step 1: Find baseline bits
    print("\n[4] Finding acceptable baseline bits...")
    baseline_bits, baseline_ppl = find_baseline_bits(
        model, layer_names, calib_loader, eval_loader_sens,
        clean_ppl, bits_range=(9, 10, 11, 12, 13)
    )

    if baseline_bits is None:
        print("ERROR: Could not find acceptable baseline. Stopping.")
        return

    probe_bits = baseline_bits - 1
    print(f"\n[5] Sensitivity measurement: {baseline_bits}b -> {probe_bits}b")
    print(f"  Baseline PPL (10-batch) = {baseline_ppl:.2f}")

    # Step 2: Group sensitivity measurement
    _, group_results = measure_group_sensitivity(
        model, layer_names, calib_loader, eval_loader_sens,
        baseline_ppl=baseline_ppl,
        nominal_bits=baseline_bits,
        probe_bits=probe_bits,
        device='cpu',
        num_calib=4,
        num_eval=10,
        clip_pct=99.0
    )

    # Step 3: Save results
    output = {
        'model': MODEL_NAME,
        'clean_ppl': clean_ppl,
        'baseline_bits': baseline_bits,
        'probe_bits': probe_bits,
        'baseline_ppl': baseline_ppl,
    }
    output.update(group_results)  # add per-group data
    out_path = OUT_DIR / 'group_sensitivity.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Step 4: Print summary and compare with OPT-125M
    print("\n" + "="*65)
    print(f"OPT-1.3B Sensitivity ({baseline_bits}b -> {probe_bits}b)")
    print(f"{'Layer type':<18} {'N':>4} {'ΔPPL/layer':>12} {'Rank':>6}")
    print("-"*65)

    # Sort by sensitivity descending
    sorted_groups = sorted(group_results.items(),
                           key=lambda x: x[1].get('delta_per_layer', 0), reverse=True)
    for rank, (ltype, g) in enumerate(sorted_groups, 1):
        print(f"  {ltype:<16} {g['n_layers']:>4d}  {g['delta_per_layer']:>+12.4f}  {rank:>4d}")

    # Compare with 125M ordering
    print("\nOPT-125M ordering (for comparison):")
    ordering_125m = ['ffn_down', 'attn_out', 'lm_head', 'ffn_up', 'attn_qkv']
    ordering_13b  = [ltype for ltype, _ in sorted_groups]
    print(f"  125M: {' > '.join(ordering_125m)}")
    print(f"  1.3B: {' > '.join(ordering_13b)}")
    match = sum(1 for a, b in zip(ordering_125m, ordering_13b) if a == b)
    print(f"  Rank agreement: {match}/5")

if __name__ == '__main__':
    main()
