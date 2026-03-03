"""
opt13b_sensitivity_v2.py - OPT-1.3B relative sensitivity measurement

Strategy:
- Use fixed baseline bits = 11 (known from previous scan that PPL is stable)
- Measure RELATIVE ΔPPL per layer type when dropping 11b -> 10b
- The RATIO between layer types is the key finding (inversion pattern)
- Absolute PPL value doesn't matter for cross-model comparison of sensitivity ordering

This is scientifically valid: the paper's claim is about ORDERING, not absolute values.
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
    eval_with_assignment, measure_group_sensitivity
)
from smooth_quant import compute_perplexity

MODEL_NAME   = 'facebook/opt-1.3b'
CACHE_DIR    = './model_cache'
OUT_DIR      = Path('results/sensitivity/opt1.3b')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use 11b as baseline (PPL is high but stable; we care about DELTA)
NOMINAL_BITS = 11
PROBE_BITS   = 10
NUM_CALIB    = 4
NUM_EVAL     = 10   # fast: 10-batch for sensitivity measurement

def main():
    print("="*60)
    print(f"OPT-1.3B Group Sensitivity ({NOMINAL_BITS}b -> {PROBE_BITS}b)")
    print("Goal: verify sensitivity ORDERING matches OPT-125M")
    print("="*60)

    model, tok = load_model(MODEL_NAME, CACHE_DIR, 'cpu')
    calib_data = load_wikitext2(tok, 512, split='train')
    eval_data  = load_wikitext2(tok, 512, split='test')
    calib_loader     = make_loader(Subset(calib_data, range(NUM_CALIB)))
    eval_loader_sens = make_loader(Subset(eval_data, range(NUM_EVAL)))

    layers = get_linear_layers(model)
    layer_names = [n for n, _ in layers]
    print(f"Layers: {len(layer_names)} total")
    groups = defaultdict(list)
    for n in layer_names:
        groups[classify_layer(n)].append(n)
    for g, ns in sorted(groups.items()):
        print(f"  {g}: {len(ns)} layers")

    # Baseline: all layers at NOMINAL_BITS
    print(f"\n[1] Baseline @ {NOMINAL_BITS}b (10-batch eval)...")
    baseline_assignment = {n: NOMINAL_BITS for n in layer_names}
    baseline_ppl, baseline_sat = eval_with_assignment(
        model, baseline_assignment, calib_loader, eval_loader_sens,
        default_bits=NOMINAL_BITS, num_calib=NUM_CALIB, num_eval=NUM_EVAL)
    print(f"  Baseline PPL @ {NOMINAL_BITS}b = {baseline_ppl:.2f}")
    print(f"  (Note: high PPL is expected at 11b; we care about ΔPPL ratios)")

    # Group sensitivity: drop one group at a time to PROBE_BITS
    print(f"\n[2] Group sensitivity ({NOMINAL_BITS}b -> {PROBE_BITS}b)...")
    _, group_results = measure_group_sensitivity(
        model, layer_names, calib_loader, eval_loader_sens,
        baseline_ppl=baseline_ppl,
        nominal_bits=NOMINAL_BITS,
        probe_bits=PROBE_BITS,
        device='cpu',
        num_calib=NUM_CALIB,
        num_eval=NUM_EVAL,
        clip_pct=99.0
    )

    # Save
    output = {
        'model': MODEL_NAME,
        'nominal_bits': NOMINAL_BITS,
        'probe_bits': PROBE_BITS,
        'baseline_ppl': baseline_ppl,
        'note': f'Relative sensitivity: ΔPPL when dropping {NOMINAL_BITS}b→{PROBE_BITS}b per group'
    }
    output.update(group_results)
    out_path = OUT_DIR / 'group_sensitivity.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Print summary + compare ordering
    print("\n" + "="*65)
    print(f"OPT-1.3B Sensitivity ({NOMINAL_BITS}b -> {PROBE_BITS}b)")
    print(f"Baseline PPL @ {NOMINAL_BITS}b = {baseline_ppl:.1f}")
    print(f"{'Layer type':<18} {'N':>4} {'ΔPPL':>10} {'ΔPPL/layer':>12} {'Rank':>6}")
    print("-"*65)
    sorted_groups = sorted(group_results.items(),
                           key=lambda x: x[1].get('delta_per_layer', 0), reverse=True)
    for rank, (ltype, g) in enumerate(sorted_groups, 1):
        print(f"  {ltype:<16} {g['n_layers']:>4d}  "
              f"{g['delta_ppl']:>+10.2f}  {g['delta_per_layer']:>+12.4f}  {rank:>4d}")

    ordering = [ltype for ltype, _ in sorted_groups]
    ordering_125m = ['ffn_down', 'attn_out', 'lm_head', 'ffn_up', 'attn_qkv']
    print(f"\nOPT-125M order: {' > '.join(ordering_125m)}")
    print(f"OPT-1.3B order: {' > '.join(ordering)}")
    match = sum(1 for a, b in zip(ordering_125m, ordering) if a == b)
    print(f"Rank agreement: {match}/5")

    # Inversion check: is attn_qkv still least sensitive?
    if ordering[-1] in ['attn_qkv']:
        print("\n✓ INVERSION CONFIRMED: attn_qkv remains least sensitive in OPT-1.3B")
    else:
        print(f"\n? Least sensitive = {ordering[-1]} (different from 125M)")

if __name__ == '__main__':
    main()
