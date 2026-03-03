"""
opt350m_sensitivity.py — OPT-350M sensitivity measurement for cross-model validation
====================================================================================
Tests whether the saturation-sensitivity inversion observed on OPT-125M
generalizes to OPT-350M (24 decoder layers, 145 linear layers, hidden=1024).

Key hypothesis to validate:
  - OPT-350M should also show attn_out > q/k/v_proj in measured sensitivity
  - Larger model → worse outliers → same layer-type ordering?

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    python3 opt350m_sensitivity.py
"""
import os, sys, json, csv, copy, time
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np

from llm_inference import load_model, load_wikitext2, make_loader
from smooth_quant import compute_perplexity
from sensitivity_analysis import (
    PerLayerCIMHook, eval_with_assignment, get_linear_layers, classify_layer,
    measure_group_sensitivity, ilp_allocation, _sensitivity_greedy,
    compute_area_from_assignment, load_ppa_sweep
)

# OPT-350M NeuroSIM PPA scaling from OPT-125M
# OPT-350M hidden=1024, 24 layers → tile count ≈ (1024/768)² × (24/12) × (4 unique types) ≈ 3.56× vs 125M
OPT350M_SCALE = 3.56  # approximate tile count ratio vs OPT-125M

OUTPUT_DIR = Path('results/sensitivity/opt350m')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PPA_CSV_125M = Path('results/ppa/opt125m/ppa_sweep_opt125m.csv')


def make_scaled_ppa(ppa_125m, scale):
    """Scale OPT-125M PPA by tile count ratio."""
    scaled = {}
    for bits, row in ppa_125m.items():
        scaled[bits] = {k: v * scale if k.endswith('_um2') or k.endswith('_pJ') else v
                        for k, v in row.items()}
    return scaled


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='facebook/opt-350m')
    p.add_argument('--model_cache', default='./model_cache')
    p.add_argument('--device', default='cpu')
    p.add_argument('--num_calib_batches', type=int, default=8)
    p.add_argument('--num_eval_batches', type=int, default=50)
    p.add_argument('--nominal_bits', type=int, default=7)
    p.add_argument('--probe_bits', type=int, default=6)
    p.add_argument('--clip_pct', type=float, default=99.0)
    args = p.parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model, args.model_cache, args.device)
    data = load_wikitext2(tokenizer, seq_len=512)
    calib_data = data[:args.num_calib_batches + 16]
    eval_data  = data[64:64 + args.num_eval_batches + 32]
    calib_loader = make_loader(calib_data)
    eval_loader  = make_loader(eval_data)

    # ── PPA (scaled from OPT-125M) ──────────────────────────────────────────
    ppa_125m = {}
    if PPA_CSV_125M.exists():
        ppa_125m = load_ppa_sweep(str(PPA_CSV_125M))
    ppa_350m = make_scaled_ppa(ppa_125m, OPT350M_SCALE)

    # Save scaled PPA
    ppa_dir = Path('results/ppa/opt350m')
    ppa_dir.mkdir(parents=True, exist_ok=True)
    with open(ppa_dir / 'ppa_sweep_opt350m.csv', 'w', newline='') as f:
        rows = [{'model': 'opt350m', 'adc_bits': b,
                 **{k: v for k, v in ppa_350m[b].items() if k != 'adc_bits'}}
                for b in sorted(ppa_350m.keys())]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    layer_names = [name for name, _ in get_linear_layers(model)]
    n = len(layer_names)
    print(f"\n[OPT-350M] {n} linear layers")
    type_counts = Counter(classify_layer(nm) for nm in layer_names)
    print(f"[OPT-350M] Types: {dict(type_counts)}")

    # ── Measure baseline PPL ────────────────────────────────────────────────
    print(f"\n[Baseline] Measuring uniform {args.nominal_bits}b PPL ({args.num_eval_batches} batches)...")
    uniform_7b = {nm: args.nominal_bits for nm in layer_names}
    baseline_ppl, baseline_sat = eval_with_assignment(
        model, uniform_7b, calib_loader, eval_loader,
        default_bits=args.nominal_bits, device=args.device,
        num_calib=args.num_calib_batches, num_eval=args.num_eval_batches,
        clip_pct=args.clip_pct)
    print(f"[Baseline] Uniform {args.nominal_bits}b PPL={baseline_ppl:.4f}")

    clean_ppl = compute_perplexity(model, eval_loader, device=args.device,
                                    max_batches=args.num_eval_batches)
    print(f"[Baseline] Clean FP32 PPL={clean_ppl:.4f}")

    # ── Group sensitivity measurement ────────────────────────────────────────
    print(f"\n[Sensitivity] Group-level {args.nominal_bits}b→{args.probe_bits}b measurement...")
    sensitivity_data, group_results = measure_group_sensitivity(
        model, layer_names, calib_loader, eval_loader,
        baseline_ppl=baseline_ppl,
        nominal_bits=args.nominal_bits, probe_bits=args.probe_bits,
        device=args.device, num_calib=args.num_calib_batches,
        num_eval=args.num_eval_batches, clip_pct=args.clip_pct)

    # Save group results
    gpath = OUTPUT_DIR / 'group_sensitivity.json'
    with open(gpath, 'w') as f:
        json.dump(group_results, f, indent=2)
    print(f"\n[Sensitivity] OPT-350M group results:")
    for lt, g in sorted(group_results.items(), key=lambda x: -x[1]['delta_per_layer']):
        print(f"  {lt:<15s}: ΔPPL/layer={g['delta_per_layer']:+.4f}  "
              f"(total={g['delta_ppl']:+.3f}, n={g['n_layers']})")

    # ── ILP Allocation ──────────────────────────────────────────────────────
    bit_choices = (4, 5, 6, 7, 8)
    TARGET = 0.20
    asgn_ilp = ilp_allocation(sensitivity_data, ppa_350m,
                               nominal_bits=args.nominal_bits,
                               bit_choices=bit_choices,
                               target_area_savings=TARGET)
    print(f"\n[ILP] Allocation: {dict(Counter(asgn_ilp))}")

    # ── Evaluate ────────────────────────────────────────────────────────────
    ppa_ref = ppa_350m.get(args.nominal_bits, {})
    configs = {
        f'Uniform {args.nominal_bits}b': {nm: args.nominal_bits for nm in layer_names},
        f'Uniform {args.probe_bits}b':   {nm: args.probe_bits for nm in layer_names},
        f'ILP {TARGET*100:.0f}%':        dict(zip(layer_names, asgn_ilp)),
    }

    eval_results = []
    for cfg_name, assignment in configs.items():
        bits_list = [assignment.get(nm, args.nominal_bits) for nm in layer_names]
        ppl, sat = eval_with_assignment(
            model, assignment, calib_loader, eval_loader,
            default_bits=args.nominal_bits, device=args.device,
            num_calib=args.num_calib_batches, num_eval=args.num_eval_batches,
            clip_pct=args.clip_pct)
        area = compute_area_from_assignment(bits_list, ppa_ref, args.nominal_bits) if ppa_ref else {}
        bc = Counter(bits_list)
        print(f"  {cfg_name:<25s}: PPL={ppl:.2f}  ADC={area.get('adc_area_mm2',0):.0f}mm²  "
              f"sav={area.get('adc_savings_pct',0):.1f}%")
        eval_results.append({
            'model': 'opt-350m',
            'config': cfg_name,
            'ppl': ppl,
            'adc_area_mm2': area.get('adc_area_mm2', 0),
            'adc_savings_pct': area.get('adc_savings_pct', 0),
            'n_6b': bc.get(6, 0),
            'n_7b': bc.get(7, 0),
        })

    # Save
    eval_path = OUTPUT_DIR / 'evaluation_results.csv'
    with open(eval_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(eval_results[0].keys()))
        w.writeheader()
        w.writerows(eval_results)

    print(f"\n[Done] OPT-350M results saved to {OUTPUT_DIR}")
    print(f"[Key Finding] Sensitivity ordering:")
    rank = sorted(group_results.items(), key=lambda x: -x[1]['delta_per_layer'])
    for i, (lt, g) in enumerate(rank):
        print(f"  {i+1}. {lt:<15s}: {g['delta_per_layer']:+.4f} ΔPPL/layer")


if __name__ == '__main__':
    main()
