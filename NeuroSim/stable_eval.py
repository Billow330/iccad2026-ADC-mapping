"""
stable_eval.py — Stable PPL evaluation with large eval set
===========================================================
Fixes the PPL variance issue by using num_eval_batches=100.
Also:
  1. Re-runs all configurations with stable eval
  2. Computes ILP vs Greedy at MULTIPLE budget points to find where ILP wins
  3. Generates the comparison data for paper Table II and Figure 5

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    python3 stable_eval.py --model facebook/opt-125m --output_dir results/stable
"""

import os, sys, json, csv, argparse, copy, time
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from llm_inference import load_model, load_wikitext2, make_loader
from smooth_quant import compute_perplexity, CIMSmoothQuant
from outlier_analysis import _is_linear_like
from sensitivity_analysis import (
    PerLayerCIMHook, eval_with_assignment, get_linear_layers, classify_layer,
    ilp_allocation, _sensitivity_greedy, adc_area_ratio, compute_area_from_assignment,
    load_ppa_sweep
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='facebook/opt-125m')
    p.add_argument('--model_cache', default='./model_cache')
    p.add_argument('--output_dir', default='results/stable/opt125m')
    p.add_argument('--ppa_csv', default='results/ppa/opt125m/ppa_sweep_opt125m.csv')
    p.add_argument('--sens_json', default='results/sensitivity/opt125m/group_sensitivity.json',
                   help='Group sensitivity JSON from previous run')
    p.add_argument('--device', default='cpu')
    p.add_argument('--num_calib_batches', type=int, default=8)
    p.add_argument('--num_eval_batches', type=int, default=100,
                   help='Use 100+ for stable PPL estimates (default: 100)')
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--nominal_bits', type=int, default=7)
    p.add_argument('--probe_bits', type=int, default=6)
    p.add_argument('--clip_pct', type=float, default=99.0)
    p.add_argument('--weight_bits', type=int, default=8)
    p.add_argument('--input_bits', type=int, default=8)
    p.add_argument('--with_smoothquant', action='store_true')
    p.add_argument('--multi_budget', action='store_true',
                   help='Compare ILP vs Greedy across multiple budget targets')
    return p.parse_args()


def sensitivity_from_json(sens_json_path, layer_names, baseline_ppl):
    """Load group sensitivity and build per-layer sensitivity list."""
    with open(sens_json_path) as f:
        group_data = json.load(f)

    per_layer = []
    for name in layer_names:
        ltype = classify_layer(name)
        g = group_data.get(ltype, {})
        # delta_per_layer from group measurement (already per-layer)
        dppl = g.get('delta_per_layer', 0.0)
        per_layer.append({
            'layer':       name,
            'layer_type':  ltype,
            'delta_ppl':   dppl,
            'sat_rate':    g.get('sat_rate', 0.0),
            'sensitivity': max(dppl, 0.0),
        })
    return per_layer


def run_greedy(sensitivity_data, layer_names, nominal_bits, bit_choices, target):
    """Run sensitivity-guided greedy allocation."""
    n = len(sensitivity_data)
    ref_area = n * (2 ** nominal_bits)
    budget = ref_area * (1.0 - target)

    sens_sorted = sorted(range(n), key=lambda i: sensitivity_data[i]['sensitivity'])
    assignments = [nominal_bits] * n
    curr_area = ref_area

    rounds = 0
    while curr_area > budget and rounds < 30:
        improved = False
        for idx in sens_sorted:
            if curr_area <= budget:
                break
            cur = assignments[idx]
            lower = [b for b in sorted(bit_choices) if b < cur]
            if lower:
                new_b = max(lower)
                curr_area += (2 ** new_b) - (2 ** cur)
                assignments[idx] = new_b
                improved = True
        if not improved:
            break
        rounds += 1

    savings = (1.0 - curr_area / ref_area) * 100
    return assignments, savings


def eval_config(model, assignment, layer_names, calib_loader, eval_loader,
                ppa_ref, nominal_bits, args):
    """Evaluate one configuration, return (ppl, sat, area_info)."""
    bits_list = [assignment.get(n, nominal_bits) for n in layer_names]
    ppl, sat = eval_with_assignment(
        model, assignment, calib_loader, eval_loader,
        default_bits=nominal_bits, device=args.device,
        num_calib=args.num_calib_batches,
        num_eval=args.num_eval_batches, clip_pct=args.clip_pct)
    area = compute_area_from_assignment(bits_list, ppa_ref, nominal_bits)
    return ppl, sat, area


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[StableEval] num_eval_batches={args.num_eval_batches} "
          f"(~{args.num_eval_batches * args.seq_len / 1000:.0f}K tokens)")

    # ── Load model & data ────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model, args.model_cache, args.device)
    data = load_wikitext2(tokenizer, seq_len=args.seq_len)

    # Use a fixed, large eval set
    calib_data = data[:args.num_calib_batches + 16]
    eval_data  = data[64:64 + args.num_eval_batches + 32]
    calib_loader = make_loader(calib_data)
    eval_loader  = make_loader(eval_data)

    # ── PPA reference ────────────────────────────────────────────────────────
    ppa_sweep = {}
    if Path(args.ppa_csv).exists():
        ppa_sweep = load_ppa_sweep(args.ppa_csv)
    ppa_ref = ppa_sweep.get(args.nominal_bits, {})

    # ── Layer names ──────────────────────────────────────────────────────────
    layer_names = [name for name, _ in get_linear_layers(model)]
    n = len(layer_names)
    print(f"[Setup] {n} linear layers")

    # ── Load sensitivity ─────────────────────────────────────────────────────
    print(f"\n[Sensitivity] Loading from {args.sens_json}")
    sensitivity_data = sensitivity_from_json(args.sens_json, layer_names, baseline_ppl=None)

    # Print sensitivity summary
    print("[Sensitivity] Per-type ΔPPL/layer:")
    by_type = defaultdict(list)
    for r in sensitivity_data:
        by_type[r['layer_type']].append(r['sensitivity'])
    for lt, vals in sorted(by_type.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {lt:<15s}: {np.mean(vals):.4f} ΔPPL/layer")

    # ── Phase 1: Stable baseline evaluation ──────────────────────────────────
    print(f"\n[Phase1] Stable evaluation of key configurations "
          f"({args.num_eval_batches} batches)...")

    bit_choices = (4, 5, 6, 7, 8)

    # Allocations at 20% target
    TARGET = 0.20
    asgn_ilp20  = ilp_allocation(sensitivity_data, ppa_sweep,
                                  nominal_bits=args.nominal_bits,
                                  bit_choices=bit_choices,
                                  target_area_savings=TARGET)
    asgn_grd20, _ = run_greedy(sensitivity_data, layer_names, args.nominal_bits,
                                bit_choices, TARGET)
    # Saturation-guided greedy (old method): high sat = protect, low sat = reduce
    # Use sat_rate as sensitivity (HIGH sat = high sensitivity = protect)
    sat_sens = copy.deepcopy(sensitivity_data)
    for r in sat_sens:
        r['sensitivity'] = r['sat_rate']  # use saturation as proxy
    asgn_satgrd20, _ = run_greedy(sat_sens, layer_names, args.nominal_bits,
                                   bit_choices, TARGET)

    configs = {
        f'Uniform {args.nominal_bits}b':  {n_: args.nominal_bits for n_ in layer_names},
        f'Uniform {args.probe_bits}b':    {n_: args.probe_bits for n_ in layer_names},
        f'Sat-Greedy {TARGET*100:.0f}%':  dict(zip(layer_names, asgn_satgrd20)),
        f'Sens-Greedy {TARGET*100:.0f}%': dict(zip(layer_names, asgn_grd20)),
        f'ILP {TARGET*100:.0f}%':         dict(zip(layer_names, asgn_ilp20)),
    }

    phase1_results = []
    for cfg_name, assignment in configs.items():
        t0 = time.time()
        ppl, sat, area = eval_config(model, assignment, layer_names,
                                      calib_loader, eval_loader, ppa_ref,
                                      args.nominal_bits, args)
        elapsed = time.time() - t0
        bc = Counter(assignment.values())
        row = {
            'config': cfg_name,
            'ppl': round(ppl, 4),
            'sat': round(sat, 5),
            'adc_area_mm2': round(area.get('adc_area_mm2', 0), 2),
            'chip_area_mm2': round(area.get('chip_area_mm2', 0), 2),
            'adc_savings_pct': round(area.get('adc_savings_pct', 0), 2),
            'n_6b': bc.get(6, 0),
            'n_7b': bc.get(7, 0),
        }
        print(f"  {cfg_name:<28s}: PPL={ppl:.2f}  ADC={area.get('adc_area_mm2',0):.1f}mm²  "
              f"sav={area.get('adc_savings_pct',0):.1f}%  ({elapsed:.0f}s)")
        phase1_results.append(row)

    # ── Phase 2: ILP vs Greedy at multiple budgets ──────────────────────────
    if args.multi_budget:
        print(f"\n[Phase2] ILP vs Greedy at multiple budget targets...")
        budget_targets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        multi_results = []

        for target in budget_targets:
            asgn_ilp = ilp_allocation(sensitivity_data, ppa_sweep,
                                       nominal_bits=args.nominal_bits,
                                       bit_choices=bit_choices,
                                       target_area_savings=target)
            asgn_grd, _ = run_greedy(sensitivity_data, layer_names,
                                      args.nominal_bits, bit_choices, target)

            ilp_assign = dict(zip(layer_names, asgn_ilp))
            grd_assign = dict(zip(layer_names, asgn_grd))

            ppl_ilp, _, area_ilp = eval_config(model, ilp_assign, layer_names,
                                                calib_loader, eval_loader, ppa_ref,
                                                args.nominal_bits, args)
            ppl_grd, _, area_grd = eval_config(model, grd_assign, layer_names,
                                                calib_loader, eval_loader, ppa_ref,
                                                args.nominal_bits, args)

            delta = ppl_grd - ppl_ilp   # positive = ILP is better
            print(f"  target={target*100:.0f}%  ILP: PPL={ppl_ilp:.2f} area={area_ilp.get('adc_savings_pct',0):.1f}%  "
                  f"Greedy: PPL={ppl_grd:.2f} area={area_grd.get('adc_savings_pct',0):.1f}%  "
                  f"ΔPPL(Grd-ILP)={delta:+.2f}")
            multi_results.append({
                'target_pct': target * 100,
                'ilp_ppl': round(ppl_ilp, 4),
                'ilp_savings': round(area_ilp.get('adc_savings_pct', 0), 2),
                'ilp_adc_mm2': round(area_ilp.get('adc_area_mm2', 0), 2),
                'grd_ppl': round(ppl_grd, 4),
                'grd_savings': round(area_grd.get('adc_savings_pct', 0), 2),
                'grd_adc_mm2': round(area_grd.get('adc_area_mm2', 0), 2),
                'delta_ppl': round(delta, 4),
                'ilp_better': delta > 0,
            })

        multi_path = out_dir / 'ilp_vs_greedy_multibudget.csv'
        with open(multi_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(multi_results[0].keys()))
            w.writeheader()
            w.writerows(multi_results)
        print(f"[Phase2] Saved to {multi_path}")

        # Summary
        better = sum(1 for r in multi_results if r['ilp_better'])
        print(f"\n[Phase2] ILP better than Greedy in {better}/{len(multi_results)} budget points")

    # ── Phase 3: SmoothQuant ────────────────────────────────────────────────
    if args.with_smoothquant:
        print(f"\n[Phase3] SmoothQuant experiments...")
        model_sq = copy.deepcopy(model)
        sq = CIMSmoothQuant(
            weight_bits=args.weight_bits, input_bits=args.input_bits,
            off_state=6e-3, on_state=6e-3 * 17, vdd=1.0, parallel_read=128,
            adc_bits=args.nominal_bits, adc_clip_pct=args.clip_pct,
            sat_lambda=0.5, verbose=False)
        sq.fit(model_sq, calib_loader, num_batches=args.num_calib_batches,
               device=args.device, task='lm')
        print("[Phase3] SmoothQuant applied.")

        sq_configs = {
            f'SQ + Uniform {args.nominal_bits}b': {n_: args.nominal_bits for n_ in layer_names},
            f'SQ + Uniform {args.probe_bits}b':   {n_: args.probe_bits for n_ in layer_names},
        }
        # Also run SQ + ILP at 20% target
        sq_configs[f'SQ + ILP {TARGET*100:.0f}%'] = dict(zip(layer_names, asgn_ilp20))

        for cfg_name, assignment in sq_configs.items():
            ppl, sat, area = eval_config(model_sq, assignment, layer_names,
                                          calib_loader, eval_loader, ppa_ref,
                                          args.nominal_bits, args)
            bc = Counter(assignment.values())
            row = {
                'config': cfg_name,
                'ppl': round(ppl, 4),
                'sat': round(sat, 5),
                'adc_area_mm2': round(area.get('adc_area_mm2', 0), 2),
                'chip_area_mm2': round(area.get('chip_area_mm2', 0), 2),
                'adc_savings_pct': round(area.get('adc_savings_pct', 0), 2),
                'n_6b': bc.get(6, 0),
                'n_7b': bc.get(7, 0),
            }
            print(f"  {cfg_name:<30s}: PPL={ppl:.2f}  ADC={area.get('adc_area_mm2',0):.1f}mm²  "
                  f"sav={area.get('adc_savings_pct',0):.1f}%")
            phase1_results.append(row)

    # ── Save & print summary ─────────────────────────────────────────────────
    result_path = out_dir / 'stable_eval_results.csv'
    with open(result_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(phase1_results[0].keys()))
        w.writeheader()
        w.writerows(phase1_results)
    print(f"\n[Done] Saved to {result_path}")

    print("\n" + "=" * 78)
    print(f"{'Config':<30} {'PPL':>8} {'ADC(mm²)':>10} {'Savings%':>9}")
    print("-" * 78)
    for r in phase1_results:
        print(f"{r['config']:<30} {r['ppl']:>8.2f} {r['adc_area_mm2']:>10.1f} "
              f"{r['adc_savings_pct']:>8.1f}%")
    print("=" * 78)


if __name__ == '__main__':
    main()
