"""
zeroshot_eval.py — A2: Zero-shot task accuracy evaluation under CIM noise
Evaluates OPT-125M on PIQA, HellaSwag, WinoGrande, ARC-Easy under:
  1. Clean FP32 (baseline)
  2. CIM 7b uniform (ADC noise injected)
  3. CIM ILP 20% (mixed-precision)
  4. SQ + CIM 6b uniform

Uses lm-evaluation-harness with a custom patched model.
"""
import os, sys, json, argparse, copy
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from llm_inference import load_model, load_wikitext2, make_loader, CIMNoiseHook
from smooth_quant import CIMSmoothQuant, compute_perplexity
from sensitivity_analysis import (
    PerLayerCIMHook, get_linear_layers, classify_layer,
    ilp_allocation, load_ppa_sweep
)
from torch.utils.data import Subset


def get_snapshot_path(model_name, cache_dir='./model_cache'):
    """Find actual snapshot directory for cached model."""
    import re
    safe = model_name.replace('/', '--')
    snap_dir = Path(cache_dir) / f'models--{safe}' / 'snapshots'
    if snap_dir.exists():
        snaps = sorted(snap_dir.iterdir())
        if snaps:
            return str(snaps[-1])
    return model_name


def run_lmeval(model_path, tasks, output_path, extra_args=''):
    """Run lm-evaluation-harness and return results dict."""
    import subprocess, json
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'lm_eval',
        '--model', 'hf',
        '--model_args', f'pretrained={model_path},dtype=float32',
        '--tasks', ','.join(tasks),
        '--num_fewshot', '0',
        '--output_path', str(out),
        '--batch_size', '1',
    ]
    print(f"  Running: {' '.join(cmd[-8:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return None
    # Parse output JSON
    result_files = sorted(out.glob('results*.json'))
    if result_files:
        with open(result_files[-1]) as f:
            return json.load(f)
    return None


def extract_accuracy(results, tasks):
    """Extract per-task accuracy from lm-eval results dict."""
    if results is None:
        return {}
    out = {}
    for task in tasks:
        r = results.get('results', {}).get(task, {})
        # Try different metric names
        for metric in ['acc,none', 'acc_norm,none', 'acc']:
            if metric in r:
                out[task] = r[metric]
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='facebook/opt-125m')
    p.add_argument('--model_cache', default='./model_cache')
    p.add_argument('--output_dir', default='results/zeroshot/opt125m')
    p.add_argument('--ppa_csv', default='results/ppa/opt125m/ppa_sweep_opt125m.csv')
    p.add_argument('--device', default='cpu')
    p.add_argument('--num_calib_batches', type=int, default=4)
    p.add_argument('--tasks', default='piqa,hellaswag,winogrande,arc_easy')
    args = p.parse_args()

    tasks = args.tasks.split(',')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = get_snapshot_path(args.model, args.model_cache)
    print(f"Model path: {model_path}")

    # ── 1. Clean FP32 ──────────────────────────────────────────────────────────
    print("\n[1/4] Clean FP32 zero-shot evaluation...")
    clean_results = run_lmeval(model_path, tasks, out_dir / 'clean')
    clean_acc = extract_accuracy(clean_results, tasks)
    print(f"  Clean: {clean_acc}")

    # ── 2. Load model for CIM evaluation ──────────────────────────────────────
    print("\n[Loading model for CIM patching...]")
    model, tok = load_model(args.model, args.model_cache, args.device)
    calib_data = load_wikitext2(tok, 512, split='train')
    calib_loader = make_loader(Subset(calib_data, range(args.num_calib_batches)))

    # ── 3. CIM 7b uniform ──────────────────────────────────────────────────────
    # For lm-eval we need to save a temporary patched model
    # Since lm-eval requires a path, we use a wrapper approach:
    # eval with our own harness instead

    print("\n[2/4] CIM 7b uniform PPL and task accuracy (direct eval)...")
    hook7 = CIMNoiseHook(model, adc_bits=7, clip_pct=99.0, weight_bits=8, input_bits=8)
    hook7.calibrate(calib_loader)
    hook7.apply()

    eval_data = load_wikitext2(tok, 512, split='test')
    eval_loader_ppl = make_loader(Subset(eval_data, range(100)))
    ppl_cim7 = compute_perplexity(model, eval_loader_ppl)
    print(f"  CIM 7b PPL: {ppl_cim7:.2f}")

    # For task accuracy under CIM, use lm-eval with saved model
    # Save model temporarily
    tmp_path = out_dir / 'tmp_cim7b'
    tmp_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(tmp_path))
    tok.save_pretrained(str(tmp_path))
    hook7.remove()

    cim7_results = run_lmeval(str(tmp_path), tasks, out_dir / 'cim7b')
    cim7_acc = extract_accuracy(cim7_results, tasks)
    print(f"  CIM 7b: {cim7_acc}")

    # ── 4. SQ + CIM 6b uniform ─────────────────────────────────────────────────
    print("\n[3/4] SQ + CIM 6b uniform...")
    model_sq, tok = load_model(args.model, args.model_cache, args.device)
    smoother = CIMSmoothQuant(model_sq)
    smoother.compute_scales(calib_loader, input_bits=6, weight_bits=8, sat_lambda=0.5)
    smoother.apply_scales()

    hook6sq = CIMNoiseHook(model_sq, adc_bits=6, clip_pct=99.0, weight_bits=8, input_bits=8)
    hook6sq.calibrate(make_loader(Subset(calib_data, range(args.num_calib_batches))))
    hook6sq.apply()

    ppl_sq6 = compute_perplexity(model_sq, eval_loader_ppl)
    print(f"  SQ+6b PPL: {ppl_sq6:.2f}")

    tmp_sq6 = out_dir / 'tmp_sq6b'
    tmp_sq6.mkdir(parents=True, exist_ok=True)
    model_sq.save_pretrained(str(tmp_sq6))
    tok.save_pretrained(str(tmp_sq6))
    hook6sq.remove()

    sq6_results = run_lmeval(str(tmp_sq6), tasks, out_dir / 'sq6b')
    sq6_acc = extract_accuracy(sq6_results, tasks)
    print(f"  SQ+6b: {sq6_acc}")

    # ── 5. ILP 20% ──────────────────────────────────────────────────────────────
    print("\n[4/4] ILP 20% mixed-precision...")
    model_ilp, tok = load_model(args.model, args.model_cache, args.device)
    ppa = load_ppa_sweep(args.ppa_csv)

    # Load sensitivity from group_sensitivity.json
    sens_path = Path('results/sensitivity/opt125m/group_sensitivity.json')
    if sens_path.exists():
        import json as _json
        with open(sens_path) as f:
            group_sens = _json.load(f)
        # Build per-layer sensitivity dict
        layers = get_linear_layers(model_ilp)
        sens_dict = {}
        for name, _ in layers:
            ltype = classify_layer(name)
            if ltype in group_sens:
                # normalize by group size
                n = group_sens[ltype]['n_layers']
                sens_dict[name] = abs(group_sens[ltype]['delta_ppl']) / n
            else:
                sens_dict[name] = 1.0

        # Run ILP at 20% budget
        budget_frac = 0.795  # ~20.5% savings → use 79.5% of 7b budget
        ref_budget = sum(2**7 for _ in layers)
        target_budget = int(ref_budget * budget_frac)
        assignment = ilp_allocation(sens_dict, target_budget, ppa,
                                     nominal_bits=7, bit_choices=[5,6,7,8])

        hook_ilp = PerLayerCIMHook(model_ilp, default_bits=7, clip_pct=99.0,
                                    weight_bits=8, input_bits=8)
        hook_ilp.set_assignment(assignment)
        hook_ilp.calibrate(make_loader(Subset(calib_data, range(args.num_calib_batches))))
        hook_ilp.apply()

        ppl_ilp = compute_perplexity(model_ilp, eval_loader_ppl)
        print(f"  ILP 20% PPL: {ppl_ilp:.2f}")

        tmp_ilp = out_dir / 'tmp_ilp'
        tmp_ilp.mkdir(parents=True, exist_ok=True)
        model_ilp.save_pretrained(str(tmp_ilp))
        tok.save_pretrained(str(tmp_ilp))
        hook_ilp.remove()

        ilp_results = run_lmeval(str(tmp_ilp), tasks, out_dir / 'ilp20')
        ilp_acc = extract_accuracy(ilp_results, tasks)
        print(f"  ILP 20%: {ilp_acc}")
    else:
        print("  [SKIP] No group_sensitivity.json found")
        ppl_ilp = None
        ilp_acc = {}

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Task':<15} {'Clean':>8} {'CIM 7b':>8} {'SQ+6b':>8} {'ILP 20%':>8}")
    print("-"*50)
    for task in tasks:
        c  = f"{clean_acc.get(task, 0)*100:.1f}%"
        c7 = f"{cim7_acc.get(task, 0)*100:.1f}%"
        s6 = f"{sq6_acc.get(task, 0)*100:.1f}%"
        ilp= f"{ilp_acc.get(task, 0)*100:.1f}%"
        print(f"  {task:<13} {c:>8} {c7:>8} {s6:>8} {ilp:>8}")

    # Save summary
    summary = {
        'clean': {'ppl': None, 'tasks': clean_acc},
        'cim_7b': {'ppl': ppl_cim7, 'tasks': cim7_acc},
        'sq_6b': {'ppl': ppl_sq6, 'tasks': sq6_acc},
        'ilp_20': {'ppl': ppl_ilp, 'tasks': ilp_acc},
    }
    with open(out_dir / 'zeroshot_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir}/zeroshot_summary.json")


if __name__ == '__main__':
    main()
