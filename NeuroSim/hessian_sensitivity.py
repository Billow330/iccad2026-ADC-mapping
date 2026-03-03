"""
hessian_sensitivity.py — A4: Hessian-trace sensitivity as comparison baseline
Implements HAWQ-style per-layer Hessian trace estimation as a baseline
for comparing against our measured CIM ADC sensitivity.

The Hessian trace for layer i approximates: tr(H_i) ≈ (1/k) * sum_j ||g_j||^2
where g_j is the gradient of the loss w.r.t. layer i's output on sample j.
This is the "Hutchinson estimator" used in HAWQ.

We then compare: Hessian-trace-guided allocation vs our measured sensitivity.
"""
import os, sys, json, csv, argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import numpy as np

from llm_inference import load_model, load_wikitext2, make_loader
from outlier_analysis import _is_linear_like
from sensitivity_analysis import get_linear_layers, classify_layer
from torch.utils.data import Subset


def compute_hessian_trace(model, data_loader, n_batches=8, n_vectors=5, device='cpu'):
    """
    Compute per-layer Hessian trace using Hutchinson estimator.

    For each layer, we estimate: tr(H_i) ≈ E[v^T H_i v] where v ~ N(0,I)
    In practice: tr(H_i) ≈ (1/n) sum_j (∂L/∂W_i)^2 averaged over data.

    Simpler approximation (gradient-squared, used in HAWQ-V2):
      sensitivity_i ≈ mean(|∂L/∂output_i|^2) * var(output_i)

    This is computed via forward+backward pass.
    """
    model.eval()
    layers = get_linear_layers(model)

    # Register gradient hooks on layer outputs
    grad_norms = defaultdict(list)
    output_vars = defaultdict(list)

    handles = []

    def make_grad_hook(name):
        def hook(grad):
            # grad is d(Loss)/d(output_i)
            grad_norms[name].append(grad.detach().float().norm().item() ** 2)
            return grad
        return hook

    def make_fwd_hook(name):
        def hook(module, inp, out):
            output_vars[name].append(out.detach().float().var().item())
            # Register backward hook on output tensor
            out.register_hook(make_grad_hook(name))
        return hook

    for name, m in layers:
        h = m.register_forward_hook(make_fwd_hook(name))
        handles.append(h)

    # Forward+backward passes
    n_done = 0
    for batch in data_loader:
        if n_done >= n_batches:
            break
        ids = batch['input_ids'].to(device) if isinstance(batch, dict) else batch[0].to(device)

        # Compute loss (next-token prediction)
        with torch.enable_grad():
            outputs = model(ids, labels=ids)
            loss = outputs.loss
            loss.backward()

        model.zero_grad()
        n_done += 1
        if n_done % 2 == 0:
            print(f"  Hessian: {n_done}/{n_batches} batches", flush=True)

    for h in handles:
        h.remove()

    # Compute Hessian trace proxy: E[||grad||^2] * E[var(output)]
    result = {}
    for name, _ in layers:
        g_norms = grad_norms.get(name, [1.0])
        o_vars  = output_vars.get(name, [1.0])
        # Hessian trace proxy: mean grad norm squared
        hess_trace = float(np.mean(g_norms))
        out_var    = float(np.mean(o_vars))
        ltype = classify_layer(name)
        result[name] = {
            'layer': name,
            'layer_type': ltype,
            'hess_trace': hess_trace,
            'output_var': out_var,
            'hawq_sensitivity': hess_trace * out_var,  # HAWQ-style
        }

    return result


def greedy_allocation_by_sensitivity(sensitivities, target_budget, nominal_bits=7,
                                      bit_choices=None):
    """
    Standard greedy: reduce bits in ascending sensitivity order.
    Same logic as sensitivity_analysis._sensitivity_greedy but uses
    Hessian-trace sensitivity instead of measured ΔPPl.
    """
    if bit_choices is None:
        bit_choices = [5, 6, 7, 8]

    assignment = {name: nominal_bits for name in sensitivities}
    sorted_names = sorted(sensitivities, key=lambda n: sensitivities[n]['hawq_sensitivity'])

    # Current budget = sum of 2^bits for each layer
    current = sum(2**nominal_bits for _ in sensitivities)
    ref_budget = current

    changed = True
    while changed:
        changed = False
        for name in sorted_names:
            curr_b = assignment[name]
            lower = [b for b in bit_choices if b < curr_b]
            if lower:
                next_b = max(lower)
                # Would reducing this layer bring us closer to budget?
                new_budget = current - 2**curr_b + 2**next_b
                if new_budget / ref_budget <= target_budget:
                    assignment[name] = next_b
                    current = new_budget
                    changed = True

    actual_savings = (1 - current / ref_budget) * 100
    return assignment, actual_savings


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='facebook/opt-125m')
    p.add_argument('--model_cache', default='./model_cache')
    p.add_argument('--output_dir', default='results/hessian/opt125m')
    p.add_argument('--device', default='cpu')
    p.add_argument('--num_batches', type=int, default=8)
    p.add_argument('--sens_json', default='results/sensitivity/opt125m/group_sensitivity.json')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Hessian] Loading {args.model}...")
    model, tok = load_model(args.model, args.model_cache, args.device)
    data = load_wikitext2(tok, 512, split='train')
    loader = make_loader(Subset(data, range(args.num_batches)))

    print(f"[Hessian] Computing Hessian trace ({args.num_batches} batches)...")
    hess = compute_hessian_trace(model, loader, n_batches=args.num_batches)

    # Save raw Hessian traces
    out_csv = out_dir / 'hessian_trace.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['layer', 'layer_type', 'hess_trace',
                                                'output_var', 'hawq_sensitivity'])
        writer.writeheader()
        writer.writerows(hess.values())
    print(f"[Hessian] Saved per-layer traces to {out_csv}")

    # Group by layer type (for comparison with our group sensitivity)
    from collections import defaultdict
    group_hess = defaultdict(list)
    for name, v in hess.items():
        group_hess[v['layer_type']].append(v['hawq_sensitivity'])

    group_summary = {}
    for ltype, vals in group_hess.items():
        group_summary[ltype] = {
            'n_layers': len(vals),
            'mean_hawq': float(np.mean(vals)),
            'std_hawq': float(np.std(vals)),
        }

    with open(out_dir / 'hessian_group.json', 'w') as f:
        json.dump(group_summary, f, indent=2)

    # Print comparison table
    print("\n--- Hessian Sensitivity (HAWQ-style) vs Measured CIM Sensitivity ---")
    print(f"{'Layer type':<15} {'HAWQ sens (mean)':>18} {'n':>4}")
    for ltype in sorted(group_summary, key=lambda x: group_summary[x]['mean_hawq'], reverse=True):
        g = group_summary[ltype]
        print(f"  {ltype:<13} {g['mean_hawq']:>18.4f} {g['n_layers']:>4}")

    # Load our measured sensitivity for comparison
    if Path(args.sens_json).exists():
        with open(args.sens_json) as f:
            measured = json.load(f)

        print("\n--- Comparison: HAWQ ordering vs Measured CIM ordering ---")
        print(f"{'Layer type':<15} {'HAWQ rank':>10} {'CIM rank':>10} {'HAWQ sens':>12} {'CIM ΔPPL/layer':>16}")

        hawq_order = sorted(group_summary.keys(),
                            key=lambda x: group_summary[x]['mean_hawq'], reverse=True)
        cim_order  = sorted(
            [k for k in measured if k in group_hess],
            key=lambda x: abs(measured[x]['delta_per_layer']), reverse=True
        )

        for ltype in hawq_order:
            if ltype not in measured:
                continue
            hr = hawq_order.index(ltype) + 1
            cr = cim_order.index(ltype) + 1 if ltype in cim_order else '-'
            hs = group_summary[ltype]['mean_hawq']
            cs = abs(measured[ltype]['delta_per_layer'])
            agree = "✓" if hr == cr else "✗"
            print(f"  {ltype:<13} {hr:>10} {cr:>10} {hs:>12.4f} {cs:>16.4f} {agree}")

        # Check correlation
        common = [k for k in hawq_order if k in measured and k in group_hess]
        hawq_ranks = [hawq_order.index(k)+1 for k in common]
        cim_ranks  = [cim_order.index(k)+1 if k in cim_order else 0 for k in common]

        if len(common) >= 3:
            from scipy.stats import spearmanr
            rho, p = spearmanr(hawq_ranks, cim_ranks)
            print(f"\n  Spearman rank correlation (HAWQ vs CIM): ρ={rho:.3f}, p={p:.3f}")
            if rho < 0:
                print(f"  → NEGATIVE correlation: HAWQ ordering INVERSELY predicts CIM sensitivity!")
            else:
                print(f"  → Positive correlation: orderings partially agree")

    print(f"\n[Done] Results saved to {out_dir}/")


if __name__ == '__main__':
    main()
