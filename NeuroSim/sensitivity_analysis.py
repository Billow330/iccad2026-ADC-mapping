"""
sensitivity_analysis.py — Per-Layer ADC Sensitivity Measurement
===============================================================
ICCAD 2026 paper key contribution: replace proxy-based sensitivity
(saturation rate) with MEASURED sensitivity (∂PPL/∂ADC_bits per layer).

This script:
  1. Measures baseline PPL at uniform-7b ADC
  2. For each layer (or layer GROUP for efficiency), drops from 7b to 6b
     and measures ΔPPl → this is the per-layer sensitivity
  3. Runs greedy + ILP allocation using measured sensitivities
  4. Evaluates all configurations (uniform-6b, uniform-7b, mixed-greedy,
     mixed-ILP) with real PPL measurements
  5. Saves a comprehensive CSV for paper tables/figures

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    python3 sensitivity_analysis.py --model facebook/opt-125m \\
        --num_calib_batches 4 --num_eval_batches 10 \\
        --output_dir results/sensitivity/opt125m

Group mode (faster, for 73-layer OPT):
    python3 sensitivity_analysis.py --model facebook/opt-125m \\
        --group_by_type    # measure 5 layer TYPES instead of 73 individual layers
"""

import os, sys, json, csv, argparse, copy, time
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from llm_inference import (
    load_model, load_wikitext2, make_loader,
    CIMNoiseHook,
)
from smooth_quant import compute_perplexity, CIMSmoothQuant
from outlier_analysis import _is_linear_like


# ─────────────────────────────────────────────────────────────────────────────
# Layer classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_layer(name):
    """Return layer type string for OPT/GPT models."""
    n = name.lower()
    if 'lm_head' in n:
        return 'lm_head'
    if 'q_proj' in n or 'k_proj' in n or 'v_proj' in n:
        return 'attn_qkv'
    if 'out_proj' in n:
        return 'attn_out'
    if 'fc1' in n:
        return 'ffn_up'
    if 'fc2' in n:
        return 'ffn_down'
    return 'other'


def get_linear_layers(model):
    """Return ordered list of (name, module) for all linear-like layers."""
    layers = []
    for name, m in model.named_modules():
        if _is_linear_like(m):
            layers.append((name, m))
    return layers


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer configurable CIM noise hook
# ─────────────────────────────────────────────────────────────────────────────

class PerLayerCIMHook:
    """
    CIM ADC noise with independently configurable bits per layer.

    bit_assignment: dict {layer_name: adc_bits}
    default_bits: fallback for layers not in bit_assignment
    """

    def __init__(self, model, bit_assignment: dict, default_bits: int = 7,
                 clip_percentile: float = 99.0):
        self.model = model
        self.bit_assignment = bit_assignment
        self.default_bits = default_bits
        self.clip_percentile = clip_percentile
        self._clip = {}   # per-layer calibrated scale
        self._hooks = []
        self.sat_counts = {}
        self.total_counts = {}

    def calibrate(self, data_loader, device='cpu', num_batches=4):
        """Calibrate per-layer ADC full-scale using the layer's assigned bits."""
        self.model.eval()
        raw = {}

        def make_cal_hook(name):
            def h(mod, inp, out):
                v = out.detach().float().abs().flatten()
                if v.numel() > 8192:
                    idx = torch.randperm(v.numel())[:8192]
                    v = v[idx]
                raw.setdefault(name, []).append(v.cpu())
            return h

        tmp = []
        for name, m in self.model.named_modules():
            if _is_linear_like(m):
                tmp.append(m.register_forward_hook(make_cal_hook(name)))

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                ids = batch['input_ids'].to(device)
                self.model(ids)

        for h in tmp:
            h.remove()

        for name, vals in raw.items():
            bits = self.bit_assignment.get(name, self.default_bits)
            n_adc = 2 ** bits - 1
            all_vals = torch.cat(vals)
            if self.clip_percentile >= 100.0:
                p = all_vals.max().item()
            else:
                p = torch.quantile(all_vals, self.clip_percentile / 100.0).item()
            self._clip[name] = max(p / n_adc, 1e-8)

    def _make_hook(self, name):
        bits = self.bit_assignment.get(name, self.default_bits)
        n_adc = 2 ** bits - 1

        def hook(mod, inp, out):
            y = out.detach().float()
            act_scale = self._clip.get(name, y.abs().max().item() / n_adc + 1e-8)
            adc_max = act_scale * n_adc
            sat_mask = y.abs() > adc_max
            self.sat_counts[name] = self.sat_counts.get(name, 0) + sat_mask.sum().item()
            self.total_counts[name] = self.total_counts.get(name, 0) + y.numel()
            y_q = (y / act_scale).round().clamp(-n_adc, n_adc) * act_scale
            return y_q.to(out.dtype)

        return hook

    def install(self):
        for name, m in self.model.named_modules():
            if _is_linear_like(m):
                h = m.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def saturation_rates(self):
        return {n: self.sat_counts.get(n, 0) / max(self.total_counts.get(n, 1), 1)
                for n in self.total_counts}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: eval PPL with a given per-layer bit assignment
# ─────────────────────────────────────────────────────────────────────────────

def eval_with_assignment(model, bit_assignment, calib_loader, eval_loader,
                         default_bits=7, device='cpu',
                         num_calib=4, num_eval=10, clip_pct=99.0):
    """Run CIM inference with given per-layer bit assignment, return (ppl, mean_sat)."""
    hook = PerLayerCIMHook(model, bit_assignment, default_bits, clip_pct)
    hook.calibrate(calib_loader, device=device, num_batches=num_calib)
    hook.install()
    ppl = compute_perplexity(model, eval_loader, device=device, max_batches=num_eval)
    sat = np.mean(list(hook.saturation_rates().values())) if hook.sat_counts else 0.0
    hook.remove()
    return ppl, sat


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer sensitivity measurement (individual layer 7b→6b)
# ─────────────────────────────────────────────────────────────────────────────

def measure_per_layer_sensitivity(model, layer_names, calib_loader, eval_loader,
                                  baseline_ppl, nominal_bits=7, probe_bits=6,
                                  device='cpu', num_calib=4, num_eval=10,
                                  clip_pct=99.0):
    """
    For each layer: drop it from nominal_bits to probe_bits, measure ΔPPL.

    Returns list of {layer, layer_type, delta_ppl, sensitivity_ratio} dicts.
    ΔPPL > 0 means accuracy degrades when reducing this layer's bits.
    sensitivity_ratio = ΔPPL / |baseline_ppl| (normalized)
    """
    results = []
    n = len(layer_names)
    t0 = time.time()

    for i, name in enumerate(layer_names):
        # Assignment: all layers at nominal_bits, except this one at probe_bits
        assignment = {n_: nominal_bits for n_ in layer_names}
        assignment[name] = probe_bits

        ppl, sat = eval_with_assignment(
            model, assignment, calib_loader, eval_loader,
            default_bits=nominal_bits, device=device,
            num_calib=num_calib, num_eval=num_eval, clip_pct=clip_pct
        )
        delta_ppl = ppl - baseline_ppl
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n - i - 1)
        print(f"  [{i+1:3d}/{n}] {name:<60s} "
              f"PPL={ppl:.2f} ΔPPL={delta_ppl:+.3f} "
              f"sat={sat*100:.1f}% ETA={eta/60:.1f}min")

        results.append({
            'layer':            name,
            'layer_type':       classify_layer(name),
            'layer_idx':        i,
            'nominal_bits':     nominal_bits,
            'probe_bits':       probe_bits,
            'ppl':              ppl,
            'delta_ppl':        delta_ppl,
            'sat_rate':         sat,
            'sensitivity':      max(delta_ppl, 0.0),  # negative = PPL improved (ignore)
        })

    return results


def measure_group_sensitivity(model, layer_names, calib_loader, eval_loader,
                              baseline_ppl, nominal_bits=7, probe_bits=6,
                              device='cpu', num_calib=4, num_eval=10,
                              clip_pct=99.0):
    """
    Group layers by type and measure sensitivity per GROUP.
    Much faster (5 measurements for 73 layers).
    Each layer in a group gets the group's ΔPPL as its sensitivity.
    """
    # Group layers by type
    groups = defaultdict(list)
    for name in layer_names:
        ltype = classify_layer(name)
        groups[ltype].append(name)

    print(f"\n[Group Sensitivity] {len(groups)} groups: {list(groups.keys())}")
    group_results = {}

    for ltype, names in groups.items():
        # Drop all layers of this type from nominal to probe bits
        assignment = {n_: nominal_bits for n_ in layer_names}
        for n_ in names:
            assignment[n_] = probe_bits

        ppl, sat = eval_with_assignment(
            model, assignment, calib_loader, eval_loader,
            default_bits=nominal_bits, device=device,
            num_calib=num_calib, num_eval=num_eval, clip_pct=clip_pct
        )
        delta_ppl = (ppl - baseline_ppl) if baseline_ppl is not None else 0.0
        delta_per_layer = delta_ppl / len(names) if names else 0.0
        print(f"  Group {ltype:<15s} ({len(names):3d} layers): "
              f"PPL={ppl:.2f} ΔPPL={delta_ppl:+.3f} "
              f"per-layer={delta_per_layer:+.4f}")

        group_results[ltype] = {
            'ltype': ltype,
            'n_layers': len(names),
            'ppl': ppl,
            'delta_ppl': delta_ppl,
            'delta_per_layer': delta_per_layer,
            'sat_rate': sat,
        }

    # Assign per-layer sensitivity from group
    per_layer = []
    for i, name in enumerate(layer_names):
        ltype = classify_layer(name)
        g = group_results.get(ltype, {})
        per_layer.append({
            'layer':        name,
            'layer_type':   ltype,
            'layer_idx':    i,
            'nominal_bits': nominal_bits,
            'probe_bits':   probe_bits,
            'ppl':          g.get('ppl', 0.0),
            'delta_ppl':    g.get('delta_per_layer', 0.0),
            'n_layers':     g.get('n_layers', 1),
            'sat_rate':     g.get('sat_rate', 0.0),
            'sensitivity':  max(g.get('delta_per_layer', 0.0), 0.0),
        })

    return per_layer, group_results


# ─────────────────────────────────────────────────────────────────────────────
# ILP-based optimal allocation
# ─────────────────────────────────────────────────────────────────────────────

def ilp_allocation(sensitivity_data, ppa_sweep, nominal_bits=7,
                   bit_choices=(4, 5, 6, 7, 8),
                   target_area_savings=0.20):
    """
    Integer Linear Program for optimal mixed-precision ADC allocation.

    Minimize: sum_i sensitivity_i * (nominal_bits - b_i)  [accuracy loss]
    Subject to:
        sum_i 2^b_i <= (1 - target_savings) * sum_i 2^nominal_bits  [area budget]
        b_i in bit_choices for all i
        b_i >= nominal_bits for layers with sensitivity > high_thresh  [protect]

    Uses scipy.optimize.milp (scipy >= 1.7) or brute-force for small n.
    Falls back to an enhanced greedy if scipy.milp unavailable.
    """
    n = len(sensitivity_data)
    nominal_area = n * (2 ** nominal_bits)
    budget = nominal_area * (1.0 - target_area_savings)

    # Sort by sensitivity descending (most sensitive = protect first)
    sens = np.array([r['sensitivity'] for r in sensitivity_data])
    layer_names = [r['layer'] for r in sensitivity_data]

    # Try scipy MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Variables: for each layer i, binary vars x_{i,b} for b in bit_choices
        # x_{i,b} = 1 means layer i uses b bits
        # Objective: minimize sum_i sum_b sensitivity_i * (nominal_bits - b) * x_{i,b}
        B = len(bit_choices)
        N_vars = n * B

        # Cost vector
        c = np.zeros(N_vars)
        for i in range(n):
            for j, b in enumerate(bit_choices):
                # We want to maximize accuracy, so minimize accuracy loss
                # Accuracy loss ∝ max(0, nominal_bits - b) * sensitivity
                c[i * B + j] = max(0.0, nominal_bits - b) * sens[i]

        # Constraint 1: Each layer uses exactly 1 bit choice
        # sum_b x_{i,b} = 1  for each i
        A_eq = np.zeros((n, N_vars))
        for i in range(n):
            for j in range(B):
                A_eq[i, i * B + j] = 1.0

        # Constraint 2: Area budget
        # sum_i sum_b 2^b * x_{i,b} <= budget
        A_area = np.zeros((1, N_vars))
        for i in range(n):
            for j, b in enumerate(bit_choices):
                A_area[0, i * B + j] = 2 ** b

        # Combine constraints
        A_ub = np.vstack([A_eq, -A_eq, A_area])  # eq + area
        b_ub_lo = np.concatenate([np.ones(n), -np.ones(n), [-np.inf]])
        b_ub_hi = np.concatenate([np.ones(n), -np.ones(n), [budget]])

        constraints = LinearConstraint(A_ub, b_ub_lo, b_ub_hi)
        bounds = Bounds(lb=0.0, ub=1.0)
        integrality = np.ones(N_vars)  # all binary

        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

        if res.success:
            assignments = []
            for i in range(n):
                best_j = max(range(B), key=lambda j: res.x[i * B + j])
                assignments.append(bit_choices[best_j])
            print(f"[ILP] Optimal solution found (status={res.status})")
            return assignments
        else:
            print(f"[ILP] MILP failed (status={res.status}), using enhanced greedy")
    except (ImportError, Exception) as e:
        print(f"[ILP] scipy.milp unavailable ({e}), using enhanced greedy")

    # Fallback: sensitivity-weighted greedy (better than simple saturation-rate greedy)
    return _sensitivity_greedy(sensitivity_data, nominal_bits, bit_choices,
                               target_area_savings)


def _sensitivity_greedy(sensitivity_data, nominal_bits=7,
                         bit_choices=(4, 5, 6, 7, 8),
                         target_area_savings=0.20):
    """
    Sensitivity-guided greedy allocation.
    Reduces bits for layers with lowest ∂PPL/∂bit first.
    Multi-step: can reduce by more than 1 bit.
    """
    n = len(sensitivity_data)
    assignments = [nominal_bits] * n
    sens = [r['sensitivity'] for r in sensitivity_data]

    ref_area = n * (2 ** nominal_bits)
    budget = ref_area * (1.0 - target_area_savings)

    # Sort by sensitivity ascending (reduce least-sensitive first)
    sorted_idx = sorted(range(n), key=lambda i: sens[i])

    current_area = ref_area
    # Cycle through sorted layers, reducing 1 bit at a time until budget met
    rounds = 0
    while current_area > budget and rounds < 20:
        improved = False
        for idx in sorted_idx:
            if current_area <= budget:
                break
            cur_bits = assignments[idx]
            # Find next lower bit choice
            lower = [b for b in sorted(bit_choices) if b < cur_bits]
            if not lower:
                continue
            new_bits = max(lower)
            delta = (2 ** new_bits) - (2 ** cur_bits)
            current_area += delta
            assignments[idx] = new_bits
            improved = True
        if not improved:
            break
        rounds += 1

    return assignments


# ─────────────────────────────────────────────────────────────────────────────
# ADC area model (matches NeuroSIM MLSA)
# ─────────────────────────────────────────────────────────────────────────────

def adc_area_ratio(bits, ref_bits=7):
    return (2 ** bits) / (2 ** ref_bits)


def compute_area_from_assignment(assignments, ref_ppa, nominal_bits=7):
    """Compute chip/ADC area from per-layer assignment using NeuroSIM ref data."""
    ref_chip = ref_ppa['chip_area_um2']
    ref_adc  = ref_ppa['adc_area_um2']
    ref_nonadc = ref_chip - ref_adc
    n = len(assignments)
    per_layer_adc = ref_adc / n

    total_adc = sum(per_layer_adc * adc_area_ratio(b, nominal_bits)
                    for b in assignments)
    total_chip = ref_nonadc + total_adc
    savings = (1.0 - total_adc / ref_adc) * 100.0
    return {
        'chip_area_mm2': total_chip / 1e6,
        'adc_area_mm2':  total_adc / 1e6,
        'adc_area_pct':  100.0 * total_adc / total_chip,
        'adc_savings_pct': savings,
    }


def load_ppa_sweep(csv_path):
    rows = {}
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            b = int(r['adc_bits'])
            rows[b] = {k: float(v) for k, v in r.items() if k != 'model' and k != 'adc_bits'}
            rows[b]['adc_bits'] = b
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SmoothQuant integration
# ─────────────────────────────────────────────────────────────────────────────

def apply_smoothquant(model, calib_loader, args, verbose=False):
    """Apply CIM-aware SmoothQuant to a model copy. Returns smoothed model."""
    model_sq = copy.deepcopy(model)
    sq = CIMSmoothQuant(
        weight_bits=args.weight_bits, input_bits=args.input_bits,
        off_state=6e-3, on_state=6e-3 * 17,
        vdd=1.0, parallel_read=128,
        adc_bits=args.nominal_bits,
        adc_clip_pct=args.clip_pct,
        sat_lambda=0.5,
        verbose=verbose,
    )
    sq.fit(model_sq, calib_loader,
           num_batches=args.num_calib_batches,
           device=args.device, task='lm')
    return model_sq


# ─────────────────────────────────────────────────────────────────────────────
# Pareto frontier computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pareto_points(sensitivity_data, eval_fn, ppa_ref,
                          nominal_bits=7, bit_choices=(4, 5, 6, 7, 8),
                          n_budgets=10):
    """
    Sweep different area budgets (5%-50% savings) and for each budget:
    - Run ILP/greedy allocation
    - Measure actual PPL
    - Record (ADC_area_mm2, PPL) point
    Returns list of (area_saving%, adc_area_mm2, ppl) tuples.
    """
    points = []
    savings_targets = np.linspace(0.05, 0.50, n_budgets)

    for target in savings_targets:
        asgn = ilp_allocation(sensitivity_data, ppa_ref,
                              nominal_bits=nominal_bits,
                              bit_choices=bit_choices,
                              target_area_savings=target)
        ppl, sat = eval_fn(asgn)
        area = compute_area_from_assignment(asgn, ppa_ref[nominal_bits], nominal_bits)
        bits_dist = Counter(asgn)
        print(f"  budget={1-target:.0%} area_sav={area['adc_savings_pct']:.1f}% "
              f"PPL={ppl:.2f} dist={dict(sorted(bits_dist.items()))}")
        points.append({
            'target_savings': target * 100,
            'actual_savings': area['adc_savings_pct'],
            'adc_area_mm2': area['adc_area_mm2'],
            'chip_area_mm2': area['chip_area_mm2'],
            'ppl': ppl,
            'sat': sat,
            'bits_dist': dict(bits_dist),
        })

    return points


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='facebook/opt-125m')
    p.add_argument('--model_cache', default='./model_cache')
    p.add_argument('--output_dir', default='results/sensitivity/opt125m')
    p.add_argument('--ppa_csv', default='results/ppa/opt125m/ppa_sweep_opt125m.csv')
    p.add_argument('--device', default='cpu')
    p.add_argument('--num_calib_batches', type=int, default=4)
    p.add_argument('--num_eval_batches', type=int, default=10)
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--nominal_bits', type=int, default=7)
    p.add_argument('--probe_bits', type=int, default=6)
    p.add_argument('--clip_pct', type=float, default=99.0)
    p.add_argument('--weight_bits', type=int, default=8)
    p.add_argument('--input_bits', type=int, default=8)
    # Mode flags
    p.add_argument('--group_by_type', action='store_true',
                   help='Measure sensitivity at group level (faster, 5 measurements)')
    p.add_argument('--skip_sensitivity', action='store_true',
                   help='Skip sensitivity measurement, load from saved CSV')
    p.add_argument('--with_smoothquant', action='store_true',
                   help='Also run SmoothQuant + mixed-precision experiments')
    p.add_argument('--pareto', action='store_true',
                   help='Compute full Pareto frontier (many budget points)')
    p.add_argument('--n_pareto', type=int, default=8,
                   help='Number of Pareto budget points')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model, args.model_cache, args.device)

    # ── Load data ───────────────────────────────────────────────────────────
    data = load_wikitext2(tokenizer, seq_len=args.seq_len)
    calib_data = data[:args.num_calib_batches + 16]
    eval_data  = data[64:64 + args.num_eval_batches + 32]
    calib_loader = make_loader(calib_data)
    eval_loader  = make_loader(eval_data)

    # ── Load PPA data ────────────────────────────────────────────────────────
    ppa_sweep = {}
    if Path(args.ppa_csv).exists():
        ppa_sweep = load_ppa_sweep(args.ppa_csv)
        print(f"[PPA] Loaded data for bits: {sorted(ppa_sweep.keys())}")
    else:
        print(f"[WARNING] PPA CSV not found: {args.ppa_csv}")

    # ── Get layer names ──────────────────────────────────────────────────────
    layer_names = [name for name, _ in get_linear_layers(model)]
    print(f"\n[Setup] {len(layer_names)} linear layers found")
    type_counts = Counter(classify_layer(n) for n in layer_names)
    print(f"[Setup] Layer types: {dict(type_counts)}")

    # ── Measure baseline PPL (uniform nominal_bits) ─────────────────────────
    sens_csv = out_dir / f'sensitivity_{args.nominal_bits}b_to_{args.probe_bits}b.csv'

    if args.skip_sensitivity and sens_csv.exists():
        print(f"\n[Sensitivity] Loading from {sens_csv}")
        sensitivity_data = []
        with open(sens_csv, newline='') as f:
            for row in csv.DictReader(f):
                sensitivity_data.append({
                    'layer':       row['layer'],
                    'layer_type':  row['layer_type'],
                    'layer_idx':   int(row['layer_idx']),
                    'nominal_bits': int(row['nominal_bits']),
                    'probe_bits':  int(row['probe_bits']),
                    'delta_ppl':   float(row['delta_ppl']),
                    'sat_rate':    float(row['sat_rate']),
                    'sensitivity': float(row['sensitivity']),
                })
        baseline_ppl = float(sensitivity_data[0].get('baseline_ppl', 0))
        # Re-read baseline if not stored
        if baseline_ppl == 0:
            print("[Baseline] Measuring baseline PPL...")
            uniform_7b = {n: args.nominal_bits for n in layer_names}
            baseline_ppl, _ = eval_with_assignment(
                model, uniform_7b, calib_loader, eval_loader,
                default_bits=args.nominal_bits, device=args.device,
                num_calib=args.num_calib_batches,
                num_eval=args.num_eval_batches, clip_pct=args.clip_pct)
            print(f"[Baseline] Uniform {args.nominal_bits}b PPL = {baseline_ppl:.4f}")
    else:
        print(f"\n[Baseline] Measuring uniform {args.nominal_bits}b PPL...")
        uniform_nb = {n: args.nominal_bits for n in layer_names}
        baseline_ppl, baseline_sat = eval_with_assignment(
            model, uniform_nb, calib_loader, eval_loader,
            default_bits=args.nominal_bits, device=args.device,
            num_calib=args.num_calib_batches,
            num_eval=args.num_eval_batches, clip_pct=args.clip_pct)
        print(f"[Baseline] Uniform {args.nominal_bits}b: PPL={baseline_ppl:.4f}, "
              f"mean_sat={baseline_sat*100:.2f}%")

        # Also measure clean PPL (no CIM noise)
        ppl_clean = compute_perplexity(model, eval_loader, device=args.device,
                                       max_batches=args.num_eval_batches)
        print(f"[Baseline] Clean FP32 PPL = {ppl_clean:.4f}")

        # ── Per-layer sensitivity measurement ──────────────────────────────
        print(f"\n[Sensitivity] Measuring {args.nominal_bits}b→{args.probe_bits}b "
              f"sensitivity ({'group' if args.group_by_type else 'per-layer'} mode)...")

        if args.group_by_type:
            sensitivity_data, group_results = measure_group_sensitivity(
                model, layer_names, calib_loader, eval_loader,
                baseline_ppl,
                nominal_bits=args.nominal_bits, probe_bits=args.probe_bits,
                device=args.device, num_calib=args.num_calib_batches,
                num_eval=args.num_eval_batches, clip_pct=args.clip_pct)
            # Save group results
            gpath = out_dir / 'group_sensitivity.json'
            with open(gpath, 'w') as f:
                json.dump(group_results, f, indent=2)
            print(f"[Sensitivity] Group results saved to {gpath}")
        else:
            sensitivity_data = measure_per_layer_sensitivity(
                model, layer_names, calib_loader, eval_loader,
                baseline_ppl,
                nominal_bits=args.nominal_bits, probe_bits=args.probe_bits,
                device=args.device, num_calib=args.num_calib_batches,
                num_eval=args.num_eval_batches, clip_pct=args.clip_pct)

        # Add baseline_ppl to each row and save
        for r in sensitivity_data:
            r['baseline_ppl'] = baseline_ppl
            r['ppl_clean'] = ppl_clean

        with open(sens_csv, 'w', newline='') as f:
            if sensitivity_data:
                w = csv.DictWriter(f, fieldnames=list(sensitivity_data[0].keys()))
                w.writeheader()
                w.writerows(sensitivity_data)
        print(f"[Sensitivity] Saved to {sens_csv}")

        # Print summary by layer type
        print(f"\n[Sensitivity Summary] ΔPPL when dropping from "
              f"{args.nominal_bits}b to {args.probe_bits}b:")
        by_type = defaultdict(list)
        for r in sensitivity_data:
            by_type[r['layer_type']].append(r['delta_ppl'])
        for ltype, deltas in sorted(by_type.items()):
            print(f"  {ltype:<15s}: mean_ΔPPL={np.mean(deltas):+.4f}  "
                  f"max_ΔPPL={np.max(deltas):+.4f}  n={len(deltas)}")

    # ── ILP Allocation ──────────────────────────────────────────────────────
    print(f"\n[Allocation] Running sensitivity-guided allocations...")
    bit_choices = (4, 5, 6, 7, 8)
    TARGET = 0.20  # 20% ADC area savings target

    # Greedy (saturation-proxy, for comparison with old method)
    # Use sensitivity as the greedy criterion (lower sensitivity = reduce first)
    sens_sorted = sorted(range(len(sensitivity_data)),
                         key=lambda i: sensitivity_data[i]['sensitivity'])
    asgn_greedy = [args.nominal_bits] * len(sensitivity_data)
    ref_area = len(sensitivity_data) * (2 ** args.nominal_bits)
    budget = ref_area * (1.0 - TARGET)
    curr_area = ref_area
    for idx in sens_sorted:
        if curr_area <= budget:
            break
        if asgn_greedy[idx] > min(bit_choices):
            lower = [b for b in sorted(bit_choices) if b < asgn_greedy[idx]]
            if lower:
                new_b = max(lower)
                curr_area += (2 ** new_b) - (2 ** asgn_greedy[idx])
                asgn_greedy[idx] = new_b
    greedy_savings = (1.0 - curr_area / ref_area) * 100
    print(f"[Greedy] Savings: {greedy_savings:.1f}% | "
          f"Dist: {dict(Counter(asgn_greedy))}")

    # ILP
    asgn_ilp = ilp_allocation(sensitivity_data, ppa_sweep,
                               nominal_bits=args.nominal_bits,
                               bit_choices=bit_choices,
                               target_area_savings=TARGET)
    ilp_area = sum(2 ** b for b in asgn_ilp)
    ilp_savings = (1.0 - ilp_area / ref_area) * 100
    print(f"[ILP]   Savings: {ilp_savings:.1f}% | "
          f"Dist: {dict(Counter(asgn_ilp))}")

    # ── Evaluate all configurations (REAL PPL measurements) ─────────────────
    print(f"\n[Evaluation] Measuring REAL PPL for all configurations...")
    configs = {
        f'Uniform {args.nominal_bits}b (baseline)': {n: args.nominal_bits for n in layer_names},
        f'Uniform {args.probe_bits}b':               {n: args.probe_bits for n in layer_names},
        f'Mixed-Greedy ({TARGET*100:.0f}% target)': dict(zip(layer_names, asgn_greedy)),
        f'Mixed-ILP ({TARGET*100:.0f}% target)':    dict(zip(layer_names, asgn_ilp)),
    }

    eval_results = []
    for config_name, assignment in configs.items():
        bits_list = [assignment.get(n, args.nominal_bits) for n in layer_names]
        ppl, sat = eval_with_assignment(
            model, assignment, calib_loader, eval_loader,
            default_bits=args.nominal_bits, device=args.device,
            num_calib=args.num_calib_batches,
            num_eval=args.num_eval_batches, clip_pct=args.clip_pct)

        area_info = {}
        if args.nominal_bits in ppa_sweep:
            area_info = compute_area_from_assignment(
                bits_list, ppa_sweep[args.nominal_bits], args.nominal_bits)

        bc = Counter(bits_list)
        result = {
            'config':           config_name,
            'ppl':              ppl,
            'sat_rate':         sat,
            'adc_area_mm2':     area_info.get('adc_area_mm2', 0),
            'chip_area_mm2':    area_info.get('chip_area_mm2', 0),
            'adc_area_pct':     area_info.get('adc_area_pct', 0),
            'adc_savings_pct':  area_info.get('adc_savings_pct', 0),
        }
        for b in bit_choices:
            result[f'n_{b}b_layers'] = bc.get(b, 0)

        print(f"  {config_name:<40s}: PPL={ppl:.2f} "
              f"ADC_area={area_info.get('adc_area_mm2', 0):.1f}mm² "
              f"savings={area_info.get('adc_savings_pct', 0):.1f}%")
        eval_results.append(result)

    # ── SmoothQuant + Mixed-Precision ────────────────────────────────────────
    if args.with_smoothquant:
        print(f"\n[SmoothQuant] Applying CIM-SQ then re-measuring sensitivity...")
        model_sq = apply_smoothquant(model, calib_loader, args, verbose=False)

        # Measure PPL of model_sq under uniform assignments
        for b in [args.probe_bits, args.nominal_bits]:
            assign = {n: b for n in layer_names}
            ppl_sq, sat_sq = eval_with_assignment(
                model_sq, assign, calib_loader, eval_loader,
                default_bits=b, device=args.device,
                num_calib=args.num_calib_batches,
                num_eval=args.num_eval_batches, clip_pct=args.clip_pct)
            area_info = {}
            if b in ppa_sweep:
                area_info = compute_area_from_assignment(
                    [b] * len(layer_names), ppa_sweep[b], args.nominal_bits)
            result = {
                'config':          f'SQ + Uniform {b}b',
                'ppl':             ppl_sq,
                'sat_rate':        sat_sq,
                'adc_area_mm2':    area_info.get('adc_area_mm2', 0),
                'chip_area_mm2':   area_info.get('chip_area_mm2', 0),
                'adc_area_pct':    area_info.get('adc_area_pct', 0),
                'adc_savings_pct': area_info.get('adc_savings_pct', 0),
            }
            for bc_ in bit_choices:
                result[f'n_{bc_}b_layers'] = len(layer_names) if bc_ == b else 0
            print(f"  SQ + Uniform {b}b: PPL={ppl_sq:.2f} "
                  f"savings={area_info.get('adc_savings_pct', 0):.1f}%")
            eval_results.append(result)

        # SQ + Group sensitivity + ILP
        print("[SmoothQuant] Measuring group sensitivity after SQ...")
        sq_sens, _ = measure_group_sensitivity(
            model_sq, layer_names, calib_loader, eval_loader,
            baseline_ppl=None,  # will use uniform-7b SQ PPL
            nominal_bits=args.nominal_bits, probe_bits=args.probe_bits,
            device=args.device, num_calib=args.num_calib_batches,
            num_eval=args.num_eval_batches, clip_pct=args.clip_pct)

        # Compute SQ baseline ppl
        sq_base = eval_with_assignment(
            model_sq, {n: args.nominal_bits for n in layer_names},
            calib_loader, eval_loader, default_bits=args.nominal_bits,
            device=args.device, num_calib=args.num_calib_batches,
            num_eval=args.num_eval_batches, clip_pct=args.clip_pct)[0]
        print(f"[SmoothQuant] Baseline PPL (uniform-{args.nominal_bits}b): {sq_base:.4f}")

        # Fix sensitivity delta_ppl: measure_group_sensitivity was called with None baseline
        # Recompute delta_ppl correctly using sq_base
        for r in sq_sens:
            r['delta_ppl'] = r['ppl'] - sq_base if 'ppl' in r else 0.0
            r['delta_per_layer'] = r['delta_ppl'] / max(r.get('n_layers', 1), 1)
            r['sensitivity'] = max(r['delta_per_layer'], 0.0)
            r['baseline_ppl'] = sq_base

        sq_sens_data = sq_sens

        asgn_sq_ilp = ilp_allocation(
            sq_sens_data, ppa_sweep,
            nominal_bits=args.nominal_bits, bit_choices=bit_choices,
            target_area_savings=TARGET)
        assign_sq_ilp = dict(zip(layer_names, asgn_sq_ilp))

        ppl_sq_ilp, sat_sq_ilp = eval_with_assignment(
            model_sq, assign_sq_ilp, calib_loader, eval_loader,
            default_bits=args.nominal_bits, device=args.device,
            num_calib=args.num_calib_batches,
            num_eval=args.num_eval_batches, clip_pct=args.clip_pct)
        area_sq_ilp = compute_area_from_assignment(
            asgn_sq_ilp, ppa_sweep.get(args.nominal_bits, {}),
            args.nominal_bits) if ppa_sweep else {}
        print(f"  SQ + Mixed-ILP: PPL={ppl_sq_ilp:.2f} "
              f"savings={area_sq_ilp.get('adc_savings_pct', 0):.1f}%")

        result = {
            'config':          f'SQ + Mixed-ILP ({TARGET*100:.0f}% target)',
            'ppl':             ppl_sq_ilp,
            'sat_rate':        sat_sq_ilp,
            'adc_area_mm2':    area_sq_ilp.get('adc_area_mm2', 0),
            'chip_area_mm2':   area_sq_ilp.get('chip_area_mm2', 0),
            'adc_area_pct':    area_sq_ilp.get('adc_area_pct', 0),
            'adc_savings_pct': area_sq_ilp.get('adc_savings_pct', 0),
        }
        bc = Counter(asgn_sq_ilp)
        for b_ in bit_choices:
            result[f'n_{b_}b_layers'] = bc.get(b_, 0)
        eval_results.append(result)

    # ── Pareto frontier ──────────────────────────────────────────────────────
    if args.pareto and args.nominal_bits in ppa_sweep:
        print(f"\n[Pareto] Computing Pareto frontier ({args.n_pareto} budget points)...")

        def eval_fn(asgn):
            assign_dict = dict(zip(layer_names, asgn))
            return eval_with_assignment(
                model, assign_dict, calib_loader, eval_loader,
                default_bits=args.nominal_bits, device=args.device,
                num_calib=args.num_calib_batches,
                num_eval=args.num_eval_batches, clip_pct=args.clip_pct)

        pareto_pts = compute_pareto_points(
            sensitivity_data, eval_fn, ppa_sweep,
            nominal_bits=args.nominal_bits,
            bit_choices=bit_choices,
            n_budgets=args.n_pareto)

        pareto_path = out_dir / 'pareto_frontier.json'
        with open(pareto_path, 'w') as f:
            json.dump(pareto_pts, f, indent=2)
        print(f"[Pareto] Saved to {pareto_path}")

    # ── Save evaluation results ──────────────────────────────────────────────
    eval_path = out_dir / 'evaluation_results.csv'
    if eval_results:
        with open(eval_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(eval_results[0].keys()))
            w.writeheader()
            w.writerows(eval_results)
    print(f"\n[Done] Evaluation results saved to {eval_path}")

    # ── Save per-layer allocations ────────────────────────────────────────────
    alloc_json = {
        'greedy': dict(zip(layer_names, asgn_greedy)),
        'ilp':    dict(zip(layer_names, asgn_ilp)),
    }
    alloc_path = out_dir / 'allocations.json'
    with open(alloc_path, 'w') as f:
        json.dump(alloc_json, f, indent=2)
    print(f"[Done] Allocations saved to {alloc_path}")

    # ── Print final summary table ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Config':<42} {'PPL':>8} {'ADC(mm²)':>10} {'ADC%':>6} {'Savings':>8}")
    print("-" * 80)
    for r in eval_results:
        print(f"{r['config']:<42} {r['ppl']:>8.2f} "
              f"{r['adc_area_mm2']:>10.1f} "
              f"{r['adc_area_pct']:>5.1f}% "
              f"{r['adc_savings_pct']:>7.1f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()
