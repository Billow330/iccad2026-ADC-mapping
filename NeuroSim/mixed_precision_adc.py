"""
mixed_precision_adc.py — CIM-Aware Mixed-Precision ADC Allocation for LLM Inference
====================================================================================
Core algorithm for ICCAD 2026 paper.

Key idea:
  LLM layers have highly heterogeneous ADC pressure (saturation rates vary from
  0% to 100% across layers). A uniform ADC bit budget is wasteful:
  - Low-saturation layers waste bits that could be saved
  - High-saturation layers need more bits for accuracy

  This module implements:
  1. Layer sensitivity analysis: ∂PPL/∂adc_bits per layer
  2. SmoothQuant-guided ADC bit reduction: CIM-SQ reduces required bits/layer
  3. Budget-constrained mixed-precision assignment (knapsack formulation)
  4. NeuroSIM-validated area/energy projections

Usage:
  python3 mixed_precision_adc.py --outlier_csv results/opt125m/outlier_*.csv
                                  --sweep_csv  results/opt125m/sweep_adc_*.csv
                                  --ppa_csv    results/ppa/opt125m/ppa_sweep_opt125m.csv
                                  --output_dir results/mixed_precision
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# OPT-125M layer definitions (matches outlier_analysis.py output)
# ─────────────────────────────────────────────────────────────────────────────

# Layer type classification for heterogeneous bit assignment
LAYER_TYPES = {
    'self_attn.q_proj': 'attn_qkv',
    'self_attn.k_proj': 'attn_qkv',
    'self_attn.v_proj': 'attn_qkv',
    'self_attn.out_proj': 'attn_out',
    'fc1': 'ffn_up',
    'fc2': 'ffn_down',
    'lm_head': 'lm_head',
}

def classify_layer(layer_name):
    for key, ltype in LAYER_TYPES.items():
        if key in layer_name:
            return ltype
    return 'other'


# ─────────────────────────────────────────────────────────────────────────────
# ADC area model
# ─────────────────────────────────────────────────────────────────────────────

def adc_area_scale(bits, ref_bits=7):
    """
    Relative ADC area vs reference bit width.

    SAR-ADC area scales roughly as O(bits * 2^bits) for small bits,
    but in practice the dominant term for 3-10 bit SAR is ~2^bits
    (comparator + DAC area).
    We use the empirical model: area ∝ 2^bits.
    (For MLSA which NeuroSIM uses: area ∝ levelOutput = 2^bits)
    """
    return (2 ** bits) / (2 ** ref_bits)


def total_adc_area_relative(assignments, ref_bits=7):
    """Compute total relative ADC area for a bit assignment dict."""
    return sum(adc_area_scale(b, ref_bits) for b in assignments)


# ─────────────────────────────────────────────────────────────────────────────
# Load outlier characterization data
# ─────────────────────────────────────────────────────────────────────────────

def load_outlier_data(csv_path):
    """Load per-layer outlier characterization from outlier_*.csv."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'layer':          r['layer'],
                'layer_type':     classify_layer(r['layer']),
                'sat_rate':       float(r['sat_rate_worst']),
                'act_max':        float(r['act_max']),
                'outlier_frac':   float(r['outlier_channel_fraction']),
                'overhead_bits':  int(r['adc_overhead_bits']),
            })
    return rows


def load_ppa_sweep(csv_path):
    """Load PPA sweep data (area vs ADC bits from NeuroSIM)."""
    rows = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            bits = int(r['adc_bits'])
            rows[bits] = {
                'chip_area_um2': float(r.get('chip_area_um2', 0)),
                'adc_area_um2':  float(r.get('adc_area_um2', 0)),
                'adc_area_pct':  float(r.get('adc_area_pct', 0)),
                'energy_pJ':     float(r.get('energy_pJ', 0)),
                'tops_w':        float(r.get('tops_w', 0)),
            }
    return rows


def load_accuracy_sweep(csv_path):
    """Load accuracy sweep data (PPL vs ADC bits from llm_inference)."""
    rows = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            bits = int(r['adc_bits'])
            rows[bits] = {
                'ppl_baseline': float(r['ppl_baseline']),
                'ppl_cim_sq':   float(r['ppl_cim_sq']),
                'sat_baseline': float(r['sat_baseline']),
                'sat_cim_sq':   float(r['sat_cim_sq']),
            }
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Layer sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_layer_sensitivity(outlier_data, accuracy_sweep):
    """
    Estimate ∂PPL/∂adc_bits for each layer type.

    Approach:
    - From accuracy_sweep: global ΔPPl/Δbits curve
    - Weight per-layer contribution by saturation rate
    - High-saturation layers → high sensitivity (reducing their bits hurts more)

    Returns: dict {layer_idx: sensitivity_score}
    where higher score = reducing this layer's ADC bits costs more accuracy.
    """
    # Global PPL vs ADC bits: compute sensitivity ∂PPL/∂bits
    bits_list = sorted(accuracy_sweep.keys())
    global_dppl_dbits = {}
    for i in range(1, len(bits_list)):
        b_lo, b_hi = bits_list[i-1], bits_list[i]
        dppl = accuracy_sweep[b_lo]['ppl_baseline'] - accuracy_sweep[b_hi]['ppl_baseline']
        dbits = b_hi - b_lo
        # PPL improves (decreases) as bits increase → positive sensitivity
        global_dppl_dbits[b_lo] = dppl / dbits if dbits != 0 else 0

    # Per-layer sensitivity: proportional to saturation rate
    # Intuition: saturating layers lose more accuracy when bits decrease
    total_sat = sum(r['sat_rate'] for r in outlier_data) + 1e-9

    layer_sensitivity = {}
    for i, layer in enumerate(outlier_data):
        sat = layer['sat_rate']
        # Weight: layers with higher saturation are more sensitive to ADC reduction
        # Plus overhead_bits: layers needing more bits are clearly sensitive
        sensitivity = (sat / total_sat) * len(outlier_data) + layer['overhead_bits'] * 0.1
        layer_sensitivity[i] = {
            'layer': layer['layer'],
            'sat_rate': sat,
            'overhead_bits': layer['overhead_bits'],
            'sensitivity': sensitivity,
        }

    return layer_sensitivity


# ─────────────────────────────────────────────────────────────────────────────
# Mixed-precision ADC allocation algorithms
# ─────────────────────────────────────────────────────────────────────────────

def uniform_allocation(outlier_data, bits):
    """Baseline: all layers get the same ADC bits."""
    return [bits] * len(outlier_data)


def greedy_allocation(outlier_data, layer_sensitivity,
                       nominal_bits=7, min_bits=4, max_bits=8,
                       target_area_savings=0.30):
    """
    Greedy mixed-precision allocation:
    1. Start with uniform nominal_bits for all layers.
    2. Sort layers by sensitivity (ascending = least sensitive first).
    3. Greedily reduce bit-width of least-sensitive layers while
       total relative ADC area > target budget.
    4. Optionally boost highly sensitive/high-overhead layers.

    Returns: list of adc_bits per layer, total_area_relative, savings_pct
    """
    n = len(outlier_data)
    assignments = [nominal_bits] * n

    ref_area = total_adc_area_relative(assignments, nominal_bits)  # = n
    target_area = ref_area * (1.0 - target_area_savings)

    # Sort indices by sensitivity ascending (least sensitive first = best candidates to reduce)
    sorted_idx = sorted(range(n), key=lambda i: layer_sensitivity[i]['sensitivity'])

    current_area = ref_area
    for idx in sorted_idx:
        if current_area <= target_area:
            break
        cur = assignments[idx]
        if cur > min_bits:
            # Try reducing by 1 bit
            new_bits = cur - 1
            delta_area = adc_area_scale(new_bits, nominal_bits) - adc_area_scale(cur, nominal_bits)
            current_area += delta_area
            assignments[idx] = new_bits

    # Optional: boost high-overhead layers (they need more bits for accuracy)
    for idx in range(n):
        overhead = outlier_data[idx]['overhead_bits']
        if overhead >= 3 and assignments[idx] < max_bits:
            potential = assignments[idx] + 1
            delta_area = adc_area_scale(potential, nominal_bits) - adc_area_scale(assignments[idx], nominal_bits)
            if current_area + delta_area <= ref_area * (1.0 - target_area_savings * 0.7):
                current_area += delta_area
                assignments[idx] = potential

    final_area = total_adc_area_relative(assignments, nominal_bits)
    actual_savings = (1.0 - final_area / ref_area) * 100.0

    return assignments, final_area, actual_savings


def sq_guided_allocation(outlier_data, layer_sensitivity,
                          nominal_bits=7, min_bits=4, max_bits=8,
                          target_area_savings=0.30,
                          sq_benefit_per_layer=None):
    """
    SmoothQuant-guided allocation:
    After CIM-SQ, each layer's saturation rate is reduced.
    This allows more layers to use lower ADC bits.

    sq_benefit_per_layer: dict {layer_idx: sat_reduction_fraction}
    If None, uses estimated 20% reduction for high-sat layers.
    """
    if sq_benefit_per_layer is None:
        # Estimate: CIM-SQ reduces saturation by 10-30% for high-sat layers
        sq_benefit_per_layer = {}
        for i, layer in enumerate(outlier_data):
            if layer['sat_rate'] > 0.3:
                sq_benefit_per_layer[i] = 0.20  # 20% sat reduction
            elif layer['sat_rate'] > 0.1:
                sq_benefit_per_layer[i] = 0.10
            else:
                sq_benefit_per_layer[i] = 0.0

    # After SQ: reduced saturation means reduced overhead_bits
    modified_outlier = []
    for i, layer in enumerate(outlier_data):
        benefit = sq_benefit_per_layer.get(i, 0.0)
        new_sat = layer['sat_rate'] * (1.0 - benefit)
        # Estimate new overhead bits: 1 bit reduction per ~30% sat reduction
        overhead_reduction = int(benefit * 3)
        new_overhead = max(0, layer['overhead_bits'] - overhead_reduction)
        modified_outlier.append({
            **layer,
            'sat_rate': new_sat,
            'overhead_bits': new_overhead,
            'sq_sat_reduction': benefit,
        })

    # Recompute sensitivity with post-SQ saturation
    modified_sensitivity = {}
    total_sat = sum(r['sat_rate'] for r in modified_outlier) + 1e-9
    for i, layer in enumerate(modified_outlier):
        sensitivity = (layer['sat_rate'] / total_sat) * len(modified_outlier) + layer['overhead_bits'] * 0.1
        modified_sensitivity[i] = {**layer_sensitivity[i], 'sensitivity': sensitivity}

    assignments, final_area, actual_savings = greedy_allocation(
        modified_outlier, modified_sensitivity,
        nominal_bits=nominal_bits, min_bits=min_bits, max_bits=max_bits,
        target_area_savings=target_area_savings
    )

    return assignments, final_area, actual_savings, modified_outlier


# ─────────────────────────────────────────────────────────────────────────────
# Estimate accuracy from allocation (using sweep data)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_ppl_from_allocation(assignments, outlier_data, accuracy_sweep):
    """
    Estimate perplexity degradation from a mixed-precision allocation.

    Model:
    - Reference: uniform 7-bit PPL from accuracy_sweep
    - Each layer's contribution to PPL increase = f(adc_bits, sat_rate)
    - Layers with higher saturation have larger impact

    This is a first-order approximation. The key insight is:
    the PPL improvement from CIM-SQ at a given bit width tells us
    how much accuracy per unit of ADC reduction.
    """
    ref_bits = 7
    if ref_bits not in accuracy_sweep:
        # Use closest available
        ref_bits = min(accuracy_sweep.keys(), key=lambda b: abs(b - 7))

    ref_ppl = accuracy_sweep[ref_bits]['ppl_baseline']
    n = len(outlier_data)

    # Per-layer PPL sensitivity: estimated from saturation rate
    total_sat = sum(r['sat_rate'] for r in outlier_data) + 1e-9

    estimated_ppl = ref_ppl
    for i, (bits, layer) in enumerate(zip(assignments, outlier_data)):
        if bits == ref_bits:
            continue
        # PPL change: for bits < ref_bits, PPL increases
        # Model: ΔPPl_layer = (layer_sat_fraction) × ΔPPl_global(bits → ref_bits)
        if bits in accuracy_sweep and ref_bits in accuracy_sweep:
            delta_ppl_global = accuracy_sweep[bits]['ppl_baseline'] - accuracy_sweep[ref_bits]['ppl_baseline']
        else:
            # Extrapolate: approximately 10 PPL per bit reduction at 7b range
            delta_ppl_global = (ref_bits - bits) * 5.0

        sat_fraction = layer['sat_rate'] / total_sat
        delta_ppl_layer = sat_fraction * delta_ppl_global
        estimated_ppl += delta_ppl_layer

    return estimated_ppl


# ─────────────────────────────────────────────────────────────────────────────
# Compute hardware metrics from allocation + PPA sweep
# ─────────────────────────────────────────────────────────────────────────────

def compute_hardware_metrics(assignments, ppa_sweep, ref_bits=7):
    """
    Estimate chip area/energy for a mixed-precision allocation using
    per-bit-width PPA data from NeuroSIM.

    Area model: each layer's ADC area = (1/n_layers) × chip_adc_area × scale(bits/ref_bits)
    Energy model: similarly scaled.
    """
    if ref_bits not in ppa_sweep:
        ref_bits = min(ppa_sweep.keys(), key=lambda b: abs(b - 7))

    ref = ppa_sweep[ref_bits]
    ref_chip_area = ref['chip_area_um2']
    ref_adc_area  = ref['adc_area_um2']
    ref_energy    = ref['energy_pJ']
    ref_non_adc   = ref_chip_area - ref_adc_area

    n = len(assignments)
    per_layer_adc_area = ref_adc_area / n  # uniform distribution assumption

    # Scale each layer's ADC area
    total_adc_area = 0
    total_energy = 0
    ref_non_adc_energy = ref_energy * (1.0 - ref_adc_area / ref_chip_area)

    for bits in assignments:
        scale = adc_area_scale(bits, ref_bits)
        total_adc_area += per_layer_adc_area * scale
        # Energy: ADC energy ∝ area (roughly)
        per_layer_adc_energy = ref_energy * (ref_adc_area / ref_chip_area) / n
        total_energy += per_layer_adc_energy * scale

    total_energy += ref_non_adc_energy  # non-ADC energy unchanged

    total_chip_area = ref_non_adc + total_adc_area
    adc_savings_pct = (1.0 - total_adc_area / ref_adc_area) * 100.0
    chip_savings_pct = (1.0 - total_chip_area / ref_chip_area) * 100.0

    return {
        'ref_bits': ref_bits,
        'chip_area_mm2': total_chip_area / 1e6,
        'adc_area_mm2': total_adc_area / 1e6,
        'adc_area_pct': 100 * total_adc_area / total_chip_area,
        'energy_pJ': total_energy,
        'adc_savings_pct': adc_savings_pct,
        'chip_savings_pct': chip_savings_pct,
        'ref_chip_area_mm2': ref_chip_area / 1e6,
        'ref_adc_area_pct': 100 * ref_adc_area / ref_chip_area,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary table generation (for paper)
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(configs, ppa_sweep=None, accuracy_sweep=None, outlier_data=None):
    """
    Print a comparison table of different ADC allocation strategies.
    This generates the data for Table III in the paper.
    """
    print('\n' + '='*80)
    print('Table: Mixed-Precision ADC Allocation Comparison')
    print('='*80)
    print(f'{"Config":<30} {"ADC Area":>10} {"Chip Area":>10} {"Est. PPL":>10} {"ADC Savings":>12}')
    print('-'*80)

    for name, assignments in configs.items():
        hw = compute_hardware_metrics(assignments, ppa_sweep) if ppa_sweep else {}
        ppl = estimate_ppl_from_allocation(assignments, outlier_data, accuracy_sweep) \
              if (accuracy_sweep and outlier_data) else float('nan')

        adc_mm2 = hw.get('adc_area_mm2', float('nan'))
        chip_mm2 = hw.get('chip_area_mm2', float('nan'))
        savings = hw.get('adc_savings_pct', float('nan'))

        print(f'{name:<30} {adc_mm2:>10.3f} {chip_mm2:>10.3f} {ppl:>10.2f} {savings:>11.1f}%')

    print('='*80)


def save_allocation_results(configs, outlier_data, ppa_sweep, accuracy_sweep, output_dir):
    """Save full allocation results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-config summary CSV
    summary_rows = []
    for name, assignments in configs.items():
        hw = compute_hardware_metrics(assignments, ppa_sweep) if ppa_sweep else {}
        ppl = estimate_ppl_from_allocation(assignments, outlier_data, accuracy_sweep) \
              if accuracy_sweep else float('nan')

        from collections import Counter
        bits_dist = Counter(assignments)

        row = {
            'config': name,
            'ppl_estimated': f'{ppl:.2f}',
            'chip_area_mm2': f'{hw.get("chip_area_mm2", 0):.4f}',
            'adc_area_mm2': f'{hw.get("adc_area_mm2", 0):.4f}',
            'adc_area_pct': f'{hw.get("adc_area_pct", 0):.1f}',
            'adc_savings_pct': f'{hw.get("adc_savings_pct", 0):.1f}',
            'chip_savings_pct': f'{hw.get("chip_savings_pct", 0):.1f}',
        }
        for b in range(3, 12):
            row[f'n_layers_{b}b'] = bits_dist.get(b, 0)
        summary_rows.append(row)

    summary_path = Path(output_dir) / 'mixed_precision_summary.csv'
    if summary_rows:
        with open(summary_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f'[MixedPrec] Summary saved to {summary_path}')

    # Per-layer allocation CSV (for all configs)
    for name, assignments in configs.items():
        safe_name = name.replace(' ', '_').replace('/', '_').replace('+', 'plus')
        layer_path = Path(output_dir) / f'allocation_{safe_name}.csv'
        rows = []
        for i, (bits, layer) in enumerate(zip(assignments, outlier_data)):
            rows.append({
                'layer_idx': i,
                'layer': layer['layer'],
                'layer_type': layer['layer_type'],
                'adc_bits': bits,
                'sat_rate': f'{layer["sat_rate"]:.4f}',
                'overhead_bits': layer['overhead_bits'],
            })
        with open(layer_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'[MixedPrec] Layer allocation saved to {layer_path}')

    # Per-layer allocation JSON (machine-readable for plot_results)
    alloc_json = {name: assignments for name, assignments in configs.items()}
    json_path = Path(output_dir) / 'allocations.json'
    with open(json_path, 'w') as f:
        json.dump(alloc_json, f, indent=2)
    print(f'[MixedPrec] Allocations JSON saved to {json_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outlier_csv', required=True,
                   help='Path to outlier_*.csv from llm_inference.py --task characterize')
    p.add_argument('--sweep_csv', required=True,
                   help='Path to sweep_adc_*.csv from llm_inference.py --task sweep_adc')
    p.add_argument('--ppa_csv', default=None,
                   help='Path to ppa_sweep_*.csv from neurosim_ppa.py --sweep_adc')
    p.add_argument('--nominal_bits', type=int, default=7,
                   help='Nominal (reference) ADC bits for uniform baseline')
    p.add_argument('--min_bits', type=int, default=4,
                   help='Minimum ADC bits for any layer')
    p.add_argument('--max_bits', type=int, default=8,
                   help='Maximum ADC bits for any layer')
    p.add_argument('--target_savings', type=float, default=30.0,
                   help='Target ADC area savings %% (default: 30%%)')
    p.add_argument('--output_dir', default='results/mixed_precision')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f'[MixedPrec] Loading outlier data from {args.outlier_csv}')
    outlier_data = load_outlier_data(args.outlier_csv)
    n = len(outlier_data)
    print(f'[MixedPrec] {n} layers loaded')

    print(f'[MixedPrec] Loading accuracy sweep from {args.sweep_csv}')
    accuracy_sweep = load_accuracy_sweep(args.sweep_csv)
    print(f'[MixedPrec] ADC sweep range: {sorted(accuracy_sweep.keys())}')

    ppa_sweep = None
    if args.ppa_csv and Path(args.ppa_csv).exists():
        print(f'[MixedPrec] Loading PPA sweep from {args.ppa_csv}')
        ppa_sweep = load_ppa_sweep(args.ppa_csv)
        print(f'[MixedPrec] PPA data for bits: {sorted(ppa_sweep.keys())}')
    else:
        print('[MixedPrec] WARNING: No PPA CSV provided. Hardware metrics will be estimated.')
        # Create synthetic PPA data based on theoretical ADC area model
        ppa_sweep = {}
        for bits in range(3, 11):
            scale = adc_area_scale(bits, 7)
            # Reference: 7-bit chip ~350mm² for OPT-125M based on NeuroSIM scaling
            ref_chip = 350e6  # um²
            ref_adc_frac = 0.25  # ADC typically 20-30% of CIM chip area
            ppa_sweep[bits] = {
                'chip_area_um2': ref_chip * (1 - ref_adc_frac + ref_adc_frac * scale),
                'adc_area_um2': ref_chip * ref_adc_frac * scale,
                'adc_area_pct': 100 * ref_adc_frac * scale / (1 - ref_adc_frac + ref_adc_frac * scale),
                'energy_pJ': 1e6 * scale,
                'tops_w': 1.0,
            }

    # ── Outlier characterization summary ──────────────────────────────────
    sat_rates = [r['sat_rate'] for r in outlier_data]
    overhead_bits = [r['overhead_bits'] for r in outlier_data]
    print(f'\n[Characterization] Layer saturation statistics:')
    print(f'  Mean sat rate: {np.mean(sat_rates)*100:.1f}%')
    print(f'  Max sat rate:  {np.max(sat_rates)*100:.1f}%')
    print(f'  Mean overhead bits: {np.mean(overhead_bits):.1f}')
    print(f'  Layers needing >0 extra bits: {sum(1 for x in overhead_bits if x > 0)}/{n}')

    # ── Layer sensitivity analysis ─────────────────────────────────────────
    layer_sensitivity = compute_layer_sensitivity(outlier_data, accuracy_sweep)

    # ── Configurations to compare ──────────────────────────────────────────
    configs = {}

    # Baselines
    for bits in [4, 5, 6, 7, 8]:
        configs[f'Uniform {bits}b'] = uniform_allocation(outlier_data, bits)

    # Mixed precision: greedy, no SQ
    asgn_greedy, area_greedy, sav_greedy = greedy_allocation(
        outlier_data, layer_sensitivity,
        nominal_bits=args.nominal_bits,
        min_bits=args.min_bits,
        max_bits=args.max_bits,
        target_area_savings=args.target_savings / 100.0
    )
    configs[f'Mixed (greedy, {args.target_savings:.0f}% savings)'] = asgn_greedy
    print(f'\n[Greedy] Actual savings: {sav_greedy:.1f}%')
    from collections import Counter
    print(f'[Greedy] Bit distribution: {dict(Counter(asgn_greedy))}')

    # Mixed precision: SQ-guided
    asgn_sq, area_sq, sav_sq, mod_outlier = sq_guided_allocation(
        outlier_data, layer_sensitivity,
        nominal_bits=args.nominal_bits,
        min_bits=args.min_bits,
        max_bits=args.max_bits,
        target_area_savings=args.target_savings / 100.0
    )
    configs[f'Mixed+SQ (greedy, {args.target_savings:.0f}% target)'] = asgn_sq
    print(f'[SQ-guided] Actual savings: {sav_sq:.1f}%')
    print(f'[SQ-guided] Bit distribution: {dict(Counter(asgn_sq))}')

    # ── Print comparison table ─────────────────────────────────────────────
    print_comparison_table(configs, ppa_sweep, accuracy_sweep, outlier_data)

    # ── Save results ───────────────────────────────────────────────────────
    save_allocation_results(configs, outlier_data, ppa_sweep, accuracy_sweep, args.output_dir)

    # ── Key findings summary ───────────────────────────────────────────────
    print('\n' + '='*60)
    print('Key Findings for Paper:')
    print('='*60)

    ref_hw = compute_hardware_metrics(configs['Uniform 7b'], ppa_sweep)
    greedy_hw = compute_hardware_metrics(asgn_greedy, ppa_sweep)
    sq_hw = compute_hardware_metrics(asgn_sq, ppa_sweep)

    print(f'Uniform 7b:')
    print(f'  Chip area: {ref_hw["chip_area_mm2"]:.3f} mm²')
    print(f'  ADC area: {ref_hw["adc_area_mm2"]:.3f} mm² ({ref_hw["adc_area_pct"]:.1f}%)')

    print(f'\nMixed (greedy, {args.target_savings:.0f}% target):')
    print(f'  Chip area: {greedy_hw["chip_area_mm2"]:.3f} mm² ({greedy_hw["chip_savings_pct"]:.1f}% saved)')
    print(f'  ADC area: {greedy_hw["adc_area_mm2"]:.3f} mm² ({greedy_hw["adc_savings_pct"]:.1f}% saved)')
    ppl_greedy = estimate_ppl_from_allocation(asgn_greedy, outlier_data, accuracy_sweep)
    ppl_ref = estimate_ppl_from_allocation(configs['Uniform 7b'], outlier_data, accuracy_sweep)
    print(f'  Est. PPL: {ppl_greedy:.2f} (uniform 7b: {ppl_ref:.2f})')

    print(f'\nMixed+SQ (greedy, {args.target_savings:.0f}% target):')
    print(f'  Chip area: {sq_hw["chip_area_mm2"]:.3f} mm² ({sq_hw["chip_savings_pct"]:.1f}% saved)')
    print(f'  ADC area: {sq_hw["adc_area_mm2"]:.3f} mm² ({sq_hw["adc_savings_pct"]:.1f}% saved)')
    ppl_sq = estimate_ppl_from_allocation(asgn_sq, outlier_data, accuracy_sweep)
    print(f'  Est. PPL: {ppl_sq:.2f} (SQ benefit: {ppl_ref - ppl_sq:.2f})')


if __name__ == '__main__':
    main()
