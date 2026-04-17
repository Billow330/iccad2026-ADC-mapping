"""
opt1_3b_ilp_ppa.py — OPT-1.3B Complete ILP Allocation + PPA Verification
=========================================================================
Uses existing sensitivity data (group_sensitivity.json) and PPA sweep
(ppa_sweep_opt1.3b.csv) to run full ILP allocation and compute
mixed-precision area/energy results for OPT-1.3B.

This addresses Plan A1: extending the main experiment to a larger model.
"""

import json, csv, sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).parent
SENS_PATH = ROOT / 'results' / 'sensitivity' / 'opt1.3b' / 'group_sensitivity.json'
PPA_CSV   = ROOT / 'results' / 'ppa' / 'opt1.3b' / 'ppa_sweep_opt1.3b.csv'
OUT_DIR   = ROOT / 'results' / 'opt1.3b_ilp'


def classify_layer_opt1_3b(name):
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


def build_layer_names_opt1_3b():
    """OPT-1.3B: 24 decoder layers x 6 linear + lm_head = 145 layers."""
    names = []
    for i in range(24):
        names.append(f'model.decoder.layers.{i}.self_attn.k_proj')
        names.append(f'model.decoder.layers.{i}.self_attn.v_proj')
        names.append(f'model.decoder.layers.{i}.self_attn.q_proj')
        names.append(f'model.decoder.layers.{i}.self_attn.out_proj')
        names.append(f'model.decoder.layers.{i}.fc1')
        names.append(f'model.decoder.layers.{i}.fc2')
    names.append('lm_head')
    return names


def load_ppa_sweep(csv_path):
    rows = {}
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            b = int(r['adc_bits'])
            rows[b] = {k: float(v) for k, v in r.items()
                       if k not in ('model', 'adc_bits')}
            rows[b]['adc_bits'] = b
    return rows


def ilp_allocation(sensitivity_data, nominal_bits=7,
                   bit_choices=(4, 5, 6, 7, 8),
                   target_area_savings=0.20):
    n = len(sensitivity_data)
    nominal_area = n * (2 ** nominal_bits)
    budget = nominal_area * (1.0 - target_area_savings)
    sens = np.array([r['sensitivity'] for r in sensitivity_data])

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        B = len(bit_choices)
        N_vars = n * B
        c = np.zeros(N_vars)
        for i in range(n):
            for j, b in enumerate(bit_choices):
                c[i * B + j] = max(0.0, nominal_bits - b) * sens[i]

        A_eq = np.zeros((n, N_vars))
        for i in range(n):
            for j in range(B):
                A_eq[i, i * B + j] = 1.0

        A_area = np.zeros((1, N_vars))
        for i in range(n):
            for j, b in enumerate(bit_choices):
                A_area[0, i * B + j] = 2 ** b

        A_ub = np.vstack([A_eq, -A_eq, A_area])
        b_ub_lo = np.concatenate([np.ones(n), -np.ones(n), [-np.inf]])
        b_ub_hi = np.concatenate([np.ones(n), -np.ones(n), [budget]])

        constraints = LinearConstraint(A_ub, b_ub_lo, b_ub_hi)
        bounds = Bounds(lb=0.0, ub=1.0)
        integrality = np.ones(N_vars)

        res = milp(c, constraints=constraints, integrality=integrality,
                   bounds=bounds)

        if res.success:
            assignments = []
            for i in range(n):
                best_j = max(range(B), key=lambda j: res.x[i * B + j])
                assignments.append(bit_choices[best_j])
            print(f"[ILP] Optimal solution found (status={res.status})")
            return assignments
        else:
            print(f"[ILP] MILP failed, using greedy fallback")
    except (ImportError, Exception) as e:
        print(f"[ILP] scipy.milp unavailable ({e}), using greedy")

    return _sensitivity_greedy(sensitivity_data, nominal_bits, bit_choices,
                               target_area_savings)


def _sensitivity_greedy(sensitivity_data, nominal_bits=7,
                        bit_choices=(4, 5, 6, 7, 8),
                        target_area_savings=0.20):
    n = len(sensitivity_data)
    assignments = [nominal_bits] * n
    sens = [r['sensitivity'] for r in sensitivity_data]
    ref_area = n * (2 ** nominal_bits)
    budget = ref_area * (1.0 - target_area_savings)
    sorted_idx = sorted(range(n), key=lambda i: sens[i])
    current_area = ref_area
    rounds = 0
    while current_area > budget and rounds < 20:
        improved = False
        for idx in sorted_idx:
            if current_area <= budget:
                break
            cur_bits = assignments[idx]
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


def adc_area_ratio(bits, ref_bits=7):
    return (2 ** bits) / (2 ** ref_bits)


def compute_area_from_assignment(assignments, ref_ppa, nominal_bits=7):
    ref_chip = ref_ppa['chip_area_um2']
    ref_adc = ref_ppa['adc_area_um2']
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


def compute_energy_from_assignment(assignments, ref_ppa, nominal_bits=7):
    ref_total = ref_ppa.get('energy_pJ', 0)
    ref_adc_e = ref_ppa.get('adc_energy_pJ', 0)
    ref_nonadc_e = ref_total - ref_adc_e
    n = len(assignments)
    per_layer_adc_e = ref_adc_e / n

    total_adc_e = sum(per_layer_adc_e * adc_area_ratio(b, nominal_bits)
                      for b in assignments)
    total_e = ref_nonadc_e + total_adc_e
    savings = (1.0 - total_e / ref_total) * 100.0 if ref_total > 0 else 0
    return {
        'total_energy_pJ': total_e,
        'adc_energy_pJ': total_adc_e,
        'energy_savings_pct': savings,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load sensitivity data
    with open(SENS_PATH) as f:
        group_sens = json.load(f)

    layer_names = build_layer_names_opt1_3b()
    n_layers = len(layer_names)
    print(f"[OPT-1.3B] {n_layers} layers")

    # Build per-layer sensitivity from group data
    sensitivity_data = []
    for name in layer_names:
        ltype = classify_layer_opt1_3b(name)
        g = group_sens.get(ltype, {})
        delta_per_layer = g.get('delta_per_layer', 0.0)
        sensitivity_data.append({
            'layer': name,
            'layer_type': ltype,
            'sensitivity': max(delta_per_layer, 0.0),
            'delta_per_layer': delta_per_layer,
        })

    # Load PPA sweep
    ppa_sweep = load_ppa_sweep(PPA_CSV)
    nominal_bits = 7
    ref_ppa = ppa_sweep[nominal_bits]
    print(f"[PPA] Ref 7-bit: chip={ref_ppa['chip_area_um2']/1e6:.1f}mm², "
          f"ADC={ref_ppa['adc_area_um2']/1e6:.1f}mm² "
          f"({ref_ppa.get('adc_area_pct', 0):.1f}%)")

    # Print sensitivity summary
    print(f"\n[Sensitivity] Group sensitivity (11b->10b, from existing data):")
    for ltype in ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']:
        g = group_sens.get(ltype, {})
        print(f"  {ltype:12s}: ΔPPL/layer={g.get('delta_per_layer', 0):+.4f} "
              f"({g.get('n_layers', 0)} layers)")

    # Run allocations at multiple savings targets
    bit_choices = (4, 5, 6, 7, 8)
    savings_targets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    all_results = []
    print(f"\n{'='*90}")
    print(f"{'Target':>8} {'Method':>12} {'ADC(mm²)':>12} {'Chip(mm²)':>12} "
          f"{'ADC_sav%':>10} {'Energy_sav%':>12} {'Bits dist':>20}")
    print(f"{'-'*90}")

    # Uniform baselines
    for bits in [6, 7, 8]:
        asgn = [bits] * n_layers
        area = compute_area_from_assignment(asgn, ref_ppa, nominal_bits)
        energy = compute_energy_from_assignment(asgn, ref_ppa, nominal_bits)
        dist = Counter(asgn)
        result = {
            'config': f'Uniform {bits}b',
            'target_savings': 0 if bits == nominal_bits else None,
            **area, **energy,
            'bits_dist': dict(dist),
            'assignment': asgn,
        }
        all_results.append(result)
        dist_str = str(dict(sorted(dist.items())))
        print(f"{'--':>8} {'Uniform '+str(bits)+'b':>12} "
              f"{area['adc_area_mm2']:>12.1f} {area['chip_area_mm2']:>12.1f} "
              f"{area['adc_savings_pct']:>10.1f} {energy['energy_savings_pct']:>12.1f} "
              f"  {dist_str}")

    # ILP and Greedy at each target
    for target in savings_targets:
        # ILP
        asgn_ilp = ilp_allocation(sensitivity_data, nominal_bits,
                                   bit_choices, target)
        area_ilp = compute_area_from_assignment(asgn_ilp, ref_ppa, nominal_bits)
        energy_ilp = compute_energy_from_assignment(asgn_ilp, ref_ppa, nominal_bits)
        dist_ilp = Counter(asgn_ilp)

        # Greedy
        asgn_grd = _sensitivity_greedy(sensitivity_data, nominal_bits,
                                        bit_choices, target)
        area_grd = compute_area_from_assignment(asgn_grd, ref_ppa, nominal_bits)
        energy_grd = compute_energy_from_assignment(asgn_grd, ref_ppa, nominal_bits)
        dist_grd = Counter(asgn_grd)

        all_results.append({
            'config': f'ILP {target*100:.0f}%',
            'target_savings': target * 100,
            **area_ilp, **energy_ilp,
            'bits_dist': dict(dist_ilp),
            'assignment': asgn_ilp,
        })
        all_results.append({
            'config': f'Greedy {target*100:.0f}%',
            'target_savings': target * 100,
            **area_grd, **energy_grd,
            'bits_dist': dict(dist_grd),
            'assignment': asgn_grd,
        })

        dist_ilp_str = str(dict(sorted(dist_ilp.items())))
        dist_grd_str = str(dict(sorted(dist_grd.items())))
        print(f"{target*100:>7.0f}% {'ILP':>12} "
              f"{area_ilp['adc_area_mm2']:>12.1f} {area_ilp['chip_area_mm2']:>12.1f} "
              f"{area_ilp['adc_savings_pct']:>10.1f} {energy_ilp['energy_savings_pct']:>12.1f} "
              f"  {dist_ilp_str}")
        print(f"{'':>8} {'Greedy':>12} "
              f"{area_grd['adc_area_mm2']:>12.1f} {area_grd['chip_area_mm2']:>12.1f} "
              f"{area_grd['adc_savings_pct']:>10.1f} {energy_grd['energy_savings_pct']:>12.1f} "
              f"  {dist_grd_str}")

    print(f"{'='*90}")

    # Save results
    out_csv = OUT_DIR / 'opt1_3b_ilp_ppa_results.csv'
    fields = ['config', 'target_savings', 'chip_area_mm2', 'adc_area_mm2',
              'adc_area_pct', 'adc_savings_pct', 'total_energy_pJ',
              'adc_energy_pJ', 'energy_savings_pct']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"\n[Saved] {out_csv}")

    # Save per-layer allocation for 20% target
    ilp_20_result = next(r for r in all_results if r['config'] == 'ILP 20%')
    alloc_json = {
        'model': 'OPT-1.3B',
        'nominal_bits': nominal_bits,
        'target_savings': 20,
        'ilp_assignment': dict(zip(layer_names, ilp_20_result['assignment'])),
        'area': {k: ilp_20_result[k] for k in ['chip_area_mm2', 'adc_area_mm2',
                                                 'adc_area_pct', 'adc_savings_pct']},
        'energy': {k: ilp_20_result[k] for k in ['total_energy_pJ', 'adc_energy_pJ',
                                                   'energy_savings_pct']},
    }
    alloc_path = OUT_DIR / 'opt1_3b_ilp_allocation.json'
    with open(alloc_path, 'w') as f:
        json.dump(alloc_json, f, indent=2)
    print(f"[Saved] {alloc_path}")

    # Print key results for paper
    print(f"\n{'='*60}")
    print(f"KEY RESULTS FOR PAPER (OPT-1.3B)")
    print(f"{'='*60}")
    print(f"Ref chip area (7b): {ref_ppa['chip_area_um2']/1e6:.1f} mm²")
    print(f"Ref ADC area (7b):  {ref_ppa['adc_area_um2']/1e6:.1f} mm² "
          f"({ref_ppa.get('adc_area_pct', 0):.1f}%)")
    print(f"Ref energy (7b):    {ref_ppa.get('energy_pJ', 0)/1e3:.1f} nJ")
    print()

    ilp20 = ilp_20_result
    print(f"ILP-20% allocation:")
    print(f"  ADC area:     {ilp20['adc_area_mm2']:.1f} mm² "
          f"(savings: {ilp20['adc_savings_pct']:.1f}%)")
    print(f"  Chip area:    {ilp20['chip_area_mm2']:.1f} mm²")
    print(f"  Energy:       {ilp20['total_energy_pJ']/1e3:.1f} nJ "
          f"(savings: {ilp20['energy_savings_pct']:.1f}%)")
    print(f"  ADC savings:  {ref_ppa['adc_area_um2']/1e6 - ilp20['adc_area_mm2']:.1f} mm²")
    print(f"  Bit dist:     {dict(sorted(Counter(ilp20['assignment']).items()))}")

    # Scaled comparison with OPT-125M
    opt125m_chip = 936.6  # mm²
    print(f"\n  Scale factor vs OPT-125M: "
          f"{ref_ppa['chip_area_um2']/1e6 / opt125m_chip:.2f}x")
    print(f"  ADC savings ({ilp20['adc_savings_pct']:.1f}%) = "
          f"{ref_ppa['adc_area_um2']/1e6 * ilp20['adc_savings_pct']/100:.1f} mm² "
          f"(> entire OPT-125M chip!)")


if __name__ == '__main__':
    main()
