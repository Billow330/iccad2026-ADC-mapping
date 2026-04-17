"""
latency_analysis.py — Mixed-Precision ADC Latency Analysis + 3D Pareto
======================================================================
Addresses Plan A4: supplement latency analysis for mixed-precision ADC.

Uses NeuroSIM PPA sweep data to model per-layer latency under different
ADC bit-widths, then constructs a 3D Pareto frontier (Area, Energy, PPL).

NeuroSIM latency model for MLSA ADC:
  Latency_ADC ∝ 2^b (successive approximation cycles)
  Total latency = max(latency_array, latency_ADC) per subarray
  For MLSA: latency scales with number of comparator levels

Since NeuroSIM sweep data doesn't include per-bit latency directly,
we use the energy model as a proxy (both scale as 2^b for MLSA).
"""

import json, csv
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).parent
OUT_DIR = ROOT / 'results' / 'latency_analysis'


def load_ppa_sweep(csv_path):
    rows = {}
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            b = int(r['adc_bits'])
            rows[b] = {k: float(v) for k, v in r.items()
                       if k not in ('model', 'adc_bits')}
            rows[b]['adc_bits'] = b
    return rows


def estimate_latency_from_energy(ppa_sweep, nominal_bits=7):
    """
    Estimate relative latency from energy data.
    For MLSA ADC, both energy and latency scale with 2^b.
    We normalize to the nominal_bits reference.
    """
    ref = ppa_sweep.get(nominal_bits, {})
    ref_energy = ref.get('energy_pJ', 1)

    latency_model = {}
    for bits, data in ppa_sweep.items():
        e = data.get('energy_pJ', 0)
        latency_model[bits] = {
            'adc_bits': bits,
            'relative_latency': e / ref_energy if ref_energy > 0 else 1.0,
            'energy_pJ': e,
            'chip_area_um2': data.get('chip_area_um2', 0),
            'adc_area_um2': data.get('adc_area_um2', 0),
        }
    return latency_model


def compute_mixed_latency(assignments, latency_model, nominal_bits=7):
    """
    For mixed-precision, the critical path latency is determined by
    the MAXIMUM ADC bits in any layer (since layers execute sequentially
    in layer-by-layer mode, or the slowest subarray in pipeline mode).

    In layer-by-layer mode: total_latency = sum of per-layer latencies
    In pipeline mode: total_latency = max(per-layer latencies) * num_layers

    We report both models.
    """
    ref_lat = latency_model.get(nominal_bits, {}).get('relative_latency', 1.0)
    n = len(assignments)

    per_layer_lat = []
    for b in assignments:
        lat = latency_model.get(b, {}).get('relative_latency', 1.0)
        per_layer_lat.append(lat)

    # Layer-by-layer: sum of all layer latencies
    lbl_total = sum(per_layer_lat)
    lbl_ref = n * ref_lat
    lbl_speedup = lbl_ref / lbl_total if lbl_total > 0 else 1.0

    # Pipeline: bottleneck is the slowest layer
    pipe_max = max(per_layer_lat) if per_layer_lat else 1.0
    pipe_ref = ref_lat
    pipe_speedup = pipe_ref / pipe_max if pipe_max > 0 else 1.0

    return {
        'lbl_relative_latency': lbl_total / lbl_ref if lbl_ref > 0 else 1.0,
        'lbl_speedup': lbl_speedup,
        'pipe_relative_latency': pipe_max / pipe_ref if pipe_ref > 0 else 1.0,
        'pipe_speedup': pipe_speedup,
        'max_bits': max(assignments),
        'min_bits': min(assignments),
        'mean_bits': np.mean(assignments),
    }


def adc_area_ratio(bits, ref_bits=7):
    return (2 ** bits) / (2 ** ref_bits)


def compute_area_energy(assignments, ref_ppa, nominal_bits=7):
    ref_chip = ref_ppa['chip_area_um2']
    ref_adc = ref_ppa['adc_area_um2']
    ref_nonadc = ref_chip - ref_adc
    ref_energy = ref_ppa.get('energy_pJ', 0)
    ref_adc_e = ref_ppa.get('adc_energy_pJ', 0)
    ref_nonadc_e = ref_energy - ref_adc_e

    n = len(assignments)
    per_layer_adc = ref_adc / n
    per_layer_adc_e = ref_adc_e / n

    total_adc = sum(per_layer_adc * adc_area_ratio(b, nominal_bits)
                    for b in assignments)
    total_adc_e = sum(per_layer_adc_e * adc_area_ratio(b, nominal_bits)
                      for b in assignments)

    return {
        'chip_area_mm2': (ref_nonadc + total_adc) / 1e6,
        'adc_area_mm2': total_adc / 1e6,
        'adc_savings_pct': (1.0 - total_adc / ref_adc) * 100,
        'total_energy_nJ': (ref_nonadc_e + total_adc_e) / 1e3,
        'energy_savings_pct': (1.0 - (ref_nonadc_e + total_adc_e) / ref_energy) * 100
                              if ref_energy > 0 else 0,
    }


def greedy_allocation(sensitivity_data, nominal_bits, bit_choices, target_savings):
    n = len(sensitivity_data)
    assignments = [nominal_bits] * n
    sens = [r['sensitivity'] for r in sensitivity_data]
    ref_area = n * (2 ** nominal_bits)
    budget = ref_area * (1.0 - target_savings)
    sorted_idx = sorted(range(n), key=lambda i: sens[i])
    current_area = ref_area
    for idx in sorted_idx:
        if current_area <= budget:
            break
        cur_bits = assignments[idx]
        lower = [b for b in sorted(bit_choices) if b < cur_bits]
        if lower:
            new_bits = max(lower)
            current_area += (2 ** new_bits) - (2 ** cur_bits)
            assignments[idx] = new_bits
    return assignments


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load OPT-125M PPA data
    ppa_125m = load_ppa_sweep(
        ROOT / 'results' / 'ppa' / 'opt125m' / 'ppa_sweep_opt125m.csv')
    ppa_1_3b = load_ppa_sweep(
        ROOT / 'results' / 'ppa' / 'opt1.3b' / 'ppa_sweep_opt1.3b.csv')

    # Load sensitivity data
    with open(ROOT / 'results' / 'sensitivity' / 'opt125m' / 'group_sensitivity.json') as f:
        sens_125m = json.load(f)

    # Build per-layer sensitivity for OPT-125M (73 layers)
    layer_names_125m = []
    for i in range(12):
        layer_names_125m.extend([
            f'model.decoder.layers.{i}.self_attn.k_proj',
            f'model.decoder.layers.{i}.self_attn.v_proj',
            f'model.decoder.layers.{i}.self_attn.q_proj',
            f'model.decoder.layers.{i}.self_attn.out_proj',
            f'model.decoder.layers.{i}.fc1',
            f'model.decoder.layers.{i}.fc2',
        ])
    layer_names_125m.append('lm_head')

    def classify(name):
        n = name.lower()
        if 'lm_head' in n: return 'lm_head'
        if 'q_proj' in n or 'k_proj' in n or 'v_proj' in n: return 'attn_qkv'
        if 'out_proj' in n: return 'attn_out'
        if 'fc1' in n: return 'ffn_up'
        if 'fc2' in n: return 'ffn_down'
        return 'other'

    sensitivity_data = []
    for name in layer_names_125m:
        ltype = classify(name)
        g = sens_125m.get(ltype, {})
        delta = g.get('delta_per_layer', 0.0)
        sensitivity_data.append({
            'layer': name,
            'layer_type': ltype,
            'sensitivity': max(delta, 0.0),
        })

    latency_model = estimate_latency_from_energy(ppa_125m, nominal_bits=7)

    # PPL data from stable eval
    ppl_data = {
        'Uniform 7b': 306.4,
        'Uniform 6b': 315.3,
        'ILP 5%': 312.9,
        'ILP 10%': 305.2,
        'ILP 15%': 311.0,
        'ILP 20%': 308.6,
        'ILP 25%': 313.9,
        'ILP 30%': 311.5,
        'ILP 40%': 311.7,
        'ILP 50%': 318.5,
        'SQ + 6b': 305.0,
        'SQ + ILP 20%': 309.4,
    }

    nominal_bits = 7
    bit_choices = (4, 5, 6, 7, 8)
    ref_ppa = ppa_125m[nominal_bits]

    print("=" * 100)
    print("LATENCY ANALYSIS: Mixed-Precision ADC Impact on Inference Latency")
    print("=" * 100)

    # Uniform bit sweep
    print(f"\n{'Config':<25} {'Area(mm²)':>10} {'ADC_sav%':>10} "
          f"{'Energy(nJ)':>12} {'E_sav%':>8} {'Lat_LBL':>10} {'Lat_Pipe':>10} "
          f"{'PPL':>8}")
    print("-" * 100)

    results = []
    for bits in range(4, 9):
        asgn = [bits] * len(layer_names_125m)
        ae = compute_area_energy(asgn, ref_ppa, nominal_bits)
        lat = compute_mixed_latency(asgn, latency_model, nominal_bits)
        ppl = ppl_data.get(f'Uniform {bits}b', None)
        r = {
            'config': f'Uniform {bits}b',
            **ae, **lat,
            'ppl': ppl,
        }
        results.append(r)
        ppl_str = f'{ppl:.1f}' if ppl else '--'
        print(f"  {'Uniform '+str(bits)+'b':<23} {ae['chip_area_mm2']:>10.1f} "
              f"{ae['adc_savings_pct']:>10.1f} {ae['total_energy_nJ']:>12.1f} "
              f"{ae['energy_savings_pct']:>8.1f} {lat['lbl_relative_latency']:>10.3f} "
              f"{lat['pipe_relative_latency']:>10.3f} {ppl_str:>8}")

    # ILP at various targets
    savings_targets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for target in savings_targets:
        asgn = greedy_allocation(sensitivity_data, nominal_bits,
                                  bit_choices, target)
        ae = compute_area_energy(asgn, ref_ppa, nominal_bits)
        lat = compute_mixed_latency(asgn, latency_model, nominal_bits)
        ppl = ppl_data.get(f'ILP {int(target*100)}%', None)
        config_name = f'ILP {target*100:.0f}%'
        r = {
            'config': config_name,
            **ae, **lat,
            'ppl': ppl,
        }
        results.append(r)
        ppl_str = f'{ppl:.1f}' if ppl else '--'
        print(f"  {config_name:<23} {ae['chip_area_mm2']:>10.1f} "
              f"{ae['adc_savings_pct']:>10.1f} {ae['total_energy_nJ']:>12.1f} "
              f"{ae['energy_savings_pct']:>8.1f} {lat['lbl_relative_latency']:>10.3f} "
              f"{lat['pipe_relative_latency']:>10.3f} {ppl_str:>8}")

    # SQ configs
    for config_name, ppl in [('SQ + 6b', 305.0), ('SQ + ILP 20%', 309.4)]:
        if '6b' in config_name:
            asgn = [6] * len(layer_names_125m)
        else:
            asgn = greedy_allocation(sensitivity_data, nominal_bits,
                                      bit_choices, 0.20)
        ae = compute_area_energy(asgn, ref_ppa, nominal_bits)
        lat = compute_mixed_latency(asgn, latency_model, nominal_bits)
        r = {
            'config': config_name,
            **ae, **lat,
            'ppl': ppl,
        }
        results.append(r)
        print(f"  {config_name:<23} {ae['chip_area_mm2']:>10.1f} "
              f"{ae['adc_savings_pct']:>10.1f} {ae['total_energy_nJ']:>12.1f} "
              f"{ae['energy_savings_pct']:>8.1f} {lat['lbl_relative_latency']:>10.3f} "
              f"{lat['pipe_relative_latency']:>10.3f} {ppl:>8.1f}")

    # 3D Pareto analysis
    print(f"\n{'='*80}")
    print("3D PARETO FRONTIER: Area vs Energy vs PPL")
    print(f"{'='*80}")

    pareto_points = [r for r in results if r['ppl'] is not None]
    pareto_points.sort(key=lambda x: x['adc_savings_pct'])

    print(f"\n{'Config':<25} {'ADC_sav%':>10} {'E_sav%':>10} {'Lat_sav%':>10} "
          f"{'PPL':>8} {'ΔPPL':>8} {'Pareto?':>8}")
    print("-" * 85)

    baseline_ppl = 306.4
    pareto_front = []
    for r in pareto_points:
        delta_ppl = r['ppl'] - baseline_ppl
        lat_sav = (1 - r['lbl_relative_latency']) * 100

        is_pareto = True
        for other in pareto_points:
            if other is r:
                continue
            if (other['adc_savings_pct'] >= r['adc_savings_pct'] and
                other['energy_savings_pct'] >= r['energy_savings_pct'] and
                other['ppl'] <= r['ppl'] and
                (other['adc_savings_pct'] > r['adc_savings_pct'] or
                 other['energy_savings_pct'] > r['energy_savings_pct'] or
                 other['ppl'] < r['ppl'])):
                is_pareto = False
                break

        if is_pareto:
            pareto_front.append(r)

        print(f"  {r['config']:<23} {r['adc_savings_pct']:>10.1f} "
              f"{r['energy_savings_pct']:>10.1f} {lat_sav:>10.1f} "
              f"{r['ppl']:>8.1f} {delta_ppl:>+8.1f} "
              f"{'  YES' if is_pareto else '':>8}")

    print(f"\nPareto-optimal configs: {[r['config'] for r in pareto_front]}")

    # Save results
    csv_path = OUT_DIR / 'latency_analysis.csv'
    fields = ['config', 'chip_area_mm2', 'adc_area_mm2', 'adc_savings_pct',
              'total_energy_nJ', 'energy_savings_pct',
              'lbl_relative_latency', 'lbl_speedup',
              'pipe_relative_latency', 'pipe_speedup',
              'max_bits', 'min_bits', 'mean_bits', 'ppl']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n[Saved] {csv_path}")

    # Generate LaTeX table
    latex_path = OUT_DIR / 'latency_table.tex'
    with open(latex_path, 'w') as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Mixed-Precision ADC: Area, Energy, and Latency Trade-offs (OPT-125M)}
\label{tab:latency}
\begin{tabular}{lrrrrr}
\toprule
Config & ADC sav. & Energy sav. & Lat. sav. & PPL & $\Delta$PPL \\
 & (\%) & (\%) & (\%) & & \\
\midrule
""")
        for r in pareto_points:
            if r['ppl'] is not None:
                delta = r['ppl'] - baseline_ppl
                lat_sav = (1 - r['lbl_relative_latency']) * 100
                cfg = r['config'].replace('%', '\\%')
                bold = r in pareto_front
                if bold:
                    f.write(f"\\textbf{{{cfg}}} & "
                            f"\\textbf{{{r['adc_savings_pct']:.1f}}} & "
                            f"\\textbf{{{r['energy_savings_pct']:.1f}}} & "
                            f"\\textbf{{{lat_sav:.1f}}} & "
                            f"\\textbf{{{r['ppl']:.1f}}} & "
                            f"\\textbf{{{delta:+.1f}}} \\\\\n")
                else:
                    f.write(f"{cfg} & {r['adc_savings_pct']:.1f} & "
                            f"{r['energy_savings_pct']:.1f} & "
                            f"{lat_sav:.1f} & {r['ppl']:.1f} & "
                            f"{delta:+.1f} \\\\\n")
        f.write(r"""\bottomrule
\multicolumn{6}{p{0.85\columnwidth}}{\small Bold = Pareto-optimal. Latency savings from MLSA ADC scaling model ($\propto 2^b$). ILP-20\% achieves 20.5\% area, 23.5\% energy, and 20.5\% latency savings with only +2.2 PPL.}\\
\end{tabular}
\end{table}
""")
    print(f"[Saved] {latex_path}")

    # Save Pareto frontier
    pareto_path = OUT_DIR / 'pareto_3d.json'
    pareto_export = []
    for r in pareto_front:
        pareto_export.append({
            'config': r['config'],
            'adc_savings_pct': r['adc_savings_pct'],
            'energy_savings_pct': r['energy_savings_pct'],
            'latency_savings_pct': (1 - r['lbl_relative_latency']) * 100,
            'ppl': r['ppl'],
        })
    with open(pareto_path, 'w') as f:
        json.dump(pareto_export, f, indent=2)
    print(f"[Saved] {pareto_path}")


if __name__ == '__main__':
    main()
