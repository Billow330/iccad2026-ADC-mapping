"""
neurosim_ppa.py  — NeuroSIM PPA Analysis for LLM CIM Deployment
================================================================
Generates per-model network CSV + dummy weight/input CSVs,
then invokes the NeuroSIM C++ backend to get area/energy/latency.

Supports sweeping ADC bits (via levelOutput in Param.cpp modification).

Usage:
    python3 neurosim_ppa.py --model opt125m --adc_bits 7
    python3 neurosim_ppa.py --model opt125m --sweep_adc
    python3 neurosim_ppa.py --model opt125m --mixed_precision allocation.csv
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEUROSIM_BIN  = ROOT / 'NeuroSIM' / 'main'
NEUROSIM_DIR  = ROOT / 'NeuroSIM'
PARAM_CPP     = ROOT / 'NeuroSIM' / 'Param.cpp'
RESULTS_DIR   = ROOT / 'results' / 'ppa'

# ─────────────────────────────────────────────────────────────────────────────
# Model network definitions  (OPT-style: all Linear layers)
# netStructure format per row:
#   ifmap_H, ifmap_W, in_ch, filter_H, filter_W, out_ch, stride, group
# For a Linear layer: 1,1,in_features,1,1,out_features,0,1
# stride=0 means "no pool after this layer" in NeuroSIM convention
# ─────────────────────────────────────────────────────────────────────────────

def opt125m_net():
    """OPT-125M: 12 decoder layers × 6 linear + lm_head = 73 layers.
    hidden=768, ffn=3072, heads=12, vocab=50272
    """
    layers = []
    for _ in range(12):
        # self_attn: q_proj, k_proj, v_proj, out_proj  (768→768)
        for _ in range(4):
            layers.append((1, 1, 768, 1, 1, 768, 0, 1))
        # ffn: fc1 (768→3072), fc2 (3072→768)
        layers.append((1, 1, 768,  1, 1, 3072, 0, 1))
        layers.append((1, 1, 3072, 1, 1, 768,  0, 1))
    # lm_head: 768 → 50272
    layers.append((1, 1, 768, 1, 1, 50272, 0, 1))
    return layers


def opt1_3b_net():
    """OPT-1.3B: 24 decoder layers × 6 linear + lm_head.
    hidden=2048, ffn=8192, heads=32, vocab=50272
    """
    layers = []
    for _ in range(24):
        for _ in range(4):
            layers.append((1, 1, 2048, 1, 1, 2048, 0, 1))
        layers.append((1, 1, 2048, 1, 1, 8192, 0, 1))
        layers.append((1, 1, 8192, 1, 1, 2048, 0, 1))
    layers.append((1, 1, 2048, 1, 1, 50272, 0, 1))
    return layers


def gpt2_net():
    """GPT-2 small: 12 transformer layers.
    Conv1D(768→2304) for c_attn, Conv1D(768→768) for c_proj,
    Conv1D(768→3072) for c_fc, Conv1D(3072→768) for c_proj_mlp
    + lm_head 768→50257  (tied, but include for completeness)
    """
    layers = []
    for _ in range(12):
        layers.append((1, 1, 768,  1, 1, 2304, 0, 1))  # c_attn (QKV merged)
        layers.append((1, 1, 768,  1, 1, 768,  0, 1))  # c_proj attn
        layers.append((1, 1, 768,  1, 1, 3072, 0, 1))  # c_fc
        layers.append((1, 1, 3072, 1, 1, 768,  0, 1))  # c_proj mlp
    layers.append((1, 1, 768, 1, 1, 50257, 0, 1))
    return layers


MODEL_NETS = {
    'opt125m': opt125m_net,
    'opt1.3b': opt1_3b_net,
    'gpt2':    gpt2_net,
}

# ─────────────────────────────────────────────────────────────────────────────
# Generate NeuroSIM network CSV
# ─────────────────────────────────────────────────────────────────────────────

def write_network_csv(layers, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        for row in layers:
            w.writerow(row)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Generate dummy weight/input CSV files for each layer
# Weight values: uniformly distributed in [algoWeightMin, algoWeightMax] = [-1, 1]
# normalized to [0, 2^synapseBit] in LoadInWeightData  → use 0.5*(maxcond+mincond) ≈ mid value
# We just use constant mid values — they determine toggle activity which is 2nd order
# ─────────────────────────────────────────────────────────────────────────────

def generate_layer_csvs(layers, layer_dir, weight_bits=8, input_bits=8):
    """Generate dummy weight and input CSVs for each layer.

    Weight CSV: rows=in_ch*filter_H*filter_W, cols=out_ch*ceil(weight_bits/cell_bits)
                values in [0,1] normalized (midpoint = 0.5)
    Input CSV:  rows=numInVector=1 (sequence processing, 1 token at a time for MVM)
                cols=in_ch*numBitInput  → but NeuroSIM just checks file size
    """
    os.makedirs(layer_dir, exist_ok=True)
    cell_bits = 1  # RRAM cellBit=1 from Param.cpp
    num_col_per_synapse = math.ceil(weight_bits / cell_bits)
    num_row_per_synapse = 1

    csv_pairs = []
    for i, layer in enumerate(layers):
        ifH, ifW, in_ch, kH, kW, out_ch, stride, group = layer
        # Weight matrix dimensions
        weight_rows = in_ch * kH * kW * num_row_per_synapse
        weight_cols = out_ch * num_col_per_synapse
        # Input vector dimensions:
        # numInVector = (ifH-kH+1)/stride * (ifW-kW+1)/stride  (=1 for Linear)
        # NeuroSIM's CopyInput accesses inputVector[i*desiredTileSizeCM + k]
        # for i=0..numTileRow-1, k=0..numRowMatrix-1.
        # Safe: provide weight_rows rows so tiled access never overflows.
        stride_act = max(1, stride)
        num_in_vector = max(1, ((ifH - kH + 1) // stride_act) * ((ifW - kW + 1) // stride_act))
        input_rows = weight_rows   # padded to weight_rows for safe CopyInput access
        input_cols = max(1, num_in_vector * input_bits)  # bitwise input width

        # Write weight CSV (midpoint conductance = 0.5)
        w_path = layer_dir / f'weight_layer{i}.csv'
        with open(w_path, 'w', newline='') as f:
            wr = csv.writer(f)
            val = 0.5  # normalized midpoint
            for _ in range(weight_rows):
                wr.writerow([f'{val:.4f}'] * weight_cols)

        # Write input CSV (midpoint input = 0.5)
        i_path = layer_dir / f'input_layer{i}.csv'
        with open(i_path, 'w', newline='') as f:
            wr = csv.writer(f)
            val = 0.5
            for _ in range(input_rows):
                wr.writerow([f'{val:.4f}'] * max(1, input_cols))

        csv_pairs.append((w_path, i_path))

    return csv_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Modify Param.cpp to set levelOutput (ADC resolution) then recompile
# ─────────────────────────────────────────────────────────────────────────────

def set_adc_bits_and_recompile(adc_bits):
    """Patch Param.cpp levelOutput and recompile NeuroSIM binary."""
    level_output = 2 ** adc_bits

    param_text = PARAM_CPP.read_text()

    import re
    # Replace the levelOutput line
    new_text = re.sub(
        r'(levelOutput\s*=\s*)\d+(\s*;)',
        f'\\g<1>{level_output}\\2',
        param_text
    )
    if new_text == param_text:
        print(f'[NeuroSIM] WARNING: could not patch levelOutput in Param.cpp')
        return False

    PARAM_CPP.write_text(new_text)

    # Recompile
    print(f'[NeuroSIM] Recompiling for ADC {adc_bits}-bit (levelOutput={level_output})...')
    make_result = subprocess.run(
        ['make', '-C', str(NEUROSIM_DIR), '-j4'],
        capture_output=True, text=True, timeout=120
    )
    if make_result.returncode != 0:
        # Try g++ direct compile
        cpp_files = list(NEUROSIM_DIR.glob('*.cpp'))
        compile_cmd = ['g++', '-O2', '-std=c++17', '-o', str(NEUROSIM_BIN)] + [str(f) for f in cpp_files]
        r = subprocess.run(compile_cmd, capture_output=True, text=True,
                          cwd=str(NEUROSIM_DIR), timeout=300)
        if r.returncode != 0:
            print(f'[NeuroSIM] Compile failed:\n{r.stderr[:500]}')
            return False
    print(f'[NeuroSIM] Compiled OK.')
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Run NeuroSIM and parse output
# ─────────────────────────────────────────────────────────────────────────────

def run_neurosim(net_csv, csv_pairs, weight_bits=8, input_bits=8,
                 subarray=128, parallel_read=128):
    """Invoke NeuroSIM binary and parse PPA results."""
    cmd = [
        str(NEUROSIM_BIN),
        str(net_csv),
        str(weight_bits),
        str(input_bits),
        str(subarray),
        str(parallel_read),
    ]
    for w_path, i_path in csv_pairs:
        cmd += [str(w_path), str(i_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                               cwd=str(ROOT))
        output = result.stdout
    except subprocess.TimeoutExpired:
        print('[NeuroSIM] Timeout after 600s')
        return None
    except Exception as e:
        print(f'[NeuroSIM] Error: {e}')
        return None

    if result.returncode != 0:
        print(f'[NeuroSIM] CRASH (rc={result.returncode}): {result.stderr[-300:]}')
        return None

    return parse_neurosim_output(output)


def _parse_val(token):
    """Parse a numeric token that may have unit suffix like um^2, pJ, ns, uW."""
    import re
    # Strip trailing unit suffix (letters, ^, /)
    m = re.match(r'([+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d+\.?\d*)', token)
    if m:
        return float(m.group(1))
    return None


def parse_neurosim_output(output):
    """Parse NeuroSIM stdout into a dict of PPA metrics.

    NeuroSIM output format examples:
      ChipArea : 9.82688e+06um^2
      Total ADC (or S/As ...) Area on chip : 2.31324e+06um^2
      Chip total readDynamicEnergy is: 108271pJ
      Energy Efficiency TOPS/W (Layer-by-Layer Process): 0
    """
    ppa = {}
    lines = output.split('\n')
    for line in lines:
        line = line.strip()
        last = line.split()[-1] if line.split() else ''
        val = _parse_val(last)

        if line.startswith('ChipArea') and val is not None:
            ppa['chip_area_um2'] = val
        elif 'Total ADC' in line and 'Area' in line and val is not None:
            ppa['adc_area_um2'] = val
        elif 'Chip total CIM array' in line and val is not None:
            ppa['array_area_um2'] = val
        elif 'Total IC Area' in line and val is not None:
            ppa['ic_area_um2'] = val
        elif 'Total Accumulation Circuits' in line and 'on chip' in line and val is not None:
            ppa['accum_area_um2'] = val
        elif 'pipeline-system-clock-cycle' in line and val is not None:
            ppa['latency_ns'] = val
        elif 'pipeline-system readDynamic' in line and val is not None:
            ppa['energy_pJ'] = val
        elif 'layer-by-layer readLatency' in line and val is not None:
            ppa['latency_ns'] = val
        elif 'Chip total readDynamicEnergy' in line and val is not None:
            ppa['energy_pJ'] = val
        elif 'TOPS/W (Pipelined' in line and val is not None:
            ppa['tops_w'] = val
        elif 'TOPS/W (Layer' in line and val is not None:
            ppa['tops_w'] = val
        elif 'ADC (or S/As and precharger for SRAM) readDynamicEnergy' in line \
             and line.startswith('-------'):
            # Chip-level ADC energy breakdown (last chip-level occurrence)
            if val is not None:
                ppa['adc_energy_pJ'] = ppa.get('adc_energy_pJ', 0) + val

    # ADC area fraction
    if 'chip_area_um2' in ppa and 'adc_area_um2' in ppa:
        ppa['adc_area_pct'] = 100.0 * ppa['adc_area_um2'] / ppa['chip_area_um2']

    return ppa if ppa else None


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer PPA using a sub-network CSV (one layer at a time)
# This is how we get per-layer ADC area to drive mixed-precision allocation
# ─────────────────────────────────────────────────────────────────────────────

def run_neurosim_per_layer(layers, layer_dir, weight_bits=8, input_bits=8,
                           subarray=128, parallel_read=128):
    """Run NeuroSIM for each individual layer to get per-layer PPA."""
    results = []
    csv_pairs = generate_layer_csvs(layers, layer_dir, weight_bits, input_bits)

    for i, (layer, (w_path, i_path)) in enumerate(zip(layers, csv_pairs)):
        # Write single-layer network CSV
        single_csv = layer_dir / f'net_layer{i}.csv'
        write_network_csv([layer], single_csv)

        ppa = run_neurosim(single_csv, [(w_path, i_path)],
                          weight_bits, input_bits, subarray, parallel_read)
        if ppa:
            ppa['layer_idx'] = i
            ppa['layer'] = layer
            results.append(ppa)
            print(f'  Layer {i:3d}: area={ppa.get("chip_area_um2",0)/1e6:.2f}mm², '
                  f'ADC_area={ppa.get("adc_area_pct",0):.1f}%, '
                  f'energy={ppa.get("energy_pJ",0):.1e}pJ')
        else:
            print(f'  Layer {i:3d}: FAILED')
            results.append({'layer_idx': i, 'layer': layer})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Mixed-precision ADC allocation
# ─────────────────────────────────────────────────────────────────────────────

def mixed_precision_allocation(outlier_csv_path, target_savings_pct=30,
                                min_adc_bits=4, max_adc_bits=8,
                                nominal_adc_bits=7):
    """
    Assign per-layer ADC bits based on outlier characterization.

    Strategy:
    1. Layers with low saturation rate → can use fewer ADC bits (save area)
    2. Layers with high saturation (outlier) → keep/add bits for accuracy
    3. Total ADC area budget = nominal area × (1 - target_savings_pct/100)

    Returns: list of (layer_name, assigned_adc_bits, sat_rate, act_max)
    """
    rows = []
    with open(outlier_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'layer':      r['layer'],
                'sat_rate':   float(r['sat_rate_worst']),
                'act_max':    float(r['act_max']),
                'overhead':   int(r['adc_overhead_bits']),
            })

    n = len(rows)
    # Start: everyone at nominal
    assignments = [nominal_adc_bits] * n

    # ADC area ∝ 2^(adc_bits) (SAR ADC area scales roughly linearly with levels,
    # or ~ bits^2 for area; we use the simpler linear-in-levels model here)
    # Relative area: 2^bits / 2^nominal
    def rel_area(bits):
        return (2 ** bits) / (2 ** nominal_adc_bits)

    total_nominal_area = sum(rel_area(nominal_adc_bits) for _ in rows)
    budget = total_nominal_area * (1.0 - target_savings_pct / 100.0)

    # Sort layers by sat_rate ascending (least saturating → candidates to reduce)
    sorted_idx = sorted(range(n), key=lambda i: rows[i]['sat_rate'])

    current_area = total_nominal_area
    # Greedily reduce ADC bits on low-saturation layers
    for idx in sorted_idx:
        if current_area <= budget:
            break
        sat = rows[idx]['sat_rate']
        cur = assignments[idx]
        # Only reduce if saturation is below threshold
        if sat < 0.05 and cur > min_adc_bits:
            new_bits = cur - 1
            delta = rel_area(new_bits) - rel_area(cur)  # negative
            current_area += delta
            assignments[idx] = new_bits

    # For layers with high saturation + high overhead: optionally add 1 bit
    for idx in range(n):
        if rows[idx]['overhead'] > 3 and assignments[idx] < max_adc_bits:
            old_area = current_area
            new_area = current_area + rel_area(assignments[idx]+1) - rel_area(assignments[idx])
            if new_area <= budget * 1.1:  # allow 10% slack for high-sat layers
                assignments[idx] += 1
                current_area = new_area

    savings = (1.0 - current_area / total_nominal_area) * 100
    print(f'[MixedPrec] Nominal area={total_nominal_area:.1f}, '
          f'Allocated area={current_area:.1f}, Savings={savings:.1f}%')

    result = []
    for i, row in enumerate(rows):
        result.append({
            'layer':     row['layer'],
            'adc_bits':  assignments[i],
            'sat_rate':  row['sat_rate'],
            'act_max':   row['act_max'],
            'overhead':  row['overhead'],
        })
    return result, savings


def save_allocation(allocation, out_path):
    """Save mixed-precision ADC allocation to CSV."""
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['layer', 'adc_bits', 'sat_rate', 'act_max', 'overhead'])
        w.writeheader()
        w.writerows(allocation)
    print(f'[MixedPrec] Allocation saved to {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fast sweep: run per unique layer type, then aggregate
# OPT-125M has only 4 unique shapes → 4 runs instead of 73
# ─────────────────────────────────────────────────────────────────────────────

def _unique_layer_types(layers):
    """Return list of (layer_spec, count) for unique layer shapes."""
    from collections import Counter
    counts = Counter(layers)
    return [(spec, counts[spec]) for spec in sorted(counts.keys())]


def sweep_adc_ppa_fast(model_name, layers, layer_dir, output_dir,
                       adc_range=range(3, 11), weight_bits=8, input_bits=8,
                       subarray=128, parallel_read=128):
    """Fast ADC sweep by running unique layer types individually.

    Instead of running all 73 layers together (slow), runs each unique
    layer shape once and multiplies by count for aggregate PPA.
    """
    os.makedirs(output_dir, exist_ok=True)
    unique_types = _unique_layer_types(layers)
    print(f'[FastSweep] {model_name}: {len(layers)} total layers, '
          f'{len(unique_types)} unique types')
    for spec, cnt in unique_types:
        print(f'  {spec} × {cnt}')

    # Pre-generate CSVs for all unique layer types
    unique_layer_specs = [spec for spec, _ in unique_types]
    csv_pairs = generate_layer_csvs(unique_layer_specs, layer_dir,
                                    weight_bits, input_bits)
    # Write single-layer net CSVs for each unique type
    single_net_csvs = []
    for i, spec in enumerate(unique_layer_specs):
        p = layer_dir / f'net_unique{i}.csv'
        write_network_csv([spec], p)
        single_net_csvs.append(p)

    results = []
    original_param = PARAM_CPP.read_text()

    try:
        for adc_bits in adc_range:
            print(f'\n[FastSweep] ADC {adc_bits}-bit...', flush=True)
            ok = set_adc_bits_and_recompile(adc_bits)
            if not ok:
                print(f'  SKIP: compile failed')
                continue

            # Run each unique layer type
            agg = {}  # aggregated PPA for this adc_bits
            all_ok = True
            for i, (spec, count) in enumerate(unique_types):
                ppa = run_neurosim(single_net_csvs[i], [csv_pairs[i]],
                                  weight_bits, input_bits, subarray, parallel_read)
                if ppa is None:
                    print(f'  FAILED for layer type {spec}')
                    all_ok = False
                    break
                # Accumulate (multiply by layer count)
                for k, v in ppa.items():
                    if isinstance(v, (int, float)) and k not in ('adc_area_pct',):
                        agg[k] = agg.get(k, 0) + v * count

            if not all_ok:
                continue

            # Recompute area percentage from aggregated totals
            if 'chip_area_um2' in agg and 'adc_area_um2' in agg:
                agg['adc_area_pct'] = 100.0 * agg['adc_area_um2'] / agg['chip_area_um2']
            agg['adc_bits'] = adc_bits
            agg['model'] = model_name
            results.append(agg)
            print(f'  ChipArea={agg.get("chip_area_um2",0)/1e6:.3f}mm², '
                  f'ADC_area={agg.get("adc_area_pct",0):.1f}%, '
                  f'energy={agg.get("energy_pJ",0)/1e3:.1f}nJ')
    finally:
        PARAM_CPP.write_text(original_param)
        subprocess.run(['make', '-C', str(NEUROSIM_DIR), '-j4'],
                      capture_output=True, timeout=120)

    # Save sweep results
    if results:
        out_csv = Path(output_dir) / f'ppa_sweep_{model_name}.csv'
        fields = ['model', 'adc_bits', 'chip_area_um2', 'adc_area_um2', 'adc_area_pct',
                  'array_area_um2', 'accum_area_um2', 'ic_area_um2',
                  'adc_energy_pJ', 'energy_pJ']
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
        print(f'\n[FastSweep] Results saved to {out_csv}')
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Estimate mixed-precision chip area from per-layer areas
# (Since running NeuroSIM per-allocation config is slow, we use the area model)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_mixed_ppa(sweep_results, allocation):
    """
    Estimate total chip area for a mixed-precision allocation,
    given per-ADC-bits PPA data from sweep.

    This is an area additive model:
      total_adc_area = sum_i(adc_area_per_layer[i] * scale_factor(bits_i))

    scale_factor: assume ADC area ∝ 2^bits (linear in output levels)
    """
    # Build lookup: adc_bits → adc_area_pct, chip_area
    adc_area_map = {}
    for r in sweep_results:
        adc_area_map[r['adc_bits']] = {
            'adc_area_um2': r.get('adc_area_um2', 0),
            'chip_area_um2': r.get('chip_area_um2', 0),
        }

    if not adc_area_map:
        return None

    # Get reference (7-bit) areas
    ref_bits = 7
    ref = adc_area_map.get(ref_bits, list(adc_area_map.values())[0])
    ref_adc_area = ref['adc_area_um2']
    ref_chip_area = ref['chip_area_um2']
    ref_non_adc = ref_chip_area - ref_adc_area

    n = len(allocation)
    # Per-layer ADC area fraction assuming uniform distribution among layers
    per_layer_adc_ref = ref_adc_area / n if n > 0 else 0

    # Scale each layer's ADC area by its assigned bits relative to ref
    total_adc_area = 0
    for a in allocation:
        bits = a['adc_bits']
        # Scale factor: levels ratio
        scale = (2 ** bits) / (2 ** ref_bits)
        total_adc_area += per_layer_adc_ref * scale

    # Non-ADC area stays the same
    total_chip_area = ref_non_adc + total_adc_area
    adc_savings_pct = (1.0 - total_adc_area / ref_adc_area) * 100

    return {
        'chip_area_mm2': total_chip_area / 1e6,
        'adc_area_mm2': total_adc_area / 1e6,
        'adc_area_pct': 100 * total_adc_area / total_chip_area,
        'adc_savings_vs_7b_pct': adc_savings_pct,
        'ref_chip_area_mm2': ref_chip_area / 1e6,
        'ref_adc_area_pct': 100 * ref_adc_area / ref_chip_area,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='opt125m',
                   choices=['opt125m', 'opt1.3b', 'gpt2'])
    p.add_argument('--adc_bits', type=int, default=7)
    p.add_argument('--weight_bits', type=int, default=8)
    p.add_argument('--input_bits', type=int, default=8)
    p.add_argument('--subarray', type=int, default=128)
    p.add_argument('--parallel_read', type=int, default=128)
    p.add_argument('--sweep_adc', action='store_true',
                   help='Sweep ADC bits 3-10 and collect PPA')
    p.add_argument('--adc_min', type=int, default=3)
    p.add_argument('--adc_max', type=int, default=10)
    p.add_argument('--mixed_precision', metavar='OUTLIER_CSV',
                   help='Run mixed-precision allocation from outlier CSV')
    p.add_argument('--savings_pct', type=float, default=30.0,
                   help='Target ADC area savings % for mixed-precision')
    p.add_argument('--output_dir', default=None)
    p.add_argument('--test_only', action='store_true',
                   help='Only test if NeuroSIM runs correctly (no recompile)')
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = str(RESULTS_DIR / args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    layers = MODEL_NETS[args.model]()
    print(f'[NeuroSIM PPA] Model: {args.model}, {len(layers)} layers')
    print(f'[NeuroSIM PPA] Binary: {NEUROSIM_BIN}')

    # Layer CSV directory
    layer_dir = Path(args.output_dir) / 'layer_data'
    os.makedirs(layer_dir, exist_ok=True)

    # Write network CSV (also copy to NeuroSIM dir)
    net_csv = layer_dir / f'NetWork_{args.model}.csv'
    write_network_csv(layers, net_csv)
    # Also place in NeuroSIM dir for convenience
    shutil.copy(net_csv, NEUROSIM_DIR / f'NetWork_{args.model}.csv')
    print(f'[NeuroSIM PPA] Network CSV: {net_csv} ({len(layers)} layers)')

    if args.test_only:
        # Quick test: generate CSVs for first 2 layers and try running
        csv_pairs = generate_layer_csvs(layers[:2], layer_dir, args.weight_bits, args.input_bits)
        test_csv = layer_dir / 'net_test2.csv'
        write_network_csv(layers[:2], test_csv)
        ppa = run_neurosim(test_csv, csv_pairs[:2], args.weight_bits, args.input_bits,
                          args.subarray, args.parallel_read)
        if ppa:
            print(f'[TEST] SUCCESS: {ppa}')
        else:
            print(f'[TEST] FAILED')
        return

    if args.sweep_adc:
        print(f'\n[Sweep] ADC bits {args.adc_min}-{args.adc_max} (fast mode: unique layers only)')
        results = sweep_adc_ppa_fast(
            args.model, layers, layer_dir, args.output_dir,
            adc_range=range(args.adc_min, args.adc_max + 1),
            weight_bits=args.weight_bits, input_bits=args.input_bits,
            subarray=args.subarray, parallel_read=args.parallel_read
        )
        if results:
            print('\n[Sweep Summary]')
            print(f'{"ADC":>4} {"Area(mm²)":>12} {"ADC_area(mm²)":>14} {"ADC%":>8} {"Energy(nJ)":>12}')
            for r in results:
                print(f'{r["adc_bits"]:>4} '
                      f'{r.get("chip_area_um2",0)/1e6:>12.3f} '
                      f'{r.get("adc_area_um2",0)/1e6:>14.3f} '
                      f'{r.get("adc_area_pct",0):>8.1f} '
                      f'{r.get("energy_pJ",0)/1e3:>12.1f}')

    elif args.mixed_precision:
        print(f'\n[MixedPrec] Loading outlier data from {args.mixed_precision}')
        allocation, savings = mixed_precision_allocation(
            args.mixed_precision,
            target_savings_pct=args.savings_pct,
        )
        alloc_csv = Path(args.output_dir) / f'mixed_precision_allocation_{args.model}.csv'
        save_allocation(allocation, alloc_csv)

        # Show allocation statistics
        from collections import Counter
        bits_dist = Counter(a['adc_bits'] for a in allocation)
        print(f'\n[MixedPrec] Bit assignment distribution:')
        for bits in sorted(bits_dist.keys()):
            print(f'  {bits}-bit: {bits_dist[bits]} layers ({100*bits_dist[bits]/len(allocation):.0f}%)')
        print(f'  Estimated area savings: {savings:.1f}%')

    else:
        # Single ADC bits run
        print(f'\n[Single] ADC {args.adc_bits}-bit')
        print(f'[Single] Generating layer CSVs...')
        csv_pairs = generate_layer_csvs(layers, layer_dir, args.weight_bits, args.input_bits)

        ok = set_adc_bits_and_recompile(args.adc_bits)
        if not ok:
            # Try without recompile
            print('[Single] Using current binary (no recompile)...')

        ppa = run_neurosim(net_csv, csv_pairs, args.weight_bits, args.input_bits,
                          args.subarray, args.parallel_read)
        if ppa:
            print(f'\n[PPA Results] {args.model} @ {args.adc_bits}-bit ADC:')
            for k, v in ppa.items():
                if isinstance(v, float):
                    print(f'  {k}: {v:.4g}')
            out_csv = Path(args.output_dir) / f'ppa_{args.model}_adc{args.adc_bits}.csv'
            with open(out_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(ppa.keys()))
                w.writeheader()
                w.writerow(ppa)
            print(f'[PPA] Saved to {out_csv}')
        else:
            print('[PPA] FAILED: NeuroSIM returned no results')


if __name__ == '__main__':
    main()
