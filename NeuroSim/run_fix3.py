#!/usr/bin/env python3
"""
3 fixes for reviewer response:
  Fix 1: Per-layer ILP vs group-ILP comparison (measure per-layer sensitivity on subset)
  Fix 2: Surrogate ranking accuracy (enumerate top configs, measure actual PPL)
  Fix 3: 10b regime complete allocation
"""
import os, sys, json, time, itertools
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/raid/privatedata/fantao/pylib")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_CACHE"] = "/raid/privatedata/fantao/model_cache"
os.environ["HF_HOME"] = "/raid/privatedata/fantao/model_cache"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS = Path("/raid/privatedata/fantao/iccad_exp/results_fix3")
RESULTS.mkdir(parents=True, exist_ok=True)

import torch
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm_inference import load_model, load_wikitext2, make_loader
from sensitivity_analysis import (
    classify_layer, get_linear_layers,
    eval_with_assignment, measure_group_sensitivity,
    measure_per_layer_sensitivity, ilp_allocation,
)
from smooth_quant import compute_perplexity

DEVICE = 'cuda:0'
GROUPS = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']

def P(msg): print(msg, flush=True)

def ev(model, assign, cld, eld, db, ne=10):
    r = eval_with_assignment(model, assign, cld, eld, default_bits=db,
                             device=DEVICE, num_calib=4, num_eval=ne, clip_pct=99.0)
    return r[0] if isinstance(r, tuple) else r

def run_ilp(per_layer, nominal, bit_choices, target):
    r = ilp_allocation(per_layer, {b: 2**b for b in range(3,11)},
                       nominal_bits=nominal, bit_choices=bit_choices,
                       target_area_savings=target)
    if isinstance(r, list) and r and isinstance(r[0], int):
        return {per_layer[i]['layer']: r[i] for i in range(len(r))}
    return r if isinstance(r, dict) else {}


def main():
    P("Loading OPT-125M...")
    model, tok = load_model("facebook/opt-125m", device=DEVICE)
    layer_names = [n for n, _ in get_linear_layers(model)]
    data = load_wikitext2(tok, seq_len=512)
    cld = make_loader(data[:20])
    eld = make_loader(data[64:106])
    NE = 10

    ppl_7b = ev(model, {n:7 for n in layer_names}, cld, eld, 7, NE)
    P(f"  7b baseline = {ppl_7b:.1f}")

    # ═══ FIX 1: Per-layer vs group-level ILP ═══════════════════
    P("\n=== FIX 1: Per-layer vs Group-level ===")

    # Group-level sensitivity (standard, 5+1 passes)
    per_layer_group, groups = measure_group_sensitivity(
        model, layer_names, cld, eld, baseline_ppl=ppl_7b,
        nominal_bits=7, probe_bits=6, device=DEVICE,
        num_calib=4, num_eval=NE, clip_pct=99.0)

    # Per-layer sensitivity (expensive, 73+1 passes)
    P("  Measuring per-layer sensitivity (73 layers)...")
    per_layer_individual = measure_per_layer_sensitivity(
        model, layer_names, cld, eld, baseline_ppl=ppl_7b,
        nominal_bits=7, probe_bits=6, device=DEVICE,
        num_calib=4, num_eval=NE, clip_pct=99.0)

    fix1 = {"baseline_7b": ppl_7b, "comparisons": []}

    for target in [0.20, 0.30, 0.40]:
        tag = f"{int(target*100)}%"

        # Group-ILP
        assign_g = run_ilp(per_layer_group, 7, (4,5,6,7), target)
        ppl_g = ev(model, assign_g, cld, eld, 7, NE) if assign_g else ppl_7b
        dist_g = Counter(assign_g.values())

        # Per-layer-ILP
        assign_p = run_ilp(per_layer_individual, 7, (4,5,6,7), target)
        ppl_p = ev(model, assign_p, cld, eld, 7, NE) if assign_p else ppl_7b
        dist_p = Counter(assign_p.values())

        # Group-greedy (sorted by group sensitivity ascending)
        sorted_layers = sorted(layer_names,
            key=lambda n: groups.get(classify_layer(n), {}).get('delta_per_layer', 0))
        assign_gg = {n: 7 for n in layer_names}
        area = len(layer_names) * 128
        budget = area * (1 - target)
        for n in sorted_layers:
            if area <= budget: break
            assign_gg[n] = 6
            area -= (128 - 64)
        ppl_gg = ev(model, assign_gg, cld, eld, 7, NE)

        row = {
            "budget": tag,
            "group_ilp_ppl": ppl_g, "group_ilp_dist": dict(dist_g),
            "perlayer_ilp_ppl": ppl_p, "perlayer_ilp_dist": dict(dist_p),
            "group_greedy_ppl": ppl_gg,
            "group_ilp_rel": (ppl_g - ppl_7b) / ppl_7b,
            "perlayer_ilp_rel": (ppl_p - ppl_7b) / ppl_7b,
        }
        fix1["comparisons"].append(row)
        P(f"  {tag}: group-ILP={ppl_g:.1f} per-layer-ILP={ppl_p:.1f} "
          f"group-greedy={ppl_gg:.1f} gap={ppl_g-ppl_p:+.1f}")

    json.dump(fix1, open(RESULTS / "fix1_perlayer_vs_group.json", 'w'), indent=2, default=float)
    P("  Saved fix1")

    # ═══ FIX 2: Surrogate ranking accuracy ═══════════════════
    P("\n=== FIX 2: Surrogate Ranking Accuracy ===")

    nominal_area = len(layer_names) * 128
    bit_choices = [4, 5, 6, 7]
    group_sens = {k: v['delta_per_layer'] for k, v in groups.items()}

    for target in [0.20, 0.30]:
        budget = nominal_area * (1 - target)
        configs = list(itertools.product(bit_choices, repeat=len(GROUPS)))
        scored = []
        for cfg in configs:
            gb = dict(zip(GROUPS, cfg))
            area = sum(groups.get(g, {}).get('n_layers', 1) * (2**b)
                      for g, b in gb.items())
            if area > budget: continue
            pred = sum(max(0, 7-b) * max(group_sens.get(g, 0), 0)
                      for g, b in gb.items())
            scored.append({"config": gb, "pred_dppl": pred,
                          "savings": 1 - area/nominal_area})
        scored.sort(key=lambda x: x["pred_dppl"])
        P(f"  {int(target*100)}% budget: {len(scored)} feasible configs")

        # Measure top-10 by surrogate prediction
        top10 = scored[:min(10, len(scored))]
        for i, s in enumerate(top10):
            assign = {n: s["config"].get(classify_layer(n), 7) for n in layer_names}
            ppl = ev(model, assign, cld, eld, 7, NE)
            s["measured_ppl"] = ppl
            s["measured_dppl"] = ppl - ppl_7b

        # Also measure 10 random feasible configs
        import random
        random.seed(42)
        rand10 = random.sample(scored[10:], min(10, len(scored)-10))
        for s in rand10:
            assign = {n: s["config"].get(classify_layer(n), 7) for n in layer_names}
            ppl = ev(model, assign, cld, eld, 7, NE)
            s["measured_ppl"] = ppl
            s["measured_dppl"] = ppl - ppl_7b

        all_measured = top10 + rand10
        all_measured.sort(key=lambda x: x["measured_dppl"])
        measured_top5 = set(str(x["config"]) for x in all_measured[:5])
        predicted_top5 = set(str(x["config"]) for x in top10[:5])
        overlap = len(measured_top5 & predicted_top5)

        P(f"  Top-5 overlap: {overlap}/5")
        P(f"  Surrogate-best PPL: {top10[0]['measured_ppl']:.1f}")
        P(f"  Measured-best PPL: {all_measured[0]['measured_ppl']:.1f}")
        P(f"  Regret: {top10[0]['measured_ppl'] - all_measured[0]['measured_ppl']:.2f}")

    json.dump({"top10": [{k: v for k, v in s.items() if k != 'config' or True}
               for s in top10],
               "rand10": [{k: v for k, v in s.items()} for s in rand10]},
              open(RESULTS / "fix2_surrogate_ranking.json", 'w'), indent=2, default=float)
    P("  Saved fix2")

    # ═══ FIX 3: 10b regime complete allocation ═══════════════════
    P("\n=== FIX 3: 10b Regime Allocation ===")

    ppl_10b = ev(model, {n:10 for n in layer_names}, cld, eld, 10, NE)
    P(f"  10b baseline = {ppl_10b:.1f}")

    # Sensitivity at 10→9
    per_layer_10, groups_10 = measure_group_sensitivity(
        model, layer_names, cld, eld, baseline_ppl=ppl_10b,
        nominal_bits=10, probe_bits=9, device=DEVICE,
        num_calib=4, num_eval=NE, clip_pct=99.0)

    sens_10 = {k: v['delta_per_layer'] for k, v in groups_10.items()}
    P(f"  10→9 sensitivity: {sens_10}")

    fix3 = {"baseline_10b": ppl_10b, "sensitivity_10to9": sens_10, "allocations": []}

    for target in [0.20, 0.30]:
        # Measured-ILP at 10b regime
        assign = run_ilp(per_layer_10, 10, tuple(range(7, 11)), target)
        ppl = ev(model, assign, cld, eld, 10, NE) if assign else ppl_10b
        dist = Counter(assign.values()) if assign else {}

        # Uniform reduced
        ppl_uni = ev(model, {n:9 for n in layer_names}, cld, eld, 9, NE)

        # Transfer from 7→6
        per_layer_7, _ = measure_group_sensitivity(
            model, layer_names, cld, eld, baseline_ppl=ppl_7b,
            nominal_bits=7, probe_bits=6, device=DEVICE,
            num_calib=4, num_eval=NE, clip_pct=99.0)
        assign_t = run_ilp(per_layer_7, 10, tuple(range(7, 11)), target)
        ppl_t = ev(model, assign_t, cld, eld, 10, NE) if assign_t else ppl_10b

        row = {
            "budget": f"{int(target*100)}%",
            "measured_ilp_ppl": ppl, "measured_ilp_rel": (ppl-ppl_10b)/ppl_10b,
            "transfer_ilp_ppl": ppl_t, "transfer_ilp_rel": (ppl_t-ppl_10b)/ppl_10b,
            "uniform_9b_ppl": ppl_uni,
            "bit_dist": dict(dist),
        }
        fix3["allocations"].append(row)
        tag = f"{int(target*100)}%"
        P(f"  {tag}: native={ppl:.1f} transfer={ppl_t:.1f} uni-9b={ppl_uni:.1f}")

    # Also compute area savings at 10b
    chip_10b = 2822  # mm2
    adc_10b = chip_10b * 0.639
    fix3["area_10b"] = {"chip_mm2": chip_10b, "adc_mm2": adc_10b,
                        "adc_20pct_saving_mm2": adc_10b * 0.20}
    P(f"  10b ADC area: {adc_10b:.0f} mm2, 20% saving: {adc_10b*0.2:.0f} mm2")

    json.dump(fix3, open(RESULTS / "fix3_10b_regime.json", 'w'), indent=2, default=float)
    P("  Saved fix3")

    P(f"\nAll done in {(time.time()-t0)/60:.1f} min")
    for f in sorted(RESULTS.glob("*.json")):
        P(f"  {f.name} ({f.stat().st_size} bytes)")

t0 = time.time()
main()
