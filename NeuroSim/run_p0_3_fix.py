#!/usr/bin/env python3
"""
P0-3 FIX: Surrogate validation with CONSISTENT num_eval=10.
Both baseline and group sensitivity use the same eval protocol.
"""
import os, sys, json, time, itertools
from pathlib import Path

sys.path.insert(0, "/raid/privatedata/fantao/pylib")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_CACHE"] = "/raid/privatedata/fantao/model_cache"
os.environ["HF_HOME"] = "/raid/privatedata/fantao/model_cache"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS = Path("/raid/privatedata/fantao/iccad_exp/results_p0")

import torch
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm_inference import load_model, load_wikitext2, make_loader
from sensitivity_analysis import (
    classify_layer, get_linear_layers,
    eval_with_assignment, measure_group_sensitivity,
)

DEVICE = 'cuda:0'
GROUPS = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']
NUM_EVAL = 10  # match original profiling protocol exactly

def P(msg): print(msg, flush=True)

def ev(model, assign, cld, eld, db):
    r = eval_with_assignment(model, assign, cld, eld, default_bits=db,
                             device=DEVICE, num_calib=4, num_eval=NUM_EVAL, clip_pct=99.0)
    return r[0] if isinstance(r, tuple) else r

def make_group_assign(layer_names, group_bits, default_bits):
    return {n: group_bits.get(classify_layer(n), default_bits) for n in layer_names}

def main():
    P(f"P0-3 FIX: Surrogate validation with num_eval={NUM_EVAL}")
    model, tok = load_model("facebook/opt-125m", device=DEVICE)
    layer_names = [n for n, _ in get_linear_layers(model)]

    data = load_wikitext2(tok, seq_len=512)
    calib_loader = make_loader(data[:20])
    eval_loader = make_loader(data[64:106])

    # Baseline with SAME num_eval
    ppl_7b = ev(model, {n:7 for n in layer_names}, calib_loader, eval_loader, 7)
    P(f"  7b baseline (num_eval={NUM_EVAL}) = {ppl_7b:.2f}")

    # Individual group sensitivity with SAME num_eval
    _, groups = measure_group_sensitivity(
        model, layer_names, calib_loader, eval_loader, baseline_ppl=ppl_7b,
        nominal_bits=7, probe_bits=6, device=DEVICE,
        num_calib=4, num_eval=NUM_EVAL, clip_pct=99.0)
    individual = {k: v['delta_ppl'] for k, v in groups.items()}
    P(f"  Individual ΔPPL: {individual}")

    # Pairwise tests
    pairs = list(itertools.combinations(GROUPS, 2))
    results = {"baseline_7b": ppl_7b, "num_eval": NUM_EVAL,
               "individual": individual, "pairs": []}

    for g1, g2 in pairs:
        assign = make_group_assign(layer_names, {g1: 6, g2: 6}, 7)
        ppl = ev(model, assign, calib_loader, eval_loader, 7)
        d_meas = ppl - ppl_7b
        d_pred = individual.get(g1, 0) + individual.get(g2, 0)
        abs_err = abs(d_pred - d_meas)
        rel_err = abs_err / max(abs(d_meas), 0.01) * 100
        results["pairs"].append({
            "pair": [g1, g2], "ppl": ppl,
            "delta_measured": d_meas, "delta_predicted": d_pred,
            "abs_error": abs_err, "rel_error_pct": rel_err
        })
        P(f"  {g1}+{g2}: pred={d_pred:+.2f}, meas={d_meas:+.2f}, "
          f"err={abs_err:.2f} ({rel_err:.0f}%)")

    # Summary
    errors = [p["abs_error"] for p in results["pairs"]]
    rel_errors = [p["rel_error_pct"] for p in results["pairs"]]
    # Filter moderate-budget pairs (both groups positive sensitivity)
    pos_pairs = [p for p in results["pairs"]
                 if individual.get(p["pair"][0], 0) > 0 and individual.get(p["pair"][1], 0) > 0]
    pos_errors = [p["rel_error_pct"] for p in pos_pairs] if pos_pairs else rel_errors

    results["summary"] = {
        "mean_abs_error": float(np.mean(errors)),
        "max_abs_error": float(np.max(errors)),
        "mean_rel_error_all": float(np.mean(rel_errors)),
        "mean_rel_error_positive_pairs": float(np.mean(pos_errors)),
        "n_positive_sensitivity_groups": sum(1 for v in individual.values() if v > 0),
    }

    out = RESULTS / "p0_3_surrogate_fixed.json"
    json.dump(results, open(out, 'w'), indent=2)
    P(f"\n  Summary: mean_abs={results['summary']['mean_abs_error']:.2f}, "
      f"mean_rel_all={results['summary']['mean_rel_error_all']:.0f}%, "
      f"mean_rel_pos_pairs={results['summary']['mean_rel_error_positive_pairs']:.0f}%")
    P(f"  Positive-sensitivity groups: {results['summary']['n_positive_sensitivity_groups']}/5")
    P(f"  Saved → {out}")

if __name__ == "__main__":
    main()
