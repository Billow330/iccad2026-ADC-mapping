#!/usr/bin/env python3
"""INT8-only control + Random x20 using SAME split as Table 3 (batches 64-164)."""
import sys, json, os, torch, numpy as np, random
from pathlib import Path
from collections import defaultdict

ROOT = Path("/raid/privatedata/fantao/iccad_exp")
sys.path.insert(0, str(ROOT))

from llm_inference import load_wikitext2, make_loader
from sensitivity_analysis import eval_with_assignment, ilp_allocation
from smooth_quant import compute_perplexity
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT = ROOT / "results_control"; OUT.mkdir(exist_ok=True)
MODEL = "facebook/opt-125m"
CACHE = str(ROOT / "model_cache")
DEVICE = "cuda"

def P(msg): print(msg, flush=True)
def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def classify_layer(name):
    nl = name.lower()
    if "q_proj" in nl or "k_proj" in nl or "v_proj" in nl: return "attn_qkv"
    if "out_proj" in nl: return "attn_out"
    if "fc1" in nl: return "ffn_up"
    if "fc2" in nl: return "ffn_down"
    if "lm_head" in nl: return "lm_head"
    return "other"

def ev(model, assign, cld, eld, db, n_c=4, n_e=100, seed=0):
    set_seed(seed)
    r = eval_with_assignment(model, assign, cld, eld, default_bits=db,
                             device=DEVICE, num_calib=n_c, num_eval=n_e, clip_pct=99.0)
    return r[0] if isinstance(r, tuple) else r

def main():
    P("="*60)
    P("CONTROL EXPERIMENT: Table 3 split (batches 64-164)")
    P("="*60)

    P("\n[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=CACHE, torch_dtype=torch.float32)
    model.to(DEVICE).eval()

    P("[2] Loading data with Table 3 split...")
    data = load_wikitext2(tokenizer, seq_len=512)
    P(f"  Total sequences: {len(data)}")
    calib_data = data[:20]
    eval_data = data[64:164]
    cld = make_loader(calib_data, batch_size=1)
    eld = make_loader(eval_data, batch_size=1)
    P(f"  Calib: {len(calib_data)}, Eval: {len(eval_data)}")

    layer_names = [n for n, m in model.named_modules()
                   if hasattr(m, 'weight') and m.weight.dim() == 2]
    R = {}

    # FP32
    P("\n[3] FP32 PPL...")
    R["fp32"] = round(compute_perplexity(model, eld, device=DEVICE, max_batches=100), 1)
    P(f"  FP32: {R['fp32']}")

    # INT8 + various ADC
    for bits in [7, 10, 12]:
        uni = {n: bits for n in layer_names}
        ppl = ev(model, uni, cld, eld, bits, n_e=100)
        R[f"int8_adc{bits}b"] = round(ppl, 1)
        P(f"  INT8 + {bits}b ADC: {R[f'int8_adc{bits}b']}")

    # Verify 7b matches Table 3
    P(f"\n  Verification: INT8+7b = {R['int8_adc7b']} (Table 3 says 306.4)")

    # Random x20
    P("\n[4] Random allocation x20...")
    # First get sensitivity to build per_layer structure
    per_layer = []
    for n in layer_names:
        per_layer.append({'layer': n, 'layer_type': classify_layer(n),
                          'sensitivity': 1.0, 'delta_ppl': 1.0,
                          'n_layers': 1, 'nominal_bits': 7, 'probe_bits': 6})

    rand_ppls = []
    for seed in range(20):
        set_seed(seed + 200)
        rand_assign = {n: random.choice([6, 7]) for n in layer_names}
        ppl = ev(model, rand_assign, cld, eld, 7, n_e=100, seed=seed+200)
        rand_ppls.append(round(ppl, 1))
        P(f"  seed {seed}: {ppl:.1f}")

    R["random_ppls"] = rand_ppls
    R["random_mean"] = round(np.mean(rand_ppls), 1)
    R["random_std"] = round(np.std(rand_ppls), 1)
    P(f"\n  Random mean: {R['random_mean']} +/- {R['random_std']}")

    # Save
    P("\n[5] Saving...")
    with open(OUT / "control_results.json", "w") as f:
        json.dump(R, f, indent=2)

    P("\n" + "="*60)
    P("RESULTS")
    P("="*60)
    for k, v in sorted(R.items()):
        if k != "random_ppls":
            P(f"  {k}: {v}")

if __name__ == "__main__":
    main()
