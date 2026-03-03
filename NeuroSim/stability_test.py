"""
stability_test.py - Repeat key PPL evaluations 3x with different random seeds
Reports mean ± std for: Uniform-7b, ILP-20%, SQ+6b

Uses different calib batch subsets (seeds) as the source of variability.
"""
import os, sys, json
from pathlib import Path

os.environ.pop('HF_HUB_OFFLINE', None)
os.environ['HF_DATASETS_OFFLINE'] = '1'

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import torch
import numpy as np
from torch.utils.data import Subset

from llm_inference import load_model, load_wikitext2, make_loader, CIMNoiseHook
from sensitivity_analysis import PerLayerCIMHook, get_linear_layers, classify_layer
from smooth_quant import CIMSmoothQuant, compute_perplexity

MODEL_NAME = 'facebook/opt-125m'
CACHE_DIR  = './model_cache'
OUT_DIR    = Path('results/stable/opt125m')
N_SEEDS    = 3
NUM_EVAL   = 100
NUM_CALIB  = 4

# Known ILP assignment from previous run
ILP_ASSIGNMENT_LTYPE = {
    'attn_qkv': 6,   # least sensitive → 6b
    'ffn_up':   6,
    'attn_out': 7,   # sensitive → keep 7b
    'ffn_down': 7,
    'lm_head':  7,
}

def main():
    print("="*60)
    print("Stability Test: 3 seeds × 3 configs (OPT-125M)")
    print("="*60)

    model, tok = load_model(MODEL_NAME, CACHE_DIR, 'cpu')
    calib_data = load_wikitext2(tok, 512, split='train')
    eval_data  = load_wikitext2(tok, 512, split='test')
    eval_loader = make_loader(Subset(eval_data, range(NUM_EVAL)))

    layers = get_linear_layers(model)
    layer_names = [n for n, _ in layers]

    # Build ILP assignment dict
    ilp_assignment = {}
    for name in layer_names:
        ltype = classify_layer(name)
        ilp_assignment[name] = ILP_ASSIGNMENT_LTYPE.get(ltype, 7)

    # 3 different calib batch offsets as "seeds"
    calib_offsets = [0, 4, 8]  # batches 0-3, 4-7, 8-11

    results = {
        'uniform_7b': [],
        'ilp_20pct':  [],
        'sq_6b':      [],
    }

    for seed_idx, offset in enumerate(calib_offsets):
        print(f"\n--- Seed {seed_idx+1}/3 (calib batches {offset}-{offset+NUM_CALIB-1}) ---")
        calib_loader = make_loader(Subset(calib_data, range(offset, offset + NUM_CALIB)))

        # Config 1: Uniform 7b
        print("[Uniform 7b]")
        hook7 = CIMNoiseHook(model, adc_bits=7, weight_bits=8, input_bits=8)
        hook7.calibrate(calib_loader, clip_percentile=99.0)
        hook7.install()
        ppl7 = compute_perplexity(model, eval_loader)
        hook7.remove()
        print(f"  PPL = {ppl7:.2f}")
        results['uniform_7b'].append(ppl7)

        # Config 2: ILP 20%
        print("[ILP 20%]")
        hook_ilp = PerLayerCIMHook(model, bit_assignment=ilp_assignment, default_bits=7)
        hook_ilp.calibrate(calib_loader)
        hook_ilp.install()
        ppl_ilp = compute_perplexity(model, eval_loader)
        hook_ilp.remove()
        print(f"  PPL = {ppl_ilp:.2f}")
        results['ilp_20pct'].append(ppl_ilp)

        # Config 3: SQ + 6b (fresh model each time for clean SQ)
        print("[SQ + 6b]")
        import copy
        model_sq = copy.deepcopy(model)
        sq = CIMSmoothQuant(weight_bits=8, input_bits=8, adc_bits=6,
                            adc_clip_pct=99.0, sat_lambda=0.5, verbose=False)
        sq.fit(model_sq, calib_loader, num_batches=NUM_CALIB, device='cpu', task='lm')
        hook6 = CIMNoiseHook(model_sq, adc_bits=6, weight_bits=8, input_bits=8)
        hook6.calibrate(calib_loader, clip_percentile=99.0)
        hook6.install()
        ppl_sq = compute_perplexity(model_sq, eval_loader)
        hook6.remove()
        del model_sq
        print(f"  PPL = {ppl_sq:.2f}")
        results['sq_6b'].append(ppl_sq)

    # Summary
    print("\n" + "="*60)
    print("STABILITY SUMMARY (mean ± std, 100-batch eval, 3 seeds)")
    print("-"*60)
    output = {}
    for cfg, ppls in results.items():
        mean = np.mean(ppls)
        std  = np.std(ppls)
        print(f"  {cfg:<18}: {mean:.1f} ± {std:.1f}  (runs: {[f'{p:.1f}' for p in ppls]})")
        output[cfg] = {'mean': mean, 'std': std, 'runs': ppls}

    out_path = OUT_DIR / 'stability_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")

if __name__ == '__main__':
    main()
