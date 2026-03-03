"""
run_zeroshot.py - Zero-shot evaluation using lm-eval Python API
Evaluates OPT-125M under: clean / CIM-7b / SQ+6b / ILP-20%
"""
import os, sys, json, copy
from pathlib import Path

# Use cached datasets without needing network access
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ['HF_DATASETS_OFFLINE'] = '1'

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import torch
from torch.utils.data import Subset

# lm_eval Python API
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, tasks

from llm_inference import load_model, load_wikitext2, make_loader, CIMNoiseHook
from smooth_quant import CIMSmoothQuant, compute_perplexity
from sensitivity_analysis import (
    PerLayerCIMHook, get_linear_layers, classify_layer,
    ilp_allocation, load_ppa_sweep
)

SNAP = 'model_cache/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6'
TASKS = ['hellaswag', 'winogrande']
OUT_DIR = Path('results/zeroshot/opt125m')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def eval_model(hf_model, tokenizer, label):
    """Wrap patched HF model in HFLM and run lm_eval."""
    lm = HFLM(pretrained=hf_model, tokenizer=tokenizer, batch_size=4)
    task_manager = tasks.TaskManager()
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=TASKS,
        num_fewshot=0,
        limit=500,
        task_manager=task_manager,
    )
    acc = {}
    for task in TASKS:
        r = results['results'].get(task, {})
        for metric in ['acc,none', 'acc_norm,none']:
            if metric in r:
                acc[task] = r[metric]
                break
    print(f"  [{label}] {acc}")
    return acc

def main():
    print("Loading model...")
    model, tok = load_model('facebook/opt-125m', './model_cache', 'cpu')
    calib_data = load_wikitext2(tok, 512, split='train')
    eval_data  = load_wikitext2(tok, 512, split='test')
    calib_loader = make_loader(Subset(calib_data, range(4)))
    eval_loader_ppl = make_loader(Subset(eval_data, range(100)))

    results = {}

    # ── 1. Clean FP32 ──────────────────────────────────────────────────────────
    print("\n[1/4] Clean FP32...")
    ppl_clean = compute_perplexity(model, eval_loader_ppl)
    print(f"  clean PPL = {ppl_clean:.2f}")
    acc_clean = eval_model(model, tok, 'clean')
    results['clean'] = {'ppl': ppl_clean, 'acc': acc_clean}

    # ── 2. CIM 7b uniform ──────────────────────────────────────────────────────
    print("\n[2/4] CIM 7b uniform...")
    hook7 = CIMNoiseHook(model, adc_bits=7, weight_bits=8, input_bits=8)
    hook7.calibrate(calib_loader, clip_percentile=99.0)
    hook7.install()
    ppl_cim7 = compute_perplexity(model, eval_loader_ppl)
    print(f"  CIM-7b PPL = {ppl_cim7:.2f}")
    acc_cim7 = eval_model(model, tok, 'cim7b')
    hook7.remove()
    results['cim_7b'] = {'ppl': ppl_cim7, 'acc': acc_cim7}

    # ── 3. SQ + CIM 6b ─────────────────────────────────────────────────────────
    print("\n[3/4] SQ + CIM 6b...")
    model_sq, tok = load_model('facebook/opt-125m', './model_cache', 'cpu')
    smoother = CIMSmoothQuant(weight_bits=8, input_bits=6, adc_bits=6, sat_lambda=0.5)
    smoother.fit(model_sq, make_loader(Subset(calib_data, range(4))), num_batches=4, device='cpu')
    hook6 = CIMNoiseHook(model_sq, adc_bits=6, weight_bits=8, input_bits=8)
    hook6.calibrate(make_loader(Subset(calib_data, range(4))), clip_percentile=99.0)
    hook6.install()
    ppl_sq6 = compute_perplexity(model_sq, eval_loader_ppl)
    print(f"  SQ+6b PPL = {ppl_sq6:.2f}")
    acc_sq6 = eval_model(model_sq, tok, 'sq6b')
    hook6.remove()
    results['sq_6b'] = {'ppl': ppl_sq6, 'acc': acc_sq6}

    # ── 4. ILP 20% ──────────────────────────────────────────────────────────────
    print("\n[4/4] ILP 20% mixed-precision...")
    sens_path = Path('results/sensitivity/opt125m/group_sensitivity.json')
    ppa_path  = Path('results/ppa/opt125m/ppa_sweep_opt125m.csv')
    if sens_path.exists() and ppa_path.exists():
        with open(sens_path) as f:
            group_sens = json.load(f)
        ppa = load_ppa_sweep(str(ppa_path))
        model_ilp, tok = load_model('facebook/opt-125m', './model_cache', 'cpu')
        layers = get_linear_layers(model_ilp)
        # Build per-layer sensitivity from group data
        sens_dict = {}
        for name, _ in layers:
            ltype = classify_layer(name)
            if ltype in group_sens:
                sens_dict[name] = abs(group_sens[ltype]['delta_per_layer'])
            else:
                sens_dict[name] = 1.0
        # ILP at ~20% savings
        ref_budget = sum(2**7 for _ in layers)
        target_budget = int(ref_budget * 0.795)
        assignment = ilp_allocation(sens_dict, target_budget, ppa,
                                     nominal_bits=7, bit_choices=[5,6,7,8])
        hook_ilp = PerLayerCIMHook(model_ilp, bit_assignment=assignment,
                                    default_bits=7)
        hook_ilp.calibrate(make_loader(Subset(calib_data, range(4))))
        hook_ilp.install()
        ppl_ilp = compute_perplexity(model_ilp, eval_loader_ppl)
        print(f"  ILP 20% PPL = {ppl_ilp:.2f}")
        acc_ilp = eval_model(model_ilp, tok, 'ilp20')
        hook_ilp.remove()
        results['ilp_20'] = {'ppl': ppl_ilp, 'acc': acc_ilp}
    else:
        print("  [SKIP] missing sensitivity or ppa data")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Config':<12} {'PPL':>8}  " + "  ".join(f"{t[:8]:>8}" for t in TASKS))
    print("-"*65)
    for key, v in results.items():
        ppl_s = f"{v['ppl']:.1f}" if v['ppl'] else "  -   "
        accs  = "  ".join(f"{v['acc'].get(t,0)*100:>7.1f}%" for t in TASKS)
        print(f"{key:<12} {ppl_s:>8}  {accs}")

    with open(OUT_DIR / 'zeroshot_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_DIR}/zeroshot_summary.json")

if __name__ == '__main__':
    main()
