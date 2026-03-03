"""
run_zeroshot_sq_ilp.py - Zero-shot for SQ+6b and ILP-20% configs only
(clean and CIM-7b already done in run_zeroshot.py)
"""
import os, sys, json, copy
from pathlib import Path

os.environ.pop('HF_HUB_OFFLINE', None)
os.environ['HF_DATASETS_OFFLINE'] = '1'

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import torch
from torch.utils.data import Subset

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, tasks

from llm_inference import load_model, load_wikitext2, make_loader, CIMNoiseHook
from smooth_quant import CIMSmoothQuant, compute_perplexity
from sensitivity_analysis import PerLayerCIMHook, get_linear_layers, classify_layer, ilp_allocation

SNAP = 'model_cache/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6'
TASKS = ['hellaswag', 'winogrande']
OUT_DIR = Path('results/zeroshot/opt125m')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def eval_model(hf_model, tokenizer, label):
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
    print("Loading base model...")
    model, tok = load_model('facebook/opt-125m', './model_cache', 'cpu')
    calib_data = load_wikitext2(tok, 512, split='train')
    eval_data  = load_wikitext2(tok, 512, split='test')
    calib_loader = make_loader(Subset(calib_data, range(4)))
    eval_loader_ppl = make_loader(Subset(eval_data, range(100)))

    results = {}

    # ── Load previously computed clean and CIM-7b results
    prev_path = OUT_DIR / 'partial_results.json'
    if prev_path.exists():
        with open(prev_path) as f:
            results = json.load(f)
        print(f"Loaded partial results: {list(results.keys())}")
    else:
        print("No partial results found, starting fresh")

    # ── SQ + CIM 6b ────────────────────────────────────────────────────────────
    if 'sq_6b' not in results:
        print("\n[SQ+6b] SQ + CIM 6b...")
        model_sq, tok = load_model('facebook/opt-125m', './model_cache', 'cpu')
        sq = CIMSmoothQuant(
            weight_bits=8, input_bits=8, adc_bits=6,
            adc_clip_pct=99.0, sat_lambda=0.5, verbose=False)
        sq.fit(model_sq, calib_loader, num_batches=4, device='cpu', task='lm')
        hook6 = CIMNoiseHook(model_sq, adc_bits=6, weight_bits=8, input_bits=8)
        hook6.calibrate(calib_loader, clip_percentile=99.0)
        hook6.install()
        ppl_sq6 = compute_perplexity(model_sq, eval_loader_ppl)
        print(f"  SQ+6b PPL = {ppl_sq6:.2f}")
        acc_sq6 = eval_model(model_sq, tok, 'sq6b')
        hook6.remove()
        results['sq_6b'] = {'ppl': ppl_sq6, 'acc': acc_sq6}
        with open(OUT_DIR / 'partial_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("  Saved partial results")
    else:
        print(f"[SKIP] sq_6b already in results: {results['sq_6b']['acc']}")

    # ── ILP 20% ─────────────────────────────────────────────────────────────────
    if 'ilp_20' not in results:
        sens_path = Path('results/sensitivity/opt125m/group_sensitivity.json')
        ppa_path  = Path('results/sensitivity/opt125m/evaluation_results.csv')
        if sens_path.exists():
            print("\n[ILP-20%] Mixed-precision ILP...")
            with open(sens_path) as f:
                group_sens = json.load(f)

            model_ilp, tok = load_model('facebook/opt-125m', './model_cache', 'cpu')
            layers = get_linear_layers(model_ilp)

            # Build per-layer sensitivity list for ilp_allocation
            # ilp_allocation expects list of {'layer': name, 'sensitivity': value}
            sensitivity_data = []
            for name, _ in layers:
                ltype = classify_layer(name)
                if ltype in group_sens:
                    s = abs(group_sens[ltype].get('delta_per_layer', 1.0))
                else:
                    s = 1.0
                sensitivity_data.append({'layer': name, 'sensitivity': s})

            assignment_list = ilp_allocation(
                sensitivity_data, {},  # ppa_sweep not used internally
                nominal_bits=7, bit_choices=(5,6,7,8),
                target_area_savings=0.20)

            # Convert list to dict {layer_name: bits}
            assignment = {sensitivity_data[i]['layer']: assignment_list[i]
                          for i in range(len(sensitivity_data))}

            hook_ilp = PerLayerCIMHook(model_ilp, bit_assignment=assignment, default_bits=7)
            hook_ilp.calibrate(calib_loader)
            hook_ilp.install()
            ppl_ilp = compute_perplexity(model_ilp, eval_loader_ppl)
            print(f"  ILP 20% PPL = {ppl_ilp:.2f}")
            acc_ilp = eval_model(model_ilp, tok, 'ilp20')
            hook_ilp.remove()
            results['ilp_20'] = {'ppl': ppl_ilp, 'acc': acc_ilp}
            with open(OUT_DIR / 'partial_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("  Saved partial results")
        else:
            print("  [SKIP] missing sensitivity data")
    else:
        print(f"[SKIP] ilp_20 already in results: {results['ilp_20']['acc']}")

    # ── Summary ──────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Config':<12} {'PPL':>8}  " + "  ".join(f"{t[:8]:>8}" for t in TASKS))
    print("-"*65)
    for key, v in results.items():
        ppl_s = f"{v['ppl']:.1f}" if v.get('ppl') else "  -   "
        accs  = "  ".join(f"{v['acc'].get(t,0)*100:>7.1f}%" for t in TASKS) if v.get('acc') else "  -  "
        print(f"{key:<12} {ppl_s:>8}  {accs}")

    with open(OUT_DIR / 'zeroshot_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_DIR}/zeroshot_summary.json")

if __name__ == '__main__':
    main()
