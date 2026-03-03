"""
run_zeroshot_arc.py - Add arc_challenge to zero-shot evaluation
Builds on existing partial_results.json (hellaswag + winogrande already done for sq_6b/ilp_20)
Runs all 4 configs on arc_challenge, and clean/cim_7b on hellaswag+winogrande if missing.
"""
import os, sys, json
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

OUT_DIR = Path('results/zeroshot/opt125m')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS_BASE   = ['hellaswag', 'winogrande']
TASKS_NEW    = ['arc_challenge']
TASKS_ALL    = TASKS_BASE + TASKS_NEW

ILP_LTYPE_BITS = {'attn_qkv': 6, 'ffn_up': 6, 'attn_out': 7, 'ffn_down': 7, 'lm_head': 7,
                   'fc1': 6, 'fc2': 7, 'out_proj': 7}

def eval_tasks(hf_model, tokenizer, task_list, label, limit=500):
    lm = HFLM(pretrained=hf_model, tokenizer=tokenizer, batch_size=4)
    task_manager = tasks.TaskManager()
    results = evaluator.simple_evaluate(
        model=lm, tasks=task_list, num_fewshot=0,
        limit=limit, task_manager=task_manager,
    )
    acc = {}
    for task in task_list:
        r = results['results'].get(task, {})
        for metric in ['acc,none', 'acc_norm,none']:
            if metric in r:
                acc[task] = round(r[metric], 4)
                break
    print(f"  [{label}] {acc}")
    return acc

def main():
    # Load existing results
    partial_path = OUT_DIR / 'partial_results.json'
    if partial_path.exists():
        with open(partial_path) as f:
            results = json.load(f)
        print(f"Loaded existing: {list(results.keys())}")
    else:
        results = {}

    print("Loading model + data...")
    model, tok = load_model('facebook/opt-125m', './model_cache', 'cpu')
    calib_data = load_wikitext2(tok, 512, split='train')
    eval_data  = load_wikitext2(tok, 512, split='test')
    calib_loader     = make_loader(Subset(calib_data, range(4)))
    eval_loader_ppl  = make_loader(Subset(eval_data, range(100)))

    def save():
        with open(partial_path, 'w') as f:
            json.dump(results, f, indent=2)
        print("  Saved partial results")

    # ── 1. Clean FP32 ──────────────────────────────────────────────────────────
    need_clean_base = 'clean' not in results
    need_clean_arc  = 'clean' in results and 'arc_challenge' not in results.get('clean', {}).get('acc', {})
    if need_clean_base or need_clean_arc:
        print("\n[Clean FP32]")
        if need_clean_base:
            ppl_clean = compute_perplexity(model, eval_loader_ppl)
            print(f"  clean PPL = {ppl_clean:.2f}")
            acc_clean = eval_tasks(model, tok, TASKS_ALL, 'clean')
            results['clean'] = {'ppl': ppl_clean, 'acc': acc_clean}
        else:
            print("  Only running arc_challenge for clean...")
            acc_arc = eval_tasks(model, tok, TASKS_NEW, 'clean_arc')
            results['clean']['acc'].update(acc_arc)
        save()
    else:
        print(f"[SKIP] clean already complete: {results['clean']['acc']}")

    # ── 2. CIM 7b uniform ──────────────────────────────────────────────────────
    need_cim7_base = 'cim_7b' not in results
    need_cim7_arc  = 'cim_7b' in results and 'arc_challenge' not in results.get('cim_7b', {}).get('acc', {})
    if need_cim7_base or need_cim7_arc:
        print("\n[CIM-7b uniform]")
        hook7 = CIMNoiseHook(model, adc_bits=7, weight_bits=8, input_bits=8)
        hook7.calibrate(calib_loader, clip_percentile=99.0)
        hook7.install()
        if need_cim7_base:
            ppl_cim7 = compute_perplexity(model, eval_loader_ppl)
            print(f"  CIM-7b PPL = {ppl_cim7:.2f}")
            acc_cim7 = eval_tasks(model, tok, TASKS_ALL, 'cim7b')
            results['cim_7b'] = {'ppl': ppl_cim7, 'acc': acc_cim7}
        else:
            print("  Only running arc_challenge for cim_7b...")
            acc_arc = eval_tasks(model, tok, TASKS_NEW, 'cim7b_arc')
            results['cim_7b']['acc'].update(acc_arc)
        hook7.remove()
        save()
    else:
        print(f"[SKIP] cim_7b already complete: {results['cim_7b']['acc']}")

    # ── 3. SQ + CIM 6b ─────────────────────────────────────────────────────────
    need_sq_arc = 'sq_6b' in results and 'arc_challenge' not in results.get('sq_6b', {}).get('acc', {})
    if 'sq_6b' not in results or need_sq_arc:
        print("\n[SQ+6b]")
        model_sq, _ = load_model('facebook/opt-125m', './model_cache', 'cpu')
        sq = CIMSmoothQuant(weight_bits=8, input_bits=8, adc_bits=6,
                            adc_clip_pct=99.0, sat_lambda=0.5, verbose=False)
        sq.fit(model_sq, calib_loader, num_batches=4, device='cpu', task='lm')
        hook6 = CIMNoiseHook(model_sq, adc_bits=6, weight_bits=8, input_bits=8)
        hook6.calibrate(calib_loader, clip_percentile=99.0)
        hook6.install()
        if 'sq_6b' not in results:
            ppl_sq6 = compute_perplexity(model_sq, eval_loader_ppl)
            print(f"  SQ+6b PPL = {ppl_sq6:.2f}")
            acc_sq6 = eval_tasks(model_sq, tok, TASKS_ALL, 'sq6b')
            results['sq_6b'] = {'ppl': ppl_sq6, 'acc': acc_sq6}
        else:
            print("  Only running arc_challenge for sq_6b...")
            acc_arc = eval_tasks(model_sq, tok, TASKS_NEW, 'sq6b_arc')
            results['sq_6b']['acc'].update(acc_arc)
        hook6.remove()
        save()
    else:
        print(f"[SKIP] sq_6b already complete: {results['sq_6b']['acc']}")

    # ── 4. ILP 20% ──────────────────────────────────────────────────────────────
    need_ilp_arc = 'ilp_20' in results and 'arc_challenge' not in results.get('ilp_20', {}).get('acc', {})
    if 'ilp_20' not in results or need_ilp_arc:
        print("\n[ILP-20%]")
        sens_path = Path('results/sensitivity/opt125m/group_sensitivity.json')
        with open(sens_path) as f:
            group_sens = json.load(f)

        model_ilp, _ = load_model('facebook/opt-125m', './model_cache', 'cpu')
        layers = get_linear_layers(model_ilp)

        # Use known ILP assignment from stable_eval
        assignment = {}
        for name, _ in layers:
            ltype = classify_layer(name)
            assignment[name] = ILP_LTYPE_BITS.get(ltype, 7)

        hook_ilp = PerLayerCIMHook(model_ilp, bit_assignment=assignment, default_bits=7)
        hook_ilp.calibrate(calib_loader)
        hook_ilp.install()
        if 'ilp_20' not in results:
            ppl_ilp = compute_perplexity(model_ilp, eval_loader_ppl)
            print(f"  ILP 20% PPL = {ppl_ilp:.2f}")
            acc_ilp = eval_tasks(model_ilp, tok, TASKS_ALL, 'ilp20')
            results['ilp_20'] = {'ppl': ppl_ilp, 'acc': acc_ilp}
        else:
            print("  Only running arc_challenge for ilp_20...")
            acc_arc = eval_tasks(model_ilp, tok, TASKS_NEW, 'ilp20_arc')
            results['ilp_20']['acc'].update(acc_arc)
        hook_ilp.remove()
        save()
    else:
        print(f"[SKIP] ilp_20 already complete: {results['ilp_20']['acc']}")

    # ── Summary ──────────────────────────────────────────────────────────────────
    print("\n" + "="*75)
    print(f"{'Config':<12} {'PPL':>7}  " + "  ".join(f"{t[:12]:>12}" for t in TASKS_ALL) + "  Avg")
    print("-"*75)
    for key, v in results.items():
        ppl_s = f"{v['ppl']:.1f}" if v.get('ppl') else "   -  "
        accs  = [v.get('acc', {}).get(t, 0) for t in TASKS_ALL]
        acc_s = "  ".join(f"{a*100:>11.1f}%" for a in accs)
        avg   = sum(accs)/len(accs) if accs else 0
        print(f"{key:<12} {ppl_s:>7}  {acc_s}  {avg*100:.1f}%")

    # Save final
    with open(OUT_DIR / 'zeroshot_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_DIR}/zeroshot_summary.json")

if __name__ == '__main__':
    main()
