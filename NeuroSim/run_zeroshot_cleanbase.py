"""
run_zeroshot_cleanbase.py - Add clean FP32 and CIM-7b to zero-shot results
(arc_challenge not available offline; hellaswag + winogrande only)
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

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, tasks

from llm_inference import load_model, load_wikitext2, make_loader, CIMNoiseHook
from smooth_quant import compute_perplexity

OUT_DIR = Path('results/zeroshot/opt125m')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ['hellaswag', 'winogrande']

def eval_tasks(hf_model, tokenizer, label, limit=500):
    lm = HFLM(pretrained=hf_model, tokenizer=tokenizer, batch_size=4)
    task_manager = tasks.TaskManager()
    results = evaluator.simple_evaluate(
        model=lm, tasks=TASKS, num_fewshot=0,
        limit=limit, task_manager=task_manager,
    )
    acc = {}
    for task in TASKS:
        r = results['results'].get(task, {})
        for metric in ['acc,none', 'acc_norm,none']:
            if metric in r:
                acc[task] = round(r[metric], 4)
                break
    print(f"  [{label}] {acc}")
    return acc

def main():
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
    calib_loader    = make_loader(Subset(calib_data, range(4)))
    eval_loader_ppl = make_loader(Subset(eval_data, range(100)))

    def save():
        with open(partial_path, 'w') as f:
            json.dump(results, f, indent=2)
        print("  Saved.")

    # ── 1. Clean FP32 ──────────────────────────────────────────────
    if 'clean' not in results:
        print("\n[Clean FP32]")
        ppl_clean = compute_perplexity(model, eval_loader_ppl)
        print(f"  clean PPL = {ppl_clean:.2f}")
        acc_clean = eval_tasks(model, tok, 'clean')
        results['clean'] = {'ppl': ppl_clean, 'acc': acc_clean}
        save()
    else:
        print(f"[SKIP] clean: {results['clean']['acc']}")

    # ── 2. CIM 7b ──────────────────────────────────────────────────
    if 'cim_7b' not in results:
        print("\n[CIM-7b]")
        hook7 = CIMNoiseHook(model, adc_bits=7, weight_bits=8, input_bits=8)
        hook7.calibrate(calib_loader, clip_percentile=99.0)
        hook7.install()
        ppl_cim7 = compute_perplexity(model, eval_loader_ppl)
        print(f"  CIM-7b PPL = {ppl_cim7:.2f}")
        acc_cim7 = eval_tasks(model, tok, 'cim7b')
        hook7.remove()
        results['cim_7b'] = {'ppl': ppl_cim7, 'acc': acc_cim7}
        save()
    else:
        print(f"[SKIP] cim_7b: {results['cim_7b']['acc']}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Config':<12} {'PPL':>7}  {'hellaswag':>10}  {'winogrande':>11}  {'Avg':>6}")
    print("-"*65)
    for key, v in results.items():
        ppl_s = f"{v['ppl']:.1f}" if v.get('ppl') else "   -  "
        hs  = v.get('acc', {}).get('hellaswag', 0)
        wg  = v.get('acc', {}).get('winogrande', 0)
        avg = (hs + wg) / 2
        print(f"{key:<12} {ppl_s:>7}  {hs*100:>9.1f}%  {wg*100:>10.1f}%  {avg*100:>5.1f}%")

    with open(OUT_DIR / 'zeroshot_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_DIR}/zeroshot_summary.json")

if __name__ == '__main__':
    main()
