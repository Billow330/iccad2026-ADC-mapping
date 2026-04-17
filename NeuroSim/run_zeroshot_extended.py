"""
run_zeroshot_extended.py — Extended Zero-Shot Evaluation (5+ tasks, full eval sets)
===================================================================================
Addresses Plan A3: expand zero-shot evaluation.

Tasks: HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, PIQA
Models: OPT-125M, OPT-1.3B
Configs: Clean FP32, CIM 7b, SQ+6b, ILP-20%

Uses lm-evaluation-harness with FULL evaluation sets (not 500-sample subsets).

Usage:
    # OPT-125M, all tasks, full eval
    python3 run_zeroshot_extended.py --model facebook/opt-125m \
        --output_dir results/zeroshot_extended/opt125m

    # OPT-1.3B
    python3 run_zeroshot_extended.py --model facebook/opt-1.3b \
        --output_dir results/zeroshot_extended/opt1.3b
"""

import os, sys, json, argparse, copy, subprocess
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

TASKS_EXTENDED = [
    'hellaswag',
    'winogrande',
    'arc_easy',
    'arc_challenge',
    'piqa',
]


def run_lmeval(model_path, tasks, output_path, batch_size=4, num_fewshot=0,
               limit=None):
    """Run lm-evaluation-harness and return results dict."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'lm_eval',
        '--model', 'hf',
        '--model_args', f'pretrained={model_path},dtype=float32',
        '--tasks', ','.join(tasks),
        '--num_fewshot', str(num_fewshot),
        '--output_path', str(out),
        '--batch_size', str(batch_size),
    ]
    if limit is not None:
        cmd += ['--limit', str(limit)]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return None

    result_files = sorted(out.glob('results*.json'))
    if result_files:
        with open(result_files[-1]) as f:
            return json.load(f)
    return None


def extract_accuracy(results, tasks):
    if results is None:
        return {}
    out = {}
    for task in tasks:
        r = results.get('results', {}).get(task, {})
        for metric in ['acc,none', 'acc_norm,none', 'acc']:
            if metric in r:
                out[task] = r[metric]
                break
    return out


def apply_cim_noise_and_save(model_name, cache_dir, adc_bits, output_path,
                              device='cpu', num_calib=4, clip_pct=99.0,
                              smooth_quant=False, smooth_alpha=0.5):
    """Load model, apply CIM noise, save patched model for lm-eval."""
    import torch
    from llm_inference import load_model, load_wikitext2, make_loader, CIMNoiseHook
    from smooth_quant import CIMSmoothQuant
    from torch.utils.data import Subset

    model, tok = load_model(model_name, cache_dir, device)
    calib_data = load_wikitext2(tok, 512, split='train')
    calib_loader = make_loader(Subset(calib_data, range(num_calib)))

    if smooth_quant:
        smoother = CIMSmoothQuant(model)
        smoother.compute_scales(calib_loader, input_bits=adc_bits,
                                weight_bits=8, sat_lambda=smooth_alpha)
        smoother.apply_scales()

    hook = CIMNoiseHook(model, adc_bits=adc_bits, clip_pct=clip_pct,
                        weight_bits=8, input_bits=8)
    hook.calibrate(calib_loader)
    hook.apply()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    hook.remove()

    return str(out)


def apply_ilp_and_save(model_name, cache_dir, output_path, device='cpu',
                        num_calib=4, clip_pct=99.0, target_savings=0.20):
    """Load model, apply ILP mixed-precision, save for lm-eval."""
    import torch
    from llm_inference import load_model, load_wikitext2, make_loader
    from sensitivity_analysis import (
        PerLayerCIMHook, get_linear_layers, classify_layer,
        ilp_allocation, load_ppa_sweep
    )
    from torch.utils.data import Subset

    model, tok = load_model(model_name, cache_dir, device)
    calib_data = load_wikitext2(tok, 512, split='train')
    calib_loader = make_loader(Subset(calib_data, range(num_calib)))

    layer_names = [name for name, _ in get_linear_layers(model)]

    sens_path = ROOT / 'results' / 'sensitivity' / 'opt125m' / 'group_sensitivity.json'
    with open(sens_path) as f:
        group_sens = json.load(f)

    sensitivity_data = []
    for name in layer_names:
        ltype = classify_layer(name)
        g = group_sens.get(ltype, {})
        delta = g.get('delta_per_layer', 0.0)
        sensitivity_data.append({
            'layer': name,
            'layer_type': ltype,
            'sensitivity': max(delta, 0.0),
        })

    ppa_csv = ROOT / 'results' / 'ppa' / 'opt125m' / 'ppa_sweep_opt125m.csv'
    ppa_sweep = load_ppa_sweep(str(ppa_csv))

    asgn = ilp_allocation(sensitivity_data, ppa_sweep, nominal_bits=7,
                           bit_choices=(4, 5, 6, 7, 8),
                           target_area_savings=target_savings)
    assignment = dict(zip(layer_names, asgn))

    hook = PerLayerCIMHook(model, assignment, default_bits=7, clip_percentile=clip_pct)
    hook.calibrate(calib_loader, device=device, num_batches=num_calib)
    hook.install()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    hook.remove()

    return str(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='facebook/opt-125m')
    p.add_argument('--model_cache', default='./model_cache')
    p.add_argument('--output_dir', default='results/zeroshot_extended/opt125m')
    p.add_argument('--ppa_csv', default='results/ppa/opt125m/ppa_sweep_opt125m.csv')
    p.add_argument('--device', default='cpu')
    p.add_argument('--num_calib_batches', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--limit', type=int, default=None,
                   help='Limit samples per task (None=full eval set)')
    p.add_argument('--tasks', default=','.join(TASKS_EXTENDED))
    p.add_argument('--skip_patching', action='store_true',
                   help='Skip model patching, only run lm-eval on existing dirs')
    args = p.parse_args()

    tasks = args.tasks.split(',')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = {}

    # 1. Clean FP32
    print("\n[1/4] Clean FP32...")
    from llm_inference import load_model
    snapshot = args.model
    try:
        safe = args.model.replace('/', '--')
        snap_dir = Path(args.model_cache) / f'models--{safe}' / 'snapshots'
        if snap_dir.exists():
            snaps = sorted(snap_dir.iterdir())
            if snaps:
                snapshot = str(snaps[-1])
    except Exception:
        pass

    clean_results = run_lmeval(snapshot, tasks, out_dir / 'clean',
                                batch_size=args.batch_size, limit=args.limit)
    configs['clean'] = extract_accuracy(clean_results, tasks)
    print(f"  Clean: {configs['clean']}")

    if not args.skip_patching:
        # 2. CIM 7b uniform
        print("\n[2/4] CIM 7b uniform...")
        cim7_path = apply_cim_noise_and_save(
            args.model, args.model_cache, 7,
            out_dir / 'tmp_cim7b', args.device, args.num_calib_batches)
        cim7_results = run_lmeval(cim7_path, tasks, out_dir / 'cim7b',
                                   batch_size=args.batch_size, limit=args.limit)
        configs['cim_7b'] = extract_accuracy(cim7_results, tasks)
        print(f"  CIM 7b: {configs['cim_7b']}")

        # 3. SQ + CIM 6b
        print("\n[3/4] SQ + CIM 6b...")
        sq6_path = apply_cim_noise_and_save(
            args.model, args.model_cache, 6,
            out_dir / 'tmp_sq6b', args.device, args.num_calib_batches,
            smooth_quant=True)
        sq6_results = run_lmeval(sq6_path, tasks, out_dir / 'sq6b',
                                  batch_size=args.batch_size, limit=args.limit)
        configs['sq_6b'] = extract_accuracy(sq6_results, tasks)
        print(f"  SQ+6b: {configs['sq_6b']}")

        # 4. ILP 20%
        print("\n[4/4] ILP 20%...")
        ilp_path = apply_ilp_and_save(
            args.model, args.model_cache,
            out_dir / 'tmp_ilp20', args.device, args.num_calib_batches)
        ilp_results = run_lmeval(ilp_path, tasks, out_dir / 'ilp20',
                                  batch_size=args.batch_size, limit=args.limit)
        configs['ilp_20'] = extract_accuracy(ilp_results, tasks)
        print(f"  ILP 20%: {configs['ilp_20']}")

    # Summary table
    print(f"\n{'='*80}")
    header = f"{'Task':<18}"
    for cfg in configs:
        header += f" {cfg:>10}"
    print(header)
    print("-" * 80)

    for task in tasks:
        row = f"  {task:<16}"
        for cfg_name, accs in configs.items():
            v = accs.get(task, 0)
            row += f" {v*100:>9.1f}%"
        print(row)

    # Average
    row = f"  {'Average':<16}"
    for cfg_name, accs in configs.items():
        vals = [accs.get(t, 0) for t in tasks if t in accs]
        avg = sum(vals) / len(vals) if vals else 0
        row += f" {avg*100:>9.1f}%"
    print(row)

    # Save summary
    summary = {
        'model': args.model,
        'tasks': tasks,
        'limit': args.limit,
        'configs': configs,
    }
    summary_path = out_dir / 'zeroshot_extended_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {summary_path}")

    # Generate LaTeX table
    latex_path = out_dir / 'zeroshot_extended_table.tex'
    with open(latex_path, 'w') as f:
        task_short = {
            'hellaswag': 'HS',
            'winogrande': 'WG',
            'arc_easy': 'ARC-E',
            'arc_challenge': 'ARC-C',
            'piqa': 'PIQA',
        }
        cols = ' & '.join(task_short.get(t, t) for t in tasks)
        f.write(r"""\begin{table}[t]
\centering
\caption{Extended Zero-Shot Accuracy (""" + args.model.split('/')[-1] +
               (f', full eval' if args.limit is None else f', {args.limit} samples') +
               r""")}
\label{tab:zeroshot_extended}
\begin{tabular}{l""" + 'r' * len(tasks) + r"""r}
\toprule
Config & """ + cols + r""" & Avg. \\
\midrule
""")
        for cfg_name, accs in configs.items():
            vals = [accs.get(t, 0) for t in tasks]
            avg = sum(vals) / len(vals) if vals else 0
            cfg_display = {
                'clean': 'FP32 (clean)',
                'cim_7b': 'CIM 7b',
                'sq_6b': 'SQ + 6b',
                'ilp_20': 'ILP 20\\%',
            }.get(cfg_name, cfg_name)
            val_strs = ' & '.join(f'{v*100:.1f}\\%' for v in vals)
            f.write(f"{cfg_display} & {val_strs} & {avg*100:.1f}\\% \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"[Saved] {latex_path}")


if __name__ == '__main__':
    main()
