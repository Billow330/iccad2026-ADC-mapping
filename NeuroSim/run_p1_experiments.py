#!/usr/bin/env python3
"""
P1 experiments:
  - P1-1 hierarchical refinement
  - P1-2 profiling cost vs gain
  - P1-3 transferability regret
  - P1-4 mechanism support (sequence length + saturated-channel consistency)
"""

import os
import sys
import json
import time
import itertools
from pathlib import Path
from collections import defaultdict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_CACHE"] = "/raid/privatedata/fantao/model_cache"
os.environ["HF_HOME"] = "/raid/privatedata/fantao/model_cache"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, "/raid/privatedata/fantao/pylib")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import run_p0_experiments as p0
from llm_inference import load_model, load_wikitext2, make_loader
from sensitivity_analysis import get_linear_layers, classify_layer, PerLayerCIMHook

DEVICE = p0.DEVICE
OUT = Path('/raid/privatedata/fantao/iccad_exp/results_p1')
OUT.mkdir(parents=True, exist_ok=True)
PHASE2 = Path('/tmp/fantaog_iccad/results/phase2')


def P(msg):
    print(msg, flush=True)


def split_thirds(names):
    n = len(names)
    a = n // 3
    b = (2 * n) // 3
    return {
        'early': names[:a],
        'mid': names[a:b],
        'late': names[b:],
    }


def prepare_loaders_for_seq(tokenizer, seq_len=512, calib_n=20, eval_start=64, eval_n=100):
    data = load_wikitext2(tokenizer, seq_len=seq_len)
    calib = data[:min(calib_n, len(data))]
    eval_data = data[eval_start:min(eval_start + eval_n, len(data))]
    return make_loader(calib, batch_size=1), make_loader(eval_data, batch_size=1), len(eval_data)


def group_counts(layer_names):
    d = defaultdict(int)
    for n in layer_names:
        d[classify_layer(n)] += 1
    return dict(d)


def measure_group_sensitivity_det(model, layer_names, cld, eld, baseline_ppl,
                                  nominal_bits=7, probe_bits=6,
                                  num_calib=4, num_eval=10, seed=0):
    groups = defaultdict(list)
    for name in layer_names:
        groups[classify_layer(name)].append(name)
    res = {}
    for gname, names in groups.items():
        assign = {n: nominal_bits for n in layer_names}
        for n in names:
            assign[n] = probe_bits
        ppl = p0.ev(model, assign, cld, eld, nominal_bits, n_c=num_calib, n_e=num_eval, seed=seed)
        dppl = ppl - baseline_ppl
        res[gname] = {
            'count': len(names),
            'ppl': ppl,
            'delta_ppl': dppl,
            'delta_per_layer': dppl / len(names),
            'layers': names,
        }
        P(f"  Group {gname:<12} ΔPPL={dppl:+.3f} per-layer={dppl/len(names):+.4f}")
    return res


def measure_depth_bin_sensitivity_det(model, layer_names, cld, eld, baseline_ppl, target_groups,
                                      nominal_bits=7, probe_bits=6,
                                      num_calib=4, num_eval=10, seed=0):
    type_to_layers = defaultdict(list)
    for n in layer_names:
        type_to_layers[classify_layer(n)].append(n)
    res = {}
    for gname in target_groups:
        thirds = split_thirds(type_to_layers[gname])
        for tag, names in thirds.items():
            key = f'{gname}_{tag}'
            assign = {n: nominal_bits for n in layer_names}
            for n in names:
                assign[n] = probe_bits
            ppl = p0.ev(model, assign, cld, eld, nominal_bits, n_c=num_calib, n_e=num_eval, seed=seed)
            dppl = ppl - baseline_ppl
            res[key] = {
                'group': gname,
                'bin': tag,
                'count': len(names),
                'layers': names,
                'ppl': ppl,
                'delta_ppl': dppl,
                'delta_per_layer': dppl / max(len(names), 1),
            }
            P(f"  Bin   {key:<18} ΔPPL={dppl:+.3f} per-layer={dppl/max(len(names),1):+.4f}")
    return res


def measure_per_layer_sensitivity_det(model, layer_names, cld, eld, baseline_ppl,
                                      nominal_bits=7, probe_bits=6,
                                      num_calib=4, num_eval=10, seed=0):
    out = []
    for i, name in enumerate(layer_names):
        assign = {n: nominal_bits for n in layer_names}
        assign[name] = probe_bits
        ppl = p0.ev(model, assign, cld, eld, nominal_bits, n_c=num_calib, n_e=num_eval, seed=seed)
        dppl = ppl - baseline_ppl
        out.append({
            'layer': name,
            'group': classify_layer(name),
            'delta_ppl': dppl,
            'count': 1,
        })
        if (i + 1) % 10 == 0 or i == len(layer_names) - 1:
            P(f"  Layer {i+1:02d}/{len(layer_names)} {name} ΔPPL={dppl:+.3f}")
    return out


def brute_force_meta(signal_per_layer, counts, nominal_bits=7,
                     bit_choices=(4, 5, 6, 7), target_savings=0.2):
    keys = list(signal_per_layer.keys())
    nominal_area = sum(counts[k] * (2 ** nominal_bits) for k in keys)
    budget = nominal_area * (1.0 - target_savings)
    best = None
    for cfg in itertools.product(bit_choices, repeat=len(keys)):
        area = sum(counts[k] * (2 ** b) for k, b in zip(keys, cfg))
        if area > budget:
            continue
        obj = sum(counts[k] * max(0, nominal_bits - b) * max(signal_per_layer[k], 0)
                  for k, b in zip(keys, cfg))
        cand = {
            'assign': dict(zip(keys, cfg)),
            'area': area,
            'objective': obj,
            'savings': 1 - area / nominal_area,
        }
        if best is None or obj < best['objective'] or (obj == best['objective'] and area < best['area']):
            best = cand
    return best


def build_group_assignment(layer_names, group_assign):
    return {n: group_assign[classify_layer(n)] for n in layer_names}


def build_hier_assignment(layer_names, meta_assign, refined_bins):
    lookup = {}
    for k, v in refined_bins.items():
        for n in v['layers']:
            lookup[n] = k
    out = {}
    for n in layer_names:
        if n in lookup:
            out[n] = meta_assign[lookup[n]]
        else:
            out[n] = meta_assign[classify_layer(n)]
    return out


def summarize_group_bits(assign, layer_names):
    acc = defaultdict(list)
    for n in layer_names:
        acc[classify_layer(n)].append(assign[n])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def evaluate_hierarchical_method(model, layer_names, cld, eld):
    P('\n' + '=' * 72)
    P('P1-1 / P1-2: hierarchical refinement + cost-vs-gain')
    P('=' * 72)

    ppl_prof = p0.ev(model, {n: 7 for n in layer_names}, cld, eld, 7, n_e=10, seed=0)
    ppl_eval = p0.ev(model, {n: 7 for n in layer_names}, cld, eld, 7, n_e=100, seed=0)
    P(f'  Profiling baseline (10): {ppl_prof:.3f}')
    P(f'  Eval baseline (100): {ppl_eval:.3f}')

    coarse = measure_group_sensitivity_det(model, layer_names, cld, eld, ppl_prof, num_eval=10, seed=0)
    coarse_signal = {k: v['delta_per_layer'] for k, v in coarse.items()}
    ranked_nonhead = [k for k in sorted(coarse_signal, key=lambda x: coarse_signal[x], reverse=True) if k != 'lm_head']
    refine_groups = ranked_nonhead[:2]
    P(f'  Refine top sensitive non-head groups: {refine_groups}')

    refined = measure_depth_bin_sensitivity_det(model, layer_names, cld, eld, ppl_prof, refine_groups, num_eval=10, seed=0)
    per_layer = measure_per_layer_sensitivity_det(model, layer_names, cld, eld, ppl_prof, num_eval=10, seed=0)

    coarse_counts = group_counts(layer_names)
    coarse_counts = {k: v for k, v in coarse_counts.items() if k in coarse_signal}

    hier_counts = {}
    hier_signal = {}
    for g, c in coarse_counts.items():
        if g not in refine_groups:
            hier_counts[g] = c
            hier_signal[g] = coarse_signal[g]
    for k, v in refined.items():
        hier_counts[k] = v['count']
        hier_signal[k] = v['delta_per_layer']

    sat_signal = {
        'attn_qkv': 1.0,
        'attn_out': 0.04,
        'ffn_up': 0.94,
        'ffn_down': 0.19,
        'lm_head': 1.0,
    }

    out = {
        'baseline_eval_7b': ppl_eval,
        'refine_groups': refine_groups,
        'coarse_signal': coarse_signal,
        'refined_signal': hier_signal,
        'profiling_cost_passes': {
            'proxy': 0,
            'group': 1 + len(coarse_signal),
            'hierarchical': 1 + len(coarse_signal) + len(refined),
            'full_per_layer': 1 + len(layer_names),
        },
        'budgets': {},
    }

    per_layer_ilp_input = [{'layer': r['layer'], 'sensitivity': max(r['delta_ppl'], 0)} for r in per_layer]

    for target in [0.20, 0.30, 0.40]:
        tag = f'{int(target*100)}pct'
        P(f'\n  --- Budget {tag} ---')
        out['budgets'][tag] = {}

        proxy_meta = brute_force_meta(sat_signal, coarse_counts, target_savings=target)
        group_meta = brute_force_meta(coarse_signal, coarse_counts, target_savings=target)
        hier_meta = brute_force_meta(hier_signal, hier_counts, target_savings=target)

        proxy_assign = build_group_assignment(layer_names, proxy_meta['assign'])
        group_assign = build_group_assignment(layer_names, group_meta['assign'])
        hier_assign = build_hier_assignment(layer_names, hier_meta['assign'], refined)
        full_assign = p0.run_ilp(per_layer_ilp_input, 7, (4, 5, 6, 7), target)
        if not full_assign:
            full_assign = {n: 7 for n in layer_names}

        for mname, assign, meta in [
            ('proxy', proxy_assign, proxy_meta),
            ('group', group_assign, group_meta),
            ('hierarchical', hier_assign, hier_meta),
            ('full_per_layer', full_assign, None),
        ]:
            ppl = p0.ev(model, assign, cld, eld, 7, n_e=100, seed=0)
            row = {
                'ppl': ppl,
                'rel_deg_pct': (ppl - ppl_eval) / ppl_eval * 100,
                'group_bits': summarize_group_bits(assign, layer_names),
            }
            if meta is not None:
                row['predicted_savings_pct'] = meta['savings'] * 100
            out['budgets'][tag][mname] = row
            P(f"    {mname:<13} PPL={ppl:.3f} rel={row['rel_deg_pct']:+.2f}%")

        full_ppl = out['budgets'][tag]['full_per_layer']['ppl']
        for mname in ['proxy', 'group', 'hierarchical']:
            out['budgets'][tag][mname]['regret_vs_full_ppl'] = out['budgets'][tag][mname]['ppl'] - full_ppl

    if PHASE2.exists() and (PHASE2 / 'depth_bin_sensitivity.json').exists():
        out['phase2_depth_bin_reference'] = json.load(open(PHASE2 / 'depth_bin_sensitivity.json'))

    json.dump(out, open(OUT / 'p1_hierarchical_cost_gain.json', 'w'), indent=2)
    P(f"  Saved → {OUT / 'p1_hierarchical_cost_gain.json'}")
    return out


def group_overlap(a, b):
    keys = sorted(set(a) | set(b))
    return float(np.mean([a.get(k) == b.get(k) for k in keys])) if keys else 1.0


def protected_overlap(a, b, nominal):
    sa = {k for k, v in a.items() if v == nominal}
    sb = {k for k, v in b.items() if v == nominal}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def evaluate_transfer_regret(model, layer_names, cld, eld):
    P('\n' + '=' * 72)
    P('P1-3: transferability regret')
    P('=' * 72)

    counts = group_counts(layer_names)
    counts = {k: v for k, v in counts.items() if k in p0.GROUPS}

    base_7_prof = p0.ev(model, {n: 7 for n in layer_names}, cld, eld, 7, n_e=10, seed=0)
    sens_7 = measure_group_sensitivity_det(model, layer_names, cld, eld, base_7_prof, nominal_bits=7, probe_bits=6, num_eval=10, seed=0)
    sig_7 = {k: v['delta_per_layer'] for k, v in sens_7.items()}
    out = {'source_signal': sig_7, 'regimes': {}}

    for deploy_nom, deploy_probe in [(8, 7), (9, 8), (10, 9)]:
        label = f'{deploy_nom}to{deploy_probe}'
        out['regimes'][label] = {}
        P(f'\n  --- Regime {label} ---')
        base_eval = p0.ev(model, {n: deploy_nom for n in layer_names}, cld, eld, deploy_nom, n_e=100, seed=0)
        base_prof = p0.ev(model, {n: deploy_nom for n in layer_names}, cld, eld, deploy_nom, n_e=10, seed=0)
        native = measure_group_sensitivity_det(model, layer_names, cld, eld, base_prof,
                                               nominal_bits=deploy_nom, probe_bits=deploy_probe,
                                               num_eval=10, seed=0)
        sig_native = {k: v['delta_per_layer'] for k, v in native.items()}

        bit_choices = tuple(range(max(3, deploy_probe - 1), deploy_nom + 1))
        for target in [0.20, 0.30, 0.40]:
            tag = f'{int(target*100)}pct'
            trans_meta = brute_force_meta(sig_7, counts, nominal_bits=deploy_nom,
                                          bit_choices=bit_choices, target_savings=target)
            native_meta = brute_force_meta(sig_native, counts, nominal_bits=deploy_nom,
                                           bit_choices=bit_choices, target_savings=target)
            trans_assign = build_group_assignment(layer_names, trans_meta['assign'])
            native_assign = build_group_assignment(layer_names, native_meta['assign'])
            ppl_t = p0.ev(model, trans_assign, cld, eld, deploy_nom, n_e=100, seed=0)
            ppl_n = p0.ev(model, native_assign, cld, eld, deploy_nom, n_e=100, seed=0)
            row = {
                'baseline_ppl': base_eval,
                'transfer_group_assign': trans_meta['assign'],
                'native_group_assign': native_meta['assign'],
                'ppl_transfer': ppl_t,
                'ppl_native': ppl_n,
                'ppl_regret': ppl_t - ppl_n,
                'relative_regret_pct': (ppl_t - ppl_n) / max(ppl_n, 1e-9) * 100,
                'group_overlap': group_overlap(trans_meta['assign'], native_meta['assign']),
                'protected_group_overlap': protected_overlap(trans_meta['assign'], native_meta['assign'], deploy_nom),
            }
            out['regimes'][label][tag] = row
            P(f"    {tag}: regret={row['ppl_regret']:+.3f}  overlap={row['group_overlap']*100:.0f}%  protected={row['protected_group_overlap']*100:.0f}%")

    json.dump(out, open(OUT / 'p1_transfer_regret.json', 'w'), indent=2)
    P(f"  Saved → {OUT / 'p1_transfer_regret.json'}")
    return out


def sequence_length_ablation(model, tokenizer, layer_names):
    P('\n' + '=' * 72)
    P('P1-4A: sequence-length ablation')
    P('=' * 72)

    out = {}
    for T in [128, 256, 512, 1024]:
        cld, eld, ne = prepare_loaders_for_seq(tokenizer, seq_len=T, eval_n=20)
        if ne == 0:
            continue
        base = p0.ev(model, {n: 7 for n in layer_names}, cld, eld, 7, n_e=ne, seed=0)
        qkv_assign = {n: (6 if classify_layer(n) == 'attn_qkv' else 7) for n in layer_names}
        fc2_assign = {n: (6 if classify_layer(n) == 'ffn_down' else 7) for n in layer_names}
        ppl_qkv = p0.ev(model, qkv_assign, cld, eld, 7, n_e=ne, seed=0)
        ppl_fc2 = p0.ev(model, fc2_assign, cld, eld, 7, n_e=ne, seed=0)
        out[str(T)] = {
            'baseline': base,
            'attn_qkv_dppl_per_layer': (ppl_qkv - base) / 36.0,
            'ffn_down_dppl_per_layer': (ppl_fc2 - base) / 12.0,
            'n_eval': ne,
        }
        P(f"  T={T:<4d} qkv={out[str(T)]['attn_qkv_dppl_per_layer']:+.4f}  fc2={out[str(T)]['ffn_down_dppl_per_layer']:+.4f}")

    return out


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return None
    return len(sa & sb) / max(len(sa | sb), 1)


def saturated_channel_consistency(model, layer_names, cld, eld):
    P('\n' + '=' * 72)
    P('P1-4B: saturated-channel consistency')
    P('=' * 72)

    sample_blocks = [0, 4, 9]
    targets = []
    for b in sample_blocks:
        targets.extend([
            f'model.decoder.layers.{b}.self_attn.q_proj',
            f'model.decoder.layers.{b}.self_attn.k_proj',
            f'model.decoder.layers.{b}.self_attn.v_proj',
            f'model.decoder.layers.{b}.fc2',
        ])
    targets = [t for t in targets if t in layer_names]
    assign = {n: 7 for n in layer_names}
    hook = PerLayerCIMHook(model, assign, 7, 99.0)
    hook.calibrate(cld, device=DEVICE, num_batches=4)

    sets_by_layer = defaultdict(list)
    handles = []
    n_adc = 2 ** 7 - 1

    def make_hook(name):
        def h(mod, inp, out):
            y = out.detach().float()
            scale = hook._clip.get(name, y.abs().max().item() / n_adc + 1e-8)
            mask = (y.abs() > scale * n_adc).any(dim=(0, 1))
            chans = torch.where(mask)[0].cpu().tolist()
            sets_by_layer[name].append(chans)
        return h

    for name, mod in model.named_modules():
        if name in targets:
            handles.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(eld):
            if bi >= 20:
                break
            ids = batch['input_ids'].to(DEVICE)
            model(ids)

    for h in handles:
        h.remove()

    layer_scores = {}
    for name, sets in sets_by_layer.items():
        js = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                v = jaccard(sets[i], sets[j])
                if v is not None:
                    js.append(v)
        layer_scores[name] = {
            'mean_jaccard': float(np.mean(js)) if js else 0.0,
            'mean_sat_channels': float(np.mean([len(s) for s in sets])) if sets else 0.0,
            'n_batches': len(sets),
        }

    qkv_vals = [v['mean_jaccard'] for k, v in layer_scores.items() if 'self_attn' in k]
    fc2_vals = [v['mean_jaccard'] for k, v in layer_scores.items() if k.endswith('.fc2')]
    out = {
        'layers': layer_scores,
        'summary': {
            'attn_qkv_mean_jaccard': float(np.mean(qkv_vals)) if qkv_vals else 0.0,
            'ffn_down_mean_jaccard': float(np.mean(fc2_vals)) if fc2_vals else 0.0,
        }
    }
    P(f"  attn_qkv mean Jaccard = {out['summary']['attn_qkv_mean_jaccard']:.3f}")
    P(f"  ffn_down mean Jaccard = {out['summary']['ffn_down_mean_jaccard']:.3f}")
    return out


def mechanism_support(model, tokenizer, layer_names, cld, eld):
    seq = sequence_length_ablation(model, tokenizer, layer_names)
    overlap = saturated_channel_consistency(model, layer_names, cld, eld)
    out = {'sequence_length': seq, 'channel_overlap': overlap}
    json.dump(out, open(OUT / 'p1_mechanism_support.json', 'w'), indent=2)
    P(f"  Saved → {OUT / 'p1_mechanism_support.json'}")
    return out


def main():
    t0 = time.time()
    P(f'Device: {DEVICE}')
    P(f'Results dir: {OUT}')
    P('\n### Loading OPT-125M ###')
    model, tok = load_model('facebook/opt-125m', device=DEVICE)
    layer_names = [n for n, _ in get_linear_layers(model)]
    P(f'  {len(layer_names)} layers')
    cld, eld = p0.prepare_loaders(tok)

    res_hier = evaluate_hierarchical_method(model, layer_names, cld, eld)
    res_transfer = evaluate_transfer_regret(model, layer_names, cld, eld)
    res_mech = mechanism_support(model, tok, layer_names, cld, eld)

    summary = {
        'hierarchical_cost_gain': res_hier,
        'transfer_regret': res_transfer,
        'mechanism_support': res_mech,
    }
    json.dump(summary, open(OUT / 'p1_summary.json', 'w'), indent=2)
    P(f"\nAll P1 experiments done in {(time.time() - t0)/60:.1f} min")
    P(f"Saved summary → {OUT / 'p1_summary.json'}")


if __name__ == '__main__':
    main()
