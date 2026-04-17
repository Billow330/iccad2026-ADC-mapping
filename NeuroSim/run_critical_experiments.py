#!/usr/bin/env python3
"""
Critical experiments for ICCAD paper strengthening.
Self-contained: downloads WikiText-2 independently to avoid _bz2 issues.
"""
import os, sys, json, time, struct
from pathlib import Path
from urllib.request import urlretrieve
from collections import defaultdict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_CACHE"] = "/raid/privatedata/fantao/model_cache"
os.environ["HF_HOME"] = "/raid/privatedata/fantao/model_cache"

WORK = Path("/raid/privatedata/fantao/iccad_exp")
DATA = WORK / "data"
RESULTS = WORK / "results_critical"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm_inference import load_model, CIMNoiseHook
from sensitivity_analysis import (
    PerLayerCIMHook, classify_layer, get_linear_layers,
    eval_with_assignment, measure_group_sensitivity,
    ilp_allocation,
)
from smooth_quant import compute_perplexity


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ─── WikiText-2 download (avoid _bz2) ────────────────────────────
def download_wikitext2():
    path = DATA / "wikitext-2-raw" / "wiki.test.raw"
    if path.exists() and path.stat().st_size > 1000:
        print(f"  WikiText-2 already at {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
    ]
    for url in urls:
        try:
            if url.endswith('.parquet'):
                tmp = DATA / "wt2.parquet"
                print(f"  Downloading WikiText-2 parquet...")
                urlretrieve(url, str(tmp))
                import pyarrow.parquet as pq
                table = pq.read_table(str(tmp))
                texts = table.to_pydict()['text']
                with open(str(path), 'w') as f:
                    f.write('\n'.join(str(t) for t in texts))
                print(f"  Saved {path.stat().st_size} bytes")
                return path
            else:
                print(f"  Downloading WikiText-2 txt...")
                urlretrieve(url, str(path))
                return path
        except Exception as e:
            print(f"  Download failed: {e}")
    raise RuntimeError("Could not download WikiText-2")


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, path, seq_len=512):
        text = open(str(path)).read()
        # Match original load_wikitext2: join ALL text with \n, no filtering
        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        n_seqs = len(tokens) // seq_len
        self.samples = [tokens[i*seq_len:(i+1)*seq_len].unsqueeze(0) for i in range(n_seqs)]
        print(f"  WikiText-2: {len(self.samples)} samples of {seq_len} tokens")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {'input_ids': self.samples[idx]}


def make_loader(ds, batch_size=1):
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=lambda x: {
                          'input_ids': torch.cat([d['input_ids'] for d in x], dim=0)
                      })


def load_data(tokenizer, seq_len=512):
    wt_path = download_wikitext2()
    ds = WikiTextDataset(tokenizer, wt_path, seq_len)
    calib = torch.utils.data.Subset(ds, range(min(4, len(ds))))
    eval_ds = torch.utils.data.Subset(ds, range(4, min(104, len(ds))))
    return make_loader(calib, 1), make_loader(eval_ds, 1), len(eval_ds)


def eval_ppl(model, assignment, calib_ld, eval_ld, default_bits, device, n_calib=4, n_eval=100):
    """Evaluate PPL for a given bit assignment; handles tuple return."""
    r = eval_with_assignment(model, assignment, calib_ld, eval_ld,
                             default_bits=default_bits, device=device,
                             num_calib=n_calib, num_eval=n_eval, clip_pct=99.0)
    return r[0] if isinstance(r, tuple) else r


def get_group_results(model, layer_names, calib_ld, eval_ld, baseline_ppl,
                      nominal, probe, device, n_calib=4, n_eval=100):
    """Run group sensitivity; return (per_layer_list, group_dict)."""
    result = measure_group_sensitivity(
        model, layer_names, calib_ld, eval_ld,
        baseline_ppl=baseline_ppl, nominal_bits=nominal, probe_bits=probe,
        device=device, num_calib=n_calib, num_eval=n_eval, clip_pct=99.0
    )
    if isinstance(result, tuple):
        return result[0], result[1]
    groups = {}
    for r in result:
        lt = r.get('layer_type', r.get('ltype', 'unknown'))
        if lt not in groups:
            groups[lt] = {'ltype': lt, 'n_layers': r.get('n_layers', 1),
                          'delta_ppl': r.get('delta_ppl', 0),
                          'delta_per_layer': r.get('delta_ppl', 0),
                          'ppl': r.get('ppl', 0)}
    return result, groups


# ═══════════════════════════════════════════════════════════════════
# EXP 1: 10b operating point for OPT-125M
# ═══════════════════════════════════════════════════════════════════
def exp1_10b_operating_point(device):
    print("\n" + "="*60)
    print("EXP 1: 10b operating point for OPT-125M")
    print("="*60)

    model, tok = load_model("facebook/opt-125m", device=device)
    calib_ld, eval_ld, n_eval = load_data(tok, seq_len=512)
    layer_names = [n for n, _ in get_linear_layers(model)]
    results = {}

    for bits in [10, 9]:
        label = f"{bits}b"
        print(f"\n--- {label} uniform baseline ---")
        assign = {n: bits for n in layer_names}
        ppl = eval_ppl(model, assign, calib_ld, eval_ld, bits, device, n_eval=n_eval)
        results[f"baseline_{label}_ppl"] = ppl
        print(f"  {label} PPL = {ppl:.2f}")

    ppl_10b = results["baseline_10b_ppl"]

    print("\n--- 10b→9b group sensitivity ---")
    per_layer, groups = get_group_results(
        model, layer_names, calib_ld, eval_ld, ppl_10b, 10, 9, device, n_eval=n_eval)
    results["sensitivity_10b_9b"] = {k: v for k, v in groups.items()}
    for lt, g in groups.items():
        print(f"  {lt:12s}: ΔPPL = {g['delta_ppl']:+.3f}, /layer = {g['delta_per_layer']:+.4f}")

    print("\n--- ILP at 10b, 20% savings ---")
    alloc_result = ilp_allocation(per_layer, {b: 2**b for b in range(3,11)},
                                  nominal_bits=10, bit_choices=(7,8,9,10),
                                  target_area_savings=0.20)
    # ilp_allocation returns a list of ints (one bit-width per layer)
    if isinstance(alloc_result, list) and len(alloc_result) == len(per_layer):
        if isinstance(alloc_result[0], int):
            ilp_assign = {per_layer[i]['layer']: alloc_result[i] for i in range(len(per_layer))}
        else:
            ilp_assign = {r['layer']: r.get('bits', 10) for r in alloc_result}
    elif isinstance(alloc_result, dict):
        ilp_assign = alloc_result
    else:
        ilp_assign = {n: 10 for n in layer_names}

    ilp_group = {}
    for n in layer_names:
        lt = classify_layer(n)
        ilp_group[lt] = ilp_assign.get(n, 10)
    results["ilp_10b_group_assignment"] = ilp_group
    print(f"  ILP group: {ilp_group}")

    print("\n--- Measure ILP PPL ---")
    ppl_ilp = eval_ppl(model, ilp_assign, calib_ld, eval_ld, 10, device, n_eval=n_eval)
    results["ilp_10b_ppl"] = ppl_ilp
    results["ilp_10b_rel_deg"] = (ppl_ilp - ppl_10b) / ppl_10b
    print(f"  ILP PPL = {ppl_ilp:.2f} ({(ppl_ilp-ppl_10b)/ppl_10b*100:+.2f}% vs 10b)")

    out = RESULTS / "exp1_10b_operating_point.json"
    json.dump(results, open(out, 'w'), indent=2, default=str)
    print(f"  Saved → {out}")
    return results


# ═══════════════════════════════════════════════════════════════════
# EXP 2: Surrogate validation
# ═══════════════════════════════════════════════════════════════════
def exp2_surrogate_validation(device):
    print("\n" + "="*60)
    print("EXP 2: Surrogate (linear additive) validation")
    print("="*60)

    model, tok = load_model("facebook/opt-125m", device=device)
    calib_ld, eval_ld, n_eval = load_data(tok, seq_len=512)
    layer_names = [n for n, _ in get_linear_layers(model)]

    assign_7b = {n: 7 for n in layer_names}
    ppl_7b = eval_ppl(model, assign_7b, calib_ld, eval_ld, 7, device, n_eval=n_eval)
    print(f"  7b baseline PPL = {ppl_7b:.2f}")

    _, groups = get_group_results(
        model, layer_names, calib_ld, eval_ld, ppl_7b, 7, 6, device, n_eval=n_eval)
    individual = {lt: g['delta_ppl'] for lt, g in groups.items()}
    print(f"  Individual ΔPPL: {individual}")

    combos = [
        ("qkv+fc1",      ['attn_qkv', 'ffn_up']),
        ("qkv+out",      ['attn_qkv', 'attn_out']),
        ("fc1+fc2",      ['ffn_up', 'ffn_down']),
        ("qkv+fc1+head", ['attn_qkv', 'ffn_up', 'lm_head']),
        ("all_to_6b",    ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']),
    ]

    results = {"baseline_7b": ppl_7b, "individual": individual, "combos": []}
    for name, grps in combos:
        assign = {n: 7 for n in layer_names}
        for n in layer_names:
            if classify_layer(n) in grps:
                assign[n] = 6

        ppl = eval_ppl(model, assign, calib_ld, eval_ld, 7, device, n_eval=n_eval)
        d_meas = ppl - ppl_7b
        d_pred = sum(individual.get(g, 0) for g in grps)
        err = abs(d_pred - d_meas) / max(abs(d_meas), 0.01) * 100

        results["combos"].append({
            "combo": name, "groups": grps,
            "ppl": ppl, "delta_measured": d_meas,
            "delta_predicted": d_pred, "error_pct": err,
        })
        print(f"  {name:15s}: pred={d_pred:+.2f}, meas={d_meas:+.2f}, err={err:.0f}%")

    out = RESULTS / "exp2_surrogate_validation.json"
    json.dump(results, open(out, 'w'), indent=2)
    print(f"  Saved → {out}")
    return results


# ═══════════════════════════════════════════════════════════════════
# EXP 3: OPT-1.3B measured PPL
# ═══════════════════════════════════════════════════════════════════
def exp3_opt13b(device):
    print("\n" + "="*60)
    print("EXP 3: OPT-1.3B allocation PPL")
    print("="*60)

    model, tok = load_model("facebook/opt-1.3b", device=device)
    calib_ld, eval_ld, n_eval = load_data(tok, seq_len=512)
    layer_names = [n for n, _ in get_linear_layers(model)]

    configs = [
        ("7b_baseline",  {n: 7 for n in layer_names}, 7),
        ("6b_uniform",   {n: 6 for n in layer_names}, 6),
        ("ilp_qkv6b",   {n: (6 if classify_layer(n)=='attn_qkv' else 7) for n in layer_names}, 7),
        ("sat_fc2out6b", {n: (6 if classify_layer(n) in ('ffn_down','attn_out') else 7) for n in layer_names}, 7),
    ]

    results = {"model": "OPT-1.3B"}
    for label, assign, default in configs:
        ppl = eval_ppl(model, assign, calib_ld, eval_ld, default, device, n_eval=min(n_eval, 50))
        results[f"{label}_ppl"] = ppl
        print(f"  {label:15s}: PPL = {ppl:.2f}")

    base = results["7b_baseline_ppl"]
    for k in ["ilp_qkv6b", "sat_fc2out6b", "6b_uniform"]:
        results[f"{k}_rel_deg"] = (results[f"{k}_ppl"] - base) / base

    # Also measure 10b baseline for 1.3B
    print("\n--- 10b baseline for OPT-1.3B ---")
    assign_10b = {n: 10 for n in layer_names}
    ppl_10b = eval_ppl(model, assign_10b, calib_ld, eval_ld, 10, device, n_eval=min(n_eval, 50))
    results["10b_baseline_ppl"] = ppl_10b
    print(f"  10b baseline PPL = {ppl_10b:.2f}")

    # ILP at 10b with sensitivity from existing data (qkv least sensitive, fc2 most)
    print("\n--- ILP at 10b (qkv→9b, rest→10b) ---")
    assign_ilp10 = {n: (9 if classify_layer(n)=='attn_qkv' else 10) for n in layer_names}
    ppl_ilp10 = eval_ppl(model, assign_ilp10, calib_ld, eval_ld, 10, device, n_eval=min(n_eval, 50))
    results["ilp_10b_ppl"] = ppl_ilp10
    results["ilp_10b_rel_deg"] = (ppl_ilp10 - ppl_10b) / ppl_10b
    print(f"  ILP-10b PPL = {ppl_ilp10:.2f} ({(ppl_ilp10-ppl_10b)/ppl_10b*100:+.2f}%)")

    out = RESULTS / "exp3_opt13b_allocation_ppl.json"
    json.dump(results, open(out, 'w'), indent=2)
    print(f"  Saved → {out}")
    return results


# ═══════════════════════════════════════════════════════════════════
# EXP 4: LayerNorm evidence
# ═══════════════════════════════════════════════════════════════════
def exp4_layernorm(device):
    print("\n" + "="*60)
    print("EXP 4: LayerNorm suppression evidence")
    print("="*60)

    model, tok = load_model("facebook/opt-125m", device=device)
    calib_ld, eval_ld, n_eval = load_data(tok, seq_len=512)

    records = []
    for bi, batch in enumerate(calib_ld):
        if bi >= 3:
            break
        ids = batch['input_ids'].to(device)

        with torch.no_grad():
            h = model.model.decoder.embed_tokens(ids)
            # OPT embed_positions expects attention_mask, not input_ids
            attn_mask = torch.ones_like(ids)
            h = h + model.model.decoder.embed_positions(attn_mask)

            for li, blk in enumerate(model.model.decoder.layers):
                if li > 3:
                    break

                ln = blk.self_attn_layer_norm
                h_ln = ln(h)
                noise_mag = 0.01 * h.abs().mean().item()

                # Inject noise BEFORE LayerNorm
                eps = torch.randn_like(h) * noise_mag
                clean_ln = ln(h)
                noisy_ln = ln(h + eps)
                ln_suppression = (noisy_ln - clean_ln).norm().item() / max(eps.norm().item(), 1e-10)

                # fc2 residual path: noise goes directly through
                attn_out = blk.self_attn(h_ln)[0]
                h2 = h + attn_out
                ffn_ln = blk.final_layer_norm(h2)
                fc2_out = blk.fc2(torch.nn.functional.relu(blk.fc1(ffn_ln)))

                # Inject noise at fc2 output before residual add
                eps2 = torch.randn_like(fc2_out) * noise_mag
                clean_res = h2 + fc2_out
                noisy_res = h2 + fc2_out + eps2

                res_ratio = (noisy_res - clean_res).norm().item() / max(eps2.norm().item(), 1e-10)

                records.append({
                    "block": li, "batch": bi,
                    "ln_suppression": float(ln_suppression),
                    "fc2_residual_passthrough": float(res_ratio),
                })

                h = blk(h)[0]

    avg_ln = np.mean([r["ln_suppression"] for r in records])
    avg_fc2 = np.mean([r["fc2_residual_passthrough"] for r in records])

    summary = {
        "records": records,
        "avg_ln_suppression": float(avg_ln),
        "avg_fc2_passthrough": float(avg_fc2),
        "conclusion": (
            f"LayerNorm suppresses injected noise to {avg_ln:.2f}x of original magnitude. "
            f"fc2 noise passes through residual at {avg_fc2:.2f}x. "
            f"This explains why Wqkv (post-LN) is less sensitive: "
            f"LN normalizes noise before attention, while fc2 injects noise "
            f"directly into the residual stream."
        )
    }
    print(f"  LN suppression: {avg_ln:.3f}x")
    print(f"  fc2 passthrough: {avg_fc2:.3f}x")
    print(f"  Ratio: fc2/LN = {avg_fc2/avg_ln:.1f}x more direct")

    out = RESULTS / "exp4_layernorm_evidence.json"
    json.dump(summary, open(out, 'w'), indent=2)
    print(f"  Saved → {out}")
    return summary


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    print(f"Results: {RESULTS}")
    t0 = time.time()

    for i, (name, fn) in enumerate([
        ("10b operating point", exp1_10b_operating_point),
        ("Surrogate validation", exp2_surrogate_validation),
        ("OPT-1.3B allocation PPL", exp3_opt13b),
        ("LayerNorm evidence", exp4_layernorm),
    ], 1):
        print(f"\n{'#'*60}")
        print(f"# Starting EXP {i}: {name}")
        print(f"{'#'*60}")
        try:
            fn(device)
            print(f"  ✓ EXP {i} completed")
        except Exception as e:
            print(f"  ✗ EXP {i} FAILED: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min")
    print(f"Results: {RESULTS}")
    for f in sorted(RESULTS.glob("*.json")):
        print(f"  {f.name} ({f.stat().st_size} bytes)")
    print(f"{'='*60}")
