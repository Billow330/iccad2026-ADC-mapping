"""
run_gpu_experiments.py — All remaining ICCAD experiments on GPU server
=====================================================================
Runs: (1) Calibration robustness, (2) Downstream tasks (PIQA/BoolQ),
      (3) Per-layer depth-refined sensitivity
Zero dependency on datasets/lm_eval — uses only torch + transformers + urllib.
"""

import os, sys, json, time, math, re
from pathlib import Path
from collections import defaultdict
from urllib.request import urlretrieve
import zipfile

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda:0"
WORK = Path("/raid/privatedata/fantao/iccad_exp")
RESULTS = WORK / "results"
DATA = WORK / "data"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/opt-125m"
CACHE_DIR = WORK / "model_cache"


# ═══════════════════════════════════════════════════════════════════
# Data Download (no datasets dependency)
# ═══════════════════════════════════════════════════════════════════

def download_wikitext2():
    """Download WikiText-2 raw text."""
    path = DATA / "wikitext-2-raw" / "wiki.test.raw"
    if path.exists():
        return path
    urls = [
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
    ]
    raw_dir = DATA / "wikitext-2-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            if url.endswith('.parquet'):
                tmp = DATA / "wt2_test.parquet"
                print(f"Downloading WikiText-2 parquet...")
                urlretrieve(url, str(tmp))
                import pyarrow.parquet as pq
                table = pq.read_table(str(tmp))
                texts = table.to_pydict()['text']
                with open(str(path), 'w') as f:
                    f.write('\n'.join(str(t) for t in texts))
                return path
            else:
                print(f"Downloading WikiText-2 txt...")
                urlretrieve(url, str(path))
                return path
        except Exception as e:
            print(f"  Failed ({e}), trying next...")
    # Last resort: generate from transformers tokenizer
    print("All downloads failed, using minimal text...")
    with open(str(path), 'w') as f:
        f.write("The quick brown fox jumps over the lazy dog.\n" * 1000)
    return path


def download_piqa():
    """Download PIQA validation set."""
    labels_path = DATA / "piqa_valid_labels.lst"
    data_path = DATA / "piqa_valid.jsonl"
    if labels_path.exists() and data_path.exists():
        return data_path, labels_path
    base = "https://yonatanbisk.com/piqa/data"
    if not data_path.exists():
        print("Downloading PIQA valid.jsonl...")
        urlretrieve(f"{base}/valid.jsonl", str(data_path))
    if not labels_path.exists():
        print("Downloading PIQA valid-labels.lst...")
        urlretrieve(f"{base}/valid-labels.lst", str(labels_path))
    return data_path, labels_path


def download_boolq():
    """Download BoolQ via HuggingFace API (JSON)."""
    path = DATA / "boolq_val.jsonl"
    if path.exists():
        return path
    print("Downloading BoolQ...")
    url = "https://huggingface.co/datasets/google/boolq/resolve/main/data/validation-00000-of-00001.parquet"
    parquet_path = DATA / "boolq_val.parquet"
    try:
        urlretrieve(url, str(parquet_path))
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_path))
        rows = table.to_pydict()
        with open(str(path), 'w') as f:
            for i in range(len(rows['question'])):
                item = {k: rows[k][i] for k in rows}
                f.write(json.dumps(item) + '\n')
        return path
    except Exception as e:
        print(f"Parquet download failed ({e}), trying JSONL fallback...")
        url2 = "https://raw.githubusercontent.com/google-research-datasets/boolean-questions/master/dev.jsonl"
        try:
            urlretrieve(url2, str(path))
            return path
        except Exception as e2:
            print(f"BoolQ download failed: {e2}")
            return None


def load_wikitext2_texts(path, min_len=50):
    with open(str(path), 'r') as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > min_len
                  and not p.strip().startswith('=')]
    return paragraphs


def load_piqa(data_path, labels_path):
    items = []
    with open(str(data_path)) as f:
        for line in f:
            items.append(json.loads(line))
    with open(str(labels_path)) as f:
        labels = [int(l.strip()) for l in f if l.strip()]
    for i, item in enumerate(items):
        if i < len(labels):
            item['label'] = labels[i]
    return items


def load_boolq(path):
    items = []
    with open(str(path)) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


# ═══════════════════════════════════════════════════════════════════
# CIM Noise Simulation
# ═══════════════════════════════════════════════════════════════════

class CIMNoiseHook:
    def __init__(self, model, tokenizer, default_bits=7, calib_method='p99'):
        self.model = model
        self.tokenizer = tokenizer
        self.default_bits = default_bits
        self.calib_method = calib_method
        self.layer_bits = {}
        self.hooks = []
        self.vfs = {}

    def set_layer_bits(self, layer_bits):
        self.layer_bits = dict(layer_bits)

    def calibrate(self, texts, n_batches=4):
        self.model.eval()
        accum = defaultdict(list)

        for text in texts[:n_batches]:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                   max_length=512).to(DEVICE)
            hooks_list = []
            def make_hook(name):
                def hook_fn(module, inp, out):
                    with torch.no_grad():
                        o = out[0] if isinstance(out, tuple) else out
                        accum[name].append(o.abs().detach().cpu())
                return hook_fn
            for name, mod in self.model.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    hooks_list.append(mod.register_forward_hook(make_hook(name)))
            with torch.no_grad():
                self.model(**inputs)
            for h in hooks_list:
                h.remove()

        for name in accum:
            vals = torch.cat([v.flatten() for v in accum[name]])
            if len(vals) > 1_000_000:
                idx = torch.randperm(len(vals))[:1_000_000]
                vals = vals[idx]
            if self.calib_method == 'max':
                self.vfs[name] = vals.max().item()
            elif self.calib_method == 'p99':
                self.vfs[name] = torch.quantile(vals.float(), 0.99).item()
            elif self.calib_method == 'p995':
                self.vfs[name] = torch.quantile(vals.float(), 0.995).item()
            else:
                self.vfs[name] = vals.max().item()

    def attach(self):
        self.hooks = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                self.hooks.append(mod.register_forward_hook(self._make_hook(name)))

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _get_bits(self, name):
        for key, bits in self.layer_bits.items():
            if key in name:
                return bits
        return self.default_bits

    def _make_hook(self, name):
        def hook_fn(module, inp, out):
            bits = self._get_bits(name)
            if bits >= 16:
                return out
            vfs = self.vfs.get(name, 1.0)
            if vfs <= 0:
                return out
            n_levels = 2 ** bits
            step = 2 * vfs / n_levels
            if isinstance(out, tuple):
                o = out[0]
                o = torch.round(torch.clamp(o, -vfs, vfs) / step) * step
                return (o,) + out[1:]
            out = torch.round(torch.clamp(out, -vfs, vfs) / step) * step
            return out
        return hook_fn


def get_layer_group(name):
    if any(x in name for x in ('q_proj', 'k_proj', 'v_proj')):
        return 'attn_qkv'
    if 'out_proj' in name:
        return 'attn_out'
    if 'fc1' in name:
        return 'ffn_up'
    if 'fc2' in name:
        return 'ffn_down'
    if 'lm_head' in name:
        return 'lm_head'
    return None


# ═══════════════════════════════════════════════════════════════════
# PPL Evaluation
# ═══════════════════════════════════════════════════════════════════

def eval_ppl(model, tokenizer, texts, n_batches=100, max_length=512):
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for text in texts[:n_batches]:
        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                          max_length=max_length).to(DEVICE)
        if inputs['input_ids'].shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(**inputs, labels=inputs['input_ids'])
            n_tok = inputs['input_ids'].shape[1] - 1
            total_loss += out.loss.item() * n_tok
            total_tokens += n_tok
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


# ═══════════════════════════════════════════════════════════════════
# EXP 1: Calibration Robustness
# ═══════════════════════════════════════════════════════════════════

def exp_calibration(model, tokenizer, texts):
    print("\n" + "="*60)
    print("EXP 1: Calibration Robustness (max / p99 / p99.5)")
    print("="*60)

    groups = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']
    methods = ['max', 'p99', 'p995']
    results = {}

    for method in methods:
        print(f"\n--- Calibration: {method} ---")
        cim = CIMNoiseHook(model, tokenizer, default_bits=7, calib_method=method)
        cim.calibrate(texts[:10])

        cim.set_layer_bits({})
        cim.attach()
        bl = eval_ppl(model, tokenizer, texts, n_batches=30)
        cim.detach()
        print(f"  Baseline 7b: {bl:.2f}")

        sens = {}
        for g in groups:
            lbits = {}
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Linear) and get_layer_group(n) == g:
                    lbits[n] = 6
            cim.set_layer_bits(lbits)
            cim.attach()
            p = eval_ppl(model, tokenizer, texts, n_batches=30)
            cim.detach()
            nl = sum(1 for n, m in model.named_modules()
                     if isinstance(m, torch.nn.Linear) and get_layer_group(n) == g)
            sens[g] = (p - bl) / max(nl, 1)
            print(f"  {g}: PPL={p:.2f}, dPPL/layer={sens[g]:+.4f}")

        ranking = sorted(groups, key=lambda g: sens[g], reverse=True)
        results[method] = {'baseline': bl, 'sensitivity': sens, 'ranking': ranking}

    print("\n--- Ranking Consistency ---")
    for m in methods:
        print(f"  {m}: {results[m]['ranking']}")
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            r1, r2 = results[methods[i]]['ranking'], results[methods[j]]['ranking']
            match = sum(1 for a, b in zip(r1, r2) if a == b)
            print(f"  {methods[i]} vs {methods[j]}: {match}/5 match")

    with open(str(RESULTS / "calibration_robustness.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: calibration_robustness.json")
    return results


# ═══════════════════════════════════════════════════════════════════
# EXP 2: Downstream Tasks
# ═══════════════════════════════════════════════════════════════════

def eval_piqa_task(model, tokenizer, items, n=500):
    correct = total = 0
    for item in items[:n]:
        goal = item['goal']
        prompts = [f"{goal} {item['sol1']}", f"{goal} {item['sol2']}"]
        scores = []
        for p in prompts:
            inp = tokenizer(p, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
            with torch.no_grad():
                out = model(**inp, labels=inp['input_ids'])
                scores.append(-out.loss.item())
        pred = 0 if scores[0] > scores[1] else 1
        if pred == item.get('label', -1):
            correct += 1
        total += 1
    return correct / total if total else 0


def eval_boolq_task(model, tokenizer, items, n=500):
    correct = total = 0
    for item in items[:n]:
        passage = str(item.get('passage', ''))[:300]
        question = str(item.get('question', ''))
        label = item.get('answer', False)
        prompts = [f"{passage} {question}? Yes", f"{passage} {question}? No"]
        scores = []
        for p in prompts:
            inp = tokenizer(p, return_tensors='pt', truncation=True, max_length=384).to(DEVICE)
            with torch.no_grad():
                out = model(**inp, labels=inp['input_ids'])
                scores.append(-out.loss.item())
        pred = True if scores[0] > scores[1] else False
        if pred == label:
            correct += 1
        total += 1
    return correct / total if total else 0


def exp_downstream(model, tokenizer, calib_texts, piqa_items, boolq_items):
    print("\n" + "="*60)
    print("EXP 2: Downstream Tasks (PIQA + BoolQ)")
    print("="*60)

    configs = [
        ('FP32', 16, {}),
        ('CIM_7b', 7, {}),
        ('ILP_20pct', 7, {'q_proj': 6, 'k_proj': 6, 'v_proj': 6, 'fc1': 6}),
        ('Uniform_6b', 6, {}),
    ]

    results = {}
    for cfg_name, bits, overrides in configs:
        print(f"\n--- {cfg_name} ---")
        cim = CIMNoiseHook(model, tokenizer, default_bits=bits, calib_method='p99')
        cim.calibrate(calib_texts[:10])

        if overrides:
            lbits = {}
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    for k, b in overrides.items():
                        if k in n:
                            lbits[n] = b
            cim.set_layer_bits(lbits)

        if bits < 16:
            cim.attach()

        piqa_acc = eval_piqa_task(model, tokenizer, piqa_items, n=500)
        boolq_acc = 0.0
        if boolq_items:
            boolq_acc = eval_boolq_task(model, tokenizer, boolq_items, n=500)

        if bits < 16:
            cim.detach()

        results[cfg_name] = {'piqa': piqa_acc, 'boolq': boolq_acc}
        print(f"  PIQA: {piqa_acc*100:.1f}%  BoolQ: {boolq_acc*100:.1f}%")

    with open(str(RESULTS / "downstream_tasks.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: downstream_tasks.json")
    return results


# ═══════════════════════════════════════════════════════════════════
# EXP 3: Per-Layer Sensitivity (sampled blocks)
# ═══════════════════════════════════════════════════════════════════

def exp_perlayer(model, tokenizer, texts):
    print("\n" + "="*60)
    print("EXP 3: Per-Layer Sensitivity (blocks 1, 5, 10)")
    print("="*60)

    cim = CIMNoiseHook(model, tokenizer, default_bits=7, calib_method='p99')
    cim.calibrate(texts[:10])

    cim.set_layer_bits({})
    cim.attach()
    bl = eval_ppl(model, tokenizer, texts, n_batches=30)
    cim.detach()
    print(f"Baseline 7b: {bl:.2f}")

    blocks = [1, 5, 10]
    layer_types_map = {
        'q_proj': 'self_attn.q_proj',
        'k_proj': 'self_attn.k_proj',
        'v_proj': 'self_attn.v_proj',
        'out_proj': 'self_attn.out_proj',
        'fc1': 'fc1',
        'fc2': 'fc2',
    }

    results = {'baseline': bl, 'per_layer': {}}

    for bidx in blocks:
        print(f"\n--- Block {bidx} ---")
        results['per_layer'][f'block_{bidx}'] = {}
        for lt, path in layer_types_map.items():
            target = f'model.decoder.layers.{bidx}.{path}'
            lbits = {}
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Linear) and target in n:
                    lbits[n] = 6
            if not lbits:
                for n, m in model.named_modules():
                    if isinstance(m, torch.nn.Linear) and f'.{bidx}.' in n and lt in n:
                        lbits[n] = 6
            if not lbits:
                print(f"  {lt}: NOT FOUND")
                continue
            cim.set_layer_bits(lbits)
            cim.attach()
            p = eval_ppl(model, tokenizer, texts, n_batches=30)
            cim.detach()
            dppl = p - bl
            grp = 'attn_qkv' if lt in ('q_proj','k_proj','v_proj') else \
                  'attn_out' if lt == 'out_proj' else \
                  'ffn_up' if lt == 'fc1' else \
                  'ffn_down' if lt == 'fc2' else 'unknown'
            results['per_layer'][f'block_{bidx}'][lt] = {
                'ppl': p, 'dppl': dppl, 'group': grp
            }
            print(f"  {lt} ({grp}): PPL={p:.2f}, dPPL={dppl:+.4f}")

    with open(str(RESULTS / "perlayer_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: perlayer_sensitivity.json")
    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print(f"Work: {WORK}")
    print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")

    # Download data
    wt2_path = download_wikitext2()
    piqa_data, piqa_labels = download_piqa()
    boolq_path = download_boolq()

    wt2_texts = load_wikitext2_texts(wt2_path)
    piqa_items = load_piqa(piqa_data, piqa_labels)
    boolq_items = load_boolq(boolq_path) if boolq_path and boolq_path.exists() else []
    print(f"WikiText-2: {len(wt2_texts)} paragraphs")
    print(f"PIQA: {len(piqa_items)} items")
    print(f"BoolQ: {len(boolq_items)} items")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))
    mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR)).to(DEVICE)
    mod.eval()
    print(f"Loaded. Params: {sum(p.numel() for p in mod.parameters()):,}")

    t0 = time.time()

    exp_calibration(mod, tok, wt2_texts)
    exp_downstream(mod, tok, wt2_texts[:10], piqa_items, boolq_items)
    exp_perlayer(mod, tok, wt2_texts)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results: {RESULTS}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
