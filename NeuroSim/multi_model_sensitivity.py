"""
multi_model_sensitivity.py — Cross-Model & Cross-Architecture Validation
=========================================================================
P1-A1: Large model validation (Pythia-1.4B, Pythia-2.8B)
P1-B1: Non-OPT architecture (Pythia = GPT-NeoX architecture, different from OPT)

Pythia models use GPT-NeoX architecture:
  - Parallel attention (QKV computed together)
  - Rotary position embeddings (RoPE)
  - Different normalization placement
This provides genuine cross-architecture validation.
"""

import sys, json, gc
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'pytorch-quantization'))

import torch
import numpy as np

RESULTS_DIR = ROOT / 'results' / 'multi_model'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = '/tmp/fantaog_iccad/model_cache'
DEVICE = 'cpu'


def classify_layer_generic(name):
    """Classify layer type for any transformer architecture."""
    n = name.lower()

    # LM head / output projection
    if any(k in n for k in ['lm_head', 'embed_out']):
        return 'lm_head'

    # Attention QKV
    if any(k in n for k in ['q_proj', 'k_proj', 'v_proj', 'query_key_value',
                             'qkv_proj', 'c_attn']):
        return 'attn_qkv'

    # Attention output
    if any(k in n for k in ['out_proj', 'o_proj', 'dense_after', 'c_proj']):
        if 'mlp' not in n and 'fc' not in n and 'ffn' not in n:
            return 'attn_out'

    # FFN up / gate
    if any(k in n for k in ['fc1', 'dense_h_to_4h', 'up_proj', 'gate_proj',
                             'w1', 'w3', 'c_fc']):
        return 'ffn_up'

    # FFN down
    if any(k in n for k in ['fc2', 'dense_4h_to_h', 'down_proj', 'w2', 'c_proj']):
        if 'attn' not in n and 'self' not in n:
            return 'ffn_down'

    # Fallback for attention output in some architectures
    if 'dense' in n and 'attention' in n:
        return 'attn_out'

    return 'other'


def load_model_generic(model_name, cache_dir, device='cpu'):
    """Load any HuggingFace causal LM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {n_params:.0f}M parameters")
    return model, tokenizer


def load_wikitext2(tokenizer, seq_len=512, split='test'):
    """Load WikiText-2 and tokenize."""
    from datasets import load_dataset
    try:
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    except Exception:
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split,
                          download_mode='force_redownload')

    text = '\n\n'.join(ds['text'])
    enc = tokenizer(text, return_tensors='pt', truncation=False)
    ids = enc['input_ids'][0]

    chunks = []
    for i in range(0, len(ids) - seq_len, seq_len):
        chunks.append(ids[i:i+seq_len].unsqueeze(0))
    print(f"  WikiText-2: {len(chunks)} chunks of {seq_len} tokens")
    return chunks


def compute_ppl(model, chunks, max_batches=10):
    """Compute perplexity on chunks."""
    import torch.nn.functional as F
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            if i >= max_batches:
                break
            ids = chunk.to(DEVICE)
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * ids.shape[1]
            total_tokens += ids.shape[1]
    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


class SimpleCIMHook:
    """Lightweight CIM ADC noise hook for sensitivity measurement."""

    def __init__(self, model, bit_assignment, default_bits=7, clip_pct=99.0):
        self.model = model
        self.bit_assignment = bit_assignment
        self.default_bits = default_bits
        self.clip_pct = clip_pct
        self.hooks = []
        self.fullscale = {}

    def calibrate(self, chunks, num_batches=4):
        stats = defaultdict(list)
        hooks = []
        for name, mod in self.model.named_modules():
            if hasattr(mod, 'weight') and mod.weight.ndim == 2:
                def make_hook(n):
                    def fn(module, inp, out):
                        if isinstance(out, tuple):
                            out = out[0]
                        stats[n].append(out.detach().abs().cpu())
                    return fn
                hooks.append(mod.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                if i >= num_batches:
                    break
                self.model(chunk.to(DEVICE))

        for h in hooks:
            h.remove()

        for name, tensors in stats.items():
            cat = torch.cat(tensors, dim=0)
            if self.clip_pct >= 100:
                self.fullscale[name] = cat.max().item()
            else:
                self.fullscale[name] = torch.quantile(
                    cat.float().flatten(), self.clip_pct / 100.0
                ).item()

    def install(self):
        for name, mod in self.model.named_modules():
            if name not in self.fullscale:
                continue
            bits = self.bit_assignment.get(name, self.default_bits)
            vfs = self.fullscale[name]

            def make_hook(b, v):
                def fn(module, inp, out):
                    is_tuple = isinstance(out, tuple)
                    tensor = out[0] if is_tuple else out
                    levels = 2 ** b
                    step = 2 * v / levels if v > 0 else 1.0
                    q = torch.clamp(tensor, -v, v)
                    q = torch.round(q / step) * step
                    return (q,) + out[1:] if is_tuple else q
                return fn

            h = mod.register_forward_hook(make_hook(bits, vfs))
            self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def measure_sensitivity_for_model(model_name, nominal_bits, probe_bits,
                                  num_calib=4, num_eval=10):
    """Full sensitivity measurement pipeline for one model."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"  Nominal: {nominal_bits}b, Probe: {probe_bits}b")
    print(f"{'='*70}")

    model, tokenizer = load_model_generic(model_name, CACHE_DIR, DEVICE)
    chunks = load_wikitext2(tokenizer, seq_len=512)

    # Identify linear layers and group them
    layer_names = []
    for name, mod in model.named_modules():
        if hasattr(mod, 'weight') and mod.weight.ndim == 2:
            layer_names.append(name)

    groups = defaultdict(list)
    for name in layer_names:
        lt = classify_layer_generic(name)
        groups[lt].append(name)

    print(f"\n  Layer groups:")
    for lt, names in sorted(groups.items()):
        print(f"    {lt:12s}: {len(names)} layers")
        if len(names) <= 3:
            for n in names:
                print(f"      - {n}")

    # Remove 'other' group if present
    groups.pop('other', None)

    # Baseline
    baseline_assign = {n: nominal_bits for n in layer_names}
    hook = SimpleCIMHook(model, baseline_assign, nominal_bits)
    hook.calibrate(chunks, num_calib)
    hook.install()
    baseline_ppl = compute_ppl(model, chunks, num_eval)
    hook.remove()
    print(f"\n  Baseline PPL ({nominal_bits}b): {baseline_ppl:.2f}")

    # Per-group sensitivity
    results = {}
    for ltype, names in sorted(groups.items()):
        assign = {n: nominal_bits for n in layer_names}
        for n in names:
            assign[n] = probe_bits

        hook = SimpleCIMHook(model, assign, nominal_bits)
        hook.calibrate(chunks, num_calib)
        hook.install()
        ppl = compute_ppl(model, chunks, num_eval)
        hook.remove()

        delta = ppl - baseline_ppl
        delta_per_layer = delta / max(len(names), 1)
        results[ltype] = {
            'n_layers': len(names),
            'ppl': float(ppl),
            'delta_ppl': float(delta),
            'delta_per_layer': float(delta_per_layer),
        }
        print(f"  {ltype:12s} ({len(names):3d} layers): PPL={ppl:.2f}, "
              f"ΔPPL/layer={delta_per_layer:+.4f}")

    # Ranking
    ranked = sorted(results.items(), key=lambda x: x[1]['delta_per_layer'], reverse=True)
    print(f"\n  Sensitivity ranking (most → least sensitive):")
    for i, (lt, d) in enumerate(ranked):
        print(f"    {i+1}. {lt:12s} ΔPPL/layer={d['delta_per_layer']:+.4f}")

    # Cleanup
    del model, tokenizer
    gc.collect()

    return {
        'model': model_name,
        'nominal_bits': nominal_bits,
        'probe_bits': probe_bits,
        'baseline_ppl': float(baseline_ppl),
        'groups': results,
        'ranking': [lt for lt, _ in ranked],
    }


def main():
    all_results = {}

    # ── Model 1: OPT-125M (reference, OPT architecture) ─────────────────
    all_results['opt-125m'] = measure_sensitivity_for_model(
        'facebook/opt-125m', nominal_bits=7, probe_bits=6,
        num_calib=4, num_eval=10
    )

    # ── Model 2: Pythia-410M (GPT-NeoX architecture, different from OPT) ─
    all_results['pythia-410m'] = measure_sensitivity_for_model(
        'EleutherAI/pythia-410m', nominal_bits=7, probe_bits=6,
        num_calib=4, num_eval=10
    )

    # ── Model 3: Pythia-1.4B (larger GPT-NeoX) ──────────────────────────
    all_results['pythia-1.4b'] = measure_sensitivity_for_model(
        'EleutherAI/pythia-1.4b', nominal_bits=8, probe_bits=7,
        num_calib=4, num_eval=10
    )

    # ── Cross-model comparison ───────────────────────────────────────────
    print("\n" + "="*70)
    print("CROSS-MODEL RANKING COMPARISON")
    print("="*70)
    print(f"{'Model':<20s} {'Most sensitive':>15s} → {'Least sensitive':>15s}")
    print("-"*70)
    for name, result in all_results.items():
        ranking = result['ranking']
        print(f"  {name:<18s}: {' → '.join(ranking)}")

    # Check if ffn_down is always most sensitive and attn_qkv least
    inversion_holds = True
    for name, result in all_results.items():
        ranking = result['ranking']
        if len(ranking) >= 2:
            if ranking[0] != 'ffn_down':
                print(f"  WARNING: {name} most sensitive is {ranking[0]}, not ffn_down")
                inversion_holds = False
            if ranking[-1] != 'attn_qkv':
                print(f"  WARNING: {name} least sensitive is {ranking[-1]}, not attn_qkv")
                inversion_holds = False

    print(f"\n  Inversion holds across all models: {'YES' if inversion_holds else 'NO'}")

    # Save
    out_path = RESULTS_DIR / 'multi_model_sensitivity.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
