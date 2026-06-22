#!/usr/bin/env python3
"""
qwen_sensitivity.py — group-level ADC sensitivity at the USABLE Qwen2-7B operating
point (per-layer MAX-clip, 8b nominal, baseline PPL ~16).

For each functional group, drop its ADC bits (probe) while the rest stay at 8b,
measure dPPL/layer. Also reduce the whole FFN block vs the whole attention block
for a direct aggregate. Tests whether FFN > attention holds at a *usable* point.
"""
import sys, json, random
from pathlib import Path

ROOT = Path("/raid/privatedata/fantao/iccad_exp")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_inference import make_loader

CACHE = str(ROOT / "model_cache")
DEV = "cuda"
ARROW = ("/home/fantao/.cache/huggingface/datasets/wikitext/"
         "wikitext-2-raw-v1/0.0.0/"
         "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow")
OUT = ROOT / "results_control" / "qwen_sensitivity.json"
NOMINAL = 8
NUM_CALIB, NUM_EVAL, SEED = 8, 20, 0
GROUPS = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_arrow(tok, sl=512):
    import pyarrow as pa
    r = pa.ipc.open_stream(pa.memory_map(ARROW, "r"))
    text = "\n".join(r.read_all().column("text").to_pylist())
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    n = len(ids) // sl
    return [{"input_ids": ids[i * sl:(i + 1) * sl].unsqueeze(0)} for i in range(n)]


def linears(m):
    return [(n, mm) for n, mm in m.named_modules() if isinstance(mm, torch.nn.Linear)]


def qclass(name):
    n = name.lower()
    if 'lm_head' in n: return 'lm_head'
    if any(k in n for k in ['q_proj', 'k_proj', 'v_proj']): return 'attn_qkv'
    if 'o_proj' in n: return 'attn_out'
    if any(k in n for k in ['gate_proj', 'up_proj']): return 'ffn_up'
    if 'down_proj' in n: return 'ffn_down'
    return 'other'


class ADC:
    """Per-layer MAX-clip ADC (exact max, no clipping). assign: name->bits."""
    def __init__(self, model, assign, default_bits):
        self.model = model; self.assign = assign; self.default = default_bits
        self.scale = {}; self.h = []

    def calibrate(self, loader, n_batches):
        acc = {}
        def mk(name):
            def h(mod, inp, out):
                m = out.detach().float().abs().max()
                acc[name] = m if name not in acc else torch.maximum(acc[name], m)
            return h
        hs = [mm.register_forward_hook(mk(n)) for n, mm in linears(self.model)]
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= n_batches:
                    break
                self.model(b['input_ids'].to(DEV))
        for x in hs:
            x.remove()
        for n, _ in linears(self.model):
            bits = self.assign.get(n, self.default); nadc = 2 ** bits - 1
            self.scale[n] = max(acc[n].item() / nadc, 1e-8)

    def install(self):
        def mk(name):
            bits = self.assign.get(name, self.default); nadc = 2 ** bits - 1
            s = self.scale[name]
            def h(mod, inp, out):
                y = out.detach().float()
                yq = (y / s).round().clamp(-nadc, nadc) * s
                return yq.to(out.dtype)
            return h
        self.h = [mm.register_forward_hook(mk(n)) for n, mm in linears(self.model)]

    def remove(self):
        for x in self.h:
            x.remove()
        self.h = []


def ppl(model, eld):
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(eld):
            if i >= NUM_EVAL:
                break
            ids = b["input_ids"].to(DEV)
            out = model(ids, labels=ids)
            tot += out.loss.item() * ids.shape[1]; ntok += ids.shape[1]
    return float(np.exp(tot / max(ntok, 1)))


def run(model, assign, cld, eld):
    set_seed(SEED)
    a = ADC(model, assign, NOMINAL)
    a.calibrate(cld, NUM_CALIB)
    a.install()
    p = ppl(model, eld)
    a.remove()
    return p


def P(m):
    print(m, flush=True)


def main():
    name = "Qwen/Qwen2-7B"
    tok = AutoTokenizer.from_pretrained(name, cache_dir=CACHE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, cache_dir=CACHE, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True).to(DEV).eval()
    data = load_arrow(tok, 512)
    cld = make_loader(data[:NUM_CALIB + 8])
    eld = make_loader(data[64:64 + NUM_EVAL + 8])
    names = [n for n, _ in linears(model)]
    grp = {g: [n for n in names if qclass(n) == g] for g in GROUPS}
    sizes = {g: len(grp[g]) for g in GROUPS}

    base = run(model, {n: NOMINAL for n in names}, cld, eld)
    res = {"baseline_max8b": round(base, 3), "group_sizes": sizes, "probes": {}}
    P(f"baseline (max-clip 8b) PPL = {base:.3f} | sizes={sizes}")
    OUT.write_text(json.dumps(res, indent=2))

    for probe in [6, 4]:
        pr = {"per_group": {}}
        for g in GROUPS:
            assign = {n: (probe if qclass(n) == g else NOMINAL) for n in names}
            p = run(model, assign, cld, eld)
            d = p - base; sig = d / max(sizes[g], 1)
            pr["per_group"][g] = {"ppl": round(p, 3), "dppl": round(d, 4),
                                  "sigma_per_layer": round(sig, 5)}
            P(f"  8->{probe} {g:9s} PPL {p:8.3f}  dPPL {d:8.3f}  sigma/l {sig:.4f}")
        # direct aggregate blocks
        for blk, members in [("FFN", ('ffn_up', 'ffn_down')),
                             ("ATTN", ('attn_qkv', 'attn_out'))]:
            assign = {n: (probe if qclass(n) in members else NOMINAL) for n in names}
            p = run(model, assign, cld, eld)
            nl = sum(sizes[m] for m in members)
            d = p - base
            pr[f"block_{blk}"] = {"ppl": round(p, 3), "dppl": round(d, 4),
                                  "n_layers": nl, "sigma_per_layer": round(d / nl, 5)}
            P(f"  8->{probe} BLOCK {blk:4s} PPL {p:8.3f}  dPPL {d:8.3f}  sigma/l {d/nl:.4f}")
        ffn_s = pr["block_FFN"]["sigma_per_layer"]; att_s = pr["block_ATTN"]["sigma_per_layer"]
        pr["FFN_gt_ATTN"] = bool(ffn_s > att_s)
        pr["FFN_over_ATTN_ratio"] = round(ffn_s / att_s, 3) if att_s > 0 else None
        P(f"  => probe 8->{probe}: FFN sigma/l={ffn_s} vs ATTN={att_s} | FFN>ATTN={pr['FFN_gt_ATTN']} ratio={pr['FFN_over_ATTN_ratio']}")
        res["probes"][f"8to{probe}"] = pr
        OUT.write_text(json.dumps(res, indent=2))

    OUT.write_text(json.dumps(res, indent=2))
    P("DONE"); P(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
