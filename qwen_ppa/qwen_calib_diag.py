#!/usr/bin/env python3
"""
qwen_calib_diag.py (v2, efficient) — find a USABLE Qwen2-7B CIM operating point.

Replicates the paper's ADC quantization EXACTLY (act_scale = clip_p / (2^b - 1);
y_q = round(y/act_scale).clamp(+-(2^b-1)) * act_scale) but with cheap strided
calibration sampling (no torch.randperm on lm_head's 78M outputs).

Sweeps full-scale clip percentile x ADC bits to locate a config with usable PPL.
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
DEVICE = "cuda"
ARROW = ("/home/fantao/.cache/huggingface/datasets/wikitext/"
         "wikitext-2-raw-v1/0.0.0/"
         "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow")
OUT = ROOT / "results_control" / "qwen_calib_diag.json"
NUM_CALIB, NUM_EVAL, SEED = 8, 20, 0


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_arrow(tok, seq_len=512):
    import pyarrow as pa
    r = pa.ipc.open_stream(pa.memory_map(ARROW, "r"))
    text = "\n".join(r.read_all().column("text").to_pylist())
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    n = len(ids) // seq_len
    return [{"input_ids": ids[i * seq_len:(i + 1) * seq_len].unsqueeze(0)} for i in range(n)]


def linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


class ADC:
    """Self-contained per-layer ADC quant; same math as PerLayerCIMHook, fast calib."""
    def __init__(self, model, assign, default_bits, clip):
        self.model = model; self.assign = assign; self.default = default_bits
        self.clip = clip; self.scale = {}; self.h = []

    def calibrate(self, loader, n_batches):
        samp = {}
        def mk(name):
            def h(mod, inp, out):
                v = out.detach().abs().float().flatten()
                k = 16384
                if v.numel() > k:
                    stride = v.numel() // k
                    v = v[::stride][:k]
                samp.setdefault(name, []).append(v.cpu())
            return h
        hs = [m.register_forward_hook(mk(n)) for n, m in linears(self.model)]
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= n_batches:
                    break
                self.model(b['input_ids'].to(DEVICE))
        for x in hs:
            x.remove()
        for n, vs in samp.items():
            bits = self.assign.get(n, self.default); n_adc = 2 ** bits - 1
            allv = torch.cat(vs)
            p = allv.max().item() if self.clip >= 100 else torch.quantile(allv, self.clip / 100.0).item()
            self.scale[n] = max(p / n_adc, 1e-8)

    def install(self):
        def mk(name):
            bits = self.assign.get(name, self.default); n_adc = 2 ** bits - 1
            s = self.scale.get(name, 1.0)
            def h(mod, inp, out):
                y = out.detach().float()
                yq = (y / s).round().clamp(-n_adc, n_adc) * s
                return yq.to(out.dtype)
            return h
        self.h = [m.register_forward_hook(mk(n)) for n, m in linears(self.model)]

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
            ids = b["input_ids"].to(DEVICE)
            out = model(ids, labels=ids)
            tot += out.loss.item() * ids.shape[1]; ntok += ids.shape[1]
    return float(np.exp(tot / max(ntok, 1)))


def ppl_adc(model, assign, default_bits, clip, cld, eld):
    set_seed(SEED)
    a = ADC(model, assign, default_bits, clip)
    a.calibrate(cld, NUM_CALIB)
    a.install()
    p = ppl(model, eld)
    a.remove()
    return p


def P(m):
    print(m, flush=True)


def main():
    name = "Qwen/Qwen2-7B"
    P(f"loading {name} (fp32) ...")
    tok = AutoTokenizer.from_pretrained(name, cache_dir=CACHE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, cache_dir=CACHE, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True).to(DEVICE).eval()
    data = load_arrow(tok, 512)
    cld = make_loader(data[:NUM_CALIB + 8])
    eld = make_loader(data[64:64 + NUM_EVAL + 8])
    names = [n for n, _ in linears(model)]
    P(f"n_linear={len(names)} | calib={NUM_CALIB} eval={NUM_EVAL}")

    res = {"clean": round(ppl(model, eld), 3)}
    P(f"clean PPL = {res['clean']}")

    for clip in [99.0, 99.9, 100.0]:
        for b in [12, 10, 8]:
            assign = {n: b for n in names}
            p = ppl_adc(model, assign, b, clip, cld, eld)
            res[f"clip{clip}_adc{b}"] = round(p, 3)
            P(f"  clip={clip:<6} adc={b:>2}b -> PPL {p:.3f}")
            OUT.write_text(json.dumps(res, indent=2))

    OUT.write_text(json.dumps(res, indent=2))
    P("DONE"); P(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
