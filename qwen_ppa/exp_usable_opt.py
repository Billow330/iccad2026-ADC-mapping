#!/usr/bin/env python3
"""
exp_usable_opt.py — locate the source of OPT-125M's high CIM PPL and test whether the
ADC sensitivity ordering (FFN > attention) survives at a USABLE operating point.

Step 1 (diagnose the 306): measure PPL at FP32 / INT8-weights-only(no ADC) /
  +ADC7 p99-clip / +ADC7 max-clip / +ADC8 max-clip.  This isolates whether the
  degradation comes from weight quantization or from p99 ADC clipping.
Step 2 (usable-point ordering): at the lowest-PPL ADC config (the usable point),
  drop each layer group by one ADC bit and measure dPPL -> FFN vs attention aggregate.

Answers #707D-Q1/C1 and #707C-C7 (does the ordering persist when the model is usable).
"""
import sys, json
from pathlib import Path
ROOT = Path("/raid/privatedata/fantao/iccad_exp"); sys.path.insert(0, str(ROOT))
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_inference import make_loader

CACHE = str(ROOT / "model_cache"); DEV = "cuda"
ARROW = ("/home/fantao/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/"
         "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow")
OUT = ROOT / "results_control" / "usable_opt.json"
MODEL = "facebook/opt-125m"
NUM_CALIB, NEVAL = 16, 40
GROUPS = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down']


def load_arrow(tok, sl=512):
    import pyarrow as pa
    r = pa.ipc.open_stream(pa.memory_map(ARROW, "r"))
    text = "\n".join(r.read_all().column("text").to_pylist())
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    n = len(ids) // sl
    return [{"input_ids": ids[i*sl:(i+1)*sl].unsqueeze(0)} for i in range(n)]


def classify(name):
    n = name.lower()
    if 'lm_head' in n: return 'lm_head'
    if any(k in n for k in ['q_proj', 'k_proj', 'v_proj']): return 'attn_qkv'
    if 'out_proj' in n: return 'attn_out'
    if 'fc1' in n: return 'ffn_up'
    if 'fc2' in n: return 'ffn_down'
    return 'other'


def linears(m): return [(n, mm) for n, mm in m.named_modules() if isinstance(mm, torch.nn.Linear)]


def quantize_weights_int8(model):
    with torch.no_grad():
        for n, m in linears(model):
            W = m.weight.data
            s = (W.abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-8)
            m.weight.data = (W / s).round().clamp(-127, 127) * s


class ADC:
    def __init__(self, model, assign, default_bits, clip):
        self.model = model; self.assign = assign; self.default = default_bits
        self.clip = clip; self.scale = {}; self.h = []

    def calibrate(self, loader, n):
        acc = {}
        def mk(name):
            def h(mod, inp, out):
                v = out.detach().float().abs().flatten()
                if v.numel() > 20000:
                    v = v[torch.randint(0, v.numel(), (20000,), device=v.device)]
                acc.setdefault(name, []).append(v)
            return h
        hs = [m.register_forward_hook(mk(nm)) for nm, m in linears(self.model)]
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= n: break
                self.model(b['input_ids'].to(DEV))
        for x in hs: x.remove()
        for nm, _ in linears(self.model):
            bits = self.assign.get(nm, self.default); nadc = 2**bits - 1
            a = torch.cat(acc[nm])
            p = a.max().item() if self.clip == 'max' else torch.quantile(a, float(self.clip)/100).item()
            self.scale[nm] = max(p / nadc, 1e-8)

    def install(self):
        def mk(name):
            bits = self.assign.get(name, self.default); nadc = 2**bits - 1; s = self.scale[name]
            def h(mod, inp, out):
                y = out.detach().float()
                return ((y/s).round().clamp(-nadc, nadc)*s).to(out.dtype)
            return h
        self.h = [m.register_forward_hook(mk(nm)) for nm, m in linears(self.model)]

    def remove(self):
        for x in self.h: x.remove()
        self.h = []


def ppl(model, ev):
    nll = 0.0; ntok = 0
    with torch.no_grad():
        for i, b in enumerate(ev):
            if i >= NEVAL: break
            ids = b['input_ids'].to(DEV)
            o = model(ids, labels=ids); t = ids.shape[1]
            nll += o.loss.item()*t; ntok += t
    return float(np.exp(nll/ntok))


def P(m): print(m, flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=CACHE,
                                                 torch_dtype=torch.float32).to(DEV).eval()
    data = load_arrow(tok, 512)
    cld = make_loader(data[:NUM_CALIB+4]); ev = make_loader(data[40:40+NEVAL+4])
    names = [n for n, _ in linears(model)]
    res = {}

    res['fp32'] = round(ppl(model, ev), 3); P(f"FP32 (no quant, no ADC) PPL = {res['fp32']}")
    quantize_weights_int8(model)
    res['int8_weights_only'] = round(ppl(model, ev), 3)
    P(f"INT8 per-channel weights, NO ADC PPL = {res['int8_weights_only']}")

    for tag, bits, clip in [('adc7_p99', 7, '99'), ('adc7_max', 7, 'max'),
                            ('adc8_max', 8, 'max'), ('adc8_p99', 8, '99')]:
        a = ADC(model, {n: bits for n in names}, bits, clip)
        a.calibrate(cld, NUM_CALIB); a.install()
        res[tag] = round(ppl(model, ev), 3); a.remove()
        P(f"INT8w + {tag} PPL = {res[tag]}")
    OUT.write_text(json.dumps(res, indent=2))

    # usable operating point = lowest-PPL ADC config
    adc_cfgs = {'adc7_p99': (7, '99'), 'adc7_max': (7, 'max'),
                'adc8_max': (8, 'max'), 'adc8_p99': (8, '99')}
    best = min(adc_cfgs, key=lambda k: res[k]); bits, clip = adc_cfgs[best]
    P(f"\n=== usable operating point: {best} (PPL {res[best]}), profiling group sensitivity {bits}->{bits-1} ===")
    grp = {g: [n for n in names if classify(n) == g] for g in GROUPS}

    base = ADC(model, {n: bits for n in names}, bits, clip)
    base.calibrate(cld, NUM_CALIB); base.install()
    ppl_base = ppl(model, ev); base.remove()
    sens = {}
    for g in GROUPS:
        assign = {n: (bits-1 if classify(n) == g else bits) for n in names}
        a = ADC(model, assign, bits, clip)
        a.calibrate(cld, NUM_CALIB); a.install()
        pg = ppl(model, ev); a.remove()
        sens[g] = round((pg - ppl_base) / len(grp[g]), 4)
        P(f"  {g:9s} dPPL/layer = {sens[g]:+.4f}")
    ffn = (sens['ffn_up'] + sens['ffn_down']) / 2
    attn = (sens['attn_qkv'] + sens['attn_out']) / 2
    res['usable_point'] = {'config': best, 'ppl_base': round(ppl_base, 3),
                           'sensitivity_per_layer': sens,
                           'ffn_aggregate': round(ffn, 4), 'attn_aggregate': round(attn, 4),
                           'FFN_gt_attn': bool(ffn > attn)}
    OUT.write_text(json.dumps(res, indent=2))
    P(f"\nFFN aggregate {ffn:+.4f}  vs  attention aggregate {attn:+.4f}  -> FFN>attn: {ffn>attn}")
    P("DONE")


if __name__ == "__main__":
    main()
