#!/usr/bin/env python3
"""
qwen_rescue.py — try to reach a USABLE Qwen2-7B CIM operating point.

Per-layer uniform ADC breaks Qwen2 at all bits/clips (best ~1380 vs clean ~10)
because of extreme per-output-channel "massive activations". Levers tested here
target the OUTPUT distribution:
  - per-channel (per-column) ADC calibration  [primary candidate]
  - mixed-precision protection of outlier layers (lm_head/down_proj)
  - SmoothQuant (expected limited: it smooths inputs, ADC quantizes outputs)
Exact per-channel max calibration (no randperm). Results saved incrementally.
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
OUT = ROOT / "results_control" / "qwen_rescue.json"
NUM_CALIB, NUM_EVAL, SEED = 8, 15, 0


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
    def __init__(self, model, assign, default_bits, clip=100.0, per_channel=False):
        self.model = model; self.assign = assign; self.default = default_bits
        self.clip = clip; self.pc = per_channel; self.scale = {}; self.h = []

    def calibrate(self, loader, n_batches):
        acc = {}
        def mk(name):
            def h(mod, inp, out):
                v = out.detach().float().abs()
                if self.pc:
                    cm = v.amax(dim=tuple(range(v.ndim - 1)))  # [F]
                    acc[name] = cm if name not in acc else torch.maximum(acc[name], cm)
                elif self.clip >= 100:
                    m = v.max()
                    acc[name] = m if name not in acc else torch.maximum(acc[name], m)
                else:
                    vf = v.flatten(); k = 16384
                    if vf.numel() > k:
                        idx = torch.randint(0, vf.numel(), (k,), device=vf.device)
                        vf = vf[idx]
                    acc.setdefault(name, []).append(vf)
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
            if self.pc:
                self.scale[n] = (acc[n] / nadc).clamp(min=1e-8)
            elif self.clip >= 100:
                self.scale[n] = max(acc[n].item() / nadc, 1e-8)
            else:
                allv = torch.cat(acc[n]); p = torch.quantile(allv, self.clip / 100.0).item()
                self.scale[n] = max(p / nadc, 1e-8)

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


def run_adc(model, assign, default_bits, clip, pc, cld, eld):
    set_seed(SEED)
    a = ADC(model, assign, default_bits, clip, pc)
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
    U = lambda b: {n: b for n in names}

    res = {"clean": round(ppl(model, eld), 3)}
    P(f"clean = {res['clean']}"); OUT.write_text(json.dumps(res, indent=2))

    res["perlayer_max_8b"] = round(run_adc(model, U(8), 8, 100.0, False, cld, eld), 3)
    P(f"perlayer_max_8b = {res['perlayer_max_8b']}"); OUT.write_text(json.dumps(res, indent=2))

    for b in [10, 8, 6]:
        res[f"perchan_max_{b}b"] = round(run_adc(model, U(b), b, 100.0, True, cld, eld), 3)
        P(f"perchan_max_{b}b = {res[f'perchan_max_{b}b']}"); OUT.write_text(json.dumps(res, indent=2))

    prot = {n: (14 if qclass(n) in ('lm_head', 'ffn_down') else 8) for n in names}
    res["protect_lmheaddown_perchan_8b"] = round(run_adc(model, prot, 8, 100.0, True, cld, eld), 3)
    P(f"protect_perchan_8b = {res['protect_lmheaddown_perchan_8b']}"); OUT.write_text(json.dumps(res, indent=2))

    # SmoothQuant (last; modifies model in-place). Expected limited for output-ADC.
    try:
        from smooth_quant import CIMSmoothQuant
        sq = CIMSmoothQuant(weight_bits=8, input_bits=8)
        try:
            sq.alpha_range = [0.5]
        except Exception:
            pass
        sq.fit(model, cld, num_batches=NUM_CALIB, device=DEV, task='lm')
        sq.apply(model)
        res["sq_perlayer_max_8b"] = round(run_adc(model, U(8), 8, 100.0, False, cld, eld), 3)
        res["sq_perchan_max_8b"] = round(run_adc(model, U(8), 8, 100.0, True, cld, eld), 3)
        P(f"sq_perlayer_max_8b = {res['sq_perlayer_max_8b']} | sq_perchan_max_8b = {res['sq_perchan_max_8b']}")
    except Exception as e:
        import traceback; traceback.print_exc()
        res["sq_error"] = str(e)[:200]
    OUT.write_text(json.dumps(res, indent=2))
    P("DONE"); P(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
