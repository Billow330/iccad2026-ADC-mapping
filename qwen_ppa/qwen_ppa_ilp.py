#!/usr/bin/env python3
"""
qwen_ppa_ilp.py — PPA + ILP allocation for Qwen2-7B at the USABLE operating point
(per-layer MAX-clip, 8b nominal). Uses the paper's ADC-area model A_ADC ∝ M·2^b
(M = output-column count = out_features), column-weighted (correct for GQA/SwiGLU),
and the measured group sensitivities (8->6) to allocate ADC bits via group-level
brute-force ILP at several ADC-area budgets. Reports area/energy savings (2^b law,
as in the paper's projection) and the ACTUAL PPL evaluated at each allocation.
"""
import sys, json, itertools, random
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
SENS = json.load(open(ROOT / "results_control" / "qwen_sensitivity.json"))
OUT = ROOT / "results_control" / "qwen_ppa_ilp.json"
NOMINAL = 8
BITS = [6, 7, 8]
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

    info = [(n, qclass(n), m.out_features) for n, m in linears(model)]
    colsum = {g: sum(of for _, gg, of in info if gg == g) for g in GROUPS}
    nlay = {g: sum(1 for _, gg, _ in info if gg == g) for g in GROUPS}
    sigma = {g: max(SENS["probes"]["8to6"]["per_group"][g]["sigma_per_layer"], 0.0) for g in GROUPS}
    ref_area = sum(colsum[g] * 2 ** NOMINAL for g in GROUPS)
    P(f"colsum={colsum}"); P(f"sigma={sigma}")

    base = run(model, {n: NOMINAL for n, _, _ in info}, cld, eld)
    out = {"baseline_ppl": round(base, 3), "sigma_8to6": sigma, "colsum": colsum,
           "nlay": nlay, "allocs": {}}
    P(f"baseline (max-clip 8b) PPL = {base:.3f}")
    OUT.write_text(json.dumps(out, indent=2))

    for budget in [0.20, 0.30, 0.50]:
        best = None
        for combo in itertools.product(BITS, repeat=len(GROUPS)):
            bg = dict(zip(GROUPS, combo))
            area = sum(colsum[g] * 2 ** bg[g] for g in GROUPS)
            if area > (1 - budget) * ref_area + 1e-6:
                continue
            cost = sum(sigma[g] * (NOMINAL - bg[g]) * nlay[g] for g in GROUPS)
            if best is None or cost < best[0]:
                best = (cost, bg, area)
        cost, bg, area = best
        save = 1 - area / ref_area
        assign = {n: bg[g] for n, g, _ in info}
        p_alloc = run(model, assign, cld, eld)
        # ADC energy follows the same 2^b law in the paper's projection
        out["allocs"][f"budget{int(budget*100)}"] = {
            "group_bits": bg,
            "adc_area_saving": round(save, 4),
            "adc_energy_saving_2b_proxy": round(save, 4),
            "ppl": round(p_alloc, 3),
            "rel_ppl_vs_base": round((p_alloc - base) / base, 4),
        }
        P(f"budget {int(budget*100)}%: bits={bg} area_save={save:.3f} ppl={p_alloc:.3f} rel={(p_alloc-base)/base:+.3%}")
        OUT.write_text(json.dumps(out, indent=2))

    # contrast: a saturation/naive baseline that reduces the MOST-sensitive group (ffn_down)
    naive = {n: (6 if g == 'ffn_down' else 8) for n, g, _ in info}
    out["naive_reduce_ffndown_6b_ppl"] = round(run(model, naive, cld, eld), 3)
    P(f"naive reduce ffn_down->6b ppl = {out['naive_reduce_ffndown_6b_ppl']}")

    OUT.write_text(json.dumps(out, indent=2))
    P("DONE"); P(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
