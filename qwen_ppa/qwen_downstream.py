#!/usr/bin/env python3
"""
qwen_downstream.py — zero-shot PIQA + BoolQ for Qwen2-7B under CIM ADC (max-clip),
for three configs: clean, CIM uniform 8b, and the profiling-ILP allocation
(ffn_up->7b, the 34%-area-saving point). Shows the allocation preserves downstream
accuracy vs the CIM-8b baseline. Self-contained scoring; exact-max ADC hook (fp16).
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
OUT = ROOT / "results_control" / "qwen_downstream.json"
LIMIT = 500
NUM_CALIB = 8
NOMINAL = 8


def set_seed(s=0):
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


@torch.no_grad()
def loglik(model, ctx_ids, cont_ids):
    ids = torch.cat([ctx_ids, cont_ids]).unsqueeze(0).to(DEV)
    logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    n_ctx = len(ctx_ids); tot = 0.0
    for i, t in enumerate(cont_ids):
        tot += logp[n_ctx - 1 + i, int(t)].item()
    return tot


def eval_piqa(model, tok):
    data = [json.loads(l) for l in open(ROOT / "data" / "piqa_valid.jsonl")]
    labels = [int(x) for x in open(ROOT / "data" / "piqa_valid_labels.lst").read().split()]
    c = n = 0
    for ex, lab in list(zip(data, labels))[:LIMIT]:
        ctx = tok(ex["goal"] + " ", return_tensors="pt").input_ids[0]
        sc = []
        for sol in [ex["sol1"], ex["sol2"]]:
            cont = tok(sol, return_tensors="pt").input_ids[0]
            sc.append(loglik(model, ctx, cont) / max(len(cont), 1))
        c += int((1 if sc[1] > sc[0] else 0) == lab); n += 1
    return round(100.0 * c / n, 2)


def eval_boolq(model, tok):
    data = [json.loads(l) for l in open(ROOT / "data" / "boolq_val.jsonl")]
    yes = tok(" yes", return_tensors="pt").input_ids[0]
    no = tok(" no", return_tensors="pt").input_ids[0]
    c = n = 0
    for ex in data[:LIMIT]:
        ctx = tok(ex["passage"] + "\nQuestion: " + ex["question"] + "?\nAnswer:",
                  return_tensors="pt").input_ids[0]
        if len(ctx) > 1024:
            ctx = ctx[-1024:]
        pred = loglik(model, ctx, yes) > loglik(model, ctx, no)
        c += int(pred == bool(ex["answer"])); n += 1
    return round(100.0 * c / n, 2)


def P(m):
    print(m, flush=True)


def main():
    name = "Qwen/Qwen2-7B"
    P(f"loading {name} (fp16) ...")
    tok = AutoTokenizer.from_pretrained(name, cache_dir=CACHE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, cache_dir=CACHE, torch_dtype=torch.float16,
        trust_remote_code=True, low_cpu_mem_usage=True).to(DEV).eval()
    wiki = load_arrow(tok, 512)
    cld = make_loader(wiki[:NUM_CALIB + 4])
    names = [n for n, _ in linears(model)]
    res = {}

    configs = {
        "clean": None,
        "cim_max8b": {n: NOMINAL for n in names},
        "ilp_ffnup7b": {n: (7 if qclass(n) == 'ffn_up' else NOMINAL) for n in names},
    }
    for cfg, assign in configs.items():
        set_seed(0)
        adc = None
        if assign is not None:
            adc = ADC(model, assign, NOMINAL)
            adc.calibrate(cld, NUM_CALIB)
            adc.install()
        piqa = eval_piqa(model, tok)
        boolq = eval_boolq(model, tok)
        if adc is not None:
            adc.remove()
        res[cfg] = {"piqa_acc": piqa, "boolq_acc": boolq}
        P(f"{cfg:14s} PIQA {piqa}  BoolQ {boolq}")
        OUT.write_text(json.dumps(res, indent=2))

    OUT.write_text(json.dumps(res, indent=2))
    P("DONE"); P(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
