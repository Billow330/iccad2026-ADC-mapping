#!/usr/bin/env python3
"""
diag_mechanism.py — validate the saturation-sensitivity inversion MECHANISM on
OPT-125M (the model with the paper's headline inversion).

Analytical claim: ADC distortion = granular (resolution, bit-dependent) + overload
(clipping, bit-INDEPENDENT). Reducing bits worsens only the granular part, so
  sensitivity_g  ~  alpha_gran_g  x  J_g
where alpha_gran = granular share of ADC error (small for high-saturation/heavy-tail
groups) and J = propagation gain to the loss (large for residual-stream W_fc2).

Per group we measure: heavy-tail kurtosis (saturation proxy), alpha_gran, dD (7->6
local output MSE increase), SQNR, residual-stream drift, output KL, and dPPL; then
correlate predictors with dPPL and compare against saturation/Hessian.
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
OUT = ROOT / "results_control" / "diag_mechanism.json"
MODEL = "facebook/opt-125m"
NOMINAL, PROBE, CLIP = 7, 6, 99.0
NUM_CALIB, NEVAL, SEED = 8, 8, 0
GROUPS = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down', 'lm_head']


def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_arrow(tok, sl=512):
    import pyarrow as pa
    r = pa.ipc.open_stream(pa.memory_map(ARROW, "r"))
    text = "\n".join(r.read_all().column("text").to_pylist())
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    n = len(ids) // sl
    return [{"input_ids": ids[i * sl:(i + 1) * sl].unsqueeze(0)} for i in range(n)]


def classify(name):
    n = name.lower()
    if 'lm_head' in n: return 'lm_head'
    if any(k in n for k in ['q_proj', 'k_proj', 'v_proj']): return 'attn_qkv'
    if 'out_proj' in n: return 'attn_out'
    if 'fc1' in n: return 'ffn_up'
    if 'fc2' in n: return 'ffn_down'
    return 'other'


def linears(m):
    return [(n, mm) for n, mm in m.named_modules() if isinstance(mm, torch.nn.Linear)]


class ADCDiag:
    """Per-layer ADC quant with clipping/granular error decomposition."""
    def __init__(self, model, assign, default_bits, clip):
        self.model = model; self.assign = assign; self.default = default_bits; self.clip = clip
        self.scale = {}; self.fs = {}; self.h = []
        self.D_ovl = {}; self.D_gran = {}; self.sig = {}; self.cnt = {}; self.kurt = {}

    def calibrate(self, loader, n):
        acc = {}
        def mk(name):
            def h(mod, inp, out):
                v = out.detach().float().flatten()
                k = 20000
                if v.numel() > k:
                    idx = torch.randint(0, v.numel(), (k,), device=v.device); v = v[idx]
                acc.setdefault(name, []).append(v)
            return h
        hs = [m.register_forward_hook(mk(n)) for n, m in linears(self.model)]
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= n: break
                self.model(b['input_ids'].to(DEV))
        for x in hs: x.remove()
        for n, _ in linears(self.model):
            bits = self.assign.get(n, self.default); nadc = 2 ** bits - 1
            v = torch.cat(acc[n]); a = v.abs()
            p = a.max().item() if self.clip >= 100 else torch.quantile(a, self.clip / 100).item()
            self.scale[n] = max(p / nadc, 1e-8); self.fs[n] = self.scale[n] * nadc
            mu = v.mean(); var = v.var().clamp(min=1e-12)
            self.kurt[n] = (((v - mu) ** 4).mean() / (var ** 2) - 3).item()

    def install(self, record=False):
        self.record = record
        def mk(name):
            bits = self.assign.get(name, self.default); nadc = 2 ** bits - 1
            s = self.scale[name]; fs = self.fs[name]
            def h(mod, inp, out):
                y = out.detach().float()
                yq = (y / s).round().clamp(-nadc, nadc) * s
                if self.record:
                    over = y.abs() > fs
                    dov = ((y.abs() - fs).clamp(min=0) ** 2).sum().item()
                    dgr = (((yq - y) ** 2) * (~over)).sum().item()
                    self.D_ovl[name] = self.D_ovl.get(name, 0.0) + dov
                    self.D_gran[name] = self.D_gran.get(name, 0.0) + dgr
                    self.sig[name] = self.sig.get(name, 0.0) + (y ** 2).sum().item()
                    self.cnt[name] = self.cnt.get(name, 0) + y.numel()
                return yq.to(out.dtype)
            return h
        self.h = [m.register_forward_hook(mk(n)) for n, m in linears(self.model)]

    def remove(self):
        for x in self.h: x.remove()
        self.h = []


def capture(model, ev):
    """Run eval set; return per-batch final hidden + logits (cpu) and PPL."""
    Hs, Ls = [], []; nll = 0.0; ntok = 0
    with torch.no_grad():
        for i, b in enumerate(ev):
            if i >= NEVAL: break
            ids = b['input_ids'].to(DEV)
            o = model(ids, labels=ids, output_hidden_states=True)
            nll += o.loss.item() * ids.shape[1]; ntok += ids.shape[1]
            Hs.append(o.hidden_states[-1].float().cpu())
            Ls.append(o.logits.float().cpu())
    return Hs, Ls, float(np.exp(nll / ntok))


def kl(Lp, Lq):
    p = torch.log_softmax(Lp, -1); q = torch.log_softmax(Lq, -1)
    return (p.exp() * (p - q)).sum(-1).mean().item()


def P(m): print(m, flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=CACHE,
                                                 torch_dtype=torch.float32).to(DEV).eval()
    data = load_arrow(tok, 512)
    cld = make_loader(data[:NUM_CALIB + 4]); ev = make_loader(data[64:64 + NEVAL + 4])
    names = [n for n, _ in linears(model)]
    grp = {g: [n for n in names if classify(n) == g] for g in GROUPS}

    # baseline: all 7b, record decomposition
    set_seed(SEED)
    base = ADCDiag(model, {n: NOMINAL for n in names}, NOMINAL, CLIP)
    base.calibrate(cld, NUM_CALIB); base.install(record=True)
    H0, L0, ppl0 = capture(model, ev); base.remove()
    P(f"baseline 7b PPL = {ppl0:.3f}")

    def gsum(d, g): return sum(d.get(n, 0.0) for n in grp[g])
    res = {"baseline_ppl": round(ppl0, 3), "groups": {}}
    for g in GROUPS:
        c = max(gsum(base.cnt, g), 1)
        dov, dgr, sig = gsum(base.D_ovl, g), gsum(base.D_gran, g), gsum(base.sig, g)
        alpha = dgr / max(dgr + dov, 1e-12)
        sqnr = 10 * np.log10(sig / max(dgr + dov, 1e-12))
        kurt = float(np.mean([base.kurt[n] for n in grp[g]]))
        mse7 = (dgr + dov) / c
        res["groups"][g] = {"kurtosis": round(kurt, 2), "alpha_gran": round(alpha, 4),
                            "sqnr_db": round(float(sqnr), 2), "mse7": mse7}

    # per group: reduce to 6b, record dD and capture drift/KL/dPPL
    for g in GROUPS:
        set_seed(SEED)
        assign = {n: (PROBE if classify(n) == g else NOMINAL) for n in names}
        a = ADCDiag(model, assign, NOMINAL, CLIP)
        a.calibrate(cld, NUM_CALIB); a.install(record=True)
        Hg, Lg, pplg = capture(model, ev); a.remove()
        c = max(sum(a.cnt.get(n, 0) for n in grp[g]), 1)
        mse6 = (sum(a.D_ovl.get(n, 0.0) for n in grp[g]) + sum(a.D_gran.get(n, 0.0) for n in grp[g])) / c
        dD = max(mse6 - res["groups"][g]["mse7"], 1e-12)
        drift = float(np.mean([ (Hg[i] - H0[i]).norm().item() / max(H0[i].norm().item(), 1e-8)
                                for i in range(len(H0))]))
        klv = float(np.mean([kl(L0[i][0], Lg[i][0]) for i in range(len(L0))]))
        dppl = pplg - ppl0
        gg = res["groups"][g]
        gg.update({"dD_7to6": dD, "resid_drift": round(drift, 5), "kl": round(klv, 5),
                   "dPPL": round(dppl, 4), "J_drift": round(drift / dD, 3) if dD > 0 else None})
        P(f"{g:9s} alpha={gg['alpha_gran']:.3f} kurt={gg['kurtosis']:.1f} dD={dD:.3e} "
          f"drift={drift:.4f} KL={klv:.4f} dPPL={dppl:.3f}")
        OUT.write_text(json.dumps(res, indent=2))
        del Hg, Lg

    # correlations (Spearman over groups) vs dPPL
    def spearman(x, y):
        x = np.argsort(np.argsort(x)); y = np.argsort(np.argsort(y))
        return float(np.corrcoef(x, y)[0, 1])
    G = [g for g in GROUPS]
    dppl = [res["groups"][g]["dPPL"] for g in G]
    preds = {
        "kurtosis (saturation proxy)": [res["groups"][g]["kurtosis"] for g in G],
        "alpha_gran (granular share)": [res["groups"][g]["alpha_gran"] for g in G],
        "resid_drift (propagation)": [res["groups"][g]["resid_drift"] for g in G],
        "KL (propagation)": [res["groups"][g]["kl"] for g in G],
        "alpha_gran x resid_drift": [res["groups"][g]["alpha_gran"] * res["groups"][g]["resid_drift"] for g in G],
        "alpha_gran x KL": [res["groups"][g]["alpha_gran"] * res["groups"][g]["kl"] for g in G],
    }
    res["spearman_vs_dPPL"] = {k: round(spearman(v, dppl), 3) for k, v in preds.items()}
    res["note"] = ("paper proxies: saturation rho=-0.70, Hessian rho=0.20 vs measured dPPL")
    OUT.write_text(json.dumps(res, indent=2))
    P("=== Spearman vs dPPL ==="); P(json.dumps(res["spearman_vs_dPPL"], indent=2))
    P("DONE")


if __name__ == "__main__":
    main()
