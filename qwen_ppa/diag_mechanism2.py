#!/usr/bin/env python3
"""
diag_mechanism2.py — corrected mechanism diagnostic for the saturation-sensitivity
inversion on OPT-125M, at the PAPER's operating point (INT8 per-channel weight-quant
floor + per-layer p99 ADC calibration + stable eval).

Analytical model (derived in supp Table S4):
  ADC error = D_gran(b) [granular, resolution, ~FS^2 4^-b]  +  D_ovl [overload/clipping,
  bit-INDEPENDENT].  Reducing b worsens ONLY D_gran, and granular noise is added only to
  the (1 - s) fraction of UNSATURATED samples (saturated ones are clipped regardless of b).
  Hence:   sensitivity_g  ~  (1 - s_g) [clip gating] x  J_g [residual-stream propagation].

Per group we report the RATE-based decomposition (not energy):
  - s_g            : hardware saturation rate (MAC output beyond a shared full-scale)
  - kurtosis       : heavy-tail proxy (parameter-free)
  - D_ovl 7b/6b    : clipping error (expect bit-INVARIANT)               [factor-1 premise]
  - D_gran 7b/6b   : granular error (expect ~4x growth 7->6)             [factor-1 premise]
  - dD_gran        : granular-error increase 7->6 (the bit-reduction local error)
  - SQNR_db
  - resid_drift    : residual-stream perturbation norm (hidden drift)    [factor-2]
  - KL             : output-logit KL                                     [factor-2 / sens]
  - dNLL, dPPL     : token-level NLL and PPL sensitivity (707C: NLL is additive)
  - J_g            : resid_drift / dD_gran (propagation gain)
  - composite (1-s)*J  vs  saturation / Hessian, Spearman vs dNLL
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
OUT = ROOT / "results_control" / "diag_mechanism2.json"
MODEL = "facebook/opt-125m"
NOMINAL, PROBE, CLIP = 7, 6, 99.0
NUM_CALIB, NEVAL, SEED = 16, 30, 0
SAT_PCTL = 99.5     # shared full-scale percentile (hardware ADC range proxy)
GROUPS = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_down']   # lm_head excluded (post-residual)


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


def quantize_weights_int8(model):
    """INT8 per-output-channel weight quantization floor (the paper's CIM baseline)."""
    with torch.no_grad():
        for n, m in linears(model):
            W = m.weight.data
            s = (W.abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-8)
            m.weight.data = (W / s).round().clamp(-127, 127) * s


class ADCDiag:
    def __init__(self, model, assign, default_bits, clip):
        self.model = model; self.assign = assign; self.default = default_bits; self.clip = clip
        self.scale = {}; self.fs = {}; self.h = []
        self.D_ovl = {}; self.D_gran = {}; self.sig = {}; self.cnt = {}

    def calibrate(self, loader, n):
        acc = {}
        def mk(name):
            def h(mod, inp, out):
                v = out.detach().float().flatten()
                if v.numel() > 20000:
                    idx = torch.randint(0, v.numel(), (20000,), device=v.device); v = v[idx]
                acc.setdefault(name, []).append(v)
            return h
        hs = [m.register_forward_hook(mk(nm)) for nm, m in linears(self.model)]
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= n: break
                self.model(b['input_ids'].to(DEV))
        for x in hs: x.remove()
        for nm, _ in linears(self.model):
            bits = self.assign.get(nm, self.default); nadc = 2 ** bits - 1
            a = torch.cat(acc[nm]).abs()
            p = a.max().item() if self.clip >= 100 else torch.quantile(a, self.clip / 100).item()
            self.scale[nm] = max(p / nadc, 1e-8); self.fs[nm] = self.scale[nm] * nadc

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
                    self.D_ovl[name] = self.D_ovl.get(name, 0.0) + (((yq - y) ** 2) * over).sum().item()
                    self.D_gran[name] = self.D_gran.get(name, 0.0) + (((yq - y) ** 2) * (~over)).sum().item()
                    self.sig[name] = self.sig.get(name, 0.0) + (y ** 2).sum().item()
                    self.cnt[name] = self.cnt.get(name, 0) + y.numel()
                return yq.to(out.dtype)
            return h
        self.h = [m.register_forward_hook(mk(nm)) for nm, m in linears(self.model)]

    def remove(self):
        for x in self.h: x.remove()
        self.h = []


def capture(model, ev):
    Hs, Ls = [], []; nll = 0.0; ntok = 0
    with torch.no_grad():
        for i, b in enumerate(ev):
            if i >= NEVAL: break
            ids = b['input_ids'].to(DEV)
            o = model(ids, labels=ids, output_hidden_states=True)
            t = ids.shape[1]; nll += o.loss.item() * t; ntok += t
            Hs.append(o.hidden_states[-1].float().cpu()); Ls.append(o.logits.float().cpu())
    mnll = nll / ntok
    return Hs, Ls, mnll, float(np.exp(mnll))


def kl(Lp, Lq):
    p = torch.log_softmax(Lp, -1); q = torch.log_softmax(Lq, -1)
    return (p.exp() * (p - q)).sum(-1).mean().item()


def measure_saturation_kurtosis(model, loader, n):
    """Paper-style hardware saturation: MAC output beyond a SHARED full-scale; + kurtosis."""
    acc = {}
    def mk(name):
        def h(mod, inp, out):
            v = out.detach().float().flatten()
            if v.numel() > 40000:
                idx = torch.randint(0, v.numel(), (40000,), device=v.device); v = v[idx]
            acc.setdefault(name, []).append(v.cpu())
        return h
    hs = [m.register_forward_hook(mk(nm)) for nm, m in linears(model)]
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= n: break
            model(b['input_ids'].to(DEV))
    for x in hs: x.remove()
    per = {nm: torch.cat(acc[nm]) for nm, _ in linears(model)}
    pooled = torch.cat([v.abs() for v in per.values()])
    if pooled.numel() > 4_000_000:                          # torch.quantile element cap
        pooled = pooled[torch.randint(0, pooled.numel(), (4_000_000,))]
    fs_hw = torch.quantile(pooled, SAT_PCTL / 100).item()   # shared hardware ADC range
    sat, kurt = {}, {}
    for nm, v in per.items():
        sat[nm] = (v.abs() > fs_hw).float().mean().item()
        mu = v.mean(); var = v.var().clamp(min=1e-12)
        kurt[nm] = (((v - mu) ** 4).mean() / (var ** 2) - 3).item()
    return sat, kurt, fs_hw


def P(m): print(m, flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=CACHE,
                                                 torch_dtype=torch.float32).to(DEV).eval()
    quantize_weights_int8(model)               # INT8 WQ floor
    data = load_arrow(tok, 512)
    cld = make_loader(data[:NUM_CALIB + 4]); ev = make_loader(data[40:40 + NEVAL + 4])
    names = [n for n, _ in linears(model)]
    grp = {g: [n for n in names if classify(n) == g] for g in GROUPS}

    sat, kurt, fs_hw = measure_saturation_kurtosis(model, cld, NUM_CALIB)
    P(f"shared hardware full-scale (p{SAT_PCTL}) = {fs_hw:.3f}")

    set_seed(SEED)
    base = ADCDiag(model, {n: NOMINAL for n in names}, NOMINAL, CLIP)
    base.calibrate(cld, NUM_CALIB); base.install(record=True)
    H0, L0, nll0, ppl0 = capture(model, ev); base.remove()
    P(f"baseline INT8WQ+ADC7 NLL={nll0:.4f} PPL={ppl0:.3f}")

    def gs(d, g): return sum(d.get(n, 0.0) for n in grp[g])
    res = {"baseline_ppl": round(ppl0, 3), "baseline_nll": round(nll0, 4),
           "fs_hw": round(fs_hw, 4), "groups": {}}
    for g in GROUPS:
        c = max(gs(base.cnt, g), 1)
        dov7, dgr7, sig = gs(base.D_ovl, g), gs(base.D_gran, g), gs(base.sig, g)
        res["groups"][g] = {
            "s": round(float(np.mean([sat[n] for n in grp[g]])), 4),
            "kurtosis": round(float(np.mean([kurt[n] for n in grp[g]])), 2),
            "D_ovl_7b": dov7 / c, "D_gran_7b": dgr7 / c,
            "sqnr_db": round(float(10 * np.log10(sig / max(dgr7 + dov7, 1e-12))), 2)}

    for g in GROUPS:
        set_seed(SEED)
        assign = {n: (PROBE if classify(n) == g else NOMINAL) for n in names}
        a = ADCDiag(model, assign, NOMINAL, CLIP)
        a.calibrate(cld, NUM_CALIB); a.install(record=True)
        Hg, Lg, nllg, pplg = capture(model, ev); a.remove()
        c = max(sum(a.cnt.get(n, 0) for n in grp[g]), 1)
        dov6 = sum(a.D_ovl.get(n, 0.0) for n in grp[g]) / c
        dgr6 = sum(a.D_gran.get(n, 0.0) for n in grp[g]) / c
        gg = res["groups"][g]
        dD_gran = max(dgr6 - gg["D_gran_7b"], 1e-12)
        drift = float(np.mean([(Hg[i] - H0[i]).norm().item() / max(H0[i].norm().item(), 1e-8)
                               for i in range(len(H0))]))
        klv = float(np.mean([kl(L0[i][0], Lg[i][0]) for i in range(len(L0))]))
        gg.update({
            "D_ovl_6b": dov6, "D_gran_6b": dgr6,
            "ovl_ratio_6to7": round(dov6 / max(gg["D_ovl_7b"], 1e-12), 3),
            "gran_ratio_6to7": round(dgr6 / max(gg["D_gran_7b"], 1e-12), 3),
            "dD_gran": dD_gran, "resid_drift": round(drift, 5), "kl": round(klv, 6),
            "dNLL": round(nllg - nll0, 5), "dPPL": round(pplg - ppl0, 4),
            "J_drift": round(drift / dD_gran, 2)})
        P(f"{g:9s} s={gg['s']:.3f} kurt={gg['kurtosis']:8.1f} "
          f"ovlR={gg['ovl_ratio_6to7']:.2f} granR={gg['gran_ratio_6to7']:.2f} "
          f"drift={drift:.4f} KL={klv:.5f} dNLL={gg['dNLL']:+.4f} dPPL={gg['dPPL']:+.3f}")
        OUT.write_text(json.dumps(res, indent=2)); del Hg, Lg

    def spearman(x, y):
        x = np.argsort(np.argsort(x)); y = np.argsort(np.argsort(y))
        return float(np.corrcoef(x, y)[0, 1])
    G = GROUPS
    for tgt in ["dNLL", "dPPL", "kl"]:
        y = [res["groups"][g][tgt] for g in G]
        preds = {
            "saturation s": [res["groups"][g]["s"] for g in G],
            "kurtosis": [res["groups"][g]["kurtosis"] for g in G],
            "(1-s)": [1 - res["groups"][g]["s"] for g in G],
            "resid_drift (J)": [res["groups"][g]["resid_drift"] for g in G],
            "(1-s)*resid_drift": [(1 - res["groups"][g]["s"]) * res["groups"][g]["resid_drift"] for g in G],
            "(1-s)*KL": [(1 - res["groups"][g]["s"]) * res["groups"][g]["kl"] for g in G],
        }
        res[f"spearman_vs_{tgt}"] = {k: round(spearman(v, y), 3) for k, v in preds.items()}
    res["note"] = "paper proxies vs measured sensitivity: saturation rho=-0.70, Hessian rho=0.20"
    OUT.write_text(json.dumps(res, indent=2))
    P("=== Spearman vs dNLL ==="); P(json.dumps(res["spearman_vs_dNLL"], indent=2))
    P("DONE")


if __name__ == "__main__":
    main()
