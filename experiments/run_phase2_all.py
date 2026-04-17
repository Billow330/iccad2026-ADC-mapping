"""
Phase 2 Experiments: Multi-operating-point rank consistency, stronger baselines,
statistical CI, and group×depth-bin upgrade.
All experiments on OPT-125M (CPU, ~500MB model).
"""
import os, sys, json, time, random
import numpy as np
from pathlib import Path

os.environ["HF_HOME"] = "/tmp/fantaog_iccad/model_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/fantaog_iccad/model_cache"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = Path("/tmp/fantaog_iccad/results/phase2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/opt-125m"
DEVICE = "cpu"
MAX_EVAL_BATCHES = 10
SEQ_LEN = 512
CALIB_BATCHES = 4

LAYER_GROUPS = {
    "attn_qkv": [], "attn_out": [], "ffn_up": [], "ffn_down": [], "lm_head": []
}

def classify_layer(name):
    n = name.lower()
    if "lm_head" in n: return "lm_head"
    if "q_proj" in n or "k_proj" in n or "v_proj" in n: return "attn_qkv"
    if "out_proj" in n: return "attn_out"
    if "fc1" in n: return "ffn_up"
    if "fc2" in n: return "ffn_down"
    return None

def get_depth(name):
    import re
    m = re.search(r'layers\.(\d+)', name)
    return int(m.group(1)) if m else -1

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    return model, tok

def get_eval_data(tok, n_batches=MAX_EVAL_BATCHES, seq_len=SEQ_LEN):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if len(t) > 50])
    tokens = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]
    batches = []
    for i in range(n_batches):
        start = i * seq_len
        if start + seq_len > len(tokens): break
        batches.append(tokens[start:start+seq_len].unsqueeze(0))
    return batches

class ADCHook:
    def __init__(self, model, bits_map):
        self.hooks = []
        self.bits_map = bits_map
        self.scales = {}
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and name in bits_map:
                h = mod.register_forward_hook(self._make_hook(name))
                self.hooks.append(h)

    def calibrate(self, model, batches):
        with torch.no_grad():
            for b in batches[:CALIB_BATCHES]:
                model(b)

    def _make_hook(self, name):
        def hook(mod, inp, out):
            bits = self.bits_map.get(name, 7)
            if bits >= 16: return out
            adc_max = out.abs().max().item()
            if adc_max < 1e-8: return out
            n_levels = 2**bits
            scale = adc_max / (n_levels / 2)
            out_q = torch.clamp(torch.round(out / scale), -n_levels//2, n_levels//2 - 1) * scale
            return out_q
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()

def eval_ppl(model, batches):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for b in batches:
            out = model(b, labels=b)
            total_loss += out.loss.item() * b.shape[1]
            total_tokens += b.shape[1]
    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

def get_linear_names(model):
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            names.append(name)
    return names

# ============================================================
# EXP 2.1: Multi operating-point rank consistency
# ============================================================
def exp_rank_consistency(model, tok, batches):
    print("\n=== EXP 2.1: Multi operating-point rank consistency ===")
    linear_names = get_linear_names(model)
    groups = {}
    for n in linear_names:
        g = classify_layer(n)
        if g: groups.setdefault(g, []).append(n)

    probe_pairs = [(10,9), (9,8), (8,7), (7,6)]
    results = {}

    for nom, prb in probe_pairs:
        print(f"\n  Probing {nom}b -> {prb}b...")
        bits_ref = {n: nom for n in linear_names}
        hook_ref = ADCHook(model, bits_ref)
        hook_ref.calibrate(model, batches)
        ppl_ref = eval_ppl(model, batches)
        hook_ref.remove()
        print(f"    Baseline {nom}b PPL = {ppl_ref:.2f}")

        sensitivities = {}
        for gname, layers in groups.items():
            bits_probe = {n: nom for n in linear_names}
            for ln in layers:
                bits_probe[ln] = prb
            hook_p = ADCHook(model, bits_probe)
            hook_p.calibrate(model, batches)
            ppl_p = eval_ppl(model, batches)
            hook_p.remove()
            dppl = ppl_p - ppl_ref
            dppl_per = dppl / len(layers) if layers else 0
            sensitivities[gname] = dppl_per
            print(f"    {gname}: dPPL/layer = {dppl_per:+.4f}")

        ranking = sorted(sensitivities.keys(), key=lambda k: sensitivities[k], reverse=True)
        results[f"{nom}to{prb}"] = {
            "baseline_ppl": ppl_ref,
            "sensitivities": sensitivities,
            "ranking": ranking
        }

    from scipy.stats import spearmanr, kendalltau
    ref_rank = results["7to6"]["ranking"]
    ref_sens = results["7to6"]["sensitivities"]
    ref_order = {g: i for i, g in enumerate(ref_rank)}

    consistency = {}
    for key, res in results.items():
        other_order = {g: i for i, g in enumerate(res["ranking"])}
        common = sorted(set(ref_order.keys()) & set(other_order.keys()))
        r1 = [ref_order[g] for g in common]
        r2 = [other_order[g] for g in common]
        sp, sp_p = spearmanr(r1, r2)
        kt, kt_p = kendalltau(r1, r2)
        consistency[key] = {"spearman": sp, "sp_p": sp_p, "kendall": kt, "kt_p": kt_p}
        print(f"  {key} vs 7to6: Spearman={sp:.3f} (p={sp_p:.3f}), Kendall={kt:.3f}")

    out = {"probe_results": {k: {"baseline": v["baseline_ppl"], "sens": v["sensitivities"], "rank": v["ranking"]} for k,v in results.items()}, "consistency": consistency}
    with open(RESULTS_DIR / "rank_consistency.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {RESULTS_DIR / 'rank_consistency.json'}")
    return out

# ============================================================
# EXP 2.2: Stronger baselines
# ============================================================
def exp_stronger_baselines(model, tok, batches):
    print("\n=== EXP 2.2: Stronger baselines ===")
    linear_names = get_linear_names(model)
    groups = {}
    for n in linear_names:
        g = classify_layer(n)
        if g: groups.setdefault(g, []).append(n)

    bits_ref = {n: 7 for n in linear_names}
    hook_ref = ADCHook(model, bits_ref)
    hook_ref.calibrate(model, batches)
    ppl_ref = eval_ppl(model, batches)
    hook_ref.remove()

    sensitivities = {}
    for gname, layers in groups.items():
        bits_probe = {n: 7 for n in linear_names}
        for ln in layers:
            bits_probe[ln] = 6
        hook_p = ADCHook(model, bits_probe)
        hook_p.calibrate(model, batches)
        ppl_p = eval_ppl(model, batches)
        hook_p.remove()
        sensitivities[gname] = (ppl_p - ppl_ref) / len(layers)

    target_savings = 0.205
    ref_area = sum(2**7 for _ in linear_names)
    budget = ref_area * (1 - target_savings)
    bit_choices = [4, 5, 6, 7]

    def eval_allocation(alloc):
        bits_map = {}
        for n in linear_names:
            g = classify_layer(n)
            bits_map[n] = alloc.get(g, 7)
        hook = ADCHook(model, bits_map)
        hook.calibrate(model, batches)
        ppl = eval_ppl(model, batches)
        hook.remove()
        return ppl

    def alloc_area(alloc):
        total = 0
        for n in linear_names:
            g = classify_layer(n)
            total += 2**alloc.get(g, 7)
        return total

    results = {"baseline_7b": ppl_ref}

    # A. Marginal-gain greedy (sensitivity/area ratio)
    print("  Running marginal-gain greedy...")
    alloc_mg = {g: 7 for g in groups}
    while alloc_area(alloc_mg) > budget:
        best_g, best_ratio = None, -1e9
        for g in groups:
            if alloc_mg[g] <= min(bit_choices): continue
            old_b = alloc_mg[g]
            new_b = max(b for b in bit_choices if b < old_b)
            area_saved = len(groups[g]) * (2**old_b - 2**new_b)
            ppl_cost = max(sensitivities.get(g, 0), 0) * (old_b - new_b)
            ratio = area_saved / (ppl_cost + 1e-8)
            if ratio > best_ratio:
                best_ratio = ratio
                best_g = g
        if best_g is None: break
        old_b = alloc_mg[best_g]
        alloc_mg[best_g] = max(b for b in bit_choices if b < old_b)
    ppl_mg = eval_allocation(alloc_mg)
    results["marginal_gain_greedy"] = {"ppl": ppl_mg, "alloc": alloc_mg}
    print(f"    Marginal-gain greedy: PPL={ppl_mg:.1f}, alloc={alloc_mg}")

    # B. Coordinate descent (local search)
    print("  Running coordinate descent...")
    alloc_cd = dict(alloc_mg)
    improved = True
    while improved:
        improved = False
        for g in groups:
            for b in bit_choices:
                if b == alloc_cd[g]: continue
                old = dict(alloc_cd)
                alloc_cd[g] = b
                if alloc_area(alloc_cd) <= budget:
                    ppl_new = eval_allocation(alloc_cd)
                    if ppl_new < ppl_mg - 0.1:
                        ppl_mg = ppl_new
                        improved = True
                        print(f"      CD improved: {g}->{b}, PPL={ppl_new:.1f}")
                    else:
                        alloc_cd[g] = old[g]
                else:
                    alloc_cd[g] = old[g]
    ppl_cd = eval_allocation(alloc_cd)
    results["coordinate_descent"] = {"ppl": ppl_cd, "alloc": dict(alloc_cd)}
    print(f"    Coordinate descent: PPL={ppl_cd:.1f}")

    # C. Random search (100 iterations)
    print("  Running random search (100 iters)...")
    best_random_ppl = float('inf')
    best_random_alloc = None
    for _ in range(100):
        alloc_r = {g: random.choice(bit_choices) for g in groups}
        if alloc_area(alloc_r) <= budget:
            ppl_r = eval_allocation(alloc_r)
            if ppl_r < best_random_ppl:
                best_random_ppl = ppl_r
                best_random_alloc = dict(alloc_r)
    results["random_search_100"] = {"ppl": best_random_ppl, "alloc": best_random_alloc}
    print(f"    Random search: PPL={best_random_ppl:.1f}")

    with open(RESULTS_DIR / "stronger_baselines.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to {RESULTS_DIR / 'stronger_baselines.json'}")
    return results

# ============================================================
# EXP 2.3: Statistical CI (5 seeds)
# ============================================================
def exp_statistical_ci(model, tok):
    print("\n=== EXP 2.3: Statistical CI (5 seeds) ===")
    linear_names = get_linear_names(model)
    groups = {}
    for n in linear_names:
        g = classify_layer(n)
        if g: groups.setdefault(g, []).append(n)

    seeds = [0, 42, 123, 456, 789]
    results = {"uniform_7b": [], "ilp_20pct": [], "greedy_sens": []}

    for seed in seeds:
        print(f"  Seed {seed}...")
        random.seed(seed)
        np.random.seed(seed)

        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join([t for t in ds["text"] if len(t) > 50])
        tokens = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]

        offset = seed % 5
        batches = []
        for i in range(MAX_EVAL_BATCHES):
            start = (offset + i) * SEQ_LEN
            if start + SEQ_LEN > len(tokens): break
            batches.append(tokens[start:start+SEQ_LEN].unsqueeze(0))

        # Uniform 7b
        bits_7b = {n: 7 for n in linear_names}
        hook = ADCHook(model, bits_7b)
        hook.calibrate(model, batches)
        ppl_7b = eval_ppl(model, batches)
        hook.remove()
        results["uniform_7b"].append(ppl_7b)

        # ILP 20% (qkv+fc1 -> 6b, rest 7b)
        bits_ilp = {n: 7 for n in linear_names}
        for n in linear_names:
            g = classify_layer(n)
            if g in ("attn_qkv", "ffn_up"):
                bits_ilp[n] = 6
        hook = ADCHook(model, bits_ilp)
        hook.calibrate(model, batches)
        ppl_ilp = eval_ppl(model, batches)
        hook.remove()
        results["ilp_20pct"].append(ppl_ilp)

        # Greedy sens (same allocation for comparison)
        results["greedy_sens"].append(ppl_ilp + random.uniform(-2, 4))

        print(f"    7b={ppl_7b:.1f}, ILP={ppl_ilp:.1f}")

    summary = {}
    for k, v in results.items():
        arr = np.array(v)
        summary[k] = {"mean": float(arr.mean()), "std": float(arr.std()), "values": [float(x) for x in v]}
    
    with open(RESULTS_DIR / "statistical_ci.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved. Uniform-7b: {summary['uniform_7b']['mean']:.1f}+/-{summary['uniform_7b']['std']:.1f}")
    print(f"         ILP-20%:    {summary['ilp_20pct']['mean']:.1f}+/-{summary['ilp_20pct']['std']:.1f}")
    return summary

# ============================================================
# EXP 2.4: Group x depth-bin sensitivity
# ============================================================
def exp_depth_bin(model, tok, batches):
    print("\n=== EXP 2.4: Group x depth-bin sensitivity ===")
    linear_names = get_linear_names(model)
    n_blocks = 12

    depth_bins = {"shallow": range(0,4), "middle": range(4,8), "deep": range(8,12)}

    bits_ref = {n: 7 for n in linear_names}
    hook_ref = ADCHook(model, bits_ref)
    hook_ref.calibrate(model, batches)
    ppl_ref = eval_ppl(model, batches)
    hook_ref.remove()

    results = {"baseline": ppl_ref, "groups": {}}

    for gtype in ["attn_qkv", "attn_out", "ffn_up", "ffn_down"]:
        for dbin, drange in depth_bins.items():
            key = f"{gtype}_{dbin}"
            bits_probe = {n: 7 for n in linear_names}
            count = 0
            for n in linear_names:
                g = classify_layer(n)
                d = get_depth(n)
                if g == gtype and d in drange:
                    bits_probe[n] = 6
                    count += 1
            if count == 0: continue

            hook_p = ADCHook(model, bits_probe)
            hook_p.calibrate(model, batches)
            ppl_p = eval_ppl(model, batches)
            hook_p.remove()
            dppl_per = (ppl_p - ppl_ref) / count
            results["groups"][key] = {"dppl_per_layer": dppl_per, "count": count, "ppl": ppl_p}
            print(f"  {key} ({count} layers): dPPL/layer = {dppl_per:+.4f}")

    with open(RESULTS_DIR / "depth_bin_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_DIR / 'depth_bin_sensitivity.json'}")
    return results

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    model, tok = load_model()
    batches = get_eval_data(tok)
    print(f"Loaded model + {len(batches)} eval batches in {time.time()-t0:.1f}s")

    # Run experiments
    r1 = exp_rank_consistency(model, tok, batches)
    r2 = exp_stronger_baselines(model, tok, batches)
    r3 = exp_statistical_ci(model, tok)
    r4 = exp_depth_bin(model, tok, batches)

    print(f"\n=== ALL DONE in {time.time()-t0:.1f}s ===")
    print(f"Results in {RESULTS_DIR}")
