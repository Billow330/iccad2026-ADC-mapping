"""
enhanced_noise_experiment.py — Enhanced CIM Noise Model Experiments
===================================================================
P0-A2: Add thermal noise + device variation to CIM model.
        Verify sensitivity inversion persists under realistic noise.
P2-B2: ADC calibration ablation (p99 vs MSE-optimal).
P2-B3: Random search baseline for ILP comparison.
"""

import sys, json, copy
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'pytorch-quantization'))

import torch
import numpy as np
from collections import defaultdict

from llm_inference import load_model, load_wikitext2, make_loader
from sensitivity_analysis import (
    PerLayerCIMHook, classify_layer, ilp_allocation, adc_area_ratio
)
from smooth_quant import compute_perplexity

RESULTS_DIR = ROOT / 'results' / 'enhanced'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'facebook/opt-125m'
CACHE_DIR = '/tmp/fantaog_iccad/model_cache'
DEVICE = 'cpu'


class EnhancedCIMHook:
    """CIM noise hook with thermal noise + device variation on top of ADC clipping."""

    def __init__(self, model, bit_assignment, adc_bits_default=7,
                 clip_percentile=99.0, thermal_sigma=0.0, device_var_sigma=0.0):
        self.model = model
        self.bit_assignment = bit_assignment
        self.adc_bits_default = adc_bits_default
        self.clip_percentile = clip_percentile
        self.thermal_sigma = thermal_sigma
        self.device_var_sigma = device_var_sigma
        self.hooks = []
        self.fullscale = {}

    def calibrate(self, data_loader, num_batches=4):
        """Calibrate ADC full-scale range per layer."""
        stats = defaultdict(list)
        hooks = []
        for name, mod in self.model.named_modules():
            if hasattr(mod, 'weight') and mod.weight.ndim == 2:
                def make_hook(n):
                    def hook_fn(module, inp, out):
                        stats[n].append(out.detach().abs().cpu())
                    return hook_fn
                hooks.append(mod.register_forward_hook(make_hook(name)))

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                if isinstance(batch, dict):
                    ids = batch['input_ids'].to(DEVICE)
                elif isinstance(batch, (list, tuple)):
                    ids = batch[0].to(DEVICE)
                else:
                    ids = batch.to(DEVICE)
                self.model(ids)

        for h in hooks:
            h.remove()

        for name, tensors in stats.items():
            cat = torch.cat(tensors, dim=0)
            if self.clip_percentile >= 100:
                self.fullscale[name] = cat.max().item()
            else:
                flat = cat.float().flatten()
                if flat.numel() > 1_000_000:
                    idx = torch.randperm(flat.numel())[:1_000_000]
                    flat = flat[idx]
                self.fullscale[name] = torch.quantile(
                    flat, self.clip_percentile / 100.0
                ).item()

    def install(self):
        """Install forward hooks with enhanced noise model."""
        for name, mod in self.model.named_modules():
            if name not in self.fullscale:
                continue
            bits = self.bit_assignment.get(name, self.adc_bits_default)
            vfs = self.fullscale[name]
            thermal = self.thermal_sigma
            dev_var = self.device_var_sigma

            def make_hook(b, v, th_s, dv_s):
                def hook_fn(module, inp, out):
                    levels = 2 ** b
                    step = 2 * v / levels if v > 0 else 1.0
                    # ADC clipping + quantization
                    q = torch.clamp(out, -v, v)
                    q = torch.round(q / step) * step

                    # Thermal noise (additive Gaussian)
                    if th_s > 0:
                        noise = torch.randn_like(q) * (th_s * step)
                        q = q + noise

                    # Device variation (multiplicative Gaussian on weights)
                    if dv_s > 0:
                        var_noise = torch.randn_like(q) * (dv_s * v)
                        q = q + var_noise

                    return q
                return hook_fn

            h = mod.register_forward_hook(make_hook(bits, vfs, thermal, dev_var))
            self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def measure_group_sensitivity(model, tokenizer, data_loader,
                              nominal_bits=7, probe_bits=6,
                              thermal_sigma=0.0, device_var_sigma=0.0,
                              num_calib=4, num_eval=10, label=""):
    """Measure per-group sensitivity under enhanced noise model."""
    print(f"\n{'='*60}")
    print(f"Sensitivity measurement: {label}")
    print(f"  thermal_sigma={thermal_sigma}, device_var_sigma={device_var_sigma}")
    print(f"{'='*60}")

    layer_names = []
    for name, mod in model.named_modules():
        if hasattr(mod, 'weight') and mod.weight.ndim == 2:
            layer_names.append(name)

    groups = defaultdict(list)
    for name in layer_names:
        lt = classify_layer(name)
        groups[lt].append(name)

    # Baseline: all layers at nominal_bits
    baseline_assign = {n: nominal_bits for n in layer_names}
    hook = EnhancedCIMHook(model, baseline_assign, nominal_bits,
                           thermal_sigma=thermal_sigma,
                           device_var_sigma=device_var_sigma)
    hook.calibrate(data_loader, num_calib)
    hook.install()
    baseline_ppl = compute_perplexity(model, data_loader, max_batches=num_eval)
    hook.remove()
    print(f"  Baseline PPL ({nominal_bits}b): {baseline_ppl:.2f}")

    results = {}
    for ltype, names in sorted(groups.items()):
        assign = {n: nominal_bits for n in layer_names}
        for n in names:
            assign[n] = probe_bits

        hook = EnhancedCIMHook(model, assign, nominal_bits,
                               thermal_sigma=thermal_sigma,
                               device_var_sigma=device_var_sigma)
        hook.calibrate(data_loader, num_calib)
        hook.install()
        ppl = compute_perplexity(model, data_loader, max_batches=num_eval)
        hook.remove()

        delta = ppl - baseline_ppl
        delta_per_layer = delta / len(names)
        results[ltype] = {
            'n_layers': len(names),
            'ppl': ppl,
            'delta_ppl': delta,
            'delta_per_layer': delta_per_layer,
        }
        print(f"  {ltype:12s} ({len(names):2d} layers): PPL={ppl:.2f}, "
              f"ΔPPL={delta:+.2f}, ΔPPL/layer={delta_per_layer:+.4f}")

    # Compute ranking
    ranked = sorted(results.items(), key=lambda x: x[1]['delta_per_layer'], reverse=True)
    print(f"\n  Sensitivity ranking (most → least):")
    for i, (lt, d) in enumerate(ranked):
        print(f"    {i+1}. {lt:12s} ΔPPL/layer={d['delta_per_layer']:+.4f}")

    return {'baseline_ppl': baseline_ppl, 'groups': results,
            'ranking': [lt for lt, _ in ranked]}


def random_search_allocation(layer_names, sensitivities, target_savings=0.20,
                             nominal_bits=7, n_trials=1000, seed=42):
    """Random search baseline for ILP comparison."""
    rng = np.random.RandomState(seed)
    choices = [4, 5, 6, 7, 8]
    n = len(layer_names)
    nominal_area = sum(2**nominal_bits for _ in range(n))
    budget = nominal_area * (1 - target_savings)

    best_cost = float('inf')
    best_assign = None

    for _ in range(n_trials):
        bits = rng.choice(choices, size=n)
        area = sum(2**b for b in bits)
        if area > budget:
            continue
        cost = sum(max(0, nominal_bits - b) * sensitivities.get(layer_names[i], 0)
                   for i, b in enumerate(bits))
        if cost < best_cost:
            best_cost = cost
            best_assign = {layer_names[i]: int(bits[i]) for i in range(n)}

    return best_assign, best_cost


def main():
    print("Loading model and data...")
    model, tokenizer = load_model(MODEL_NAME, cache_dir=CACHE_DIR, device=DEVICE)
    data = load_wikitext2(tokenizer, seq_len=512, split='test')
    loader = make_loader(data, batch_size=1)
    print(f"  Model: {MODEL_NAME}, Data batches: {len(loader)}")

    all_results = {}

    # ── Experiment 1: Enhanced noise model sensitivity ──────────────────
    noise_configs = [
        ("baseline_adc_only",     0.0,  0.0),
        ("thermal_low",           0.1,  0.0),
        ("thermal_high",          0.3,  0.0),
        ("device_var_low",        0.0,  0.01),
        ("device_var_high",       0.0,  0.03),
        ("combined_realistic",    0.1,  0.01),
        ("combined_worst",        0.3,  0.03),
    ]

    for config_name, thermal, dev_var in noise_configs:
        result = measure_group_sensitivity(
            model, tokenizer, loader,
            nominal_bits=7, probe_bits=6,
            thermal_sigma=thermal, device_var_sigma=dev_var,
            num_calib=4, num_eval=10,
            label=config_name
        )
        all_results[config_name] = result

    # ── Experiment 2: Check ranking consistency ──────────────────────────
    print("\n" + "="*60)
    print("RANKING CONSISTENCY ACROSS NOISE MODELS")
    print("="*60)
    baseline_ranking = all_results['baseline_adc_only']['ranking']
    for config_name, result in all_results.items():
        ranking = result['ranking']
        match = sum(1 for a, b in zip(baseline_ranking, ranking) if a == b)
        kendall = match / len(baseline_ranking)
        print(f"  {config_name:25s}: ranking={ranking}, "
              f"match={match}/{len(baseline_ranking)}")

    # ── Experiment 3: Random search baseline ─────────────────────────────
    print("\n" + "="*60)
    print("RANDOM SEARCH BASELINE")
    print("="*60)

    layer_names = []
    for name, mod in model.named_modules():
        if hasattr(mod, 'weight') and mod.weight.ndim == 2:
            layer_names.append(name)

    # Build per-layer sensitivity from group data
    group_data = all_results['baseline_adc_only']['groups']
    per_layer_sens = {}
    for name in layer_names:
        lt = classify_layer(name)
        if lt in group_data:
            per_layer_sens[name] = group_data[lt]['delta_per_layer']

    for n_trials in [100, 1000, 5000]:
        assign, cost = random_search_allocation(
            layer_names, per_layer_sens, target_savings=0.20, n_trials=n_trials
        )
        if assign:
            hook = EnhancedCIMHook(model, assign, 7)
            hook.calibrate(loader, 4)
            hook.install()
            ppl = compute_perplexity(model, loader, max_batches=10)
            hook.remove()
            print(f"  Random search ({n_trials:5d} trials): PPL={ppl:.2f}, cost={cost:.4f}")
        else:
            print(f"  Random search ({n_trials:5d} trials): no feasible solution found")

    # ── Save results ─────────────────────────────────────────────────────
    out_path = RESULTS_DIR / 'enhanced_noise_results.json'
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {
            'baseline_ppl': v['baseline_ppl'],
            'ranking': v['ranking'],
            'groups': v['groups'],
        }
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
