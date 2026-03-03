"""
Outlier Analysis Module for LLM on CIM Accelerators
=====================================================
Characterizes per-layer activation outliers and quantifies their impact on:
  1. ADC saturation rate
  2. Inference accuracy degradation
  3. Required ADC dynamic range

Usage:
    from outlier_analysis import OutlierAnalyzer
    analyzer = OutlierAnalyzer(model, adc_bits=7, vdd=1.0, parallel_read=128)
    stats = analyzer.run(data_loader, num_batches=16)
    analyzer.report(stats)
"""

import torch
import torch.nn as nn
import numpy as np
import csv
import os
from collections import defaultdict

# GPT-2 uses transformers.pytorch_utils.Conv1D (weight shape [in, out])
# rather than nn.Linear (weight shape [out, in]).
try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except ImportError:
    HFConv1D = None

def _is_linear_like(m):
    """Return True for nn.Linear and HuggingFace Conv1D."""
    if isinstance(m, nn.Linear):
        return True
    if HFConv1D is not None and isinstance(m, HFConv1D):
        return True
    return False

def _get_in_features(m):
    """Return in_features regardless of layer type."""
    if isinstance(m, nn.Linear):
        return m.in_features
    # Conv1D: weight is [in_features, out_features]
    return m.weight.shape[0]


# ────────────────────────────────────────────────────────────────────────────
# Core analyzer
# ────────────────────────────────────────────────────────────────────────────

class OutlierAnalyzer:
    """
    Hooks into every nn.Linear layer of a model and collects activation
    statistics to characterise outlier behaviour relative to CIM hardware
    constraints.

    Args:
        model        : PyTorch model (already in eval mode)
        adc_bits     : ADC precision used in CIM (default 7)
        vdd          : Supply voltage (V); determines ADC input range
        parallel_read: Number of rows activated simultaneously; sets the
                       maximum analog MAC output before ADC saturation
        threshold_sigma: Activations whose absolute value exceeds
                         threshold_sigma * channel_std are classed as outliers
    """

    def __init__(self, model, adc_bits=7, vdd=1.0, parallel_read=128,
                 threshold_sigma=6.0):
        self.model = model
        self.adc_bits = adc_bits
        self.vdd = vdd
        self.parallel_read = parallel_read
        self.threshold_sigma = threshold_sigma

        # ADC full-scale corresponds to parallel_read inputs each at vdd
        # (worst-case all-ones pattern)
        self.adc_fullscale = parallel_read * vdd

        self._hooks = []
        self._raw_stats = defaultdict(list)   # layer_name -> list of tensors

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _make_hook(self, name):
        def hook(module, inp, out):
            x = inp[0].detach().float()   # [B, *, in_features]
            self._raw_stats[name].append(x.reshape(-1, x.shape[-1]))
        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if _is_linear_like(module):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Main analysis loop
    # ------------------------------------------------------------------

    def run(self, data_loader, num_batches=16, device='cpu', task='lm'):
        """
        Forward-pass ``num_batches`` batches and collect activation statistics.

        Args:
            data_loader : DataLoader yielding (input_ids, ...) or (images, labels)
            num_batches : How many batches to process
            device      : 'cpu' or 'cuda'
            task        : 'lm' for language models, 'vision' for image classifiers

        Returns:
            dict mapping layer name -> per-layer stats dict
        """
        self.model.eval()
        self.model.to(device)
        self._raw_stats.clear()
        self._register_hooks()

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                if task == 'lm':
                    input_ids = batch['input_ids'].to(device) if isinstance(batch, dict) else batch[0].to(device)
                    self.model(input_ids)
                else:
                    images = batch[0].to(device)
                    self.model(images)

        self._remove_hooks()
        return self._compute_stats()

    # ------------------------------------------------------------------
    # Statistics computation
    # ------------------------------------------------------------------

    def _compute_stats(self):
        stats = {}
        for name, tensor_list in self._raw_stats.items():
            X = torch.cat(tensor_list, dim=0)   # [N_tokens, in_features]
            stats[name] = self._layer_stats(X)
        return stats

    def _layer_stats(self, X):
        """
        Compute outlier metrics for a single layer's collected activations.

        X : FloatTensor [N_tokens, C]  (C = in_features)
        """
        N, C = X.shape

        # ── Per-channel statistics ──────────────────────────────────────
        ch_mean = X.mean(dim=0)          # [C]
        ch_std  = X.std(dim=0) + 1e-8   # [C]
        ch_max  = X.abs().max(dim=0).values  # [C]

        # ── Outlier mask (per token, per channel) ──────────────────────
        # A value is an outlier if |x - mu| > threshold_sigma * sigma
        z_score = (X - ch_mean.unsqueeze(0)).abs() / ch_std.unsqueeze(0)
        outlier_mask = z_score > self.threshold_sigma  # [N, C]

        # Fraction of tokens that contain at least one outlier channel
        token_has_outlier = outlier_mask.any(dim=1).float().mean().item()

        # Fraction of channels that are persistently outlier-prone
        # (outlier in >10% of tokens)
        ch_outlier_rate = outlier_mask.float().mean(dim=0)   # [C]
        outlier_channels = (ch_outlier_rate > 0.10).sum().item()
        outlier_channel_fraction = outlier_channels / C

        # ── ADC saturation analysis ─────────────────────────────────────
        # Simulate what happens when quantised integer activations drive
        # parallel_read rows of a CIM array.
        # We estimate the MAC output magnitude as ||x||_1 scaled by
        # a weight ≈ 0.5 (random weights, zero-mean).
        # Saturation occurs when the analog sum > adc_fullscale.
        # Here we use a simpler per-token upper-bound: max(|x|) * parallel_read
        # as a worst-case, and the actual distribution median as typical.
        token_max_abs = X.abs().max(dim=1).values   # [N]
        worst_case_mac = token_max_abs * self.parallel_read
        sat_rate_worst = (worst_case_mac > self.adc_fullscale).float().mean().item()

        # Typical-case: use token L1-norm / C * parallel_read
        token_l1 = X.abs().mean(dim=1)   # [N]
        typical_mac = token_l1 * self.parallel_read
        sat_rate_typical = (typical_mac > self.adc_fullscale).float().mean().item()

        # ── Required ADC bits to avoid saturation ──────────────────────
        # ADC needs to represent values up to max observed MAC output
        # (assuming weights in [-1, 1]).
        max_mac_observed = worst_case_mac.max().item()
        if max_mac_observed > 0:
            bits_needed = np.ceil(np.log2(max_mac_observed + 1)).astype(int)
        else:
            bits_needed = 1
        adc_overhead_bits = max(0, int(bits_needed) - self.adc_bits)

        # ── Global activation range ─────────────────────────────────────
        act_max  = X.abs().max().item()
        act_mean = X.abs().mean().item()
        act_std  = X.std().item()

        # ── Outlier channel indices (top-20) ───────────────────────────
        top_outlier_channels = ch_outlier_rate.topk(min(20, C)).indices.tolist()

        return {
            'num_tokens'               : N,
            'in_features'              : C,
            'act_max'                  : act_max,
            'act_mean'                 : act_mean,
            'act_std'                  : act_std,
            'outlier_channel_fraction' : outlier_channel_fraction,
            'outlier_channels_count'   : int(outlier_channels),
            'token_outlier_rate'       : token_has_outlier,
            'sat_rate_worst'           : sat_rate_worst,
            'sat_rate_typical'         : sat_rate_typical,
            'bits_needed'              : int(bits_needed),
            'adc_overhead_bits'        : adc_overhead_bits,
            'ch_max'                   : ch_max.numpy(),
            'ch_std'                   : ch_std.numpy(),
            'top_outlier_channels'     : top_outlier_channels,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, stats, verbose=True):
        """Print a formatted summary table and return a summary dict."""
        print("\n" + "="*80)
        print(f"{'Layer':<45} {'OutlierCh%':>10} {'SatRate%':>9} {'ActMax':>9} {'ADCbits+':>8}")
        print("="*80)

        summary = {
            'mean_outlier_ch_frac' : 0.0,
            'mean_sat_rate_worst'  : 0.0,
            'max_act_max'          : 0.0,
            'max_bits_needed'      : 0,
            'layers_needing_extra_bits': 0,
        }
        n = len(stats)

        for name, s in stats.items():
            short = name[-44:] if len(name) > 44 else name
            print(f"{short:<45} "
                  f"{s['outlier_channel_fraction']*100:>9.1f}% "
                  f"{s['sat_rate_worst']*100:>8.1f}% "
                  f"{s['act_max']:>9.3f} "
                  f"{s['adc_overhead_bits']:>7}b")

            summary['mean_outlier_ch_frac'] += s['outlier_channel_fraction'] / n
            summary['mean_sat_rate_worst']  += s['sat_rate_worst'] / n
            summary['max_act_max']           = max(summary['max_act_max'], s['act_max'])
            summary['max_bits_needed']       = max(summary['max_bits_needed'], s['bits_needed'])
            if s['adc_overhead_bits'] > 0:
                summary['layers_needing_extra_bits'] += 1

        print("="*80)
        print(f"\n[Summary]")
        print(f"  Mean outlier-channel fraction : {summary['mean_outlier_ch_frac']*100:.1f}%")
        print(f"  Mean worst-case ADC sat. rate : {summary['mean_sat_rate_worst']*100:.1f}%")
        print(f"  Max activation magnitude      : {summary['max_act_max']:.3f}")
        print(f"  Max ADC bits needed           : {summary['max_bits_needed']}")
        print(f"  Layers needing extra ADC bits : {summary['layers_needing_extra_bits']} / {n}")
        print()
        return summary

    def save_csv(self, stats, path='outlier_stats.csv'):
        """Save per-layer scalar statistics to CSV for plotting."""
        rows = []
        for name, s in stats.items():
            rows.append({
                'layer'                    : name,
                'in_features'              : s['in_features'],
                'act_max'                  : s['act_max'],
                'act_mean'                 : s['act_mean'],
                'act_std'                  : s['act_std'],
                'outlier_channel_fraction' : s['outlier_channel_fraction'],
                'outlier_channels_count'   : s['outlier_channels_count'],
                'token_outlier_rate'       : s['token_outlier_rate'],
                'sat_rate_worst'           : s['sat_rate_worst'],
                'sat_rate_typical'         : s['sat_rate_typical'],
                'bits_needed'              : s['bits_needed'],
                'adc_overhead_bits'        : s['adc_overhead_bits'],
            })
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OutlierAnalyzer] Statistics saved to {path}")


# ────────────────────────────────────────────────────────────────────────────
# ADC saturation rate sweep (for paper figure: sat_rate vs adc_bits)
# ────────────────────────────────────────────────────────────────────────────

def sweep_adc_bits(stats, adc_bits_range=range(3, 12), parallel_read=128, vdd=1.0):
    """
    For each ADC precision in adc_bits_range, compute the average worst-case
    ADC saturation rate across all layers given the collected activation stats.

    Returns: dict {adc_bits: mean_sat_rate}
    """
    results = {}
    for bits in adc_bits_range:
        fullscale = (2**bits - 1) * vdd   # simplified: ADC range ∝ 2^bits
        sat_rates = []
        for name, s in stats.items():
            # Recompute saturation using stored ch_max
            ch_max = torch.tensor(s['ch_max'])
            worst_case = ch_max.max().item() * parallel_read
            sat_rates.append(float(worst_case > fullscale))
        results[bits] = np.mean(sat_rates)
    return results
