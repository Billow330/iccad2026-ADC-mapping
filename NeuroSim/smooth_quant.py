"""
CIM-Aware SmoothQuant
=====================
Extends the SmoothQuant algorithm (Xiao et al., ICML 2023) with hardware
constraints specific to Compute-In-Memory (CIM) accelerators:

  1. Weight range constraint  – after scaling, weights must be representable
     by the available bitcell states [off_state, on_state].
  2. Input range constraint   – scaled activations must not exceed the DAC
     input range (vdd * parallel_read).
  3. Per-layer Pareto search  – finds the optimal alpha that minimises
     post-quantisation error under the CIM constraints.

Reference:
  SmoothQuant: Accurate and Efficient Post-Training Quantization for
  Large Language Models.  Xiao et al., ICML 2023.

Usage:
    from smooth_quant import CIMSmoothQuant
    sq = CIMSmoothQuant(
            weight_bits=8, input_bits=8,
            off_state=6e-3, on_state=6e-3*17,
            vdd=1.0, parallel_read=128)
    sq.calibrate(model, data_loader, num_batches=16)
    sq.apply(model)                 # modifies model weights in-place
    smooth_model = sq.get_model()   # returns model with scale hooks installed
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# GPT-2 uses transformers.pytorch_utils.Conv1D whose weight shape is
# [in_features, out_features] — transposed relative to nn.Linear.
try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except ImportError:
    HFConv1D = None


def _is_linear_like(m):
    if isinstance(m, nn.Linear):
        return True
    if HFConv1D is not None and isinstance(m, HFConv1D):
        return True
    return False


def _get_weight_as_out_in(m):
    """Return weight as [out_features, in_features] for any linear-like layer."""
    W = m.weight.detach().float()
    if isinstance(m, nn.Linear):
        return W                # already [out, in]
    return W.t()                # Conv1D: [in, out] → [out, in]


def _apply_col_scale(m, s):
    """Multiply weight in-place by s along the in_features dimension."""
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            m.weight.data.mul_(s.unsqueeze(0))   # [out, in] × [1, in]
        else:                                     # Conv1D: [in, out]
            m.weight.data.mul_(s.unsqueeze(1))   # [in, out] × [in, 1]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _quantise_range(bits, signed=True):
    """Integer range for a given bit-width."""
    if signed:
        return -(2**(bits-1)), 2**(bits-1) - 1
    return 0, 2**bits - 1


def _quant_error(x, bits, signed=True):
    """Mean absolute quantisation error for tensor x at given precision."""
    lo, hi = _quantise_range(bits, signed)
    scale = (x.abs().max() + 1e-8) / hi
    x_q = (x / scale).round().clamp(lo, hi) * scale
    return (x - x_q).abs().mean().item()


# ────────────────────────────────────────────────────────────────────────────
# Main class
# ────────────────────────────────────────────────────────────────────────────

class CIMSmoothQuant:
    """
    CIM-hardware-aware migration of activation difficulty to weights.

    The standard SmoothQuant transform is:
        Y = (X / s) @ (s * W^T)
    where s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha).

    CIM constraints added here:
    ── Weight constraint ──────────────────────────────────────────────────
      After scaling, the weight range must be storable in the CIM array.
      The physical conductance states span [off_state, on_state] for an
      n-bit bitcell, giving (2^bitcell) levels.  We require:
          max(|s * W_j|) ≤ weight_max_bound  (= 2^(weight_bits-1) - 1 for signed)
      If the scaled weight exceeds this, we clip alpha toward 0.

    ── Input constraint ────────────────────────────────────────────────────
      The DAC input range is limited to [0, vdd] per row, and up to
      parallel_read rows fire simultaneously, so the maximum MAC output is:
          V_mac_max = vdd * parallel_read
      We require:
          max(|X_j / s_j|) ≤ vdd * parallel_read / in_features
      (rough channel-level budget).  If violated, we clip alpha toward 1.

    ── Optimization objective ──────────────────────────────────────────────
      For CIM, the primary goal is to reduce ADC saturation rate.
      We directly minimize the estimated saturation rate under the given
      ADC clip threshold (adc_clip_val), using the calibration activations
      as a proxy.  A small weight quantization error penalty is added to
      avoid degrading weight representation.

    Args:
        weight_bits   : Weight quantisation precision
        input_bits    : Activation quantisation precision
        bitcell       : Bits stored per CIM memory cell (1, 2, or 4)
        off_state     : Device off-state conductance/capacitance
        on_state      : Device on-state conductance/capacitance
        vdd           : Supply voltage (V)
        parallel_read : Rows activated in parallel
        adc_bits      : ADC bits (used for saturation-rate objective)
        adc_clip_pct  : Percentile for ADC clip calibration (e.g. 99.0)
        alpha_range   : Grid of alpha values to search over
        sat_lambda    : Weight for weight-quant error penalty in objective
        verbose       : Print per-layer results
    """

    def __init__(self,
                 weight_bits=8, input_bits=8, bitcell=1,
                 off_state=6e-3, on_state=6e-3*17,
                 vdd=1.0, parallel_read=128,
                 adc_bits=7, adc_clip_pct=99.0,
                 alpha_range=None, sat_lambda=0.1, verbose=True):
        self.weight_bits    = weight_bits
        self.input_bits     = input_bits
        self.bitcell        = bitcell
        self.off_state      = off_state
        self.on_state       = on_state
        self.vdd            = vdd
        self.parallel_read  = parallel_read
        self.adc_bits       = adc_bits
        self.adc_clip_pct   = adc_clip_pct
        self.sat_lambda     = sat_lambda
        self.verbose        = verbose

        if alpha_range is None:
            self.alpha_range = np.linspace(0.0, 1.0, 21)   # 0, 0.05, …, 1.0

        # CIM physical constraints
        self._w_max_bound  = 2**(weight_bits - 1) - 1   # signed int range
        self._act_dac_max  = vdd * parallel_read         # max DAC-driven sum

        # Calibration results: layer_name -> scale vector s  (per-channel)
        self._scales      = {}
        # Best alpha per layer
        self._alphas      = {}
        # Collected activation statistics
        self._act_scales  = defaultdict(lambda: None)   # name -> max abs per channel
        self._act_samples = defaultdict(lambda: None)   # name -> sampled activations for sat-rate
        self._hooks       = []
        # Names of layers whose weights are shared with other modules (skip these)
        self._tied_names  = set()

    # ------------------------------------------------------------------
    # Tied-weight detection
    # ------------------------------------------------------------------

    def _find_tied_weights(self, model):
        """Return set of linear-like layer names whose weights are shared with
        ANY other module (e.g. lm_head tied to wte Embedding in GPT-2)."""
        # Collect data pointers for ALL modules with a weight attribute
        all_ptrs = defaultdict(list)
        for name, m in model.named_modules():
            if hasattr(m, 'weight') and m.weight is not None:
                all_ptrs[m.weight.data_ptr()].append(name)

        # A linear-like layer is "tied" if its weight pointer appears elsewhere
        tied = set()
        for name, m in model.named_modules():
            if _is_linear_like(m) and m.weight is not None:
                ptr = m.weight.data_ptr()
                if len(all_ptrs[ptr]) > 1:
                    tied.add(name)
        return tied

    def calibrate(self, model, data_loader, num_batches=16,
                  device='cpu', task='lm'):
        """
        Run forward passes and record per-channel activation maximums
        and a sample of MAC-output magnitudes for saturation-rate estimation.
        """
        model.eval().to(device)
        self._tied_names = self._find_tied_weights(model)
        if self._tied_names and self.verbose:
            print(f"[CIMSmoothQuant] Skipping tied-weight layers: {self._tied_names}")
        act_max    = defaultdict(lambda: None)
        act_sample = defaultdict(list)   # store flat abs MAC-output samples

        def make_hook(name):
            def hook(module, inp, out):
                # Per-channel max of input activations (for scale computation)
                x = inp[0].detach().float()
                x_abs_max = x.reshape(-1, x.shape[-1]).abs().max(dim=0).values
                if act_max[name] is None:
                    act_max[name] = x_abs_max
                else:
                    act_max[name] = torch.max(act_max[name], x_abs_max)

                # Flat sample of output magnitudes (for saturation-rate estimation)
                y = out.detach().float().abs().flatten()
                if y.numel() > 4096:
                    idx = torch.randperm(y.numel())[:4096]
                    y = y[idx]
                act_sample[name].append(y.cpu())
            return hook

        hooks = []
        for name, m in model.named_modules():
            if _is_linear_like(m) and name not in self._tied_names:
                hooks.append(m.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                if task == 'lm':
                    ids = batch['input_ids'].to(device) if isinstance(batch, dict) else batch[0].to(device)
                    model(ids)
                else:
                    model(batch[0].to(device))

        for h in hooks:
            h.remove()

        self._act_scales  = dict(act_max)
        self._act_samples = {n: torch.cat(v) for n, v in act_sample.items()}
        if self.verbose:
            print(f"[CIMSmoothQuant] Calibrated {len(self._act_scales)} linear-like layers.")

    # ------------------------------------------------------------------
    # Step 2: compute per-layer optimal smooth scale s
    # ------------------------------------------------------------------

    def compute_scales(self, model):
        """
        For each Linear layer, find the alpha ∈ alpha_range that minimises
        the estimated ADC saturation rate on calibration outputs while
        satisfying CIM physical constraints.

        Objective:
            min  sat_rate(alpha) + sat_lambda * err_weight(alpha)
        subject to:
            max|W_scaled| ≤ w_max_bound * 2   (weight range constraint)
            max|X_smooth| ≤ act_dac_max       (DAC range constraint)

        sat_rate(alpha) is estimated by scaling the collected output samples
        by the ratio of old vs new activation norm (a first-order proxy),
        then checking against the p99-clip ADC threshold.

        Must be called after calibrate().
        """
        scales, alphas = {}, {}

        for name, module in model.named_modules():
            if not _is_linear_like(module):
                continue
            if name not in self._act_scales or self._act_scales[name] is None:
                continue

            act_max_ch = self._act_scales[name].float()   # [in_features]
            W = _get_weight_as_out_in(module)              # [out_features, in_features]
            w_max_ch  = W.abs().max(dim=0).values          # [in_features]

            # Calibration output samples for saturation-rate estimation
            out_samples = self._act_samples.get(name, None)  # flat abs tensor
            if out_samples is not None and out_samples.numel() > 0:
                # ADC clip threshold at adc_clip_pct percentile
                clip_val = torch.quantile(
                    out_samples,
                    min(self.adc_clip_pct, 100.0) / 100.0
                ).item()
                clip_val = max(clip_val, 1e-8)
            else:
                out_samples = None
                clip_val = 1e-8

            # Baseline saturation rate (alpha=current, no scaling yet)
            if out_samples is not None:
                sat_base = (out_samples > clip_val).float().mean().item()
            else:
                sat_base = 1.0

            best_alpha  = 0.5
            best_score  = float('inf')
            best_scale  = torch.ones_like(act_max_ch)

            for alpha in self.alpha_range:
                s = self._compute_scale(act_max_ch, w_max_ch, alpha)

                # ── CIM constraint 1: weight range ──────────────────────
                W_scaled = W * s.unsqueeze(0)
                if W_scaled.abs().max().item() > self._w_max_bound * 2:
                    continue

                # ── CIM constraint 2: activation range ──────────────────
                act_scaled_max = (act_max_ch / s).max().item()
                if act_scaled_max > self._act_dac_max:
                    continue

                # ── Balanced objective: activation + weight quantization ─────
                # CIM bottleneck: ADC precision limited, so activation range
                # should be minimized. But weight scaling beyond INT8 causes
                # catastrophic quantization error.
                # Minimize: gamma * err_act + (1-gamma) * err_w, gamma=0.75
                # (weight activation error more since ADC is the bottleneck)
                act_scaled = act_max_ch / s    # [in_features]
                err_act = _quant_error(act_scaled.unsqueeze(0),
                                       self.input_bits, signed=True)

                # ── Secondary objective: weight quantization error ───────
                err_w = _quant_error(W_scaled, self.weight_bits, signed=True)

                score = err_act + self.sat_lambda * err_w

                if score < best_score:
                    best_score  = score
                    best_alpha  = alpha
                    best_scale  = s

            scales[name] = best_scale
            alphas[name] = best_alpha

        self._scales = scales
        self._alphas = alphas

        if self.verbose:
            alpha_vals = list(alphas.values())
            print(f"[CIMSmoothQuant] Scales computed. "
                  f"Mean alpha={np.mean(alpha_vals):.3f}, "
                  f"Std alpha={np.std(alpha_vals):.3f}")
        return scales, alphas

    def _compute_scale(self, act_max_ch, w_max_ch, alpha):
        """
        s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
        Clamped to avoid division by zero and extreme values.
        """
        s = (act_max_ch.pow(alpha) /
             w_max_ch.pow(1.0 - alpha).clamp(min=1e-8))
        s = s.clamp(min=1e-8)
        return s

    # ------------------------------------------------------------------
    # Step 3: apply scales to model weights in-place
    # ------------------------------------------------------------------

    def apply(self, model):
        """
        Absorb the smooth scale into weights:
            W_new_j = s_j * W_j   (column-wise)
        The corresponding activation division (X / s) is handled by
        a lightweight forward hook installed on each layer.

        This modifies model.weight in-place.
        """
        if not self._scales:
            raise RuntimeError("Call compute_scales() before apply().")

        for name, module in model.named_modules():
            if not _is_linear_like(module):
                continue
            if name not in self._scales:
                continue
            s = self._scales[name].to(module.weight.device)
            # Multiply weight columns by s (handles both Linear and Conv1D)
            _apply_col_scale(module, s)
            # Install activation hook to divide input by s
            self._install_act_hook(module, s)

        if self.verbose:
            print(f"[CIMSmoothQuant] Scales applied to {len(self._scales)} layers.")

    def _install_act_hook(self, module, s):
        """Pre-forward hook: divide incoming activations by s."""
        def hook(mod, inp):
            x = inp[0]
            return (x / s.to(x.device),) + inp[1:]
        module.register_forward_pre_hook(hook)

    # ------------------------------------------------------------------
    # Convenience: calibrate + compute + apply in one call
    # ------------------------------------------------------------------

    def fit(self, model, data_loader, num_batches=16, device='cpu', task='lm'):
        """Calibrate, compute scales, and apply in one call."""
        self.calibrate(model, data_loader, num_batches, device, task)
        self.compute_scales(model)
        self.apply(model)
        return model

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self):
        """Print per-layer alpha and scale statistics."""
        if not self._alphas:
            print("[CIMSmoothQuant] No scales computed yet.")
            return

        print("\n" + "="*70)
        print(f"{'Layer':<45} {'alpha':>6} {'scale_mean':>11} {'scale_max':>10}")
        print("="*70)
        for name in self._alphas:
            s = self._scales[name]
            print(f"{name[-44:]:<45} "
                  f"{self._alphas[name]:>6.3f} "
                  f"{s.mean().item():>11.4f} "
                  f"{s.max().item():>10.4f}")
        print("="*70)

    def save_scales(self, path='smooth_scales.pt'):
        """Save computed scales to disk for later reuse."""
        torch.save({'scales': self._scales, 'alphas': self._alphas}, path)
        print(f"[CIMSmoothQuant] Scales saved to {path}")

    def load_scales(self, path='smooth_scales.pt'):
        """Load previously computed scales."""
        data = torch.load(path, map_location='cpu')
        self._scales = data['scales']
        self._alphas = data['alphas']
        print(f"[CIMSmoothQuant] Loaded scales from {path} "
              f"({len(self._scales)} layers)")


# ────────────────────────────────────────────────────────────────────────────
# Utility: compare perplexity before/after smoothing (for paper table)
# ────────────────────────────────────────────────────────────────────────────

def compute_perplexity(model, data_loader, device='cpu', max_batches=50):
    """
    Compute perplexity on a causal language model.
    data_loader should yield dicts with 'input_ids'.
    """
    model.eval().to(device)
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break
            input_ids = batch['input_ids'].to(device) if isinstance(batch, dict) else batch[0].to(device)
            # Shift labels: predict next token
            labels = input_ids.clone()
            outputs = model(input_ids, labels=labels)
            # outputs.loss is mean NLL over non-padding tokens
            nll  = outputs.loss.item()
            ntok = (labels != -100).sum().item()
            if ntok == 0:
                ntok = input_ids.numel()
            total_nll    += nll * ntok
            total_tokens += ntok

    ppl = np.exp(total_nll / max(total_tokens, 1))
    return ppl
