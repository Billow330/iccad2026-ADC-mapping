"""
LLM + CIM Inference Pipeline
==============================
Main experiment entry point for:
  "Outlier-Aware CIM Mapping for Large Language Model Inference:
   Characterization, Mitigation, and Hardware Co-Design"

Experiments:
  1. Outlier characterisation on GPT-2 / OPT
  2. Baseline perplexity under CIM noise (ADC saturation)
  3. CIM-Aware SmoothQuant vs. standard SmoothQuant
  4. ADC bits sweep (Pareto: perplexity vs. ADC area)
  5. PPA analysis via NeuroSIM C++ backend

Usage:
    python llm_inference.py --model gpt2 --task characterize
    python llm_inference.py --model gpt2 --task baseline  --adc_bits 7
    python llm_inference.py --model gpt2 --task smooth    --adc_bits 7
    python llm_inference.py --model gpt2 --task sweep_adc
    python llm_inference.py --model gpt2 --task ppa
"""

import os, sys, argparse, json, csv
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'pytorch-quantization'))

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GPT2LMHeadModel, GPT2TokenizerFast,
    OPTForCausalLM,
)
from torch.utils.data import DataLoader

from outlier_analysis import OutlierAnalyzer, sweep_adc_bits, _is_linear_like, _get_in_features
from smooth_quant import CIMSmoothQuant, compute_perplexity

try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except ImportError:
    HFConv1D = None


# ────────────────────────────────────────────────────────────────────────────
# Argument parser
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='LLM + CIM Inference Experiment')

    # Model
    p.add_argument('--model',      default='gpt2',
                   help='HuggingFace model name or local path to model directory')
    p.add_argument('--model_cache', default='./model_cache',
                   help='Local cache directory for downloaded models')
    p.add_argument('--hf_endpoint', default='',
                   help='HuggingFace mirror endpoint, e.g. https://hf-mirror.com '
                        '(can also be set via HF_ENDPOINT env var)')

    # Task
    p.add_argument('--task', default='characterize',
                   choices=['characterize', 'baseline', 'smooth', 'sweep_adc', 'ppa', 'all'],
                   help='Experiment to run')

    # Dataset
    p.add_argument('--dataset',   default='wikitext2',
                   choices=['wikitext2', 'ptb'])
    p.add_argument('--seq_len',   type=int, default=512)
    p.add_argument('--num_calib_batches', type=int, default=32,
                   help='Batches for calibration / characterisation')
    p.add_argument('--num_eval_batches',  type=int, default=100,
                   help='Batches for perplexity evaluation')
    p.add_argument('--batch_size', type=int, default=1)

    # CIM hardware
    p.add_argument('--adc_bits',      type=int,   default=7)
    p.add_argument('--adc_clip_pct',  type=float, default=99.0,
                   help='ADC calibration percentile (99=p99-clip, 100=max-clip)')
    p.add_argument('--weight_bits',   type=int,   default=8)
    p.add_argument('--input_bits',    type=int,   default=8)
    p.add_argument('--bitcell',       type=int,   default=1)
    p.add_argument('--parallel_read', type=int,   default=128)
    p.add_argument('--sub_array',     type=int,   default=128)
    p.add_argument('--mem_type',      default='resistive',
                   choices=['resistive', 'capacitive'])
    p.add_argument('--off_state',     type=float, default=6e-3)
    p.add_argument('--on_state',      type=float, default=6e-3*17)
    p.add_argument('--vdd',           type=float, default=1.0)
    p.add_argument('--read_noise',    type=float, default=0.0)

    # SmoothQuant
    p.add_argument('--alpha', type=float, default=0.5,
                   help='SmoothQuant migration strength (0=all to weight, 1=all to act)')
    p.add_argument('--cim_constrained', type=int, default=1,
                   help='1=use CIM-aware constraints, 0=standard SmoothQuant')

    # ADC sweep
    p.add_argument('--adc_min', type=int, default=3)
    p.add_argument('--adc_max', type=int, default=11)

    # Output
    p.add_argument('--output_dir', default='./results')
    p.add_argument('--device',     default='cpu')

    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Dataset utilities
# ────────────────────────────────────────────────────────────────────────────

def load_wikitext2(tokenizer, seq_len=512, split='test'):
    """
    Load WikiText-2 via HuggingFace datasets and tokenise.
    Falls back to a tiny stub if the dataset is unavailable (offline mode).
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1',
                               split=split)
        text = '\n'.join(dataset['text'])
    except Exception as e:
        print(f"[WARNING] Could not load WikiText-2: {e}")
        print("[WARNING] Using synthetic data for testing.")
        text = "The quick brown fox jumps over the lazy dog. " * 10000

    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids'][0]

    # Chunk into sequences of seq_len
    n_seqs = len(input_ids) // seq_len
    chunks = [input_ids[i*seq_len:(i+1)*seq_len]
              for i in range(n_seqs)]
    data = [{'input_ids': c.unsqueeze(0)} for c in chunks]
    return data


def make_loader(data, batch_size=1):
    return DataLoader(data, batch_size=batch_size, shuffle=False,
                      collate_fn=lambda x: {
                          'input_ids': torch.cat([d['input_ids'] for d in x], dim=0)
                      })


# ────────────────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────────────────

def load_model(model_name, cache_dir='./model_cache', device='cpu', hf_endpoint=''):
    """Load a causal LM and its tokeniser.

    model_name can be:
      - a HuggingFace model ID  (e.g. 'gpt2', 'facebook/opt-125m')
      - a local directory path  (e.g. './model_cache/gpt2_local')

    hf_endpoint: optional mirror, e.g. 'https://hf-mirror.com'
    """
    print(f"[INFO] Loading model: {model_name}")
    os.makedirs(cache_dir, exist_ok=True)

    # ── HuggingFace mirror support ────────────────────────────────────────
    endpoint = hf_endpoint or os.environ.get('HF_ENDPOINT', '')
    if endpoint:
        os.environ['HF_ENDPOINT'] = endpoint
        print(f"[INFO] Using HF endpoint: {endpoint}")

    # ── Detect local path ─────────────────────────────────────────────────
    local_path = Path(model_name)
    if local_path.exists() and local_path.is_dir():
        load_from = str(local_path)
        print(f"[INFO] Loading from local path: {load_from}")
    else:
        load_from = model_name

    tokenizer = AutoTokenizer.from_pretrained(
        load_from, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        load_from, cache_dir=cache_dir,
        torch_dtype=torch.float32)
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_layers  = sum(1 for _, m in model.named_modules() if _is_linear_like(m))
    print(f"[INFO] Loaded {model_name}: {n_params:.1f}M params, {n_layers} linear-like layers")
    return model, tokenizer


# ────────────────────────────────────────────────────────────────────────────
# CIM noise injection (simulate ADC saturation and read noise)
# ────────────────────────────────────────────────────────────────────────────

class CIMNoiseHook:
    """
    Simulates CIM inference noise: DAC input quantization + ADC output quantization.

    CIM Hardware Model
    ------------------
    In Compute-In-Memory (CIM) arrays:
    - Weights are stored as analog conductance values (NOT digitally quantized)
    - Input activations pass through a DAC (Digital-to-Analog Converter)
      with `input_bits` precision
    - The analog MAC output (dot product) is read by an ADC with `adc_bits`
      precision and a fixed dynamic range [0, adc_fullscale]

    The key problem (this paper's motivation):
    - LLM activations have severe outliers (some channels 100x larger than median)
    - These outliers drive the MAC output beyond adc_fullscale → ADC saturation
    - ADC saturation clips the output → accuracy loss

    SmoothQuant's role:
    - Migrates activation difficulty to weights via smooth scaling
    - Reduces peak activation magnitude → fewer ADC saturations
    - CIM-aware SQ optimizes the scale per-layer under CIM hardware constraints

    Read noise (sigma * |output|) models conductance variation in memristors.
    """

    def __init__(self, model, adc_bits=7, weight_bits=8, input_bits=8,
                 parallel_read=128, vdd=1.0, read_noise=0.0,
                 clip_percentile=99.0):
        self.model         = model
        self.adc_bits      = adc_bits
        self.weight_bits   = weight_bits
        self.input_bits    = input_bits
        self.parallel_read = parallel_read
        self.vdd           = vdd
        self.read_noise    = read_noise
        self.clip_percentile = clip_percentile
        self._hooks        = []
        self.sat_counts    = {}
        self.total_counts  = {}
        self._clip         = {}   # per-layer ADC clip scale (calibrated)

    # ------------------------------------------------------------------
    # Quantisation helper (for DAC input quantization)
    # ------------------------------------------------------------------

    @staticmethod
    def _quantise_act(x, bits):
        """Per-token symmetric INT quantization for activations (DAC model).

        Each token is quantized independently to handle varying dynamic range.
        This matches the DAC model: each MAC cycle uses one token's activations.
        """
        n = 2 ** (bits - 1) - 1   # e.g. 127 for 8-bit
        orig_shape = x.shape
        xf = x.reshape(-1, orig_shape[-1])       # [N_tokens, C]
        scale = xf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / n
        xq = (xf / scale).round().clamp(-n, n) * scale
        return xq.reshape(orig_shape)

    def calibrate(self, data_loader, device='cpu', num_batches=4,
                  clip_percentile=99.0):
        """Calibrate per-layer ADC full-scale from FP32 MAC outputs.

        clip_percentile=99.0 (p99-clip): sets ADC range to the 99th percentile
        of observed MAC output magnitudes.  This deliberately allows ~1% of
        outputs to saturate, trading slight clipping noise for better resolution
        on the main distribution.  SmoothQuant then directly reduces this 1%
        saturation, yielding measurable accuracy gains.

        clip_percentile=100.0 is equivalent to the old max-clip behavior.
        """
        self.model.eval()
        n_adc = 2 ** self.adc_bits - 1
        raw = {}   # name -> list of flat abs tensors (subsampled)

        def make_cal_hook(name):
            def h(mod, inp, out):
                v = out.detach().float().abs().flatten()
                # Subsample to keep memory bounded (max 8192 values per batch)
                if v.numel() > 8192:
                    idx = torch.randperm(v.numel())[:8192]
                    v = v[idx]
                raw.setdefault(name, []).append(v.cpu())
            return h

        tmp = []
        for name, m in self.model.named_modules():
            if _is_linear_like(m):
                tmp.append(m.register_forward_hook(make_cal_hook(name)))

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches: break
                ids = batch['input_ids'].to(device) if isinstance(batch, dict) else batch[0].to(device)
                self.model(ids)

        for h in tmp: h.remove()

        for name, vals in raw.items():
            all_vals = torch.cat(vals)
            if clip_percentile >= 100.0:
                p = all_vals.max().item()
            else:
                p = torch.quantile(all_vals,
                                   clip_percentile / 100.0).item()
            self._clip[name] = max(p / n_adc, 1e-8)

        mode = f"p{clip_percentile:.0f}-clip"
        print(f"[CIMNoiseHook] Calibrated {len(self._clip)} layers "
              f"(ADC {self.adc_bits}b, {mode}).")

    def _make_hook(self, name, module):
        n_adc      = 2 ** self.adc_bits - 1
        read_noise = self.read_noise

        def hook(mod, inp, out):
            # Use the FP32 MAC output directly (analog CIM model:
            # weights stored in memristors, inputs applied via DAC).
            # We model ADC quantization as noise on top of the FP32 result.
            y = out.detach().float()

            # ── Read noise (memristor conductance variation) ──────────────
            if read_noise > 0:
                y = y + torch.randn_like(y) * read_noise * y.abs()

            # ── ADC quantization with saturation clipping ─────────────────
            if name in self._clip:
                act_scale = self._clip[name]
            else:
                act_scale = (y.abs().max().item() / n_adc + 1e-8)

            adc_max = act_scale * n_adc

            sat_mask = y.abs() > adc_max
            self.sat_counts[name]   = self.sat_counts.get(name, 0) + sat_mask.sum().item()
            self.total_counts[name] = self.total_counts.get(name, 0) + y.numel()

            y_q = (y / act_scale).round().clamp(-n_adc, n_adc) * act_scale
            return y_q.to(out.dtype)
        return hook

    def install(self):
        for name, m in self.model.named_modules():
            if _is_linear_like(m):
                h = m.register_forward_hook(self._make_hook(name, m))
                self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def saturation_rates(self):
        return {n: self.sat_counts.get(n, 0) / max(self.total_counts.get(n, 1), 1)
                for n in self.total_counts}


# ────────────────────────────────────────────────────────────────────────────
# PPA lookup / NeuroSIM wrapper
# ────────────────────────────────────────────────────────────────────────────

def run_neurosim_ppa(model_name, weight_bits, input_bits, sub_array,
                     parallel_read, output_dir, adc_bits):
    """
    Call NeuroSIM C++ backend to compute PPA for the given model/config.
    Returns dict with area, latency, energy, TOPS_W.
    """
    net_csv_map = {
        'gpt2'              : 'NetWork_gpt2.csv',
        'gpt2-medium'       : 'NetWork_gpt2_medium.csv',
        'facebook/opt-125m' : 'NetWork_opt125m.csv',
        'facebook/opt-350m' : 'NetWork_opt350m.csv',
    }
    net_csv = net_csv_map.get(model_name)
    if net_csv is None or not (ROOT / 'NeuroSIM' / net_csv).exists():
        print(f"[WARNING] NetWork CSV not found for {model_name}. "
              "Skipping C++ PPA (run inference.py with --ppa 1 first to generate it).")
        return None

    import subprocess
    cmd = [
        str(ROOT / 'NeuroSIM' / 'main'),
        str(ROOT / 'NeuroSIM' / net_csv),
        str(weight_bits),
        str(input_bits),
        str(sub_array),
        str(parallel_read),
    ]
    # Append weight/input CSV paths from layer_record if present
    layer_dir = ROOT / f'layer_record_{model_name.replace("/", "_")}'
    if layer_dir.exists():
        for f in sorted(layer_dir.glob('weight_*.csv')):
            layer_id = f.stem.replace('weight_layer', '')
            inp_f = layer_dir / f'input_layer{layer_id}.csv'
            if inp_f.exists():
                cmd += [str(f), str(inp_f)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout
        # Parse key metrics from NeuroSIM output
        ppa = {}
        for line in output.split('\n'):
            if 'Chip area' in line:
                ppa['area_um2'] = float(line.split()[-1])
            elif 'Total latency' in line:
                ppa['latency_ns'] = float(line.split()[-1])
            elif 'Energy' in line and 'dynamic' in line.lower():
                ppa['energy_pJ'] = float(line.split()[-1])
            elif 'TOPS/W' in line:
                ppa['tops_w'] = float(line.split()[-1])
        return ppa
    except Exception as e:
        print(f"[ERROR] NeuroSIM failed: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# Experiment functions
# ────────────────────────────────────────────────────────────────────────────

def task_characterize(args, model, tokenizer, calib_loader):
    """Experiment 1: Outlier characterisation."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Activation Outlier Characterisation")
    print("="*60)

    analyzer = OutlierAnalyzer(
        model,
        adc_bits=args.adc_bits,
        vdd=args.vdd,
        parallel_read=args.parallel_read,
        threshold_sigma=6.0,
    )
    stats = analyzer.run(calib_loader, num_batches=args.num_calib_batches,
                         device=args.device, task='lm')
    summary = analyzer.report(stats)

    # Save CSV for plotting
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.replace('/', '_')
    analyzer.save_csv(stats, path=str(out_dir / f'outlier_{model_tag}_adc{args.adc_bits}.csv'))

    # ADC bits sweep for saturation rate figure
    sweep = sweep_adc_bits(stats, adc_bits_range=range(args.adc_min, args.adc_max),
                            parallel_read=args.parallel_read, vdd=args.vdd)
    print("\n[ADC Saturation Rate vs. Bits]")
    for bits, rate in sweep.items():
        print(f"  ADC {bits:2d} bits: sat_rate = {rate*100:.1f}%")

    # Save sweep CSV
    with open(str(out_dir / f'adc_sweep_{model_tag}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['adc_bits', 'sat_rate'])
        writer.writerows(sweep.items())

    return stats, summary


def task_baseline(args, model, tokenizer, calib_loader, eval_loader):
    """Experiment 2: Baseline perplexity with CIM ADC saturation noise."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Baseline Perplexity under CIM ADC Saturation")
    print("="*60)

    # Without noise
    ppl_clean = compute_perplexity(model, eval_loader,
                                   device=args.device,
                                   max_batches=args.num_eval_batches)
    print(f"Clean perplexity (no CIM noise):  {ppl_clean:.2f}")

    # With CIM noise
    noise_hook = CIMNoiseHook(model, adc_bits=args.adc_bits,
                               weight_bits=args.weight_bits, input_bits=args.input_bits,
                               parallel_read=args.parallel_read,
                               vdd=args.vdd, read_noise=args.read_noise)
    noise_hook.calibrate(calib_loader, device=args.device, num_batches=8,
                         clip_percentile=args.adc_clip_pct)
    noise_hook.install()
    ppl_cim = compute_perplexity(model, eval_loader,
                                 device=args.device,
                                 max_batches=args.num_eval_batches)
    noise_hook.remove()
    print(f"CIM perplexity (ADC {args.adc_bits}b, noise={args.read_noise}): {ppl_cim:.2f}")
    print(f"Perplexity degradation: +{ppl_cim - ppl_clean:.2f}")

    sat_rates = noise_hook.saturation_rates()
    mean_sat = np.mean(list(sat_rates.values()))
    print(f"Mean ADC saturation rate: {mean_sat*100:.2f}%")

    result = {
        'model': args.model, 'adc_bits': args.adc_bits,
        'ppl_clean': ppl_clean, 'ppl_cim': ppl_cim,
        'ppl_delta': ppl_cim - ppl_clean,
        'mean_sat_rate': mean_sat,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.replace('/', '_')
    with open(str(out_dir / f'baseline_{model_tag}_adc{args.adc_bits}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def task_smooth(args, model, tokenizer, calib_loader, eval_loader):
    """Experiment 3: CIM-Aware SmoothQuant vs. standard SmoothQuant."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: CIM-Aware SmoothQuant")
    print("="*60)

    import copy

    # ── Clean baseline ──────────────────────────────────────────────────
    ppl_clean = compute_perplexity(model, eval_loader,
                                   device=args.device,
                                   max_batches=args.num_eval_batches)
    print(f"Clean (fp32):       {ppl_clean:.2f}")

    # ── CIM baseline (no smoothing) ─────────────────────────────────────
    nh = CIMNoiseHook(model, adc_bits=args.adc_bits,
                      weight_bits=args.weight_bits, input_bits=args.input_bits,
                      parallel_read=args.parallel_read, vdd=args.vdd)
    nh.calibrate(calib_loader, device=args.device, num_batches=8,
                 clip_percentile=args.adc_clip_pct)
    nh.install()
    ppl_cim_base = compute_perplexity(model, eval_loader, device=args.device,
                                      max_batches=args.num_eval_batches)
    nh.remove()
    sat_base = np.mean(list(nh.saturation_rates().values()))
    print(f"CIM (ADC {args.adc_bits}b, no smooth): {ppl_cim_base:.2f}  "
          f"[sat={sat_base*100:.2f}%]")

    # ── Standard SmoothQuant (alpha fixed, no CIM constraints) ──────────
    model_std = copy.deepcopy(model)
    sq_std = CIMSmoothQuant(
        weight_bits=args.weight_bits, input_bits=args.input_bits,
        off_state=args.off_state, on_state=args.on_state,
        vdd=args.vdd, parallel_read=args.parallel_read,
        verbose=False,
    )
    # Force alpha_range to a single value (standard SmoothQuant)
    sq_std.alpha_range = [args.alpha]
    sq_std.fit(model_std, calib_loader,
               num_batches=args.num_calib_batches,
               device=args.device, task='lm')

    nh_std = CIMNoiseHook(model_std, adc_bits=args.adc_bits,
                           weight_bits=args.weight_bits, input_bits=args.input_bits,
                           parallel_read=args.parallel_read, vdd=args.vdd)
    nh_std.calibrate(calib_loader, device=args.device, num_batches=8,
                     clip_percentile=args.adc_clip_pct)
    nh_std.install()
    ppl_smooth_std = compute_perplexity(model_std, eval_loader, device=args.device,
                                        max_batches=args.num_eval_batches)
    sat_std = np.mean(list(nh_std.saturation_rates().values()))
    nh_std.remove()
    print(f"Standard SmoothQuant (alpha={args.alpha:.2f}): {ppl_smooth_std:.2f}  "
          f"[sat={sat_std*100:.2f}%]")

    # ── CIM-Aware SmoothQuant (per-layer optimal alpha + constraints) ───
    model_cim = copy.deepcopy(model)
    sq_cim = CIMSmoothQuant(
        weight_bits=args.weight_bits, input_bits=args.input_bits,
        bitcell=args.bitcell,
        off_state=args.off_state, on_state=args.on_state,
        vdd=args.vdd, parallel_read=args.parallel_read,
        adc_bits=args.adc_bits, adc_clip_pct=args.adc_clip_pct,
        sat_lambda=0.5,
        verbose=True,
    )
    sq_cim.fit(model_cim, calib_loader,
               num_batches=args.num_calib_batches,
               device=args.device, task='lm')
    sq_cim.report()

    nh_cim = CIMNoiseHook(model_cim, adc_bits=args.adc_bits,
                           weight_bits=args.weight_bits, input_bits=args.input_bits,
                           parallel_read=args.parallel_read, vdd=args.vdd)
    nh_cim.calibrate(calib_loader, device=args.device, num_batches=8,
                     clip_percentile=args.adc_clip_pct)
    nh_cim.install()
    ppl_smooth_cim = compute_perplexity(model_cim, eval_loader, device=args.device,
                                        max_batches=args.num_eval_batches)
    sat_cim = np.mean(list(nh_cim.saturation_rates().values()))
    nh_cim.remove()
    print(f"CIM-Aware SmoothQuant:             {ppl_smooth_cim:.2f}  "
          f"[sat={sat_cim*100:.2f}%]")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n[Perplexity Summary]")
    print(f"  fp32 baseline        : {ppl_clean:.2f}")
    print(f"  CIM no smooth        : {ppl_cim_base:.2f}  (+{ppl_cim_base-ppl_clean:.2f})")
    print(f"  Std SmoothQuant      : {ppl_smooth_std:.2f}  (+{ppl_smooth_std-ppl_clean:.2f})")
    print(f"  CIM-Aware SmoothQuant: {ppl_smooth_cim:.2f}  (+{ppl_smooth_cim-ppl_clean:.2f})")

    result = {
        'model': args.model, 'adc_bits': args.adc_bits,
        'adc_clip_pct': args.adc_clip_pct,
        'ppl_clean': ppl_clean, 'ppl_cim_base': ppl_cim_base,
        'ppl_smooth_std': ppl_smooth_std, 'ppl_smooth_cim': ppl_smooth_cim,
        'sat_base': float(sat_base), 'sat_std': float(sat_std),
        'sat_cim': float(sat_cim),
        'alpha_fixed': args.alpha,
        'alpha_per_layer': sq_cim._alphas,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.replace('/', '_')
    sq_cim.save_scales(str(out_dir / f'smooth_scales_{model_tag}.pt'))
    with open(str(out_dir / f'smooth_{model_tag}_adc{args.adc_bits}.json'), 'w') as f:
        # alpha_per_layer values are floats, safe to serialise
        result['alpha_per_layer'] = {k: float(v)
                                     for k, v in result['alpha_per_layer'].items()}
        json.dump(result, f, indent=2)
    return result


def task_sweep_adc(args, model, tokenizer, calib_loader, eval_loader):
    """Experiment 4: Perplexity vs. ADC bits (Pareto curve for paper)."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: ADC Bits Sweep (Pareto Curve)")
    print("="*60)

    import copy

    adc_range = range(args.adc_min, args.adc_max)
    rows = []

    # Prepare smoothed model once
    model_sm = copy.deepcopy(model)
    sq = CIMSmoothQuant(
        weight_bits=args.weight_bits, input_bits=args.input_bits,
        bitcell=args.bitcell, off_state=args.off_state, on_state=args.on_state,
        vdd=args.vdd, parallel_read=args.parallel_read,
        adc_bits=args.adc_bits, adc_clip_pct=args.adc_clip_pct,
        sat_lambda=0.5,
        verbose=False)
    sq.fit(model_sm, calib_loader, num_batches=args.num_calib_batches,
           device=args.device, task='lm')

    for adc_b in adc_range:
        # Without smoothing
        nh = CIMNoiseHook(model, adc_bits=adc_b,
                          weight_bits=args.weight_bits, input_bits=args.input_bits,
                          parallel_read=args.parallel_read, vdd=args.vdd)
        nh.calibrate(calib_loader, device=args.device, num_batches=8,
                     clip_percentile=args.adc_clip_pct)
        nh.install()
        ppl_base = compute_perplexity(model, eval_loader, device=args.device,
                                      max_batches=min(args.num_eval_batches, 30))
        sat_b = np.mean(list(nh.saturation_rates().values()))
        nh.remove()

        # With CIM-Aware smoothing
        nh_sm = CIMNoiseHook(model_sm, adc_bits=adc_b,
                              weight_bits=args.weight_bits, input_bits=args.input_bits,
                              parallel_read=args.parallel_read, vdd=args.vdd)
        nh_sm.calibrate(calib_loader, device=args.device, num_batches=8,
                        clip_percentile=args.adc_clip_pct)
        nh_sm.install()
        ppl_smooth = compute_perplexity(model_sm, eval_loader, device=args.device,
                                        max_batches=min(args.num_eval_batches, 30))
        sat_sm = np.mean(list(nh_sm.saturation_rates().values()))
        nh_sm.remove()

        print(f"  ADC {adc_b:2d}b | baseline: {ppl_base:10.2f} [sat={sat_b*100:.1f}%] "
              f"| CIM-SQ: {ppl_smooth:10.2f} [sat={sat_sm*100:.1f}%]")
        rows.append({'adc_bits': adc_b,
                     'ppl_baseline': ppl_base, 'sat_baseline': sat_b,
                     'ppl_cim_sq': ppl_smooth, 'sat_cim_sq': sat_sm})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.replace('/', '_')
    with open(str(out_dir / f'sweep_adc_{model_tag}.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Sweep] Results saved to {out_dir / f'sweep_adc_{model_tag}.csv'}")
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"\n[Config] model={args.model}, task={args.task}, "
          f"adc={args.adc_bits}b, device={args.device}")

    # Load model and tokeniser
    model, tokenizer = load_model(args.model, args.model_cache, args.device,
                                  hf_endpoint=args.hf_endpoint)

    # Load dataset
    data = load_wikitext2(tokenizer, seq_len=args.seq_len)
    calib_data = data[:args.num_calib_batches * args.batch_size + 64]
    eval_data  = data[64:]   # held-out eval set
    calib_loader = make_loader(calib_data, batch_size=args.batch_size)
    eval_loader  = make_loader(eval_data,  batch_size=args.batch_size)

    results = {}

    if args.task in ('characterize', 'all'):
        results['characterize'] = task_characterize(args, model, tokenizer, calib_loader)

    if args.task in ('baseline', 'all'):
        results['baseline'] = task_baseline(args, model, tokenizer, calib_loader, eval_loader)

    if args.task in ('smooth', 'all'):
        results['smooth'] = task_smooth(args, model, tokenizer, calib_loader, eval_loader)

    if args.task in ('sweep_adc', 'all'):
        results['sweep_adc'] = task_sweep_adc(args, model, tokenizer, calib_loader, eval_loader)

    if args.task in ('ppa', 'all'):
        ppa = run_neurosim_ppa(
            args.model, args.weight_bits, args.input_bits,
            args.sub_array, args.parallel_read,
            args.output_dir, args.adc_bits)
        if ppa:
            print(f"\n[NeuroSIM PPA]")
            for k, v in ppa.items():
                print(f"  {k}: {v}")
        results['ppa'] = ppa

    print("\n[Done] All experiments completed.")
    return results


if __name__ == '__main__':
    main()
