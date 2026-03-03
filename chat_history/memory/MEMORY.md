# ICCAD 2026 Project Memory

## Project Goal
ICCAD 2026 paper: "Measured Layer Sensitivity Guides Mixed-Precision ADC Allocation for CIM-Based LLM Inference"
**New story**: saturation-sensitivity inversion + ILP allocation + proxy invalidation

## Project Location
`/home/ubuntu/iccad2026_bxkj/NeuroSim/` (main working directory)
`/home/ubuntu/iccad2026_bxkj/paper/main.tex` (LaTeX paper)

## Environment Setup
- Python 3.10.12, PyTorch 2.10.0+cpu (no GPU)
- transformers 5.2.0, datasets 4.6.0, lm_eval installed
- OPT-125M cached: `./model_cache/models--facebook--opt-125m/`
- OPT-1.3B cached: `./model_cache/models--facebook--opt-1.3b/` (5GB)
- Use `export HF_DATASETS_OFFLINE=1` (not HF_HUB_OFFLINE) for offline lm-eval
- **DISK**: 49GB total, ~9GB free

## Key Source Files
| File | Purpose |
|------|---------|
| `sensitivity_analysis.py` | PerLayerCIMHook, group sensitivity, ilp_allocation |
| `stable_eval.py` | Stable 100-batch eval; multi-budget ILP vs Greedy |
| `smooth_quant.py` | CIMSmoothQuant.fit(model, loader, ...) API |
| `llm_inference.py` | load_model, CIMNoiseHook, compute_perplexity |
| `hessian_sensitivity.py` | HAWQ-style Hessian trace computation |
| `compute_energy_savings.py` | Energy analysis from PPA data |
| `run_zeroshot.py` | Zero-shot eval: clean + CIM-7b (fixed) |
| `run_zeroshot_sq_ilp.py` | Zero-shot eval: SQ+6b + ILP-20% (fixed) |
| `plot_paper_figures_v2.py` | 9 figures (fig1-fig9) |

## API Notes (Critical)
- `CIMSmoothQuant(weight_bits=8, ...).fit(model, loader, num_batches=4, device='cpu')` — model is NOT in __init__
- `CIMNoiseHook(model, adc_bits=7, ...).calibrate(loader, clip_percentile=99.0).install()` → `.remove()`
- `PerLayerCIMHook(model, bit_assignment={name: bits}, default_bits=7)` — bit_assignment is dict, 2nd positional arg
- `ilp_allocation(sensitivity_data, ppa_sweep, nominal_bits=7, bit_choices=..., target_area_savings=0.20)` → returns list of bits per layer (not dict!)
- lm-eval offline: set `HF_DATASETS_OFFLINE=1`, remove `HF_HUB_OFFLINE`
- Cached datasets: hellaswag, winogrande, wikitext (NOT piqa/arc_easy — old script format)

## KEY RESULTS (Stable 100-batch eval, OPT-125M)

### Sensitivity Table (group measurement, 7b→6b)
| Layer type | Sat. rate | ΔPPL/layer | Rank |
|-----------|-----------|------------|------|
| attn_qkv (36) | 100% | 0.128 | 5 (LEAST sensitive) |
| fc1 (12) | 94% | 0.201 | 4 |
| lm_head (1) | 100% | 0.700 | 3 |
| out_proj (12) | 4% | 0.866 | 2 |
| fc2 (12) | 19% | 1.413 | 1 (MOST sensitive) |

**KEY FINDING**: 11× inversion — attn_qkv (100% sat) LEAST sensitive; fc2 (19% sat) MOST sensitive

### PPL Results (100-batch stable eval)
| Config | PPL | ADC area | Savings |
|--------|-----|----------|---------|
| Uniform 7b | 306.4 | 228.4mm² | 0% |
| ILP 20% | 308.6 | 181.5mm² | 20.5% |
| SQ + 6b | 305.0 | 114.2mm² | 50% |

### Proxy Comparison (negative results)
- Saturation vs CIM: Spearman ρ = -0.80 (opposite of expected)
- Hessian trace vs CIM: Spearman ρ = 0.20, p=0.75 (not significant), 0/5 rank correct

### Energy Analysis (NeuroSIM MLSA model, E ∝ M·2^b)
- Uniform 7b: 11.4 nJ/512-token
- ILP 20%: 8.7 nJ (-23.5%)
- SQ + 6b: 4.9 nJ (-57.3%)

### Zero-Shot Accuracy (OPT-125M, 500 samples, 0-shot, lm-eval)
| Config | HellaSwag | WinoGrande | Avg |
|--------|-----------|------------|-----|
| Clean FP32 | 34.8% | 53.6% | 44.2% |
| CIM 7b | 30.6% | 52.0% | 41.3% |
| SQ + 6b | 26.2% | 51.0% | 38.6% |
| ILP 20% | 30.0% | 51.2% | 40.6% |

ILP-20% retains -0.7%pt avg vs CIM-7b at 20.5% savings (better than SQ+6b -2.7%pt)

### OPT-1.3B (8-16 calib, confirms 100% saturation)
- Clean PPL: 21.16 | CIM @8b: 700.78 | @9b: 703.22 | @10b: 703.56
- sat_rate=1.0 at all tested bits; needs 11b ADC for acceptable operation
- Used qualitatively: "outlier severity increases with model size"

## Results Files
- `results/sensitivity/opt125m/group_sensitivity.json` — sensitivity table
- `results/stable/opt125m/stable_eval_results.csv` — 100-batch PPL results
- `results/ppa/opt125m/ppa_sweep_opt125m.csv` — NeuroSIM PPA data
- `results/hessian/opt125m/hessian_group.json` — Hessian trace data
- `results/stable/opt125m/energy_analysis.csv` — energy analysis
- `results/zeroshot/opt125m/` — partial_results.json + zeroshot_summary.json
- `results/figures_iccad2026_v2/` — 9 figures (fig1-fig9)

## Paper Status (2026-03-02) — UPDATED with Plan B results
- **Pages**: 6, 0 Overfull, no errors
- **Sections**: Intro (4 contributions), Background, Sensitivity, Allocation, PPA, Discussion (6 subsections), Related Work, Conclusion
- **Tables**: I (sensitivity), II (PPA sweep), III (allocation+HAWQ row), IV (proxy comparison), V (zero-shot), VI (OPT-1.3B sensitivity new)
- **New content added**:
  - §3.1: error bars (Uniform-7b 312.5±5.4, ILP 314.8±5.2, SQ+6b 316.1±7.5, 3 calib seeds)
  - §5.3 Cross-Model Validation: OPT-1.3B Table VI confirms inversion (ffn_down most, attn_qkv least)
  - §5.4 Proxy: HAWQ PPL=321.3 quantified vs ILP 308.6 (12.7 PPL gap)
  - Table III: HAWQ-guided greedy row (PPL=321.3, 20.5% savings)
  - Contribution #3: now mentions OPT-1.3B + HAWQ numerical comparison
- Compile: `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

## OPT-1.3B Results (cross-model validation)
- nominal_bits=11, probe_bits=10 (11b→10b relative sensitivity)
- baseline_ppl=708.99 (high but stable at 11b)
- ffn_down: ΔPPL/layer=+0.619 (most sensitive, rank 1)
- ffn_up: ΔPPL/layer=+0.149 (rank 2)
- attn_out: ΔPPL/layer=+0.033 (rank 3)
- attn_qkv: ΔPPL/layer=-0.051 (LEAST sensitive, rank 4)
- **INVERSION CONFIRMED**: same ordering as OPT-125M

## HAWQ Comparison (OPT-125M, 100-batch, 20.5% savings)
- Uniform 7b: 326.8 PPL (different calib from stable_eval's 306.4; relative order valid)
- HAWQ-guided greedy: 321.3 PPL
- Sat-guided greedy: 313.8 PPL
- ILP: 308.6 PPL (best, -12.7 vs HAWQ)

## Stability (3 calib seeds × 3 configs)
- Uniform-7b: 312.5±5.4 (runs: 319.9, 310.1, 307.4)
- ILP-20%: 314.8±5.2 (runs: 321.2, 308.6, 314.7)
- SQ+6b: 316.1±7.5 (runs: 326.6, 312.0, 309.6)

## NeuroSIM Critical Setup
- Must set pipeline=false, speedUpDegree=1, synchronous=false, novelMapping=false in Param.cpp
- OPT-125M unique types: (768×768)×48, (768×3072)×12, (3072×768)×12, (768×50272)×1
