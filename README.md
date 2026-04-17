# ICCAD 2026: Mixed-Precision ADC Allocation for CIM-Based LLM Inference

> **Paper**: *Measured Layer Sensitivity Guides Mixed-Precision ADC Allocation for CIM-Based LLM Inference*
>
> **Venue**: ICCAD 2026 (IEEE/ACM International Conference on Computer-Aided Design)
>
> **Status**: Submitted (April 2026)

## Overview

This repository contains the complete codebase, experimental data, and LaTeX source for our ICCAD 2026 paper. We discover a **saturation–sensitivity inversion** in CIM-based LLM inference: the highest-saturation attention layers (W_qkv, ~100% saturation) are the *least* accuracy-sensitive, while low-saturation FFN layers (W_fc2, ~19%) are the *most* sensitive — an 11× inversion. Based on this finding, we propose a profiling-guided ILP allocation flow that achieves 20% ADC-area savings at <1% relative PPL degradation.

## Repository Structure

```
.
├── paper/                          # LaTeX paper source
│   ├── main.tex                    # Main LaTeX file (8 pages, ACM sigconf format)
│   ├── refs.bib                    # Bibliography (22 references)
│   ├── fig_*.pdf                   # All 15 paper figures
│   ├── acmart.cls                  # ACM template class
│   └── ACM-Reference-Format.bst    # Bibliography style
│
├── NeuroSim/                       # Experiment code
│   ├── llm_inference.py            # Core: CIMNoiseHook, model loading, ADC quantization
│   ├── sensitivity_analysis.py     # Core: PerLayerCIMHook, ILP allocation, group profiling
│   ├── stable_eval.py              # 100-batch stable evaluation on D_test
│   ├── smooth_quant.py             # SmoothQuant integration
│   ├── hessian_sensitivity.py      # Hessian trace baseline (diagonal Fisher approx.)
│   ├── mixed_precision_adc.py      # Mixed-precision allocation experiments
│   ├── neurosim_ppa.py             # NeuroSIM PPA wrapper
│   ├── compute_energy_savings.py   # Energy/latency analysis
│   ├── zeroshot_eval.py            # Zero-shot downstream task evaluation
│   ├── regen_final.py              # Final figure generation script
│   ├── plot_paper_figures_final.py # Paper figure plotting
│   ├── environment.yml             # Conda environment specification
│   ├── NeuroSIM/                   # NeuroSIM C++ circuit simulator (Princeton)
│   │   ├── main.cpp                # Entry point
│   │   ├── Chip.cpp/h              # Chip-level modeling
│   │   ├── SubArray.cpp/h          # 128×128 subarray modeling
│   │   ├── MultilevelSAEncoder.*   # MLSA ADC modeling
│   │   └── makefile                # Build: `make -C NeuroSIM`
│   └── results/                    # All experimental results
│       ├── sensitivity/            # Per-group sensitivity measurements
│       ├── stable/                 # 100-batch D_test evaluation results
│       ├── ppa/                    # NeuroSIM area/energy/latency data
│       ├── hessian/                # Hessian trace baselines
│       ├── mixed_precision/        # Allocation comparison results
│       ├── zeroshot/               # Downstream task accuracy
│       └── figures_iccad2026_v2/   # Generated figure PDFs
│
├── experiments/                    # Experiment runner scripts
├── HANDOVER.md                     # Detailed project handover document
├── paper_chinese.md                # Full Chinese translation of the paper
├── paper_explanation.md            # Beginner-friendly paper explanation (Chinese)
└── paper_data_complete.md          # Complete data reference for all figures/tables
```

## Key Results

| Configuration | PPL | Rel. Δ | ADC Area Saving |
|---|---|---|---|
| Uniform 7b (baseline) | 306.4 | — | 0% |
| **Profiling-ILP (ours)** | **308.6** | **+0.7%** | **20.5%** |
| Saturation-ILP | 309.2 | +0.9% | 20.5% |
| Hessian-ILP | 309.7 | +1.1% | 20.5% |
| Random (20 trials) | 314.5±4.4 | +2.6%±1.4% | 20.5% |

## Setup

### Prerequisites

- Python 3.10+
- LaTeX distribution (for paper compilation)
- C++ compiler (for NeuroSIM)

### Install Dependencies

```bash
pip install torch transformers datasets scipy matplotlib openpyxl
# For zero-shot evaluation:
pip install lm_eval
```

### Download Model Weights

Models are not included in the repo (~500MB for OPT-125M, ~5GB for OPT-1.3B):

```bash
cd NeuroSim
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('facebook/opt-125m', cache_dir='./model_cache')
AutoTokenizer.from_pretrained('facebook/opt-125m', cache_dir='./model_cache')
"
```

### Build NeuroSIM

```bash
make -C NeuroSim/NeuroSIM
```

### Compile Paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Hardware Configuration

- **CIM Array**: 128×128 RRAM crossbar
- **ADC Type**: MLSA (Multi-Level Sense Amplifier)
- **Process**: 45nm CMOS (NeuroSIM)
- **Weight Mapping**: INT8 with bit-slicing
- **DAC Input**: 1-bit
- **ADC Calibration**: p99-clip

## Data Splits

| Split | Batches | Tokens | Purpose |
|---|---|---|---|
| D_cal | 4 | ~2K | ADC full-scale calibration |
| D_prof | 10 | ~5K | Sensitivity profiling (Algorithm 1) |
| D_test | 100 | ~51K | Final PPL evaluation (Table 3) |

## Citation

```bibtex
@inproceedings{gao2026measured,
  title={Measured Layer Sensitivity Guides Mixed-Precision ADC Allocation
         for CIM-Based LLM Inference},
  author={Gao, Fantao and Xiao, Cancheng and Zhao, Jiahao
          and Tang, Jianshi and Nan, Tianxiang},
  booktitle={Proc. IEEE/ACM International Conference on
             Computer-Aided Design (ICCAD)},
  year={2026}
}
```
