# ICCAD 2026 — CIM ADC Mixed-Precision Allocation

> **Paper**: *Measured Layer Sensitivity Guides Mixed-Precision ADC Allocation for CIM-Based LLM Inference*
> **Target**: ICCAD 2026
> **Authors**: Fantao Gao, Cancheng Xiao, Jiahao Zhao, Jianshi Tang, Tianxiang Nan*
> **Affiliation**: School of Integrated Circuits, Tsinghua University
> **Status**: Draft complete (6 pages, 0 Overfull, no errors)

---

## What This Repo Contains

```
iccad2026-cim-adc/
├── paper/                         # LaTeX paper (IEEEtran, 6 pages)
│   ├── main.tex                   # Full paper source
│   ├── refs.bib                   # Bibliography
│   ├── main.pdf                   # Compiled PDF
│   └── fig{1-9}_*.pdf            # All 9 figures
├── NeuroSim/                      # Experiment code & results
│   ├── sensitivity_analysis.py    # Core: PerLayerCIMHook, ILP allocation
│   ├── stable_eval.py            # 100-batch stable PPL eval
│   ├── smooth_quant.py           # CIMSmoothQuant implementation
│   ├── llm_inference.py          # load_model, CIMNoiseHook
│   ├── hessian_sensitivity.py    # HAWQ Hessian trace
│   ├── compute_energy_savings.py # Energy analysis
│   ├── plot_paper_figures_v2.py  # Generate all 9 paper figures
│   ├── update_fig7.py            # Fig7 update (with HAWQ bar)
│   ├── run_zeroshot*.py          # Zero-shot evaluation scripts
│   ├── export_figures_tables.py  # Export all data to Excel
│   └── results/                  # All experiment results
│       ├── ppa/opt125m/ppa_sweep_opt125m.csv
│       ├── sensitivity/opt125m/group_sensitivity.json
│       ├── sensitivity/opt125m/hawq_comparison.json
│       ├── sensitivity/opt1.3b/group_sensitivity.json
│       ├── hessian/opt125m/hessian_group.json
│       ├── stable/opt125m/stable_eval_results.csv
│       ├── stable/opt125m/ilp_vs_greedy_multibudget.csv
│       ├── stable/opt125m/stability_results.json
│       ├── stable/opt125m/energy_analysis.csv
│       ├── zeroshot/opt125m/zeroshot_summary.json
│       ├── figures_iccad2026_v2/  # Final figures (PDF + PNG)
│       └── figures_tables_data.xlsx  # All figure/table data (11 sheets)
├── chat_history/                  # Full Claude AI conversation logs
│   ├── d0d8408d-*.jsonl          # Main conversation (26MB, all core work)
│   ├── *.jsonl                   # Other session files
│   └── memory/MEMORY.md          # AI persistent project memory
├── HANDOVER.md                    # Complete project handover document
├── PROMPT_FOR_NEXT_AI.md         # Ready-to-use prompt for next AI assistant
├── paper_data_complete.md        # All figure & table data in Markdown
└── README.md                     # This file
```

> **Model cache** (~7.2 GB) is **not included** in this repo. See download instructions below.

---

## Core Research Finding

**The saturation-sensitivity inversion** — prior CIM ADC allocation methods assume high-saturation layers are most sensitive. We prove this is wrong:

| Layer type | Saturation rate | ΔPPL/layer (7b→6b) | Sensitivity rank |
|-----------|----------------|-------------------|-----------------|
| `q/k/v_proj` | **≈100%** | **0.128** | 5 — LEAST sensitive |
| `fc1` | ≈94% | 0.201 | 4 |
| `lm_head` | ≈100% | 0.700 | 3 |
| `out_proj` | ≈4% | 0.866 | 2 |
| `fc2` | **≈19%** | **1.413** | 1 — MOST sensitive |

**11× inversion**: `fc2` (low saturation) is 11× more sensitive than `q/k/v_proj` (high saturation). Confirmed on OPT-1.3B.

**ILP result**: 20.5% ADC area savings, +2.2 PPL — beats HAWQ-guided by **12.7 PPL** at equal area budget.

---

## Environment Setup

### 1. Python Dependencies

```bash
pip install torch transformers datasets lm_eval scipy openpyxl
```

Tested with:
- Python 3.10.12
- PyTorch 2.10.0+cpu (CPU-only, no GPU needed)
- transformers 5.2.0
- datasets 4.6.0

### 2. Download Model Weights

The model cache is **not in this repo** (too large). Download before running any experiments:

```bash
cd NeuroSim

# OPT-125M (~500 MB)
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('facebook/opt-125m', cache_dir='./model_cache')
AutoTokenizer.from_pretrained('facebook/opt-125m', cache_dir='./model_cache')
print('OPT-125M downloaded.')
"

# OPT-1.3B (~5 GB) — only needed for cross-model validation experiments
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b', cache_dir='./model_cache')
AutoTokenizer.from_pretrained('facebook/opt-1.3b', cache_dir='./model_cache')
print('OPT-1.3B downloaded.')
"
```

After downloading, your directory should look like:
```
NeuroSim/model_cache/
├── models--facebook--opt-125m/   (~500 MB)
└── models--facebook--opt-1.3b/   (~5 GB)
```

### 3. Environment Variables

```bash
# Required for offline lm-eval evaluation (use this, NOT HF_HUB_OFFLINE)
export HF_DATASETS_OFFLINE=1
```

---

## Compile the Paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
# Output: main.pdf (6 pages)
```

---

## Reproduce Key Results

All results are already in `NeuroSim/results/`. To reproduce from scratch:

```bash
cd NeuroSim

# 1. Measure per-layer sensitivity (OPT-125M, 7b→6b)
python3 sensitivity_analysis.py

# 2. Run stable 100-batch eval (all 8 configurations)
python3 stable_eval.py

# 3. Hessian trace (HAWQ baseline)
python3 hessian_sensitivity.py

# 4. Zero-shot accuracy eval
export HF_DATASETS_OFFLINE=1
python3 run_zeroshot.py          # FP32 + CIM-7b
python3 run_zeroshot_sq_ilp.py   # SQ+6b + ILP-20%

# 5. Regenerate all 9 paper figures
python3 plot_paper_figures_v2.py
python3 update_fig7.py

# 6. Export all data to Excel
python3 export_figures_tables.py
```

---

## Key Results Summary

| Config | PPL | ADC Area | Savings |
|--------|-----|----------|---------|
| Uniform 7b (baseline) | 306.4 | 228.4 mm² | 0% |
| **ILP 20%** | **308.6** | **181.5 mm²** | **20.5%** |
| HAWQ-guided greedy | 321.3 | 181.5 mm² | 20.5% |
| Sat-guided greedy | 313.8 | 181.5 mm² | 20.5% |
| SQ + Uniform 6b | **305.0** | 114.2 mm² | **50%** |

- ILP beats HAWQ by **12.7 PPL** at equal area budget
- SQ+6b achieves PPL below 7b baseline while saving 50% ADC area
- Energy: ILP −23.5%, SQ+6b −57.3%

---

## Chat History with Claude AI

All conversations with Claude (the AI that developed this project) are in `chat_history/`.

**Reading order** (by timestamp):

| File | Date | Size | Content |
|------|------|------|---------|
| `38bd39fc-*.jsonl` | Feb 27 | 1.8 KB | File snapshots only |
| `085523f9-*.jsonl` | Feb 28 | 24 KB | Short session |
| `a81bb881-*.jsonl` | Mar 2 | 1.9 KB | Short session |
| `770af3a5-*.jsonl` | Mar 3 | 1.9 KB | Short session |
| **`d0d8408d-*.jsonl`** | **Mar 3** | **26 MB** | **Main conversation — all core work** |
| `memory/MEMORY.md` | — | — | AI cross-session persistent memory |

The `d0d8408d-*.jsonl` file contains **the complete development history**: cloning NeuroSim, writing all experiment code, running experiments, writing the paper, generating figures, and the final handover.

**How to read** (each line is one JSON object, read top-to-bottom):

```bash
# Extract all user messages to see conversation flow
python3 -c "
import json, re
with open('chat_history/d0d8408d-c634-4140-8e0b-d443b26233fd.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('type') == 'user':
            msg = obj.get('message', {}).get('content', '')
            if isinstance(msg, list):
                msg = msg[0].get('text', '') if msg else ''
            msg = re.sub(r'<[^>]+>.*?</[^>]+>', '', str(msg), flags=re.DOTALL).strip()
            if msg:
                print(msg[:120])
"
```

---

## For the Next AI Assistant

If you are an AI assistant taking over this project, read these files in order:

1. **`PROMPT_FOR_NEXT_AI.md`** — paste this as your first message, contains everything needed
2. **`HANDOVER.md`** — detailed technical handover with all data tables
3. **`paper/main.tex`** — the actual paper
4. **`paper_data_complete.md`** — all figure/table data for redrawing figures

The paper is in final draft state. Typical remaining tasks:
- English proofreading and polish
- Figure redrawing in user's preferred style
- Submission format check for ICCAD 2026
- Reference completeness check

---

## Important Notes

1. **Two saturation rate definitions** (do not confuse):
   - `sat_rate_worst` (max-clip, 100th percentile): ≈100%/94%/19%/4% — **used in paper Fig 2 & Table I**
   - `sat_rate` (p99-clip): ≈3.5% for all layers — internal experiment metric only

2. **ILP optimal assignment** (same across all calibration seeds):
   - `q/k/v_proj`, `fc1` → 6b (least sensitive)
   - `fc2`, `out_proj`, `lm_head` → 7b (most sensitive)

3. **NeuroSIM settings** (must set in Param.cpp):
   ```
   pipeline=false, speedUpDegree=1, synchronous=false, novelMapping=false
   ```

---

*Project initiated: Feb 27, 2026 | Last updated: Mar 3, 2026*
*Developed with Claude Sonnet 4.6 (Anthropic)*
