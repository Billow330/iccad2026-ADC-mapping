#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh  —  One-click GPU experiment runner
# ICCAD 2026: "Outlier-Aware CIM Mapping for LLM Inference"
#
# Usage:
#   bash run_experiments.sh [OPTIONS]
#
# Options:
#   --model      gpt2 | gpt2-medium | facebook/opt-125m | facebook/opt-350m
#                (default: gpt2)
#   --skip_setup skip pip install / compile steps (if already done)
#   --task       characterize | baseline | smooth | sweep_adc | all
#                (default: all)
#   --device     cuda | cpu  (auto-detected if not set)
#   --adc_bits   int (default: 7)
#   --no_opt     skip OPT-125m run (default: also run OPT-125m after GPT-2)
#
# Example:
#   bash run_experiments.sh --model gpt2 --task all
#   bash run_experiments.sh --skip_setup --model facebook/opt-125m --task sweep_adc
# =============================================================================

set -euo pipefail

# ── colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
sep()   { echo -e "${BOLD}────────────────────────────────────────────────────────────${NC}"; }

# ── default parameters ───────────────────────────────────────────────────────
MODEL="gpt2"
SKIP_SETUP=0
TASK="all"
DEVICE="auto"
ADC_BITS=7
WEIGHT_BITS=8
INPUT_BITS=8
PARALLEL_READ=128
NUM_CALIB=32
NUM_EVAL=100
ADC_MIN=3
ADC_MAX=12
READ_NOISE=0.0
RUN_OPT=1
SMOKE_TEST=0
HF_ENDPOINT=""       # e.g. https://hf-mirror.com  (empty = use official HF)
LOCAL_MODEL_DIR=""   # e.g. /data/models/gpt2  (overrides --model download)

# ── parse CLI arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --task)         TASK="$2";         shift 2 ;;
        --device)       DEVICE="$2";       shift 2 ;;
        --adc_bits)     ADC_BITS="$2";     shift 2 ;;
        --weight_bits)  WEIGHT_BITS="$2";  shift 2 ;;
        --input_bits)   INPUT_BITS="$2";   shift 2 ;;
        --parallel_read) PARALLEL_READ="$2"; shift 2 ;;
        --num_calib)    NUM_CALIB="$2";    shift 2 ;;
        --num_eval)     NUM_EVAL="$2";     shift 2 ;;
        --adc_min)      ADC_MIN="$2";      shift 2 ;;
        --adc_max)      ADC_MAX="$2";      shift 2 ;;
        --read_noise)   READ_NOISE="$2";   shift 2 ;;
        --hf_endpoint)  HF_ENDPOINT="$2";  shift 2 ;;
        --local_model)  LOCAL_MODEL_DIR="$2"; shift 2 ;;
        --skip_setup)   SKIP_SETUP=1;      shift ;;
        --no_opt)       RUN_OPT=0;         shift ;;
        --smoke_test)   SMOKE_TEST=1; NUM_CALIB=2; NUM_EVAL=5; shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# ── resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
MODEL_CACHE="${SCRIPT_DIR}/model_cache"
NEUROSIM_BIN="${SCRIPT_DIR}/NeuroSIM/main"
PY="python3"

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${HOME}/.local/bin:${PATH}"

sep
echo -e "${BOLD}  ICCAD 2026 — LLM + CIM Outlier Experiment Suite${NC}"
sep
info "Script dir : ${SCRIPT_DIR}"
info "Results dir: ${RESULTS_DIR}"
info "Model      : ${MODEL}"
info "Task       : ${TASK}"
info "ADC bits   : ${ADC_BITS}"
echo

# =============================================================================
# STEP 1: Environment setup
# =============================================================================
if [[ ${SKIP_SETUP} -eq 0 ]]; then
    sep; info "STEP 1: Setting up Python environment"

    # ── detect CUDA version ──────────────────────────────────────────────────
    CUDA_VER=""
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[\d.]+')
        info "Detected CUDA ${CUDA_VER}"
    elif command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' | head -1)
        info "Detected CUDA ${CUDA_VER} (from nvidia-smi)"
    else
        warn "No CUDA detected. Installing CPU-only PyTorch."
    fi

    # ── select PyTorch wheel ─────────────────────────────────────────────────
    # Compare as a single integer: MAJOR*100 + MINOR to avoid &&-logic bugs
    TORCH_INDEX=""
    if [[ -n "${CUDA_VER}" ]]; then
        MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
        MINOR=$(echo "${CUDA_VER}" | cut -d. -f2)
        VER_INT=$(( MAJOR * 100 + MINOR ))
        if   [[ ${VER_INT} -ge 1204 ]]; then
            # CUDA 12.4+ → use cu124 (latest stable as of 2025)
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        elif [[ ${VER_INT} -ge 1201 ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        elif [[ ${VER_INT} -ge 1108 ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        else
            # CUDA too old or too new (e.g. 13.x) — cu124 is the safest bet
            warn "CUDA ${CUDA_VER}: no exact wheel match, falling back to cu124."
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        fi
        info "Using PyTorch index: ${TORCH_INDEX}"
    fi

    # ── install pip if missing ───────────────────────────────────────────────
    if ! ${PY} -m pip --version &>/dev/null; then
        info "Installing pip..."
        curl -sS https://bootstrap.pypa.io/get-pip.py | ${PY}
    fi

    # ── install/upgrade PyTorch ──────────────────────────────────────────────
    info "Installing PyTorch..."
    if [[ -n "${TORCH_INDEX}" ]]; then
        ${PY} -m pip install --quiet \
            torch torchvision \
            --index-url "${TORCH_INDEX}"
    else
        ${PY} -m pip install --quiet \
            torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu
    fi

    # ── install other dependencies ───────────────────────────────────────────
    info "Installing Python dependencies..."
    ${PY} -m pip install --quiet \
        transformers>=4.40.0 \
        datasets \
        accelerate \
        absl-py \
        numpy \
        matplotlib \
        scipy \
        pandas \
        tqdm

    ok "Python environment ready."

    # ── compile NeuroSIM C++ backend ─────────────────────────────────────────
    sep; info "STEP 2: Compiling NeuroSIM C++ backend"
    if [[ -f "${NEUROSIM_BIN}" ]]; then
        ok "NeuroSIM already compiled: ${NEUROSIM_BIN}"
    else
        NEUROSIM_SRC="${SCRIPT_DIR}/NeuroSIM"
        if [[ ! -d "${NEUROSIM_SRC}" ]]; then
            die "NeuroSIM source not found at ${NEUROSIM_SRC}"
        fi
        CPP_FILES=$(find "${NEUROSIM_SRC}" -name '*.cpp' | tr '\n' ' ')
        g++ -fopenmp -O3 -std=c++0x -Wall \
            -I"${NEUROSIM_SRC}" \
            ${CPP_FILES} \
            -o "${NEUROSIM_BIN}" 2>&1 | tail -5
        ok "NeuroSIM compiled → ${NEUROSIM_BIN}"
    fi
fi  # end SKIP_SETUP

# ── auto-detect device ────────────────────────────────────────────────────────
if [[ "${DEVICE}" == "auto" ]]; then
    HAS_CUDA=$(${PY} -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
    if [[ "${HAS_CUDA}" == "1" ]]; then
        DEVICE="cuda"
        GPU_NAME=$(${PY} -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
        ok "GPU detected: ${GPU_NAME}"
    else
        DEVICE="cpu"
        warn "No GPU available — running on CPU (slow for large models)"
    fi
fi

# ── create output directories ─────────────────────────────────────────────────
mkdir -p "${RESULTS_DIR}" "${MODEL_CACHE}"

# =============================================================================
# Helper: run one experiment
# =============================================================================
run_task() {
    local task="$1"
    local model="$2"
    local safe_model
    safe_model=$(echo "${model}" | tr '/' '_')

    # If a local directory override is given for this model, use it
    local model_arg="${model}"
    if [[ -n "${LOCAL_MODEL_DIR}" && -d "${LOCAL_MODEL_DIR}" ]]; then
        model_arg="${LOCAL_MODEL_DIR}"
        info "Using local model directory: ${LOCAL_MODEL_DIR}"
    fi

    sep
    info "Running task=${task}  model=${model}  device=${DEVICE}"

    # HF_ENDPOINT must be set as an env var BEFORE Python starts,
    # because huggingface_hub reads it at import time.
    if [[ -n "${HF_ENDPOINT}" ]]; then
        export HF_ENDPOINT="${HF_ENDPOINT}"
        info "HF_ENDPOINT=${HF_ENDPOINT}"
    fi

    ${PY} "${SCRIPT_DIR}/llm_inference.py" \
        --model             "${model_arg}" \
        --model_cache       "${MODEL_CACHE}" \
        --task              "${task}" \
        --device            "${DEVICE}" \
        --adc_bits          "${ADC_BITS}" \
        --weight_bits       "${WEIGHT_BITS}" \
        --input_bits        "${INPUT_BITS}" \
        --parallel_read     "${PARALLEL_READ}" \
        --num_calib_batches "${NUM_CALIB}" \
        --num_eval_batches  "${NUM_EVAL}" \
        --adc_min           "${ADC_MIN}" \
        --adc_max           "${ADC_MAX}" \
        --read_noise        "${READ_NOISE}" \
        --output_dir        "${RESULTS_DIR}/${safe_model}"

    ok "Finished: ${task} / ${model}"
}

# =============================================================================
# STEP 3: Run experiments
# =============================================================================
sep; info "STEP 3: Running experiments — model=${MODEL}, task=${TASK}"
SAFE_MODEL=$(echo "${MODEL}" | tr '/' '_')

START_TIME=$(date +%s)

if [[ "${TASK}" == "all" ]]; then
    run_task "characterize" "${MODEL}"
    run_task "baseline"     "${MODEL}"
    run_task "smooth"       "${MODEL}"
    run_task "sweep_adc"    "${MODEL}"
else
    run_task "${TASK}" "${MODEL}"
fi

# =============================================================================
# STEP 4 (optional): also run OPT-125m for comparison table
# =============================================================================
if [[ ${RUN_OPT} -eq 1 && "${MODEL}" == "gpt2" && "${TASK}" == "all" ]]; then
    sep; info "STEP 4: Running OPT-125m for cross-model comparison"
    run_task "characterize"  "facebook/opt-125m"
    run_task "baseline"      "facebook/opt-125m"
    run_task "smooth"        "facebook/opt-125m"
    run_task "sweep_adc"     "facebook/opt-125m"
fi

# =============================================================================
# STEP 5: Generate paper figures
# =============================================================================
sep; info "STEP 5: Generating paper figures"
${PY} "${SCRIPT_DIR}/plot_results.py" \
    --results_dir "${RESULTS_DIR}" \
    --output_dir  "${RESULTS_DIR}/figures" 2>/dev/null \
    && ok "Figures saved to ${RESULTS_DIR}/figures/" \
    || warn "plot_results.py not found or failed — skipping figure generation"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

sep
echo -e "${GREEN}${BOLD}  All experiments complete!${NC}"
echo -e "  Elapsed time : ${MINUTES}m ${SECONDS}s"
echo -e "  Results in   : ${RESULTS_DIR}/"
sep
