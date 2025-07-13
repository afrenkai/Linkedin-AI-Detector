#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/.venv"
BASH_LOG_DIR="$PROJECT_DIR/bash_logs"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'


log() {
  echo "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
  echo "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
  echo "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    PYTHON_CMD="python3"
    VENV_ACTIVATE="$VENV_DIR/bin/activate"
    TORCH_REQ="gpu_requirements.txt"
  elif [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win32"* ]]; then
    OS="windows"
    PYTHON_CMD="python"
    VENV_ACTIVATE="$VENV_DIR/Scripts/activate"
    TORCH_REQ="requirements_cuda.txt"
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    PYTHON_CMD="python3"
    VENV_ACTIVATE="$VENV_DIR/bin/activate"
    TORCH_REQ="requirements_mps.txt"
  else
    error "Unsupported OS: $OSTYPE"
    exit 1
  fi
  log "Detected OS: $OS"
}


setup_environment() {
  log "Setting up environment..."

  if ! command_exists uv; then
    error "uv is not installed. Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi

  if [ ! -d "$VENV_DIR" ]; then
    log "Creating venv..."
    cd "$PROJECT_DIR"
    uv venv
  fi

  log "Activating venv..."
  source "$VENV_ACTIVATE"

  log "Installing dependencies..."
  uv pip install -r "$TORCH_REQ"
  uv pip install -r requirements.txt
  log "Checking Torch Backend Status ..."
  TORCH_BACKEND_CHECK_OUTPUT=$(uv run --no-sync check_health.py 2>&1)
  echo "$TORCH_BACKEND_CHECK_OUTPUT"

  if echo "$TORCH_BACKEND_CHECK_OUTPUT=" | grep -iq '\bno\b'; then
    warn "No GPU detected. Training will use CPU (this will be slow)."
    DEVICE="cpu"
  elif echo "$TORCH_BACKEND_CHECK_OUTPUT=" | grep -iq '\bMPS detected, yes\b'; then
    log "Mac silicon detected, training will use MPS backend. Note that this is experimental. Stuff will break!!!"
    DEVICE="mps"
  elif echo "$TORCH_BACKEND_CHECK_OUTPUT=" | grep -iq '\bCuda detected, yes\b'; then
    log "Cuda Compute Capable GPU found, training will use Cuda Backend"
    DEVICE="cuda"
  fi
}


# corpus=${1:-"slop_sample_1.txt"}
# config=${2:-"model_arch/model_config.json"}
# tokenizer=${3:-""}
# output_dir=${4:-"slop_ckpts"}
# epochs=${5:-50}
# batch_size=${6:-16}
# lr=${7:-1e-4}
# warmup_steps=${8:-100}
# block_size=${9:-512}
# grad_accum=${10:-8}
# optim_weight_decay=${11:-0.1}
# optim_beta_1=${12:-0.9}
# optim_beta_2=${13:-0.95}
# eval_every=${15:-5}
# save_every=${16:-10}

run_training_test_init() {

    local corpus=${1:-"slop_sample_1.txt"}
    local config=${2:-"model_arch/model_config.json"}
    local tokenizer=${3:-""}
    local output_dir=${4:-"slop_ckpts"}
    local epochs=${5:-50}
    local batch_size=${6:-16}
    local lr=${7:-1e-4}
    local warmup_steps=${8:-100}
    local block_size=${9:-512}
    local grad_accum=${10:-8}
    local optim_weight_decay=${11:-0.1}
    local optim_beta_1=${12:-0.9}
    local optim_beta_2=${13:-0.95}
    local eval_every=${15:-5}
    local save_every=${16:-10}
  }

    log "Training with params:"
    log " Corpus: $corpus"
    log " Config: $config"
    log " Tokenizer (pickled ?): $tokenizer"
    log " Checkpoints are gonna go to: $output_dir"
    log " Epochs: $epochs"
    log " Batch Size: $batch_size"
    log " Learning Rate: $lr"
Running" Optimizer Warmup Steps: $warmup_steps}"
    log " Context Length: $block_size"
    log " Gradient Accumulation: $grad_accum"
    log " Optimizer Weight Decay: $optim_weight decay"
    log " First Gradient Moment Estimation: $optim_beta_1"
    log " Second Gradient Moment Estimation: $optim_beta_2"
    log " Frequency with Which to Evaluate the Model: $eval_every"
    log " How Often I'm Checkpointing: $save_every"
    
    mkdir -p "$BASH_LOG_DIR"
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$BASH_LOG_DIR/_${TIMESTAMP}.log"
    CMD="uv run --no-sync -m train.train
    CMD="$CMD --corpus $corpus"
    CMD="$CMD --config $config"
    CMD="$CMD --output_dir $output_dir"
    CMD="$CMD --epochs $epochs"
    CMD="$CMD --batch_size $batch_size"
    CMD="$CMD --lr $lr"
    CMD="$CMD --warmup_steps $warmup_steps"
    CMD="$CMD --block_size $block_size"
    CMD="$CMD --grad_accum $grad_accum"
    CMD="$CMD --optim_weight_decay $optim_weight_decay"
    CMD="$CMD --optim_beta_1 $optim_beta_1"
    CMD="$CMD --optim_beta_2 $optim_beta_2"
    CMD="$CMD --eval_every $eval_every"
    CMD="$CMD --save_every $save_every"
    
    if [ -n "$tokenizer" ]; then
        CMD="$CMD --tokenizer $tokenizer"
    fi

    log "Running: $CMD"

    log "Logging to: $LOG_FILE"
    cd "$PROJECT_DIR"
    source "$VENV_ACTIVATE"
    eval "$CMD" 2>&1 | tee "$LOG_FILE"
    ls
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
      log "Pipeline completed successfully!"
      log "Log file: $LOG_FILE"
    else
      error "Pipeline failed! Check log file: $LOG_FILE"
      exit 1
    fi
}

