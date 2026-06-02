#!/usr/bin/env bash
# Overview:
# Run the paper reproduction workflow from the repository root. By default this
# evaluates downloaded paper checkpoints; set MODEL_ACTION=train to train the
# full FNO/scOT sweep before evaluation and figure generation.
set -euo pipefail
shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-const_back}"
SEED="${SEED:-123}"
MODEL_ACTION="${MODEL_ACTION:-evaluate}"
WANDB_PROJECT="${WANDB_PROJECT-hybrid-architectures}"

if [[ -z "${MODEL_VERSION:-}" ]]; then
  if [[ "$MODEL_ACTION" == "train" ]]; then
    MODEL_VERSION="seed${SEED}"
  else
    MODEL_VERSION="paper"
  fi
fi

RAW_ROOT="data/raw"
DATA_ROOT="data/processed"
CHECKPOINT_ROOT="outputs/checkpoints/${DATASET}/${MODEL_VERSION}"
RESULTS_ROOT="results/${DATASET}/${MODEL_VERSION}"
FIGURES_ROOT="outputs/figures/${DATASET}/${MODEL_VERSION}"
LOG_ROOT="outputs/logs/${DATASET}/${MODEL_VERSION}"
EVALUATION_ROOT="${RESULTS_ROOT}/evaluation"

FNO_CONFIG="${FNO_CONFIG:-configs/fno_paper.yaml}"
SCOT_CONFIG="${SCOT_CONFIG:-configs/scot_paper.yaml}"
SIZES=(2 4 6 8 10)
TASKS=(smooth2smooth contrast sharp2sharp)
RAW_FILES=(velocity_sharp.npy velocity_smooth.npy pressure_sharp.npy pressure_smooth.npy)
PROCESSED_FILES=(velocity_sharp velocity_smooth velocity_delta pressure_sharp pressure_smooth pressure_delta)

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

fno_name() {
  echo "fno_${DATASET}_${1}_layers${2}"
}

scot_name() {
  echo "scot_${DATASET}_${1}_depths${2}-${2}-${2}-${2}"
}

fno_checkpoint() {
  echo "${CHECKPOINT_ROOT}/fno/$(fno_name "$1" "$2")"
}

scot_checkpoint() {
  echo "${CHECKPOINT_ROOT}/scot/$(scot_name "$1" "$2")"
}

processed_data_ready() {
  local split name
  for split in train validation test; do
    for name in "${PROCESSED_FILES[@]}"; do
      [[ -f "${DATA_ROOT}/${DATASET}/${split}/${name}.npy" ]] || return 1
    done
  done
}

prepare_data() {
  if processed_data_ready; then
    echo "Processed data already present under ${DATA_ROOT}/${DATASET}"
    return
  fi

  local missing=()
  local name
  for name in "${RAW_FILES[@]}"; do
    [[ -f "${RAW_ROOT}/${DATASET}/${name}" ]] || missing+=("${RAW_ROOT}/${DATASET}/${name}")
  done
  if (( ${#missing[@]} > 0 )); then
    printf 'Missing raw dataset files:\n' >&2
    printf '  - %s\n' "${missing[@]}" >&2
    die "Download the Hugging Face dataset into data/raw, or make data/raw a symlink containing the required arrays."
  fi

  "$PYTHON" scripts/prepare_data.py \
    --raw-root "$RAW_ROOT" \
    --output-root "$DATA_ROOT" \
    --dataset "$DATASET"
}

run_logged() {
  local log_file="$1"
  local gpu="$2"
  shift 2
  mkdir -p "$(dirname "$log_file")"
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  } >> "$log_file"

  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$@" 2>&1 | tee -a "$log_file"
  else
    "$@" 2>&1 | tee -a "$log_file"
  fi
}

train_task() {
  local task="$1"
  local gpu="${2:-}"
  local log_file="${LOG_ROOT}/train_${DATASET}_${task}.log"
  local size
  local -a wandb=()
  if [[ -n "$WANDB_PROJECT" ]]; then
    wandb=(--wandb-project "$WANDB_PROJECT")
  fi

  for size in "${SIZES[@]}"; do
    run_logged "$log_file" "$gpu" \
      "$PYTHON" scripts/train_fno.py \
      --config "$FNO_CONFIG" \
      --data-root "$DATA_ROOT" \
      --dataset "$DATASET" \
      --task "$task" \
      --output-root "${CHECKPOINT_ROOT}/fno" \
      --num-layers "$size" \
      --run-name "$(fno_name "$task" "$size")" \
      --seed "$SEED" \
      "${wandb[@]}"

    run_logged "$log_file" "$gpu" \
      "$PYTHON" scripts/train_scot.py \
      --config "$SCOT_CONFIG" \
      --data-root "$DATA_ROOT" \
      --dataset "$DATASET" \
      --task "$task" \
      --output-root "${CHECKPOINT_ROOT}/scot" \
      --depths "$size" "$size" "$size" "$size" \
      --run-name "$(scot_name "$task" "$size")" \
      --seed "$SEED" \
      "${wandb[@]}"
  done
}

train_models() {
  case "$MODEL_ACTION" in
    evaluate)
      echo "Skipping training because MODEL_ACTION=evaluate"
      return
      ;;
    train)
      ;;
    *)
      die "MODEL_ACTION must be either evaluate or train"
      ;;
  esac

  mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local -a gpus
    IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
  else
    local -a gpus=()
  fi

  if (( ${#gpus[@]} >= 3 )); then
    train_task smooth2smooth "${gpus[0]}" &
    local p1=$!
    train_task contrast "${gpus[1]}" &
    local p2=$!
    train_task sharp2sharp "${gpus[2]}" &
    local p3=$!
    local status=0
    wait "$p1" || status=$?
    wait "$p2" || status=$?
    wait "$p3" || status=$?
    (( status == 0 )) || exit "$status"
  else
    echo "Training sequentially. Set CUDA_VISIBLE_DEVICES=0,1,2 to train the three tasks in parallel."
    local task
    for task in "${TASKS[@]}"; do
      train_task "$task" ""
    done
  fi
}

evaluate_models() {
  mkdir -p "$EVALUATION_ROOT"
  local -a device_args=()
  if [[ -n "${EVALUATION_DEVICES:-}" ]]; then
    device_args+=(--devices "$EVALUATION_DEVICES")
  fi
  if [[ -n "${EVALUATION_MAX_PARALLEL:-}" ]]; then
    device_args+=(--max-parallel "$EVALUATION_MAX_PARALLEL")
  fi

  "$PYTHON" scripts/evaluate.py \
    --sweep \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-dir "$EVALUATION_ROOT" \
    --data-root "$DATA_ROOT" \
    --dataset "$DATASET" \
    --workers "${EVALUATION_WORKERS:-4}" \
    --sizes "${SIZES[@]}" \
    --tasks "${TASKS[@]}" \
    "${device_args[@]}"
}

plot_figures() {
  mkdir -p "$RESULTS_ROOT" "$FIGURES_ROOT"
  local metrics_json=("${EVALUATION_ROOT}"/*.json)
  (( ${#metrics_json[@]} > 0 )) || die "No evaluation metrics found under ${EVALUATION_ROOT}"

  "$PYTHON" scripts/plot_data_examples.py \
    --data-root "$DATA_ROOT" \
    --dataset "$DATASET" \
    --output-dir "$FIGURES_ROOT"

  "$PYTHON" scripts/collect_parameter_scaling.py \
    "${metrics_json[@]}" \
    --output "${RESULTS_ROOT}/parameter_scaling.csv"

  "$PYTHON" scripts/plot_parameter_scaling.py \
    --metrics-csv "${RESULTS_ROOT}/parameter_scaling.csv" \
    --output "${FIGURES_ROOT}/parameter_scaling.png"

  "$PYTHON" scripts/plot_result_comparison.py \
    --data-root "$DATA_ROOT" \
    --dataset "$DATASET" \
    --index 0 \
    --fno-sharp-checkpoint "$(fno_checkpoint sharp2sharp 10)" \
    --scot-sharp-checkpoint "$(scot_checkpoint sharp2sharp 10)" \
    --fno-smooth-checkpoint "$(fno_checkpoint smooth2smooth 10)" \
    --scot-contrast-checkpoint "$(scot_checkpoint contrast 10)" \
    --output "${FIGURES_ROOT}/result_comparison.png"

  echo "Wrote figures to ${FIGURES_ROOT}"
}

prepare_data
train_models
evaluate_models
plot_figures
