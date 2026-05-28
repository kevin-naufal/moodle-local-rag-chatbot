#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

load_dotenv() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    if [[ "$line" == *"="* ]]; then
      local key="${line%%=*}"
      local value="${line#*=}"
      key="${key%"${key##*[![:space:]]}"}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"
      export "${key}=${value}"
    fi
  done < "$env_file"
}

abs_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "${PROJECT_ROOT}/${p}" | sed 's#//*#/#g'
  fi
}

to_lower() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

split_csv_models() {
  local raw="$1"
  local out=()
  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    part="${part#"${part%%[![:space:]]*}"}"
    part="${part%"${part##*[![:space:]]}"}"
    [[ -n "$part" ]] && out+=("$part")
  done
  echo "${out[@]}"
}

timestamp_now() {
  date +"%Y-%m-%d %H:%M:%S"
}

stop_model_with_logs() {
  local model="$1"
  local start_ts end_ts elapsed status
  start_ts="$(date +%s)"
  echo "[$(timestamp_now)] [stop] mulai: $model"
  if ollama stop "$model" >/dev/null 2>&1; then
    status="stopped"
  else
    status="skip/tidak_aktif"
  fi
  end_ts="$(date +%s)"
  elapsed="$((end_ts - start_ts))"
  echo "[$(timestamp_now)] [stop] selesai: $model | status=$status | durasi=${elapsed}s"
}

run_preparse_for_modes() {
  local modes_csv="$1"
  local data_dir_path="$2"
  local -a mode_items=()
  local -a targets=()
  local item
  IFS=',' read -r -a mode_items <<< "$modes_csv"
  for item in "${mode_items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    item="$(to_lower "$item")"
    [[ -z "$item" || "$item" == "llm_only" ]] && continue
    case "$item" in
      rag_ollama) targets+=("ollama|") ;;
      rag_bert) targets+=("bert|bert-base-uncased") ;;
      rag_msmarco) targets+=("bert|sentence-transformers/msmarco-bert-base-dot-v5") ;;
      *) echo "[WARN] Lewati preparse mode tidak dikenal: $item" ;;
    esac
  done

  local unique_targets
  unique_targets="$(printf '%s\n' "${targets[@]}" | awk 'NF && !seen[$0]++')"
  local target backend bert_model label
  while IFS= read -r target; do
    [[ -z "$target" ]] && continue
    backend="${target%%|*}"
    bert_model="${target#*|}"
    if [[ -n "$bert_model" ]]; then
      label="backend=${backend} model=${bert_model}"
    else
      label="backend=${backend}"
    fi
    echo "[preparse] mode=derived ${label}"
    if [[ -n "$bert_model" ]]; then
      EMBED_BACKEND="$backend" BERT_MODEL="$bert_model" \
        "$PYTHON_BIN" app/moodle_rag_runner.py --data-dir "$data_dir_path" --preparse
    else
      EMBED_BACKEND="$backend" \
        "$PYTHON_BIN" app/moodle_rag_runner.py --data-dir "$data_dir_path" --preparse
    fi
  done <<< "$unique_targets"
}

ENV_FILE="${PROJECT_ROOT}/.env"
load_dotenv "$ENV_FILE"

PROFILE="${DEMO_EVAL_PROFILE:-prod}"
DATASET="${DEMO_EVAL_DATASET:-./data/answer_run_questions/ch03_running_time_in_scope_40q.json}"
DEV_DATASET="${DEMO_EVAL_DEV_DATASET:-./data/answer_run_questions/ch03_running_time_in_scope_3q.json}"
DATA_DIR="${DEMO_EVAL_DATA_DIR:-./data/eval_ch03_only}"
RUNS="${DEMO_EVAL_RUNS:-3}"
MODES="${DEMO_EVAL_MODES:-llm_only,rag_bert,rag_msmarco}"
CHAT_MODELS="${DEMO_EVAL_CHAT_MODELS:-qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b}"
EXISTING_ANSWER_RUNS="${DEMO_EVAL_EXISTING_ANSWER_RUNS:-}"
USE_EXISTING_ANSWER_RUNS="${DEMO_EVAL_USE_EXISTING_ANSWER_RUNS:-false}"
SKIP_PREPARSE="false"
SKIP_SYSTEM_PLOTS="false"
DRY_RUN="false"
DATASET_FROM_CLI="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; DATASET_FROM_CLI="true"; shift 2 ;;
    --dev-dataset) DEV_DATASET="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --chat-models) CHAT_MODELS="$2"; shift 2 ;;
    --existing-answer-runs) EXISTING_ANSWER_RUNS="$2"; shift 2 ;;
    --use-existing-answer-runs) USE_EXISTING_ANSWER_RUNS="true"; shift ;;
    --skip-preparse) SKIP_PREPARSE="true"; shift ;;
    --skip-system-plots) SKIP_SYSTEM_PLOTS="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    *)
      echo "[ERROR] Unknown arg: $1"
      exit 1
      ;;
  esac
done

PROFILE="$(to_lower "$PROFILE")"
if [[ "$PROFILE" != "dev" && "$PROFILE" != "prod" ]]; then
  echo "[ERROR] DEMO_EVAL_PROFILE must be 'dev' or 'prod'. Current value: $PROFILE"
  exit 1
fi
if [[ "$PROFILE" == "dev" && "$DATASET_FROM_CLI" != "true" ]]; then
  DATASET="$DEV_DATASET"
fi

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python venv tidak ditemukan di ${PYTHON_BIN}"
  echo "Buat dulu: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

DATASET_PATH="$(abs_path "$DATASET")"
if [[ ! -f "$DATASET_PATH" ]]; then
  echo "[ERROR] Dataset tidak ditemukan: $DATASET_PATH"
  exit 1
fi

DATA_DIR_PATH="$(abs_path "$DATA_DIR")"
if [[ "$(to_lower "$USE_EXISTING_ANSWER_RUNS")" != "true" && ! -d "$DATA_DIR_PATH" ]]; then
  echo "[ERROR] Folder corpus tidak ditemukan: $DATA_DIR_PATH"
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DATASET_STEM="$(basename "$DATASET_PATH")"
DATASET_STEM="${DATASET_STEM%.*}"
SAFE_DATASET_STEM="$(echo "$DATASET_STEM" | sed 's/[^A-Za-z0-9_-]/_/g')"
[[ -n "$SAFE_DATASET_STEM" ]] || SAFE_DATASET_STEM="dataset"

EVAL_OUTPUT_DIR="${PROJECT_ROOT}/data/eval_results/demo_eval_${SAFE_DATASET_STEM}_${TIMESTAMP}"
ANSWER_RUNS_PATH="${PROJECT_ROOT}/data/answer_runs/demo_answer_runs_resume_${SAFE_DATASET_STEM}.jsonl"
ANSWER_RUNS_SNAPSHOT="${EVAL_OUTPUT_DIR}/answer_runs.jsonl"
SYSTEM_RUNS_PATH="${EVAL_OUTPUT_DIR}/system_eval_runs.jsonl"
SYSTEM_SUMMARY_PATH="${EVAL_OUTPUT_DIR}/system_eval_summary.json"
SYSTEM_PLOT_DIR="${EVAL_OUTPUT_DIR}/system_eval_plots"
JUDGED_RUNS_PATH="${EVAL_OUTPUT_DIR}/judged_runs.jsonl"
QUALITY_RUNS_PATH="${EVAL_OUTPUT_DIR}/quality_eval_runs.jsonl"
QUALITY_SUMMARY_PATH="${EVAL_OUTPUT_DIR}/quality_eval_summary.json"
QUALITY_PLOT_DIR="${EVAL_OUTPUT_DIR}/quality_eval_plots"
TRACE_LOG_PATH="${TRACE_LOG_PATH:-${PROJECT_ROOT}/data/logs/e2e_trace_python.jsonl}"

if [[ "$(to_lower "$USE_EXISTING_ANSWER_RUNS")" == "true" ]]; then
  if [[ -z "$EXISTING_ANSWER_RUNS" ]]; then
    echo "[ERROR] --use-existing-answer-runs aktif tapi --existing-answer-runs belum diisi."
    exit 1
  fi
  ANSWER_RUNS_PATH="$(abs_path "$EXISTING_ANSWER_RUNS")"
  [[ -f "$ANSWER_RUNS_PATH" ]] || { echo "[ERROR] File answer-runs tidak ditemukan: $ANSWER_RUNS_PATH"; exit 1; }
fi

echo "== Demo Evaluation Runner (macOS/Linux) =="
echo "Project : $PROJECT_ROOT"
echo "Profile : $PROFILE"
echo "Dataset : $DATASET_PATH"
echo "Corpus  : $DATA_DIR_PATH"
echo "Modes   : $MODES"
echo "Models  : $CHAT_MODELS"
echo "Runs    : $RUNS"
echo "Output  : $EVAL_OUTPUT_DIR"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Mode    : dry-run"
  exit 0
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "[ERROR] Command 'ollama' tidak ditemukan."
  exit 1
fi

mkdir -p "$EVAL_OUTPUT_DIR"
mkdir -p "$(dirname "$TRACE_LOG_PATH")"

if [[ "$(to_lower "$USE_EXISTING_ANSWER_RUNS")" != "true" ]]; then
  if [[ "$SKIP_PREPARSE" != "true" ]]; then
    echo
    echo "== Preparse =="
    run_preparse_for_modes "$MODES" "$DATA_DIR_PATH"
  fi

  echo
  echo "== Generate answer runs =="
  MODELS=($(split_csv_models "$CHAT_MODELS"))
  TOTAL_MODELS="${#MODELS[@]}"
  if [[ "$TOTAL_MODELS" -eq 0 ]]; then
    echo "[ERROR] Tidak ada chat model yang valid."
    exit 1
  fi
  for ((i=0; i<TOTAL_MODELS; i++)); do
    model="${MODELS[$i]}"
    echo
    echo "-- [Model $((i+1))/$TOTAL_MODELS] $model --"
    echo "Mematikan model lain sebelum evaluasi..."
    for other in "${MODELS[@]}"; do
      if [[ "$other" != "$model" ]]; then
        stop_model_with_logs "$other"
      fi
    done

    RUN_EVAL_ARGS=(
      "scripts/eval/run_eval.py"
      "--dataset" "$DATASET_PATH"
      "--data-dir" "$DATA_DIR_PATH"
      "--output" "$ANSWER_RUNS_PATH"
      "--runs" "$RUNS"
      "--modes" "$MODES"
      "--chat-models" "$model"
      "--trace-log" "$TRACE_LOG_PATH"
      "--resume"
      "--skip-dedupe"
      "--skip-preparse"
    )
    "$PYTHON_BIN" "${RUN_EVAL_ARGS[@]}"

    echo "Mematikan model selesai evaluasi: $model"
    stop_model_with_logs "$model"
  done
else
  echo
  echo "== Generate answer runs =="
  echo "Skip generate. Menggunakan file existing: $ANSWER_RUNS_PATH"
fi

cp -f "$ANSWER_RUNS_PATH" "$ANSWER_RUNS_SNAPSHOT"

echo
echo "== Evaluate system performance =="
"$PYTHON_BIN" scripts/eval/evaluate_system_performance.py \
  --questions "$DATASET_PATH" \
  --answer-runs "$ANSWER_RUNS_SNAPSHOT" \
  --output-runs "$SYSTEM_RUNS_PATH" \
  --output-summary "$SYSTEM_SUMMARY_PATH"

if [[ "$SKIP_SYSTEM_PLOTS" != "true" ]]; then
  echo
  echo "== Generate system plots =="
  "$PYTHON_BIN" scripts/eval/plot_system_eval.py \
    --summary "$SYSTEM_SUMMARY_PATH" \
    --output-dir "$SYSTEM_PLOT_DIR"
fi

echo
echo "== Auto-judge answer quality =="
"$PYTHON_BIN" scripts/eval/auto_judge_answer_quality.py \
  --questions "$DATASET_PATH" \
  --answer-runs "$ANSWER_RUNS_SNAPSHOT" \
  --output "$JUDGED_RUNS_PATH"

echo
echo "== Summarize answer quality =="
"$PYTHON_BIN" scripts/eval/evaluate_answer_quality.py \
  --judged-runs "$JUDGED_RUNS_PATH" \
  --output-runs "$QUALITY_RUNS_PATH" \
  --output-summary "$QUALITY_SUMMARY_PATH" \
  --plot \
  --plot-output-dir "$QUALITY_PLOT_DIR"

echo
echo "== Output Penting =="
echo "eval_output_dir      : $EVAL_OUTPUT_DIR"
echo "answer_runs          : $ANSWER_RUNS_SNAPSHOT"
echo "system_eval_runs     : $SYSTEM_RUNS_PATH"
echo "system_eval_summary  : $SYSTEM_SUMMARY_PATH"
if [[ "$SKIP_SYSTEM_PLOTS" != "true" ]]; then
  echo "system_eval_plots    : $SYSTEM_PLOT_DIR"
fi
echo "judged_runs          : $JUDGED_RUNS_PATH"
echo "quality_eval_runs    : $QUALITY_RUNS_PATH"
echo "quality_eval_summary : $QUALITY_SUMMARY_PATH"
echo "quality_eval_plots   : $QUALITY_PLOT_DIR"
echo
echo "Demo evaluation completed successfully."
