from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4


MSMARCO_BERT_MODEL = "sentence-transformers/msmarco-bert-base-dot-v5"
BASE_BERT_MODEL = "bert-base-uncased"
DEFAULT_CHAT_MODELS = ("qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b")

MODE_CONFIGS: dict[str, dict[str, str]] = {
    "llm_only": {"runner_mode": "general", "embed_backend": "auto"},
    "rag_ollama": {"runner_mode": "auto", "embed_backend": "ollama"},
    "rag_bert": {"runner_mode": "auto", "embed_backend": "bert", "bert_model": BASE_BERT_MODEL},
    "rag_msmarco": {
        "runner_mode": "auto",
        "embed_backend": "bert",
        "bert_model": MSMARCO_BERT_MODEL,
    },
}

DEFAULT_MODES = ("llm_only", "rag_bert", "rag_msmarco")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return [item for item in data["questions"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError("Dataset must be a JSON array or an object with a 'questions' list.")


def normalize_modes(raw: str) -> list[str]:
    items = [part.strip().lower() for part in str(raw or "").split(",") if part.strip()]
    if not items:
        return list(DEFAULT_MODES)
    valid = set(MODE_CONFIGS)
    invalid = [item for item in items if item not in valid]
    if invalid:
        raise ValueError(f"Unsupported mode(s): {', '.join(invalid)}")
    return items


def normalize_chat_models(raw: str) -> list[str]:
    items = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not items:
        return list(DEFAULT_CHAT_MODELS)
    return items


def get_mode_config(mode: str) -> dict[str, str]:
    config = MODE_CONFIGS.get(str(mode or "").strip().lower())
    if config is None:
        raise ValueError(f"Unsupported mode: {mode}")
    return dict(config)


def load_existing_completed_keys(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()

    completed: set[tuple[str, str, str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status and status != "success":
            continue
        question_id = str(payload.get("question_id") or "").strip()
        mode = str(payload.get("mode") or "").strip().lower()
        model_name = str(payload.get("model_name") or "").strip()
        try:
            run_id = int(payload.get("run_id") or 0)
        except (TypeError, ValueError):
            continue
        if question_id and mode and model_name and run_id > 0:
            completed.add((question_id, mode, model_name, run_id))
    return completed


def dedupe_answer_runs_file(path: Path, planned_keys: set[tuple[str, str, str, int]]) -> None:
    if not path.exists():
        return

    latest_rows: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        question_id = str(payload.get("question_id") or "").strip()
        mode = str(payload.get("mode") or "").strip().lower()
        model_name = str(payload.get("model_name") or "").strip()
        try:
            run_id = int(payload.get("run_id") or 0)
        except (TypeError, ValueError):
            continue
        key = (question_id, mode, model_name, run_id)
        if key not in planned_keys:
            continue
        latest_rows[key] = payload

    ordered_rows = [latest_rows[key] for key in sorted(latest_rows)]
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_rows)
    path.write_text(content, encoding="utf-8")


def parse_runner_payload(stdout_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Runner produced no JSON output.")
    return json.loads(lines[-1])


def run_preparse(
    *,
    python_bin: str,
    runner_path: Path,
    data_dir: Path,
    embed_backend: str,
    bert_model: str,
    project_root: Path,
) -> None:
    env = os.environ.copy()
    env["EMBED_BACKEND"] = embed_backend
    if bert_model:
        env["BERT_MODEL"] = bert_model
    cmd = [
        python_bin,
        str(runner_path),
        "--data-dir",
        str(data_dir),
        "--preparse",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Preparse failed for backend '{embed_backend}'.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluation runner for my-llm.")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset JSON.")
    parser.add_argument("--data-dir", required=True, help="Path to corpus directory used by RAG.")
    parser.add_argument(
        "--output",
        default="data/answer_runs/llm_answer_results.jsonl",
        help="Path to JSONL file where LLM answer runs will be appended.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of repetitions per mode.")
    parser.add_argument(
        "--modes",
        default="llm_only,rag_bert,rag_msmarco",
        help="Comma-separated list of modes to run.",
    )
    parser.add_argument(
        "--chat-models",
        default=",".join(DEFAULT_CHAT_MODELS),
        help="Comma-separated list of Ollama chat models to evaluate.",
    )
    parser.add_argument(
        "--runner",
        default="app/moodle_rag_runner.py",
        help="Path to the Python runner script.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to invoke the runner.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output JSONL file before running.",
    )
    parser.add_argument(
        "--skip-preparse",
        action="store_true",
        help="Skip prebuilding cached vectorstores for RAG modes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume into an existing output JSONL file by skipping completed question/mode/run rows.",
    )
    parser.add_argument(
        "--trace-log",
        default="",
        help="Optional JSONL trace log path passed to each runner process.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dataset_path = Path(args.dataset).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    runner_path = Path(args.runner)
    if not runner_path.is_absolute():
        runner_path = project_root / runner_path

    questions = load_dataset(dataset_path)
    modes = normalize_modes(args.modes)
    chat_models = normalize_chat_models(args.chat_models)
    runs = max(1, int(args.runs or 1))

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    if args.overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    planned_keys = {
        (
            str(question.get("id") or question.get("question_id") or f"q{index:02d}"),
            mode,
            chat_model,
            run_id,
        )
        for index, question in enumerate(questions, start=1)
        if str(question.get("question", "")).strip()
        for mode in modes
        for chat_model in chat_models
        for run_id in range(1, runs + 1)
    }
    completed_keys = set()
    if args.resume:
        completed_keys = planned_keys & load_existing_completed_keys(output_path)
        print(f"[resume] output={output_path}")
        print(f"[resume] completed_jobs_found={len(completed_keys)}")

    if not args.skip_preparse:
        for mode in modes:
            if mode == "llm_only":
                continue
            config = get_mode_config(mode)
            embed_backend = config["embed_backend"]
            embed_model = config.get("bert_model", "") if embed_backend == "bert" else ""
            embed_model_display = f" model={embed_model}" if embed_model else ""
            print(f"[preparse] mode={mode} backend={embed_backend}{embed_model_display}")
            run_preparse(
                python_bin=args.python_bin,
                runner_path=runner_path,
                data_dir=data_dir,
                embed_backend=embed_backend,
                bert_model=config.get("bert_model", ""),
                project_root=project_root,
            )

    total_jobs = len(questions) * len(modes) * len(chat_models) * runs
    completed = len(completed_keys)
    failures = 0

    for index, question in enumerate(questions, start=1):
        question_text = str(question.get("question", "")).strip()
        if not question_text:
            continue
        question_id = str(question.get("id") or question.get("question_id") or f"q{index:02d}")

        for chat_model in chat_models:
            for mode in modes:
                config = get_mode_config(mode)
                runner_mode = config["runner_mode"]
                embed_backend = config["embed_backend"]
                for run_id in range(1, runs + 1):
                    current_key = (question_id, mode, chat_model, run_id)
                    if current_key in completed_keys:
                        continue
                    completed += 1
                    print(f"[{completed}/{total_jobs}] {question_id} | {mode} | {chat_model} | run {run_id}")
                    env = os.environ.copy()
                    env["CHAT_MODEL"] = chat_model
                    env["EMBED_BACKEND"] = embed_backend
                    if config.get("bert_model"):
                        env["BERT_MODEL"] = config["bert_model"]
                    request_id = f"eval-{uuid4().hex[:12]}"
                    cmd = [
                        args.python_bin,
                        str(runner_path),
                        "--data-dir",
                        str(data_dir),
                        "--query",
                        question_text,
                        "--mode",
                        runner_mode,
                        "--request-id",
                        request_id,
                        "--question-number",
                        str(index),
                        "--attempt",
                        str(run_id),
                        "--eval-mode",
                        "--question-id",
                        question_id,
                        "--run-id",
                        str(run_id),
                        "--raw-results-path",
                        str(output_path),
                        "--eval-mode-name",
                        mode,
                    ]
                    if str(args.trace_log or "").strip():
                        cmd.extend(["--trace-log", str(Path(args.trace_log).resolve())])
                    result = subprocess.run(
                        cmd,
                        cwd=str(project_root),
                        env=env,
                        text=True,
                        capture_output=True,
                        encoding="utf-8",
                    )
                    if result.returncode != 0:
                        failures += 1
                        print("[error] runner process failed")
                        print(result.stderr.strip() or result.stdout.strip())
                        continue

                    try:
                        payload = parse_runner_payload(result.stdout)
                    except Exception as exc:
                        failures += 1
                        print(f"[error] invalid runner JSON: {exc}")
                        print(result.stdout.strip())
                        continue

                    status = str(payload.get("status", "success"))
                    summary = payload.get("mode", mode)
                    print(f"  -> status={status} payload_mode={summary} model={payload.get('model_name')}")
                    if status != "success":
                        failures += 1
                        print(f"  -> error_message={payload.get('error_message')}")

    print("\nEvaluation run completed.")
    print(f"- output: {output_path}")
    print(f"- total_questions: {len(questions)}")
    print(f"- total_jobs: {total_jobs}")
    print(f"- failures: {failures}")
    dedupe_answer_runs_file(output_path, planned_keys)


if __name__ == "__main__":
    main()
