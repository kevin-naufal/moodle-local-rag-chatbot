from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_MODES = ("llm_only", "rag_ollama", "rag_bert")


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
    valid = {"llm_only", "rag_ollama", "rag_bert"}
    invalid = [item for item in items if item not in valid]
    if invalid:
        raise ValueError(f"Unsupported mode(s): {', '.join(invalid)}")
    return items


def mode_to_runner_args(mode: str) -> tuple[str, str]:
    if mode == "llm_only":
        return "general", "auto"
    if mode == "rag_ollama":
        return "auto", "ollama"
    if mode == "rag_bert":
        return "auto", "bert"
    raise ValueError(f"Unsupported mode: {mode}")


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
    project_root: Path,
) -> None:
    env = os.environ.copy()
    env["EMBED_BACKEND"] = embed_backend
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
        default="llm_only,rag_ollama,rag_bert",
        help="Comma-separated list of modes to run.",
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
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
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
    runs = max(1, int(args.runs or 1))

    if args.overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_preparse:
        for mode in modes:
            if mode == "llm_only":
                continue
            _, embed_backend = mode_to_runner_args(mode)
            print(f"[preparse] backend={embed_backend}")
            run_preparse(
                python_bin=args.python_bin,
                runner_path=runner_path,
                data_dir=data_dir,
                embed_backend=embed_backend,
                project_root=project_root,
            )

    total_jobs = len(questions) * len(modes) * runs
    completed = 0
    failures = 0

    for index, question in enumerate(questions, start=1):
        question_text = str(question.get("question", "")).strip()
        if not question_text:
            continue
        question_id = str(question.get("id") or question.get("question_id") or f"q{index:02d}")

        for mode in modes:
            runner_mode, embed_backend = mode_to_runner_args(mode)
            for run_id in range(1, runs + 1):
                completed += 1
                print(f"[{completed}/{total_jobs}] {question_id} | {mode} | run {run_id}")
                env = os.environ.copy()
                env["EMBED_BACKEND"] = embed_backend
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
                print(f"  -> status={status} payload_mode={summary}")
                if status != "success":
                    failures += 1
                    print(f"  -> error_message={payload.get('error_message')}")

    print("\nEvaluation run completed.")
    print(f"- output: {output_path}")
    print(f"- total_questions: {len(questions)}")
    print(f"- total_jobs: {total_jobs}")
    print(f"- failures: {failures}")


if __name__ == "__main__":
    main()
