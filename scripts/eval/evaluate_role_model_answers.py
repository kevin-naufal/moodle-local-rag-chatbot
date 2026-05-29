from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "app"))

from quality_eval import build_quality_eval_summary, load_judged_quality_runs_from_text, write_json, write_jsonl_rows
from scripts.eval.auto_judge_answer_quality import (
    build_question_lookup,
    judge_row,
    load_json,
    load_semantic_quality_evaluator,
    write_jsonl,
)


ROLE_MODEL_MODE = "role_model_semantic"
ROLE_MODEL_NAME = "role_model_answer"
NO_RETRIEVAL_MODE = "llm_only"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def load_retrieved_context_lookup(path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Retrieved contexts file not found: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        if not isinstance(row, dict):
            raise ValueError(f"Invalid retrieved context row at line {line_number}: expected object.")
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"Missing question_id in retrieved context row at line {line_number}.")
        lookup[question_id] = {
            "embedding_backend": str(row.get("embedding_backend") or "").strip(),
            "embedding_model": str(row.get("embedding_model") or row.get("embedding_model_name") or "").strip(),
            "retrieved_context": list(row.get("retrieved_context") or []),
        }
    return lookup


def build_role_model_answer_runs(
    dataset: dict[str, Any],
    runs: int,
    retrieved_context_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if runs < 1:
        raise ValueError("--runs must be at least 1.")

    context_lookup = retrieved_context_lookup or {}
    rows: list[dict[str, Any]] = []
    for item in list(dataset.get("questions") or []):
        question_id = str(item.get("id") or "").strip()
        question = str(item.get("question") or "").strip()
        answer = str(item.get("role_model_answer") or "").strip()
        if not question_id:
            raise ValueError("Every question must have an id.")
        if not answer:
            raise ValueError(f"Missing role_model_answer for question: {question_id}")

        context_payload = context_lookup.get(question_id, {})
        retrieved_context = list(context_payload.get("retrieved_context") or [])
        has_retrieval_context = bool(retrieved_context)
        mode = ROLE_MODEL_MODE if has_retrieval_context else NO_RETRIEVAL_MODE

        for run_id in range(1, runs + 1):
            rows.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "mode": mode,
                    "run_id": run_id,
                    "model_name": ROLE_MODEL_NAME,
                    "embedding_backend": str(context_payload.get("embedding_backend") or "").strip() or None,
                    "embedding_model_name": str(context_payload.get("embedding_model") or "").strip() or None,
                    "status": "success",
                    "model_answer": answer,
                    "retrieved_context": retrieved_context,
                }
            )
    return rows


def summarize_by_question(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            continue
        grouped.setdefault(question_id, []).append(row)

    summaries: list[dict[str, Any]] = []
    for question_id in sorted(grouped):
        question_rows = grouped[question_id]
        quality_scores = [
            float(row["quality_score"])
            for row in question_rows
            if row.get("quality_score") is not None
        ]
        pass_count = sum(1 for score in quality_scores if score >= 0.8)
        run_count = len(question_rows)
        pass_rate = round(pass_count / run_count, 4) if run_count else 0.0
        if pass_rate >= 1.0:
            final_label = "strong_pass"
        elif pass_rate >= 0.8:
            final_label = "pass"
        elif pass_rate >= 0.6:
            final_label = "unstable"
        else:
            final_label = "fail"

        summaries.append(
            {
                "question_id": question_id,
                "question": str(question_rows[0].get("question") or "").strip(),
                "run_count": run_count,
                "pass_count": pass_count,
                "pass_rate": pass_rate,
                "final_label": final_label,
                "avg_answer_correctness": _mean(
                    [float(row["answer_correctness"]) for row in question_rows if row.get("answer_correctness") is not None]
                ),
                "avg_answer_groundedness": _mean(
                    [float(row["answer_groundedness"]) for row in question_rows if row.get("answer_groundedness") is not None]
                ),
                "avg_answer_relevance": _mean(
                    [float(row["answer_relevance"]) for row in question_rows if row.get("answer_relevance") is not None]
                ),
                "avg_quality_score": _mean(quality_scores),
            }
        )

    return {
        "total_questions": len(summaries),
        "questions": summaries,
    }


def create_output_dir(base_dir: Path, dataset_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"role_model_answer_eval_{dataset_path.stem}_{stamp}"


def evaluate_role_model_answers(
    questions_path: Path,
    *,
    runs: int,
    output_dir: Path,
    retrieved_contexts_path: Path | None = None,
    embed_model: str | None = None,
) -> dict[str, Path]:
    dataset = load_json(questions_path)
    retrieved_context_lookup = (
        load_retrieved_context_lookup(retrieved_contexts_path)
        if retrieved_contexts_path is not None
        else None
    )
    answer_runs = build_role_model_answer_runs(dataset, runs, retrieved_context_lookup=retrieved_context_lookup)
    question_lookup = build_question_lookup(dataset)
    default_scope = str(dataset.get("scope") or "unknown").strip() or "unknown"
    evaluator = load_semantic_quality_evaluator(embed_model or os.getenv("QUALITY_EVAL_EMBED_MODEL"))

    judged_rows = [
        judge_row(
            run,
            question_lookup.get(str(run.get("question_id") or "").strip(), {}),
            default_scope,
            evaluator=evaluator,
        )
        for run in answer_runs
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    answer_runs_path = output_dir / "answer_runs_role_model.jsonl"
    judged_runs_path = output_dir / "judged_runs_role_model.jsonl"
    quality_runs_path = output_dir / "quality_eval_runs_role_model.jsonl"
    summary_path = output_dir / "quality_eval_summary_role_model.json"
    question_summary_path = output_dir / "question_summary_role_model.json"

    write_jsonl(answer_runs_path, answer_runs)
    write_jsonl(judged_runs_path, judged_rows)

    normalized_rows = load_judged_quality_runs_from_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in judged_rows)
    )
    summary = build_quality_eval_summary(
        normalized_rows,
        judged_runs_file=str(judged_runs_path),
        scoring_version="role_model_semantic_v1",
    )
    question_summary = summarize_by_question(normalized_rows)
    write_jsonl_rows(quality_runs_path, normalized_rows)
    write_json(summary_path, summary)
    write_json(question_summary_path, question_summary)

    return {
        "output_dir": output_dir,
        "answer_runs": answer_runs_path,
        "judged_runs": judged_runs_path,
        "quality_runs": quality_runs_path,
        "summary": summary_path,
        "question_summary": question_summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic evaluation for role_model_answer fields.")
    parser.add_argument("--questions", required=True, help="Path to question dataset JSON.")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated evaluations per role model answer.")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    parser.add_argument(
        "--retrieved-contexts",
        default="",
        help="Optional JSONL from retrieve_question_contexts.py. If omitted, groundedness is N/A.",
    )
    parser.add_argument("--embed-model", default="", help="Optional sentence-transformers model override.")
    args = parser.parse_args()

    questions_path = Path(args.questions).resolve()
    base_output_dir = PROJECT_ROOT / "data" / "eval_results"
    output_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else create_output_dir(base_output_dir, questions_path)

    paths = evaluate_role_model_answers(
        questions_path,
        runs=args.runs,
        output_dir=output_dir,
        retrieved_contexts_path=Path(args.retrieved_contexts).resolve() if str(args.retrieved_contexts).strip() else None,
        embed_model=str(args.embed_model or "").strip() or None,
    )

    print("Role model semantic evaluation completed.")
    print(f"- output_dir: {paths['output_dir']}")
    print(f"- answer_runs: {paths['answer_runs']}")
    print(f"- judged_runs: {paths['judged_runs']}")
    print(f"- quality_runs: {paths['quality_runs']}")
    print(f"- summary: {paths['summary']}")
    print(f"- question_summary: {paths['question_summary']}")


if __name__ == "__main__":
    main()
