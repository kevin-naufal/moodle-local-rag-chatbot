from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from uuid import uuid4


def create_output_path(base_dir: Path, prefix: str, suffix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = uuid4().hex[:6]
    return base_dir / f"{prefix}_{stamp}_{token}.{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Objective system-performance evaluator for answer runs.")
    parser.add_argument("--questions", required=True, help="Path to the question dataset JSON file.")
    parser.add_argument("--answer-runs", required=True, help="Path to the answer-run JSONL file.")
    parser.add_argument("--top-k", type=int, default=4, help="Top-k retrieved chunks used for retrieval metrics.")
    parser.add_argument(
        "--retrieval-match-level",
        choices=["page", "evidence"],
        default="page",
        help="Relevance criterion for retrieval metrics: page only, or page plus anchor-term evidence.",
    )
    parser.add_argument(
        "--coverage-method",
        choices=["semantic", "anchor"],
        default="semantic",
        help="Coverage@K matching method: semantic gold-point similarity, or deterministic anchor-term matching.",
    )
    parser.add_argument("--output-runs", default="", help="Optional JSONL output path for per-run objective scores.")
    parser.add_argument("--output-summary", default="", help="Optional JSON output path for aggregated mode summary.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "app") not in sys.path:
        sys.path.insert(0, str(project_root / "app"))

    from system_eval import (  # pylint: disable=import-outside-toplevel
        build_objective_eval_summary,
        evaluate_answer_runs,
        load_answer_runs_from_text,
        load_question_specs_from_text,
        write_json,
        write_jsonl_rows,
    )
    from scripts.eval.auto_judge_answer_quality import load_semantic_quality_evaluator  # pylint: disable=import-outside-toplevel

    question_path = Path(args.questions).resolve()
    answer_runs_path = Path(args.answer_runs).resolve()
    output_dir = project_root / "data" / "system_eval_results"
    output_runs = Path(args.output_runs).resolve() if str(args.output_runs).strip() else create_output_path(output_dir, "objective_eval_runs", "jsonl")
    output_summary = Path(args.output_summary).resolve() if str(args.output_summary).strip() else create_output_path(output_dir, "objective_eval_summary", "json")

    question_specs = load_question_specs_from_text(question_path.read_text(encoding="utf-8"))
    answer_runs = load_answer_runs_from_text(answer_runs_path.read_text(encoding="utf-8"))
    coverage_evaluator = load_semantic_quality_evaluator() if args.coverage_method == "semantic" else None
    evaluated_rows = evaluate_answer_runs(
        answer_runs,
        question_specs,
        top_k=args.top_k,
        retrieval_match_level=args.retrieval_match_level,
        coverage_evaluator=coverage_evaluator,
    )
    summary = build_objective_eval_summary(
        evaluated_rows,
        top_k=args.top_k,
        retrieval_match_level=args.retrieval_match_level,
        answer_runs_file=str(answer_runs_path),
        question_dataset_file=str(question_path),
    )

    write_jsonl_rows(output_runs, evaluated_rows)
    write_json(output_summary, summary)

    print("Objective system-performance evaluation completed.")
    print(f"- per_run_output: {output_runs}")
    print(f"- summary_output: {output_summary}")
    print(f"- total_runs: {len(evaluated_rows)}")


if __name__ == "__main__":
    main()
