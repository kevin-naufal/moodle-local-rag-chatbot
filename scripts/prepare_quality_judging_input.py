from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def create_output_path(base_dir: Path, prefix: str, suffix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = uuid4().hex[:6]
    return base_dir / f"{prefix}_{stamp}_{token}.{suffix}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSONL row at line {line_number}: expected object.")
        rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_question_lookup(dataset: dict) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for item in list(dataset.get("questions") or []):
        question_id = str(item.get("id") or "").strip()
        if question_id:
            lookup[question_id] = item
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare answer-quality judging template from answer runs.")
    parser.add_argument("--questions", required=True, help="Path to question dataset JSON.")
    parser.add_argument("--answer-runs", required=True, help="Path to answer runs JSONL.")
    parser.add_argument("--output", default="", help="Optional output JSONL path.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "quality_eval_inputs"

    questions_path = Path(args.questions).resolve()
    answer_runs_path = Path(args.answer_runs).resolve()
    output_path = Path(args.output).resolve() if str(args.output).strip() else create_output_path(output_dir, "judged_quality_template", "jsonl")

    dataset = load_json(questions_path)
    answer_runs = load_jsonl(answer_runs_path)
    question_lookup = build_question_lookup(dataset)
    default_scope = str(dataset.get("scope") or "unknown").strip() or "unknown"

    rows: list[dict] = []
    for run in answer_runs:
        question_id = str(run.get("question_id") or "").strip()
        spec = question_lookup.get(question_id, {})
        gold_points = list(spec.get("gold_points") or [])
        gold_source = list(spec.get("gold_source") or [])

        rows.append(
            {
                "question_id": question_id,
                "question": str(run.get("question") or spec.get("question") or "").strip(),
                "mode": str(run.get("mode") or "").strip(),
                "run_id": int(run.get("run_id") or 0),
                "model_name": str(run.get("model_name") or "").strip(),
                "embedding_backend": run.get("embedding_backend"),
                "embedding_model_name": str(run.get("embedding_model_name") or "").strip() or None,
                "scope": str(spec.get("scope") or default_scope or "unknown").strip(),
                "expected_behavior": "answer",
                "status": str(run.get("status") or "").strip().lower() or "success",
                "latency_total": run.get("latency_total"),
                "latency_retrieval": run.get("latency_retrieval"),
                "latency_generation": run.get("latency_generation"),
                "model_answer": str(run.get("model_answer") or "").strip(),
                "retrieved_context": run.get("retrieved_context") or [],
                "gold_points": gold_points,
                "gold_source": gold_source,
                "key_points_total": len(gold_points),
                "key_points_covered": None,
                "answer_correctness": None,
                "answer_completeness": None,
                "answer_groundedness": None,
                "answer_relevance": None,
                "refusal_appropriateness": None,
                "answer_clarity": None,
                "instruction_compliance": None,
                "need_alignment": None,
                "scaffolding_quality": None,
                "pedagogical_actionability": None,
                "unsupported_claim_count": None,
                "must_not_claim_violations": None,
                "judge_label": "",
                "judge_reason": "",
            }
        )

    write_jsonl(output_path, rows)
    print("Quality judging template prepared.")
    print(f"- output: {output_path}")
    print(f"- total_rows: {len(rows)}")


if __name__ == "__main__":
    main()
