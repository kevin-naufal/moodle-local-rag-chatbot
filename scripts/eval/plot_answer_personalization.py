from __future__ import annotations

import argparse
import sys
from pathlib import Path


PERSONALIZATION_FIELDS = (
    "instruction_compliance",
    "need_alignment",
    "answer_clarity",
    "scaffolding_quality",
    "pedagogical_actionability",
)


def ensure_personalization_fields_present(rows: list[dict]) -> None:
    missing: list[str] = []
    for row in rows:
        row_id = f"{row.get('question_id')}|{row.get('mode')}|run{row.get('run_id')}"
        for field in PERSONALIZATION_FIELDS:
            if row.get(field) is None:
                missing.append(f"{row_id}:{field}")
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(
            "Answer personalization fields are empty. "
            "Run the LLM personalization judge first. "
            f"Missing: {preview}{suffix}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answer-personalization plots after LLM judging.")
    parser.add_argument("--judged-runs", required=True, help="Path to judged runs JSONL with LLM-filled personalization fields.")
    parser.add_argument("--output-dir", required=True, help="Directory for personalization PNG/Markdown artifacts.")
    parser.add_argument("--output-summary", default="", help="Optional summary JSON path to update from judged runs.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "app") not in sys.path:
        sys.path.insert(0, str(project_root / "app"))
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))

    from quality_eval import build_quality_eval_summary, load_judged_quality_runs_from_text, write_json
    from plot_quality_eval import build_plots, build_tables

    judged_runs_path = Path(args.judged_runs).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_judged_quality_runs_from_text(judged_runs_path.read_text(encoding="utf-8"))
    ensure_personalization_fields_present(rows)
    summary = build_quality_eval_summary(rows, judged_runs_file=str(judged_runs_path), scoring_version="llm_personalization_judged_v1")

    if str(args.output_summary or "").strip():
        write_json(Path(args.output_summary).resolve(), summary)

    files = build_plots(summary, output_dir, include_personalization=True)
    files.extend(build_tables(summary, output_dir, include_personalization=True))
    personalization_files = [
        file_path
        for file_path in files
        if "personalization" in Path(file_path).name
    ]
    if not personalization_files:
        raise ValueError("No answer personalization artifacts were generated.")

    print("Answer personalization plots completed.")
    print(f"- judged_runs: {judged_runs_path}")
    print(f"- output_dir: {output_dir}")
    print(f"- plot_files: {len(personalization_files)}")


if __name__ == "__main__":
    main()
