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
    parser = argparse.ArgumentParser(description="Answer evaluation summarizer for pre-scored judged runs.")
    parser.add_argument("--judged-runs", required=True, help="Path to the judged quality JSONL or JSON file.")
    parser.add_argument("--output-runs", default="", help="Optional JSONL output path for normalized judged rows.")
    parser.add_argument("--output-summary", default="", help="Optional JSON output path for aggregated quality summary.")
    parser.add_argument("--plot", action="store_true", help="Also generate PNG charts and markdown tables.")
    parser.add_argument("--plot-output-dir", default="", help="Optional output directory for charts and tables.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "app") not in sys.path:
        sys.path.insert(0, str(project_root / "app"))

    from quality_eval import (  # pylint: disable=import-outside-toplevel
        build_quality_eval_summary,
        load_judged_quality_runs_from_text,
        write_json,
        write_jsonl_rows,
    )

    judged_runs_path = Path(args.judged_runs).resolve()
    output_dir = project_root / "data" / "quality_eval_results"
    output_runs = Path(args.output_runs).resolve() if str(args.output_runs).strip() else create_output_path(output_dir, "quality_eval_runs", "jsonl")
    output_summary = Path(args.output_summary).resolve() if str(args.output_summary).strip() else create_output_path(output_dir, "quality_eval_summary", "json")

    evaluated_rows = load_judged_quality_runs_from_text(judged_runs_path.read_text(encoding="utf-8"))
    summary = build_quality_eval_summary(
        evaluated_rows,
        judged_runs_file=str(judged_runs_path),
    )

    write_jsonl_rows(output_runs, evaluated_rows)
    write_json(output_summary, summary)

    print("Answer evaluation summary completed.")
    print(f"- normalized_runs_output: {output_runs}")
    print(f"- summary_output: {output_summary}")
    print(f"- total_runs: {len(evaluated_rows)}")

    if args.plot:
        from plot_quality_eval import build_plots, build_tables, ensure_output_dir  # pylint: disable=import-outside-toplevel

        plot_dir = ensure_output_dir(output_summary, args.plot_output_dir)
        files = build_plots(summary, plot_dir)
        files.extend(build_tables(summary, plot_dir))
        print(f"- plot_output_dir: {plot_dir}")
        print(f"- plot_files: {len(files)}")


if __name__ == "__main__":
    main()
