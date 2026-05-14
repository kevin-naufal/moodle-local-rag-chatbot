from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(summary_path: Path, output_dir: str) -> Path:
    if output_dir.strip():
        target = Path(output_dir).resolve()
    else:
        target = summary_path.with_suffix("")
        target = target.parent / f"{target.name}_plots"
    target.mkdir(parents=True, exist_ok=True)
    return target


def sanitize_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name or "").strip())
    return safe or "plot"


def format_cell(value: float | int | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def save_bar_chart(
    *,
    title: str,
    ylabel: str,
    categories: list[str],
    series: list[tuple[str, list[float | None]]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    width = 0.8 / max(1, len(series))
    positions = list(range(len(categories)))

    for series_index, (label, values) in enumerate(series):
        xs = [position + (series_index - (len(series) - 1) / 2.0) * width for position in positions]
        ys = [0.0 if value is None else float(value) for value in values]
        plt.bar(xs, ys, width=width, label=label)

    plt.xticks(positions, categories)
    plt.title(title)
    plt.ylabel(ylabel)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_plots(summary: dict, output_dir: Path) -> list[str]:
    mode_rows = list(summary.get("by_mode") or [])
    if not mode_rows:
        return []

    categories = [str(row.get("mode") or "-") for row in mode_rows]
    files: list[str] = []
    chart_specs = [
        (
            "success_rate",
            "Success Rate by Mode",
            "Rate",
            [("success_rate", [row.get("success_rate") for row in mode_rows])],
        ),
        (
            "quality_core",
            "Core Answer Quality by Mode",
            "Score",
            [
                ("correctness", [row.get("avg_answer_correctness") for row in mode_rows]),
                ("completeness", [row.get("avg_answer_completeness") for row in mode_rows]),
                ("relevance", [row.get("avg_answer_relevance") for row in mode_rows]),
            ],
        ),
        (
            "grounding_quality",
            "Grounding and Quality Score by Mode",
            "Score",
            [
                ("groundedness", [row.get("avg_answer_groundedness") for row in mode_rows]),
                ("refusal_appropriateness", [row.get("avg_refusal_appropriateness") for row in mode_rows]),
                ("quality_score", [row.get("avg_quality_score") for row in mode_rows]),
            ],
        ),
        (
            "quality_detail",
            "Coverage and Consistency by Mode",
            "Score",
            [
                ("key_point_coverage", [row.get("avg_key_point_coverage_rate") for row in mode_rows]),
                ("consistency", [row.get("consistency_score") for row in mode_rows]),
            ],
        ),
        (
            "latency",
            "Average Latency by Mode",
            "Seconds",
            [
                ("total", [row.get("avg_latency_total") for row in mode_rows]),
                ("retrieval", [row.get("avg_latency_retrieval") for row in mode_rows]),
                ("generation", [row.get("avg_latency_generation") for row in mode_rows]),
            ],
        ),
        (
            "quality_risks",
            "Average Risk Counts by Mode",
            "Count",
            [
                ("unsupported_claim_count", [row.get("avg_unsupported_claim_count") for row in mode_rows]),
                ("must_not_claim_violations", [row.get("avg_must_not_claim_violations") for row in mode_rows]),
            ],
        ),
    ]

    for slug, title, ylabel, series in chart_specs:
        has_any_data = any(any(value is not None for value in values) for _, values in series)
        if not has_any_data:
            continue
        output_path = output_dir / f"{sanitize_filename(slug)}.png"
        save_bar_chart(
            title=title,
            ylabel=ylabel,
            categories=categories,
            series=series,
            output_path=output_path,
        )
        files.append(str(output_path))

    return files


def build_mode_metric_table(summary: dict) -> tuple[list[str], list[dict[str, object]]]:
    mode_rows = list(summary.get("by_mode") or [])
    columns = [
        "mode",
        "total_runs",
        "successful_runs",
        "failed_runs",
        "success_rate",
        "avg_answer_correctness",
        "avg_answer_completeness",
        "avg_answer_groundedness",
        "avg_answer_relevance",
        "avg_refusal_appropriateness",
        "avg_key_point_coverage_rate",
        "avg_quality_score",
        "consistency_score",
        "avg_unsupported_claim_count",
        "avg_must_not_claim_violations",
        "avg_latency_total",
        "avg_latency_retrieval",
        "avg_latency_generation",
    ]
    rows: list[dict[str, object]] = []
    for row in mode_rows:
        rows.append({column: row.get(column) for column in columns})
    return columns, rows


def write_markdown_table(output_path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tables(summary: dict, output_dir: Path) -> list[str]:
    columns, rows = build_mode_metric_table(summary)
    if not rows:
        return []
    md_path = output_dir / "mode_vs_quality_metrics.md"
    write_markdown_table(md_path, columns, rows)
    return [str(md_path)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate charts from answer-quality evaluation summary.")
    parser.add_argument("--summary", required=True, help="Path to quality evaluation summary JSON.")
    parser.add_argument("--output-dir", default="", help="Optional directory for generated chart PNGs.")
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    summary = load_summary(summary_path)
    output_dir = ensure_output_dir(summary_path, args.output_dir)
    files = build_plots(summary, output_dir)
    files.extend(build_tables(summary, output_dir))

    payload = {
        "ok": True,
        "summary_file": str(summary_path),
        "output_dir": str(output_dir),
        "files": files,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
