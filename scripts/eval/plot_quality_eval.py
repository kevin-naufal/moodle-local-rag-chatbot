from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from matplotlib.patches import Patch

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
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def annotate_bar_values(bars, values: list[float | None], max_y: float) -> None:
    offset = max(max_y * 0.015, 0.02)
    for bar, value in zip(bars, values):
        if value is None:
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                offset,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#666666",
            )
            continue
        height = float(value)
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            format_cell(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_bar_chart(
    *,
    title: str,
    ylabel: str,
    categories: list[str],
    category_models: list[str],
    series: list[tuple[str, list[float | None]]],
    output_path: Path,
) -> None:
    unique_models = sorted({m for m in category_models if m})
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    model_colors = {model: palette[i % len(palette)] for i, model in enumerate(unique_models)}
    bar_colors = [model_colors.get(model, "#4e79a7") for model in category_models]

    width_inches = max(11, min(20, 0.75 * len(categories)))
    plt.figure(figsize=(width_inches, 6))
    width = 0.8 / max(1, len(series))
    positions = list(range(len(categories)))
    all_present_values = [float(value) for _, values in series for value in values if value is not None]
    max_y = max(all_present_values) if all_present_values else 1.0

    for series_index, (label, values) in enumerate(series):
        xs = [position + (series_index - (len(series) - 1) / 2.0) * width for position in positions]
        ys = [math.nan if value is None else float(value) for value in values]
        bars = plt.bar(xs, ys, width=width, label=label, color=bar_colors)
        annotate_bar_values(bars, values, max_y)

    plt.xticks(positions, categories, rotation=25, ha="right", fontsize=8)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, max(max_y * 1.15, 0.1))
    if unique_models:
        legend_handles = [Patch(facecolor=model_colors[m], label=m) for m in unique_models]
        plt.legend(handles=legend_handles, title="Model", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def get_group_metric(row: dict, group: str, metric: str, fallback_key: str) -> float | None:
    grouped = row.get(group)
    if isinstance(grouped, dict) and metric in grouped:
        return grouped.get(metric)
    return row.get(fallback_key)


def build_plots(summary: dict, output_dir: Path, *, include_personalization: bool = False) -> list[str]:
    mode_rows = list(summary.get("by_model_mode") or summary.get("by_mode") or [])
    if not mode_rows:
        return []

    categories = []
    category_models = []
    for row in mode_rows:
        model_name = str(row.get("model_name") or "").strip()
        short_model = model_name.split(":", 1)[1] if ":" in model_name else model_name or "-"
        categories.append(f"{row.get('mode')}\n{short_model}")
        category_models.append(model_name or "-")
    files: list[str] = []
    answer_quality_specs = [
        (
            "answer_quality_correctness",
            "Answer Quality: Correctness by Mode",
            "Score",
            [("correctness", [get_group_metric(row, "answer_quality", "correctness", "avg_answer_correctness") for row in mode_rows])],
        ),
        (
            "answer_quality_groundedness",
            "Answer Quality: Groundedness by Mode",
            "Score",
            [("groundedness", [get_group_metric(row, "answer_quality", "groundedness", "avg_answer_groundedness") for row in mode_rows])],
        ),
        (
            "answer_quality_relevance",
            "Answer Quality: Relevance by Mode",
            "Score",
            [("relevance", [get_group_metric(row, "answer_quality", "relevance", "avg_answer_relevance") for row in mode_rows])],
        ),
    ]
    answer_personalization_specs = [
        (
            "answer_personalization_instruction_compliance",
            "Answer Personalization: Instruction Compliance by Mode",
            "Score",
            [
                (
                    "instruction_compliance",
                    [get_group_metric(row, "answer_personalization", "instruction_compliance", "avg_instruction_compliance") for row in mode_rows],
                )
            ],
        ),
        (
            "answer_personalization_need_alignment",
            "Answer Personalization: Need Alignment by Mode",
            "Score",
            [("need_alignment", [get_group_metric(row, "answer_personalization", "need_alignment", "avg_need_alignment") for row in mode_rows])],
        ),
        (
            "answer_personalization_scaffolding_quality",
            "Answer Personalization: Scaffolding Quality by Mode",
            "Score",
            [
                (
                    "scaffolding_quality",
                    [get_group_metric(row, "answer_personalization", "scaffolding_quality", "avg_scaffolding_quality") for row in mode_rows],
                )
            ],
        ),
    ]

    plot_specs = list(answer_quality_specs)
    if include_personalization:
        plot_specs.extend(answer_personalization_specs)

    for slug, title, ylabel, series in plot_specs:
        has_any_data = any(any(value is not None for value in values) for _, values in series)
        if not has_any_data:
            continue
        output_path = output_dir / f"{sanitize_filename(slug)}.png"
        save_bar_chart(
            title=title,
            ylabel=ylabel,
            categories=categories,
            category_models=category_models,
            series=series,
            output_path=output_path,
        )
        files.append(str(output_path))

    return files


def build_answer_quality_table(summary: dict) -> tuple[list[str], list[dict[str, object]]]:
    mode_rows = list(summary.get("by_model_mode") or summary.get("by_mode") or [])
    include_model = any(row.get("model_name") for row in mode_rows)
    columns = (["model_name"] if include_model else []) + ["mode", "total_runs", "correctness", "groundedness", "relevance"]
    rows: list[dict[str, object]] = []
    for row in mode_rows:
        rows.append(
            {
                "model_name": row.get("model_name"),
                "mode": row.get("mode"),
                "total_runs": row.get("total_runs"),
                "correctness": get_group_metric(row, "answer_quality", "correctness", "avg_answer_correctness"),
                "groundedness": get_group_metric(row, "answer_quality", "groundedness", "avg_answer_groundedness"),
                "relevance": get_group_metric(row, "answer_quality", "relevance", "avg_answer_relevance"),
            }
        )
    return columns, rows


def build_answer_personalization_table(summary: dict) -> tuple[list[str], list[dict[str, object]]]:
    mode_rows = list(summary.get("by_model_mode") or summary.get("by_mode") or [])
    include_model = any(row.get("model_name") for row in mode_rows)
    columns = (["model_name"] if include_model else []) + ["mode", "total_runs", "instruction_compliance", "need_alignment", "scaffolding_quality"]
    rows: list[dict[str, object]] = []
    for row in mode_rows:
        rows.append(
            {
                "model_name": row.get("model_name"),
                "mode": row.get("mode"),
                "total_runs": row.get("total_runs"),
                "instruction_compliance": get_group_metric(
                    row, "answer_personalization", "instruction_compliance", "avg_instruction_compliance"
                ),
                "need_alignment": get_group_metric(row, "answer_personalization", "need_alignment", "avg_need_alignment"),
                "scaffolding_quality": get_group_metric(
                    row, "answer_personalization", "scaffolding_quality", "avg_scaffolding_quality"
                ),
            }
        )
    return columns, rows


def write_markdown_table(output_path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tables(summary: dict, output_dir: Path, *, include_personalization: bool = False) -> list[str]:
    files: list[str] = []

    quality_columns, quality_rows = build_answer_quality_table(summary)
    if quality_rows:
        quality_md_path = output_dir / "mode_vs_answer_quality.md"
        write_markdown_table(quality_md_path, quality_columns, quality_rows)
        files.append(str(quality_md_path))

    personalization_columns, personalization_rows = build_answer_personalization_table(summary)
    has_personalization_data = any(
        any(row.get(column) is not None for column in personalization_columns if column not in {"mode", "total_runs"})
        for row in personalization_rows
    )
    if include_personalization and personalization_rows and has_personalization_data:
        personalization_md_path = output_dir / "mode_vs_answer_personalization.md"
        write_markdown_table(personalization_md_path, personalization_columns, personalization_rows)
        files.append(str(personalization_md_path))

    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate charts from answer evaluation summary.")
    parser.add_argument("--summary", required=True, help="Path to quality evaluation summary JSON.")
    parser.add_argument("--output-dir", default="", help="Optional directory for generated chart PNGs.")
    parser.add_argument("--include-personalization", action="store_true", help="Also generate answer personalization charts and tables.")
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    summary = load_summary(summary_path)
    output_dir = ensure_output_dir(summary_path, args.output_dir)
    files = build_plots(summary, output_dir, include_personalization=args.include_personalization)
    files.extend(build_tables(summary, output_dir, include_personalization=args.include_personalization))

    payload = {
        "ok": True,
        "summary_file": str(summary_path),
        "output_dir": str(output_dir),
        "files": files,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
