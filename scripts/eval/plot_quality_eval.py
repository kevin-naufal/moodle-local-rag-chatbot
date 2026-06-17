from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from matplotlib.patches import Patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class HeatmapMatrix:
    models: list[str]
    modes: list[str]
    values: list[list[float | None]]


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


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def build_quality_heatmap_matrix(summary: dict, metric: str, fallback_key: str) -> HeatmapMatrix:
    mode_rows = list(summary.get("by_model_mode") or summary.get("by_mode") or [])
    models = _ordered_unique([str(row.get("model_name") or "-") for row in mode_rows])
    modes = _ordered_unique([str(row.get("mode") or "-") for row in mode_rows])
    value_by_key: dict[tuple[str, str], float | None] = {}
    for row in mode_rows:
        model = str(row.get("model_name") or "-")
        mode = str(row.get("mode") or "-")
        value_by_key[(model, mode)] = get_group_metric(row, "answer_quality", metric, fallback_key)
    values = [[value_by_key.get((model, mode)) for mode in modes] for model in models]
    return HeatmapMatrix(models=models, modes=modes, values=values)


def save_heatmap(
    *,
    title: str,
    matrix: HeatmapMatrix,
    output_path: Path,
    cmap: str = "YlGnBu",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
) -> None:
    plot_values = [[math.nan if value is None else float(value) for value in row] for row in matrix.values]
    width_inches = max(7, 1.35 * len(matrix.modes) + 3)
    height_inches = max(4.5, 0.65 * len(matrix.models) + 2.5)
    plt.figure(figsize=(width_inches, height_inches))
    image = plt.imshow(plot_values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(matrix.modes)), matrix.modes, rotation=20, ha="right")
    y_labels = [model.split(":", 1)[1] if ":" in model else model for model in matrix.models]
    plt.yticks(range(len(matrix.models)), y_labels)
    plt.title(title)

    for row_index, row in enumerate(matrix.values):
        for column_index, value in enumerate(row):
            label = format_cell(value) if value is not None else "N/A"
            plt.text(column_index, row_index, label, ha="center", va="center", fontsize=9, color="#111111")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def contrast_limits(matrix: HeatmapMatrix, *, padding_ratio: float = 0.08) -> tuple[float, float]:
    values = [float(value) for row in matrix.values for value in row if value is not None]
    if not values:
        return 0.0, 1.0

    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        padding = 0.05
    else:
        padding = (max_value - min_value) * padding_ratio
    return max(0.0, min_value - padding), min(1.0, max_value + padding)


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
            "correctness",
            "avg_answer_correctness",
            False,
        ),
        (
            "answer_quality_completeness",
            "Answer Quality: Completeness by Mode",
            "completeness",
            "avg_answer_completeness",
            False,
        ),
        (
            "answer_quality_groundedness",
            "Answer Quality: Groundedness by Mode",
            "groundedness",
            "avg_answer_groundedness",
            True,
        ),
        (
            "answer_quality_relevance",
            "Answer Quality: Relevance by Mode",
            "relevance",
            "avg_answer_relevance",
            True,
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

    for slug, title, metric, fallback_key, use_contrast_scale in answer_quality_specs:
        matrix = build_quality_heatmap_matrix(summary, metric, fallback_key)
        has_any_data = any(any(value is not None for value in row) for row in matrix.values)
        if not has_any_data:
            continue
        output_path = output_dir / f"{sanitize_filename(slug)}.png"
        vmin, vmax = contrast_limits(matrix) if use_contrast_scale else (0.0, 1.0)
        save_heatmap(
            title=title,
            matrix=matrix,
            output_path=output_path,
            vmin=vmin,
            vmax=vmax,
        )
        files.append(str(output_path))

    if include_personalization:
        for slug, title, ylabel, series in answer_personalization_specs:
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
    columns = (["model_name"] if include_model else []) + ["mode", "total_runs", "correctness", "completeness", "groundedness", "relevance"]
    rows: list[dict[str, object]] = []
    for row in mode_rows:
        rows.append(
            {
                "model_name": row.get("model_name"),
                "mode": row.get("mode"),
                "total_runs": row.get("total_runs"),
                "correctness": get_group_metric(row, "answer_quality", "correctness", "avg_answer_correctness"),
                "completeness": get_group_metric(row, "answer_quality", "completeness", "avg_answer_completeness"),
                "groundedness": get_group_metric(row, "answer_quality", "groundedness", "avg_answer_groundedness"),
                "relevance": get_group_metric(row, "answer_quality", "relevance", "avg_answer_relevance"),
            }
        )
    return columns, rows


def build_answer_personalization_table(summary: dict) -> tuple[list[str], list[dict[str, object]]]:
    mode_rows = list(summary.get("by_model_mode") or summary.get("by_mode") or [])
    include_model = any(row.get("model_name") for row in mode_rows)
    columns = (["model_name"] if include_model else []) + [
        "mode",
        "total_runs",
        "format_compliance",
        "instruction_compliance",
        "need_alignment",
        "scaffolding_quality",
    ]
    rows: list[dict[str, object]] = []
    for row in mode_rows:
        rows.append(
            {
                "model_name": row.get("model_name"),
                "mode": row.get("mode"),
                "total_runs": row.get("total_runs"),
                "format_compliance": get_group_metric(row, "answer_personalization", "format_compliance", "avg_format_compliance"),
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
