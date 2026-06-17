from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.patches import Patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name or "").strip())
    return safe or "plot"


def format_cell(value: float | int | str | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def pick_rows(summary: dict) -> list[dict]:
    rows = list(summary.get("by_model_mode") or summary.get("by_mode") or [])
    return [row for row in rows if isinstance(row, dict)]


def get_quality_metric(row: dict, metric: str, fallback_key: str) -> float | None:
    grouped = row.get("answer_quality")
    if isinstance(grouped, dict) and metric in grouped:
        return grouped.get(metric)
    return row.get(fallback_key)


def make_categories(rows: list[dict]) -> tuple[list[str], list[str]]:
    categories: list[str] = []
    models: list[str] = []
    for row in rows:
        model_name = str(row.get("model_name") or "").strip()
        short_model = model_name.split(":", 1)[1] if ":" in model_name else model_name or "-"
        categories.append(f"{row.get('mode')}\\n{short_model}")
        models.append(model_name or "-")
    return categories, models


def annotate_bar_values(bars, values: list[float | None], max_y: float) -> None:
    offset = max(max_y * 0.015, 0.02)
    for bar, value in zip(bars, values):
        if value is None:
            plt.text(bar.get_x() + bar.get_width() / 2.0, offset, "N/A", ha="center", va="bottom", fontsize=8, color="#666666")
            continue
        height = float(value)
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + offset, format_cell(value), ha="center", va="bottom", fontsize=8)


def save_bar_chart(
    *,
    title: str,
    ylabel: str,
    categories: list[str],
    category_models: list[str],
    values: list[float | None],
    output_path: Path,
) -> None:
    unique_models = sorted({m for m in category_models if m})
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    model_colors = {model: palette[i % len(palette)] for i, model in enumerate(unique_models)}
    bar_colors = [model_colors.get(model, "#4e79a7") for model in category_models]

    width_inches = max(11, min(20, 0.75 * len(categories)))
    plt.figure(figsize=(width_inches, 6))
    positions = list(range(len(categories)))
    ys = [0.0 if value is None else float(value) for value in values]
    bars = plt.bar(positions, ys, width=0.65, color=bar_colors)

    all_present_values = [float(value) for value in values if value is not None]
    max_y = max(all_present_values) if all_present_values else 1.0
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


def write_metric_md(output_path: Path, metric_label: str, key: str, rows: list[dict], values: list[float | None]) -> None:
    include_model = any(row.get("model_name") for row in rows)
    columns = (["model_name"] if include_model else []) + ["mode", "total_runs", key]
    lines = [
        f"# Core Metric: {metric_label}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row, value in zip(rows, values):
        payload = {
            "model_name": row.get("model_name"),
            "mode": row.get("mode"),
            "total_runs": row.get("total_runs"),
            key: value,
        }
        lines.append("| " + " | ".join(format_cell(payload.get(column)) for column in columns) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate core-5 evaluation report folders (plot + markdown per metric).")
    parser.add_argument("--system-summary", required=True, help="Path to system_eval_summary.json")
    parser.add_argument("--quality-summary", required=True, help="Path to quality_eval_summary.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for core metric folders")
    args = parser.parse_args()

    system_summary = load_json(Path(args.system_summary).resolve())
    quality_summary = load_json(Path(args.quality_summary).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    system_rows = pick_rows(system_summary)
    quality_rows = pick_rows(quality_summary)

    metric_specs = [
        {
            "slug": "hit_at_k",
            "label": "Hit@K",
            "source": "system",
            "key": "hit_at_k",
            "title": "Core Metric: Hit@K by Mode",
            "ylabel": "Score",
            "extractor": lambda row: row.get("source_hit_at_k_rate"),
        },
        {
            "slug": "mrr",
            "label": "MRR",
            "source": "system",
            "key": "mrr",
            "title": "Core Metric: MRR by Mode",
            "ylabel": "Score",
            "extractor": lambda row: row.get("mrr"),
        },
        {
            "slug": "coverage_at_k",
            "label": "Coverage@K",
            "source": "system",
            "key": "coverage_at_k",
            "title": "Core Metric: Coverage@K by Mode",
            "ylabel": "Score",
            "extractor": lambda row: row.get("avg_coverage_at_k"),
        },
        {
            "slug": "latency",
            "label": "Latency",
            "source": "system",
            "key": "latency",
            "title": "Core Metric: Average Total Latency by Mode",
            "ylabel": "Seconds",
            "extractor": lambda row: row.get("avg_latency_total"),
        },
        {
            "slug": "correctness",
            "label": "Correctness",
            "source": "quality",
            "key": "correctness",
            "title": "Core Metric: Correctness by Mode",
            "ylabel": "Score",
            "extractor": lambda row: get_quality_metric(row, "correctness", "avg_answer_correctness"),
        },
        {
            "slug": "groundedness",
            "label": "Groundedness",
            "source": "quality",
            "key": "groundedness",
            "title": "Core Metric: Groundedness by Mode",
            "ylabel": "Score",
            "extractor": lambda row: get_quality_metric(row, "groundedness", "avg_answer_groundedness"),
        },
    ]

    generated_files: list[str] = []
    for spec in metric_specs:
        rows = system_rows if spec["source"] == "system" else quality_rows
        if not rows:
            continue
        categories, models = make_categories(rows)
        values = [spec["extractor"](row) for row in rows]

        metric_dir = output_dir / sanitize_filename(spec["slug"])
        metric_dir.mkdir(parents=True, exist_ok=True)

        png_path = metric_dir / f"{sanitize_filename(spec['slug'])}.png"
        md_path = metric_dir / f"{sanitize_filename(spec['slug'])}.md"

        if any(value is not None for value in values):
            save_bar_chart(
                title=str(spec["title"]),
                ylabel=str(spec["ylabel"]),
                categories=categories,
                category_models=models,
                values=values,
                output_path=png_path,
            )
            generated_files.append(str(png_path))

        write_metric_md(md_path, str(spec["label"]), str(spec["key"]), rows, values)
        generated_files.append(str(md_path))

    payload = {
        "ok": True,
        "output_dir": str(output_dir),
        "files": generated_files,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
