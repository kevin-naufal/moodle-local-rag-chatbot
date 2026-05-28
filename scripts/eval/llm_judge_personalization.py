from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a strict evaluator for answer personalization quality. "
    "Return ONLY valid JSON with these numeric fields in [0.0,1.0] rounded to one decimal: "
    "instruction_compliance, need_alignment, answer_clarity, scaffolding_quality, pedagogical_actionability. "
    "Use null only when the answer is empty. Also return one short string field: judge_reason. "
    "Do not include markdown."
)


def _round_tenth(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    number = max(0.0, min(1.0, number))
    return round(number * 10.0) / 10.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def build_answer_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], str]:
    lookup: dict[tuple[str, str, str, int], str] = {}
    for row in rows:
        key = (
            str(row.get("question_id") or "").strip(),
            str(row.get("mode") or "").strip(),
            str(row.get("model_name") or "").strip(),
            int(row.get("run_id") or 0),
        )
        lookup[key] = str(row.get("model_answer") or "").strip()
    return lookup


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_ollama_json(base_url: str, model: str, question: str, answer: str, timeout_sec: int = 120) -> dict[str, Any]:
    user_prompt = (
        "Evaluate the following QA pair.\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Scoring guide:\n"
        "- instruction_compliance: follows explicit format/constraint from question.\n"
        "- need_alignment: matches user need and detail level.\n"
        "- answer_clarity: clear, coherent, readable.\n"
        "- scaffolding_quality: structured explanation helpful for learning.\n"
        "- pedagogical_actionability: gives actionable next steps or practical guidance.\n"
    )
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    content = (
        payload.get("message", {}).get("content", "")
        if isinstance(payload, dict)
        else ""
    )
    result = json.loads(str(content or "{}"))
    if not isinstance(result, dict):
        return {}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge for answer personalization fields.")
    parser.add_argument("--input", required=True, help="Input judged-runs JSONL path.")
    parser.add_argument("--output", required=True, help="Output judged-runs JSONL path.")
    parser.add_argument("--answer-runs", default="", help="Optional answer-runs JSONL path to read model_answer.")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama chat model for judging.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    rows = load_jsonl(input_path)
    answer_lookup: dict[tuple[str, str, str, int], str] = {}
    if str(args.answer_runs or "").strip():
        answer_rows = load_jsonl(Path(args.answer_runs).resolve())
        answer_lookup = build_answer_lookup(answer_rows)
    updated: list[dict[str, Any]] = []

    fields = (
        "instruction_compliance",
        "need_alignment",
        "answer_clarity",
        "scaffolding_quality",
        "pedagogical_actionability",
    )

    for row in rows:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("model_answer") or "").strip()
        if not answer:
            key = (
                str(row.get("question_id") or "").strip(),
                str(row.get("mode") or "").strip(),
                str(row.get("model_name") or "").strip(),
                int(row.get("run_id") or 0),
            )
            answer = answer_lookup.get(key, "")
        if not answer:
            for field in fields:
                row[field] = None
            row["judge_reason"] = "Answer is empty."
            updated.append(row)
            continue

        result = call_ollama_json(args.ollama_url, args.model, question, answer)
        for field in fields:
            row[field] = _round_tenth(result.get(field))
        if not row.get("judge_reason"):
            row["judge_reason"] = str(result.get("judge_reason") or row.get("judge_reason") or "").strip()
        row["personalization_judge_method"] = "llm_as_judge_v1"
        row["personalization_judge_model"] = args.model
        updated.append(row)

    write_jsonl(output_path, updated)
    print(f"LLM personalization judging completed: {output_path}")
    print(f"rows: {len(updated)}")


if __name__ == "__main__":
    main()
