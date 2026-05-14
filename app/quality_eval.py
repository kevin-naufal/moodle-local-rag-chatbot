from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

from eval_schema import utc_timestamp
from system_eval import write_json, write_jsonl_rows


DEFAULT_QUALITY_WEIGHTS = {
    "correctness": 0.4,
    "completeness": 0.2,
    "groundedness": 0.25,
    "relevance": 0.15,
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, round(float(value or 0.0), 4)))


def _quantize_tenth(value: float | None) -> float | None:
    if value is None:
        return None
    clipped = _clip01(value)
    return round(math.floor((clipped * 10.0) + 0.5) / 10.0, 1)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_scope(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"in-scope", "inscope", "in_scope", "answerable"}:
        return "in-scope"
    if text in {"out-of-scope", "outofscope", "out_of_scope", "unanswerable"}:
        return "out-of-scope"
    return text or "unknown"


def _normalize_expected_behavior(value: str, scope: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"answer", "refuse"}:
        return text
    if scope == "in-scope":
        return "answer"
    if scope == "out-of-scope":
        return "refuse"
    return "unknown"


def _normalize_mode(value: str) -> str:
    return str(value or "").strip()


def _normalize_weights(value: Any) -> dict[str, float]:
    base = dict(DEFAULT_QUALITY_WEIGHTS)
    if not isinstance(value, dict):
        return base

    parsed: dict[str, float] = {}
    for key in ("correctness", "completeness", "groundedness", "relevance"):
        raw = _parse_float(value.get(key))
        if raw is not None and raw > 0:
            parsed[key] = raw

    if not parsed:
        return base

    total = sum(parsed.values())
    return {key: round(parsed[key] / total, 4) for key in parsed}


def _compute_quality_score(row: dict[str, Any], weights: dict[str, float]) -> float | None:
    explicit_score = _parse_float(row.get("quality_score"))
    if explicit_score is not None:
        return _quantize_tenth(explicit_score)

    scope = _normalize_scope(row.get("scope"))
    expected_behavior = _normalize_expected_behavior(row.get("expected_behavior"), scope)
    if expected_behavior == "refuse":
        refusal_score = _parse_float(row.get("refusal_appropriateness"))
        return _quantize_tenth(refusal_score) if refusal_score is not None else None

    metrics = {
        "correctness": _parse_float(row.get("answer_correctness")),
        "completeness": _parse_float(row.get("answer_completeness")),
        "groundedness": _parse_float(row.get("answer_groundedness")),
        "relevance": _parse_float(row.get("answer_relevance")),
    }
    usable = [(float(weights.get(name, 0.0)), value) for name, value in metrics.items() if value is not None]
    total_weight = sum(weight for weight, _ in usable)
    if total_weight <= 0:
        return None
    weighted_score = sum(weight * float(value) for weight, value in usable) / total_weight
    return _quantize_tenth(weighted_score)


def _derive_key_point_coverage_rate(row: dict[str, Any]) -> float | None:
    explicit = _parse_float(row.get("key_point_coverage_rate"))
    if explicit is not None:
        return _clip01(explicit)

    total = _parse_float(row.get("key_points_total"))
    covered = _parse_float(row.get("key_points_covered"))
    if total is None or covered is None or total <= 0:
        return None
    return _clip01(covered / total)


def _compute_consistency(rows: list[dict[str, Any]]) -> float | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = row.get("quality_score")
        if score is None:
            continue
        grouped[str(row.get("question_id") or "")].append(float(score))

    spreads: list[float] = []
    for scores in grouped.values():
        if len(scores) < 2:
            continue
        spreads.append(max(scores) - min(scores))

    if not spreads:
        return None
    return _clip01(1.0 - (sum(spreads) / len(spreads)))


def load_judged_quality_runs_from_text(raw_text: str) -> list[dict[str, Any]]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("No judged quality runs found in the input file.")

    rows: list[dict[str, Any]] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for line_number, line in enumerate(text.splitlines(), start=1):
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                raise ValueError(f"Invalid judged quality JSONL at line {line_number}: row must be a JSON object.")
    else:
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            items = payload["rows"]
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError("Judged quality input must be a JSON array, JSONL, or an object with a 'rows' list.")
        for item in items:
            if isinstance(item, dict):
                rows.append(item)

    if not rows:
        raise ValueError("No valid judged quality rows found in the input file.")

    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        question_id = str(row.get("question_id") or row.get("id") or f"auto-q{index:03d}").strip()
        mode = _normalize_mode(row.get("mode"))
        if not mode:
            raise ValueError(f"Missing mode for judged quality row: {question_id}")

        scope = _normalize_scope(row.get("scope"))
        expected_behavior = _normalize_expected_behavior(row.get("expected_behavior"), scope)
        status = str(row.get("status") or "success").strip().lower()
        weights = _normalize_weights(row.get("quality_weights"))

        normalized_row = {
            "question_id": question_id,
            "question": str(row.get("question") or "").strip(),
            "mode": mode,
            "run_id": int(row.get("run_id") or 0),
            "model_name": str(row.get("model_name") or "").strip(),
            "embedding_backend": row.get("embedding_backend"),
            "scope": scope,
            "expected_behavior": expected_behavior,
            "status": status,
            "success_score": 1 if status == "success" else 0,
            "latency_total": float(row.get("latency_total") or 0.0),
            "latency_retrieval": float(row.get("latency_retrieval") or 0.0),
            "latency_generation": float(row.get("latency_generation") or 0.0),
            "predicted_behavior": str(row.get("predicted_behavior") or "").strip().lower() or None,
            "answer_correctness": _quantize_tenth(_parse_float(row.get("answer_correctness"))),
            "answer_completeness": _quantize_tenth(_parse_float(row.get("answer_completeness"))),
            "answer_groundedness": _quantize_tenth(_parse_float(row.get("answer_groundedness"))),
            "answer_relevance": _quantize_tenth(_parse_float(row.get("answer_relevance"))),
            "refusal_appropriateness": _quantize_tenth(_parse_float(row.get("refusal_appropriateness"))),
            "key_points_total": int(row.get("key_points_total") or 0),
            "key_points_covered": int(row.get("key_points_covered") or 0),
            "key_point_coverage_rate": _derive_key_point_coverage_rate(row),
            "unsupported_claim_count": float(row.get("unsupported_claim_count") or 0.0),
            "must_not_claim_violations": float(row.get("must_not_claim_violations") or 0.0),
            "judge_label": str(row.get("judge_label") or "").strip() or None,
            "judge_reason": str(row.get("judge_reason") or "").strip() or None,
            "quality_weights": weights,
            "timestamp": str(row.get("timestamp") or utc_timestamp()),
        }
        normalized_row["quality_score"] = _compute_quality_score(normalized_row, weights)
        normalized_rows.append(normalized_row)

    return normalized_rows


def _summarize_quality_rows(rows: list[dict[str, Any]], mode: str, scope: str | None = None) -> dict[str, Any]:
    success_scores = [int(row["success_score"]) for row in rows]
    successful_rows = [row for row in rows if int(row["success_score"]) == 1]

    def collect(metric: str) -> list[float]:
        return [float(row[metric]) for row in rows if row.get(metric) is not None]

    return {
        "mode": mode,
        "scope": scope,
        "total_runs": len(rows),
        "successful_runs": sum(success_scores),
        "failed_runs": len(rows) - sum(success_scores),
        "success_rate": _mean([float(value) for value in success_scores]),
        "avg_answer_correctness": _mean(collect("answer_correctness")),
        "avg_answer_completeness": _mean(collect("answer_completeness")),
        "avg_answer_groundedness": _mean(collect("answer_groundedness")),
        "avg_answer_relevance": _mean(collect("answer_relevance")),
        "avg_refusal_appropriateness": _mean(collect("refusal_appropriateness")),
        "avg_key_point_coverage_rate": _mean(collect("key_point_coverage_rate")),
        "avg_unsupported_claim_count": _mean(collect("unsupported_claim_count")),
        "avg_must_not_claim_violations": _mean(collect("must_not_claim_violations")),
        "avg_quality_score": _mean(collect("quality_score")),
        "consistency_score": _compute_consistency(rows),
        "avg_latency_total": _mean([float(row.get("latency_total") or 0.0) for row in successful_rows]),
        "avg_latency_retrieval": _mean([float(row.get("latency_retrieval") or 0.0) for row in successful_rows]),
        "avg_latency_generation": _mean([float(row.get("latency_generation") or 0.0) for row in successful_rows]),
    }


def build_quality_eval_summary(
    evaluated_rows: list[dict[str, Any]],
    *,
    judged_runs_file: str = "",
    scoring_version: str = "manual_judged_quality_v1",
) -> dict[str, Any]:
    by_mode: list[dict[str, Any]] = []
    unique_modes = sorted({str(row.get("mode") or "").strip() for row in evaluated_rows if str(row.get("mode") or "").strip()})
    for mode in unique_modes:
        mode_rows = [row for row in evaluated_rows if str(row.get("mode") or "").strip() == mode]
        mode_summary = _summarize_quality_rows(mode_rows, mode)
        by_scope: list[dict[str, Any]] = []
        unique_scopes = sorted({str(row.get("scope") or "unknown") for row in mode_rows})
        for scope in unique_scopes:
            scope_rows = [row for row in mode_rows if str(row.get("scope") or "unknown") == scope]
            by_scope.append(_summarize_quality_rows(scope_rows, mode, scope))
        mode_summary["by_scope"] = by_scope
        by_mode.append(mode_summary)

    return {
        "generated_at": utc_timestamp(),
        "scoring_version": str(scoring_version or "manual_judged_quality_v1"),
        "judged_runs_file": str(judged_runs_file or ""),
        "total_runs": len(evaluated_rows),
        "by_mode": by_mode,
    }


__all__ = [
    "build_quality_eval_summary",
    "load_judged_quality_runs_from_text",
    "write_json",
    "write_jsonl_rows",
]
