from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from eval_schema import utc_timestamp


REFUSAL_PATTERNS = (
    "not found in context",
    "not found in the provided context",
    "not found in the provided material",
    "the context does not contain",
    "the provided context does not contain",
    "the material does not contain",
    "the provided material does not contain",
    "cannot answer from the provided material",
    "cannot answer from the material",
    "cannot be answered from the provided material",
    "cannot be answered from the material",
    "insufficient context",
    "insufficient information in the provided context",
    "no relevant information in the provided context",
)


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


def _normalize_source_name(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    # Keep matching stable across OSes: question datasets may store Windows paths
    # while retrieved context may store only filenames.
    normalized = raw.replace("\\", "/")
    return Path(normalized).name.strip().lower()


def _extract_page_range_from_text(value: str) -> tuple[int | None, int | None]:
    text = str(value or "")
    match = re.search(r"PDF\s+page\s+(\d+)(?:\s*[-to]+\s*(\d+))?", text, flags=re.IGNORECASE)
    if not match:
        return None, None
    start = int(match.group(1))
    end = int(match.group(2) or match.group(1))
    return start, end


def _normalize_gold_sources(item: dict[str, Any], default_source_name: str) -> list[dict[str, Any]]:
    system_eval = item.get("system_eval") if isinstance(item.get("system_eval"), dict) else {}
    raw_structured = system_eval.get("gold_sources") or item.get("gold_sources") or []
    gold_sources: list[dict[str, Any]] = []

    if isinstance(raw_structured, list) and raw_structured:
        for entry in raw_structured:
            if not isinstance(entry, dict):
                continue
            source_name = _normalize_source_name(entry.get("source") or default_source_name)
            if not source_name:
                continue
            page_start_raw = entry.get("page_start")
            page_end_raw = entry.get("page_end")
            page_start = int(page_start_raw) if page_start_raw not in (None, "") else None
            page_end = int(page_end_raw) if page_end_raw not in (None, "") else page_start
            gold_sources.append(
                {
                    "source": source_name,
                    "page_start": page_start,
                    "page_end": page_end,
                }
            )
        return gold_sources

    raw_legacy = item.get("gold_source") or []
    if isinstance(raw_legacy, str):
        raw_legacy = [raw_legacy]
    if not isinstance(raw_legacy, list):
        return []

    for entry in raw_legacy:
        source_name = _normalize_source_name(default_source_name)
        page_start, page_end = _extract_page_range_from_text(str(entry or ""))
        if not source_name and page_start is None and page_end is None:
            continue
        gold_sources.append(
            {
                "source": source_name,
                "page_start": page_start,
                "page_end": page_end,
            }
        )
    return gold_sources


def load_question_specs_from_text(raw_text: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(str(raw_text or ""))
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        items = payload["questions"]
        top_scope = payload.get("scope")
        top_source_document = payload.get("source_document")
    elif isinstance(payload, list):
        items = payload
        top_scope = ""
        top_source_document = ""
    else:
        raise ValueError("Question dataset must be a JSON array or an object with a 'questions' list.")

    default_source_name = _normalize_source_name(top_source_document)
    question_specs: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        question_text = str(item.get("question", "")).strip()
        if not question_text:
            continue
        question_id = str(item.get("question_id") or item.get("id") or f"auto-q{index:03d}").strip()
        scope = _normalize_scope(item.get("scope") or top_scope)
        system_eval = item.get("system_eval") if isinstance(item.get("system_eval"), dict) else {}
        expected_behavior = _normalize_expected_behavior(system_eval.get("expected_behavior") or item.get("expected_behavior"), scope)
        question_specs[question_id] = {
            "question_id": question_id,
            "question": question_text,
            "scope": scope,
            "expected_behavior": expected_behavior,
            "gold_sources": _normalize_gold_sources(item, default_source_name),
            "gold_points": item.get("gold_points") if isinstance(item.get("gold_points"), list) else [],
            "gold_point_anchor_terms": item.get("gold_point_anchor_terms") or item.get("anchor_terms") or [],
        }

    if not question_specs:
        raise ValueError("No valid questions found in the question dataset.")
    return question_specs


def load_answer_runs_from_text(raw_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(str(raw_text or "").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    if not rows:
        raise ValueError("No valid answer runs found in the JSONL file.")
    return rows


def _normalize_page_value(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_anchor_term(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _anchor_group_matches_context(anchor_group: Any, context_text: str) -> bool:
    if isinstance(anchor_group, str):
        terms = [anchor_group]
    elif isinstance(anchor_group, list):
        terms = anchor_group
    else:
        terms = []

    normalized_context = _normalize_anchor_term(context_text)
    context_tokens = set(re.findall(r"[a-z0-9][a-z0-9\-]{1,}", normalized_context))
    for term in terms:
        normalized_term = _normalize_anchor_term(term)
        if not normalized_term:
            continue
        if " " in normalized_term and normalized_term in normalized_context:
            return True
        term_tokens = re.findall(r"[a-z0-9][a-z0-9\-]{1,}", normalized_term)
        if any(token in context_tokens for token in term_tokens):
            return True
    return False


def _anchor_terms_match_context(anchor_terms: Any, context_text: str) -> bool:
    if not isinstance(anchor_terms, list) or not anchor_terms:
        return True
    return any(_anchor_group_matches_context(group, context_text) for group in anchor_terms)


def _anchor_group_has_terms(anchor_group: Any) -> bool:
    if isinstance(anchor_group, str):
        return bool(_normalize_anchor_term(anchor_group))
    if isinstance(anchor_group, list):
        return any(_normalize_anchor_term(term) for term in anchor_group)
    return False


def _valid_anchor_groups(anchor_terms: Any) -> list[tuple[int, Any]]:
    if not isinstance(anchor_terms, list):
        return []
    return [(index, group) for index, group in enumerate(anchor_terms) if _anchor_group_has_terms(group)]


def _required_evidence_count(anchor_terms: Any) -> int:
    valid_count = len(_valid_anchor_groups(anchor_terms))
    if valid_count <= 0:
        return 0
    return max(1, (valid_count + 1) // 2)


def _coverage_groups_for_gold_points(gold_points: Any, anchor_terms: Any) -> list[Any]:
    if isinstance(gold_points, list) and gold_points:
        groups: list[Any] = []
        anchors = anchor_terms if isinstance(anchor_terms, list) else []
        for index, point in enumerate(gold_points):
            if not str(point or "").strip():
                continue
            group = anchors[index] if index < len(anchors) else []
            if _anchor_group_has_terms(group):
                groups.append(group)
        return groups
    return [group for _, group in _valid_anchor_groups(anchor_terms)]


def _gold_point_coverage_at_k(
    top_context: list[dict[str, Any]],
    gold_points: Any,
    anchor_terms: Any,
) -> tuple[int, int, float | None]:
    coverage_groups = _coverage_groups_for_gold_points(gold_points, anchor_terms)
    total = len(coverage_groups)
    if total <= 0:
        return 0, 0, None

    covered = 0
    context_texts = [str(item.get("text") or "") for item in top_context if isinstance(item, dict)]
    for group in coverage_groups:
        if any(_anchor_group_matches_context(group, context_text) for context_text in context_texts):
            covered += 1
    return total, covered, round(covered / total, 4)


def _semantic_gold_point_coverage_at_k(
    top_context: list[dict[str, Any]],
    gold_points: Any,
    coverage_evaluator: Any,
) -> tuple[int, int, float | None]:
    points = [str(point or "").strip() for point in gold_points if str(point or "").strip()] if isinstance(gold_points, list) else []
    if not points:
        return 0, 0, None

    context_texts: list[str] = []
    for item in top_context:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text:
            context_texts.append(text)
    if not context_texts:
        return len(points), 0, 0.0

    covered, coverage_rate, _ = coverage_evaluator.compute_gold_coverage(
        "\n\n".join(context_texts),
        points,
    )
    return len(points), int(covered), round(float(coverage_rate), 4)


def _gold_source_evidence_threshold_rank(
    gold_source: dict[str, Any],
    top_context: list[dict[str, Any]],
    anchor_terms: Any,
) -> int | None:
    valid_groups = _valid_anchor_groups(anchor_terms)
    required_count = _required_evidence_count(anchor_terms)
    matched_group_indexes: set[int] = set()

    for index, context_item in enumerate(top_context, start=1):
        if not _gold_source_matches_context(gold_source, context_item):
            continue
        if not valid_groups:
            return index
        context_text = str(context_item.get("text") or "")
        for group_index, group in valid_groups:
            if group_index in matched_group_indexes:
                continue
            if _anchor_group_matches_context(group, context_text):
                matched_group_indexes.add(group_index)
        if len(matched_group_indexes) >= required_count:
            return index
    return None


def infer_predicted_behavior(answer_text: str, status: str) -> str:
    if str(status or "").strip().lower() != "success":
        return "error"

    normalized = str(answer_text or "").strip().lower()
    if not normalized:
        return "error"
    for pattern in REFUSAL_PATTERNS:
        if pattern in normalized:
            return "refuse"
    return "answer"


def _gold_source_matches_context(gold_source: dict[str, Any], context_item: dict[str, Any]) -> bool:
    gold_name = _normalize_source_name(gold_source.get("source"))
    context_name = _normalize_source_name(context_item.get("source"))
    if gold_name and context_name and gold_name != context_name:
        return False

    page = _normalize_page_value(context_item.get("page"))
    page_start = _normalize_page_value(gold_source.get("page_start"))
    page_end = _normalize_page_value(gold_source.get("page_end"))
    if page_start is None and page_end is None:
        return True
    if page is None:
        return False
    if page_end is None:
        page_end = page_start
    if page_start is None:
        page_start = page_end
    return bool(page_start is not None and page_end is not None and page_start <= page <= page_end)


def evaluate_answer_runs(
    answer_runs: list[dict[str, Any]],
    question_specs: dict[str, dict[str, Any]],
    *,
    top_k: int = 4,
    retrieval_match_level: str = "page",
    coverage_evaluator: Any | None = None,
) -> list[dict[str, Any]]:
    effective_top_k = max(1, int(top_k or 1))
    match_level = str(retrieval_match_level or "page").strip().lower()
    if match_level not in {"page", "evidence"}:
        raise ValueError("retrieval_match_level must be 'page' or 'evidence'.")
    missing_ids = sorted(
        {
            str(row.get("question_id") or "").strip()
            for row in answer_runs
            if str(row.get("question_id") or "").strip() not in question_specs
        }
    )
    if missing_ids:
        raise ValueError(f"Missing question metadata for question_id: {', '.join(missing_ids)}")

    evaluated_rows: list[dict[str, Any]] = []
    for row in answer_runs:
        question_id = str(row.get("question_id") or "").strip()
        spec = question_specs[question_id]
        status = str(row.get("status", "success")).strip().lower()
        answer_text = str(row.get("model_answer") or "")
        predicted_behavior = infer_predicted_behavior(answer_text, status)
        retrieved_context = row.get("retrieved_context") if isinstance(row.get("retrieved_context"), list) else []
        top_context = [item for item in retrieved_context[:effective_top_k] if isinstance(item, dict)]
        gold_sources = list(spec.get("gold_sources") or [])
        gold_points = list(spec.get("gold_points") or [])
        gold_point_anchor_terms = spec.get("gold_point_anchor_terms") or []
        evidence_group_count = len(_valid_anchor_groups(gold_point_anchor_terms))
        required_evidence_count = _required_evidence_count(gold_point_anchor_terms)
        coverage_eval_method = "semantic_embedding" if coverage_evaluator is not None else "anchor_terms"
        if coverage_evaluator is not None:
            gold_points_total, gold_points_covered_at_k, coverage_at_k = _semantic_gold_point_coverage_at_k(
                top_context,
                gold_points,
                coverage_evaluator,
            )
        else:
            gold_points_total, gold_points_covered_at_k, coverage_at_k = _gold_point_coverage_at_k(
                top_context,
                gold_points,
                gold_point_anchor_terms,
            )
        matched_gold_sources = 0
        first_match_rank: int | None = None

        if gold_sources:
            for gold_source in gold_sources:
                source_match_rank: int | None = None
                if match_level == "evidence":
                    source_match_rank = _gold_source_evidence_threshold_rank(
                        gold_source,
                        top_context,
                        gold_point_anchor_terms,
                    )
                else:
                    for index, context_item in enumerate(top_context, start=1):
                        if _gold_source_matches_context(gold_source, context_item):
                            source_match_rank = index
                            break
                if source_match_rank is not None:
                    matched_gold_sources += 1
                    if first_match_rank is None or source_match_rank < first_match_rank:
                        first_match_rank = source_match_rank

        source_hit_at_k: int | None = None
        source_recall_at_k: float | None = None
        rank_of_gold_source: int | None = None
        mrr: float | None = None
        if gold_sources:
            source_hit_at_k = 1 if matched_gold_sources > 0 else 0
            source_recall_at_k = round(matched_gold_sources / len(gold_sources), 4)
            rank_of_gold_source = first_match_rank
            mrr = round(1.0 / first_match_rank, 4) if first_match_rank else 0.0

        expected_behavior = str(spec.get("expected_behavior") or "unknown")
        answerable_detection_correct: int | None = None
        refusal_correct: int | None = None
        if expected_behavior in {"answer", "refuse"}:
            answerable_detection_correct = 1 if predicted_behavior == expected_behavior else 0
            if expected_behavior == "refuse":
                refusal_correct = answerable_detection_correct

        evaluated_rows.append(
            {
                "question_id": question_id,
                "question": spec.get("question") or row.get("question") or "",
                "mode": str(row.get("mode") or "").strip(),
                "model_name": str(row.get("model_name") or "").strip(),
                "run_id": int(row.get("run_id") or 0),
                "scope": spec.get("scope", "unknown"),
                "expected_behavior": expected_behavior,
                "status": status,
                "success_score": 1 if status == "success" else 0,
                "latency_total": float(row.get("latency_total") or 0.0),
                "latency_retrieval": float(row.get("latency_retrieval") or 0.0),
                "latency_generation": float(row.get("latency_generation") or 0.0),
                "top_k": effective_top_k,
                "retrieval_match_level": match_level,
                "evidence_group_count": evidence_group_count,
                "required_evidence_count": required_evidence_count,
                "gold_points_total": gold_points_total,
                "gold_points_covered_at_k": gold_points_covered_at_k,
                "coverage_at_k": coverage_at_k,
                "coverage_eval_method": coverage_eval_method,
                "coverage_eval_embedding_model": str(getattr(coverage_evaluator, "model_name", "") or "").strip() or None,
                "retrieved_context_count": len(retrieved_context),
                "gold_source_count": len(gold_sources),
                "matched_gold_sources": matched_gold_sources,
                "source_hit_at_k": source_hit_at_k,
                "source_recall_at_k": source_recall_at_k,
                "rank_of_gold_source": rank_of_gold_source,
                "mrr": mrr,
                "predicted_behavior": predicted_behavior,
                "answerable_detection_correct": answerable_detection_correct,
                "refusal_correct": refusal_correct,
                "timestamp": row.get("timestamp") or utc_timestamp(),
            }
        )

    return evaluated_rows


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _summarize_rows(rows: list[dict[str, Any]], mode: str, scope: str | None = None, model_name: str | None = None) -> dict[str, Any]:
    success_scores = [int(row["success_score"]) for row in rows]
    successful_rows = [row for row in rows if int(row["success_score"]) == 1]
    total_runs = len(rows)
    source_hit_values = [float(row["source_hit_at_k"]) for row in rows if row.get("source_hit_at_k") is not None]
    source_recall_values = [float(row["source_recall_at_k"]) for row in rows if row.get("source_recall_at_k") is not None]
    coverage_values = [float(row["coverage_at_k"]) for row in rows if row.get("coverage_at_k") is not None]
    rank_values = [float(row["rank_of_gold_source"]) for row in rows if row.get("rank_of_gold_source") is not None]
    mrr_values = [float(row["mrr"]) for row in rows if row.get("mrr") is not None]
    detection_values = [float(row["answerable_detection_correct"]) for row in rows if row.get("answerable_detection_correct") is not None]
    refusal_values = [float(row["refusal_correct"]) for row in rows if row.get("refusal_correct") is not None]

    return {
        "mode": mode,
        "model_name": model_name,
        "scope": scope,
        "total_runs": total_runs,
        "successful_runs": sum(success_scores),
        "failed_runs": total_runs - sum(success_scores),
        "success_rate": _mean([float(value) for value in success_scores]),
        "avg_latency_total": _mean([float(row["latency_total"]) for row in successful_rows]),
        "avg_latency_retrieval": _mean([float(row["latency_retrieval"]) for row in successful_rows]),
        "avg_latency_generation": _mean([float(row["latency_generation"]) for row in successful_rows]),
        "source_hit_at_k_rate": _mean(source_hit_values),
        "avg_source_recall_at_k": _mean(source_recall_values),
        "avg_coverage_at_k": _mean(coverage_values),
        "avg_rank_of_gold_source": _mean(rank_values),
        "mrr": _mean(mrr_values),
        "answerable_detection_accuracy": _mean(detection_values),
        "refusal_accuracy": _mean(refusal_values),
    }


def build_objective_eval_summary(
    evaluated_rows: list[dict[str, Any]],
    *,
    top_k: int,
    retrieval_match_level: str = "page",
    answer_runs_file: str = "",
    question_dataset_file: str = "",
) -> dict[str, Any]:
    by_mode: list[dict[str, Any]] = []
    unique_modes = sorted({str(row.get("mode") or "").strip() for row in evaluated_rows if str(row.get("mode") or "").strip()})
    for mode in unique_modes:
        mode_rows = [row for row in evaluated_rows if str(row.get("mode") or "").strip() == mode]
        mode_summary = _summarize_rows(mode_rows, mode)
        by_scope: list[dict[str, Any]] = []
        unique_scopes = sorted({str(row.get("scope") or "unknown") for row in mode_rows})
        for scope in unique_scopes:
            scope_rows = [row for row in mode_rows if str(row.get("scope") or "unknown") == scope]
            by_scope.append(_summarize_rows(scope_rows, mode, scope))
        mode_summary["by_scope"] = by_scope
        by_mode.append(mode_summary)

    by_model_mode: list[dict[str, Any]] = []
    unique_model_modes = sorted(
        {
            (str(row.get("model_name") or "").strip(), str(row.get("mode") or "").strip())
            for row in evaluated_rows
            if str(row.get("mode") or "").strip()
        }
    )
    for model_name, mode in unique_model_modes:
        rows = [
            row
            for row in evaluated_rows
            if str(row.get("model_name") or "").strip() == model_name
            and str(row.get("mode") or "").strip() == mode
        ]
        by_model_mode.append(_summarize_rows(rows, mode, model_name=model_name))

    match_level = str(retrieval_match_level or "page").strip().lower()
    coverage_methods = sorted({str(row.get("coverage_eval_method") or "").strip() for row in evaluated_rows if str(row.get("coverage_eval_method") or "").strip()})
    coverage_embedding_models = sorted({str(row.get("coverage_eval_embedding_model") or "").strip() for row in evaluated_rows if str(row.get("coverage_eval_embedding_model") or "").strip()})
    return {
        "generated_at": utc_timestamp(),
        "top_k": int(top_k or 0),
        "retrieval_match_level": match_level,
        "evidence_match_policy": "half_of_valid_anchor_groups" if match_level == "evidence" else None,
        "coverage_eval_method": coverage_methods[0] if len(coverage_methods) == 1 else coverage_methods,
        "coverage_eval_embedding_model": coverage_embedding_models[0] if len(coverage_embedding_models) == 1 else coverage_embedding_models or None,
        "answer_runs_file": str(answer_runs_file or ""),
        "question_dataset_file": str(question_dataset_file or ""),
        "total_runs": len(evaluated_rows),
        "by_mode": by_mode,
        "by_model_mode": by_model_mode,
    }


def write_jsonl_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
