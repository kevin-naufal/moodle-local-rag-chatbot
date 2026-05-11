from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ms_to_seconds(value_ms: int | float) -> float:
    value = max(0.0, float(value_ms or 0.0))
    return round(value / 1000.0, 3)


def normalize_embedding_backend(mode: str, backend: str | None) -> str | None:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode == "llm_only":
        return "none"

    text = str(backend or "").strip().lower()
    if text in {"bert", "ollama"}:
        return text
    return text or None


def build_raw_result_payload(
    *,
    question_id: str,
    question: str,
    mode: str,
    run_id: int,
    model_name: str,
    embedding_backend: str | None,
    model_answer: str,
    retrieved_context: list[dict[str, Any]] | None,
    latency_total_ms: int | float,
    latency_retrieval_ms: int | float,
    latency_generation_ms: int | float,
    status: str,
    error_message: str | None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    return {
        "question_id": str(question_id or "").strip(),
        "question": str(question or "").strip(),
        "mode": str(mode or "").strip(),
        "run_id": int(run_id or 0),
        "model_name": str(model_name or "").strip(),
        "embedding_backend": normalize_embedding_backend(mode, embedding_backend),
        "model_answer": str(model_answer or ""),
        "retrieved_context": list(retrieved_context or []),
        "latency_total": ms_to_seconds(latency_total_ms),
        "latency_retrieval": ms_to_seconds(latency_retrieval_ms),
        "latency_generation": ms_to_seconds(latency_generation_ms),
        "status": str(status or "success").strip().lower(),
        "error_message": str(error_message) if error_message is not None else None,
        "timestamp": str(timestamp or utc_timestamp()),
    }
