from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4


DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "can", "for", "from",
    "has", "have", "if", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "we", "when", "which", "while", "with", "why", "what",
    "how", "does", "do", "not", "than", "then", "they", "them", "into", "about", "only",
    "also", "other", "most", "more", "such", "these", "those", "using", "used", "use",
    "will", "would", "should", "could", "may", "might", "your", "you", "our",
}


def create_output_path(base_dir: Path, prefix: str, suffix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = uuid4().hex[:6]
    return base_dir / f"{prefix}_{stamp}_{token}.{suffix}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSONL row at line {line_number}: expected object.")
        rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-zA-Z0-9]+", normalize_text(text)) if tok and tok not in DEFAULT_STOPWORDS]


def keyword_set(text: str) -> set[str]:
    return set(tokenize(text))


def split_sentences(text: str) -> list[str]:
    cleaned = str(text or "").replace("\r", "\n")
    parts = re.split(r"(?:\n{2,}|[\.\?\!]\s+|\n- |\n\* )", cleaned)
    items = [part.strip(" -\n\t") for part in parts if part and part.strip(" -\n\t")]
    return items


def overlap_ratio(source: str, target: str) -> float:
    source_keys = keyword_set(source)
    target_keys = keyword_set(target)
    if not source_keys or not target_keys:
        return 0.0
    return len(source_keys & target_keys) / max(1, len(source_keys))


def quantize_tenth(value: float) -> float:
    clipped = max(0.0, min(1.0, float(value)))
    return round(math.floor((clipped * 10.0) + 0.5) / 10.0, 1)


def count_distinct_content_sentences(text: str) -> int:
    seen: set[str] = set()
    count = 0
    for sentence in split_sentences(text):
        normalized = normalize_text(sentence)
        if len(normalized) < 20:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        count += 1
    return count


def compute_gold_coverage(answer: str, gold_points: list[str]) -> tuple[int, float, list[float]]:
    if not gold_points:
        return 0, 0.0, []
    per_point: list[float] = []
    covered = 0
    for point in gold_points:
        score = overlap_ratio(point, answer)
        per_point.append(score)
        if score >= 0.45:
            covered += 1
    return covered, covered / max(1, len(gold_points)), per_point


def compute_context_support(answer: str, retrieved_context: list[dict]) -> tuple[float, int]:
    if not retrieved_context:
        return 0.0, 0
    context_text = "\n".join(str(item.get("text") or "") for item in retrieved_context if isinstance(item, dict))
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0, 0
    supports: list[float] = []
    unsupported = 0
    for sentence in sentences:
        if len(keyword_set(sentence)) < 3:
            continue
        score = overlap_ratio(sentence, context_text)
        supports.append(score)
        if score < 0.18:
            unsupported += 1
    if not supports:
        return 0.0, 0
    return sum(supports) / len(supports), unsupported


def compute_question_focus(question: str, answer: str) -> float:
    qkeys = keyword_set(question)
    akeys = keyword_set(answer)
    if not qkeys or not akeys:
        return 0.0
    return len(qkeys & akeys) / max(1, min(len(qkeys), 8))


def detect_question_type(question: str) -> str:
    lowered = normalize_text(question)
    if lowered.startswith("why "):
        return "why"
    if lowered.startswith("how "):
        return "how"
    if lowered.startswith("what ") or lowered.startswith("what are") or lowered.startswith("what is"):
        return "what"
    return "other"


def score_clarity(answer: str) -> float:
    words = len(str(answer or "").split())
    sentence_count = max(1, count_distinct_content_sentences(answer))
    has_bullets = 1 if re.search(r"(^|\n)\s*[-*]", str(answer or "")) else 0
    duplicate_penalty = 0.15 if len(split_sentences(answer)) - sentence_count >= 1 else 0.0
    length_bonus = 0.15 if 20 <= words <= 130 else (0.05 if 10 <= words <= 180 else -0.1)
    structure_bonus = 0.1 if has_bullets else 0.0
    base = 0.65 + length_bonus + structure_bonus - duplicate_penalty
    return quantize_tenth(base)


def score_instruction_compliance(answer: str) -> float:
    text = str(answer or "")
    if not text.strip():
        return 0.0
    score = 1.0
    if "<think>" in text.lower():
        score -= 0.4
    if "?" in text and len(text.split()) < 30:
        score -= 0.2
    return quantize_tenth(score)


def score_scaffolding(answer: str, question_type: str) -> float:
    text = str(answer or "")
    has_steps = bool(re.search(r"(^|\n)\s*[-*]|\bfirst\b|\bsecond\b|\bfor example\b|\bthis means\b", text.lower()))
    has_explanation = bool(re.search(r"\bbecause\b|\bso that\b|\btherefore\b|\bthis means\b", text.lower()))
    base = 0.2
    if has_steps:
        base += 0.3
    if has_explanation:
        base += 0.2
    if question_type == "how":
        base += 0.1 if has_steps else 0.0
    return quantize_tenth(base)


def score_actionability(answer: str) -> float:
    text = str(answer or "").lower()
    action_markers = [
        "you should", "try", "consider", "review", "focus on", "check", "look for",
        "next", "step", "practice", "compare",
    ]
    hits = sum(1 for marker in action_markers if marker in text)
    if hits <= 0:
        return 0.1
    if hits == 1:
        return 0.3
    if hits == 2:
        return 0.5
    return 0.7


def score_need_alignment(question_type: str, completeness: float, relevance: float, clarity: float, answer: str) -> float:
    words = len(str(answer or "").split())
    size_fit = 0.1 if 15 <= words <= 140 else 0.0
    type_bonus = 0.1 if question_type in {"what", "why", "how"} else 0.0
    base = (completeness * 0.35) + (relevance * 0.35) + (clarity * 0.2) + size_fit + type_bonus
    return quantize_tenth(min(base, 1.0))


def maybe_contradiction_penalty(answer: str, question_id: str) -> float:
    text = normalize_text(answer)
    penalties: dict[str, list[str]] = {
        "ch03-full-q01": ["prioritizing efficiency over simplicity", "choose an algorithm that runs quickly", "sacrificing some ease"],
        "ch03-full-q21": ["exponential growth of quadratic"],
    }
    if question_id not in penalties:
        return 0.0
    return 0.25 if any(fragment in text for fragment in penalties[question_id]) else 0.0


def build_question_lookup(dataset: dict) -> dict[str, dict]:
    return {
        str(item.get("id") or "").strip(): item
        for item in list(dataset.get("questions") or [])
        if str(item.get("id") or "").strip()
    }


def judge_row(run: dict, spec: dict, default_scope: str) -> dict:
    question_id = str(run.get("question_id") or "").strip()
    question = str(run.get("question") or spec.get("question") or "").strip()
    answer = str(run.get("model_answer") or "").strip()
    gold_points = list(spec.get("gold_points") or [])
    retrieved_context = list(run.get("retrieved_context") or [])
    scope = str(spec.get("scope") or default_scope or "unknown").strip()
    question_type = detect_question_type(question)

    covered, coverage_rate, per_point = compute_gold_coverage(answer, gold_points)
    question_focus = compute_question_focus(question, answer)
    context_support, unsupported_sentences = compute_context_support(answer, retrieved_context)
    contradiction_penalty = maybe_contradiction_penalty(answer, question_id)

    correctness_raw = (coverage_rate * 0.65) + (question_focus * 0.15)
    if retrieved_context:
        correctness_raw += context_support * 0.2
    else:
        correctness_raw += 0.1
    correctness_raw -= contradiction_penalty

    completeness_raw = min(1.0, coverage_rate + (0.1 if count_distinct_content_sentences(answer) >= max(1, len(gold_points)) else 0.0))
    if len(gold_points) >= 3 and covered == 1:
        completeness_raw = min(completeness_raw, 0.45)

    groundedness_raw = context_support if retrieved_context else max(0.15, coverage_rate * 0.45)
    if retrieved_context and unsupported_sentences > 0:
        groundedness_raw -= min(0.25, unsupported_sentences * 0.05)
    if not retrieved_context and unsupported_sentences > 0:
        groundedness_raw -= min(0.15, unsupported_sentences * 0.04)

    relevance_raw = min(1.0, (question_focus * 0.55) + (coverage_rate * 0.35) + 0.15)
    if len(answer.split()) <= 3:
        relevance_raw -= 0.1

    answer_correctness = quantize_tenth(correctness_raw)
    answer_completeness = quantize_tenth(completeness_raw)
    answer_groundedness = quantize_tenth(groundedness_raw)
    answer_relevance = quantize_tenth(relevance_raw)
    answer_clarity = score_clarity(answer)
    instruction_compliance = score_instruction_compliance(answer)
    scaffolding_quality = score_scaffolding(answer, question_type)
    pedagogical_actionability = quantize_tenth(score_actionability(answer))
    need_alignment = score_need_alignment(
        question_type,
        answer_completeness,
        answer_relevance,
        answer_clarity,
        answer,
    )

    unsupported_claim_count = unsupported_sentences
    if not retrieved_context and answer_correctness <= 0.4 and len(answer.split()) > 80:
        unsupported_claim_count += 1
    must_not_claim_violations = 0
    refusal_appropriateness = None

    quality_score = quantize_tenth(
        (answer_correctness * 0.4)
        + (answer_completeness * 0.2)
        + (answer_groundedness * 0.25)
        + (answer_relevance * 0.15)
    )

    if quality_score >= 0.8:
        judge_label = "high_quality"
    elif quality_score >= 0.5:
        judge_label = "medium_quality"
    else:
        judge_label = "low_quality"

    strongest_point = max(range(len(per_point)), key=lambda idx: per_point[idx]) if per_point else None
    strongest_text = gold_points[strongest_point] if strongest_point is not None and strongest_point < len(gold_points) else ""
    judge_reason_parts = [
        f"Covered {covered}/{len(gold_points)} key points" if gold_points else "No gold points available",
        f"relevance {answer_relevance:.1f}",
        f"groundedness {answer_groundedness:.1f}",
    ]
    if strongest_text:
        judge_reason_parts.append(f"strongest match: {strongest_text[:80]}")
    judge_reason = "; ".join(judge_reason_parts) + "."

    return {
        "question_id": question_id,
        "question": question,
        "mode": str(run.get("mode") or "").strip(),
        "run_id": int(run.get("run_id") or 0),
        "model_name": str(run.get("model_name") or "").strip(),
        "embedding_backend": run.get("embedding_backend"),
        "embedding_model_name": str(run.get("embedding_model_name") or "").strip() or None,
        "scope": scope,
        "expected_behavior": "answer",
        "status": str(run.get("status") or "").strip().lower() or "success",
        "latency_total": run.get("latency_total"),
        "latency_retrieval": run.get("latency_retrieval"),
        "latency_generation": run.get("latency_generation"),
        "answer_correctness": answer_correctness,
        "answer_completeness": answer_completeness,
        "answer_groundedness": answer_groundedness,
        "answer_relevance": answer_relevance,
        "refusal_appropriateness": refusal_appropriateness,
        "answer_clarity": answer_clarity,
        "instruction_compliance": instruction_compliance,
        "need_alignment": need_alignment,
        "scaffolding_quality": scaffolding_quality,
        "pedagogical_actionability": pedagogical_actionability,
        "key_points_total": len(gold_points),
        "key_points_covered": covered,
        "unsupported_claim_count": unsupported_claim_count,
        "must_not_claim_violations": must_not_claim_violations,
        "judge_label": judge_label,
        "judge_reason": judge_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-fill answer quality judgments from answer runs and gold points.")
    parser.add_argument("--questions", required=True, help="Path to question dataset JSON.")
    parser.add_argument("--answer-runs", required=True, help="Path to answer runs JSONL.")
    parser.add_argument("--output", default="", help="Optional output JSONL path.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "quality_eval_inputs"

    questions_path = Path(args.questions).resolve()
    answer_runs_path = Path(args.answer_runs).resolve()
    output_path = Path(args.output).resolve() if str(args.output).strip() else create_output_path(output_dir, "judged_answer_runs_auto", "jsonl")

    dataset = load_json(questions_path)
    answer_runs = load_jsonl(answer_runs_path)
    question_lookup = build_question_lookup(dataset)
    default_scope = str(dataset.get("scope") or "unknown").strip() or "unknown"

    judged_rows = [
        judge_row(run, question_lookup.get(str(run.get("question_id") or "").strip(), {}), default_scope)
        for run in answer_runs
    ]
    write_jsonl(output_path, judged_rows)
    print("Auto answer-quality judging completed.")
    print(f"- output: {output_path}")
    print(f"- total_rows: {len(judged_rows)}")


if __name__ == "__main__":
    main()
