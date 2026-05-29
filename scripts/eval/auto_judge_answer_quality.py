from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4


DEFAULT_QUALITY_EVAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_FULL_COVERAGE_THRESHOLD = 0.72
SEMANTIC_PARTIAL_COVERAGE_THRESHOLD = 0.60
SEMANTIC_PARTIAL_COVERAGE_CREDIT = 0.75
SEMANTIC_UNSUPPORTED_THRESHOLD = 0.55


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "Semantic answer-quality evaluation requires `sentence-transformers`. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(vector) for vector in vectors]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


class SemanticQualityEvaluator:
    method = "semantic_embedding_v1"

    def __init__(self, embedder, model_name: str):
        self.embedder = embedder
        self.model_name = str(model_name or "").strip()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        cleaned = [str(text or "").strip() for text in texts]
        if not cleaned:
            return []
        return self.embedder.encode(cleaned)

    def _best_similarity(self, source: str, candidates: list[str]) -> float:
        source_text = str(source or "").strip()
        candidate_texts = [str(candidate or "").strip() for candidate in candidates if str(candidate or "").strip()]
        if not source_text or not candidate_texts:
            return 0.0
        vectors = self._embed([source_text] + candidate_texts)
        source_vector = vectors[0]
        return max(cosine_similarity(source_vector, candidate_vector) for candidate_vector in vectors[1:])

    def compute_gold_coverage(self, answer: str, gold_points: list[str]) -> tuple[int, float, list[float]]:
        points = [str(point or "").strip() for point in gold_points if str(point or "").strip()]
        if not points:
            return 0, 0.0, []

        answer_units = split_sentences(answer)
        full_answer = str(answer or "").strip()
        if full_answer:
            answer_units.append(full_answer)

        per_point: list[float] = []
        covered = 0
        coverage_credit = 0.0
        for point in points:
            score = self._best_similarity(point, answer_units)
            per_point.append(score)
            if score >= SEMANTIC_FULL_COVERAGE_THRESHOLD:
                covered += 1
                coverage_credit += 1.0
            elif score >= SEMANTIC_PARTIAL_COVERAGE_THRESHOLD:
                coverage_credit += SEMANTIC_PARTIAL_COVERAGE_CREDIT
        return covered, coverage_credit / max(1, len(points)), per_point

    def compute_question_focus(self, question: str, answer: str) -> float:
        return self._best_similarity(question, [answer])

    def compute_context_support(self, answer: str, retrieved_context: list[dict]) -> tuple[float, int]:
        details = self.compute_context_support_details(answer, retrieved_context, [])
        return details["context_support_raw"], details["unsupported_sentence_count"]

    def compute_context_support_details(
        self,
        answer: str,
        retrieved_context: list[dict],
        gold_points: list[str],
    ) -> dict[str, float | int]:
        context_units: list[str] = []
        seen_context_units: set[str] = set()
        for item in retrieved_context:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            candidates = [
                sentence
                for sentence in split_sentences(text)
                if len(sentence.split()) >= 4
            ]
            candidates.append(text)
            for candidate in candidates:
                normalized = normalize_text(candidate)
                if not normalized or normalized in seen_context_units:
                    continue
                seen_context_units.add(normalized)
                context_units.append(candidate)
        if not context_units:
            return {
                "context_support_raw": 0.0,
                "core_context_support": 0.0,
                "supported_sentence_ratio": 0.0,
                "unsupported_sentence_count": 0,
                "unsupported_extra_claim_count": 0,
            }

        sentence_units = [
            sentence
            for sentence in split_sentences(answer)
            if len(sentence.split()) >= 4
        ]
        if not sentence_units:
            return {
                "context_support_raw": 0.0,
                "core_context_support": 0.0,
                "supported_sentence_ratio": 0.0,
                "unsupported_sentence_count": 0,
                "unsupported_extra_claim_count": 0,
            }

        scores = [self._best_similarity(sentence, context_units) for sentence in sentence_units]
        unsupported = sum(1 for score in scores if score < SEMANTIC_UNSUPPORTED_THRESHOLD)
        supported = len(scores) - unsupported

        core_points = [
            str(point or "").strip()
            for point in gold_points
            if str(point or "").strip()
            and self._best_similarity(str(point or "").strip(), sentence_units) >= SEMANTIC_PARTIAL_COVERAGE_THRESHOLD
        ]
        if core_points:
            core_scores = [self._best_similarity(point, context_units) for point in core_points]
            core_context_support = sum(core_scores) / len(core_scores)
        else:
            core_context_support = sum(scores) / len(scores)

        unsupported_extra_claim_count = 0
        point_units = [str(point or "").strip() for point in gold_points if str(point or "").strip()]
        for sentence, score in zip(sentence_units, scores):
            if score >= SEMANTIC_UNSUPPORTED_THRESHOLD:
                continue
            if point_units and self._best_similarity(sentence, point_units) >= SEMANTIC_PARTIAL_COVERAGE_THRESHOLD:
                continue
            unsupported_extra_claim_count += 1

        return {
            "context_support_raw": sum(scores) / len(scores),
            "core_context_support": core_context_support,
            "supported_sentence_ratio": supported / len(scores),
            "unsupported_sentence_count": unsupported,
            "unsupported_extra_claim_count": unsupported_extra_claim_count,
        }


def load_semantic_quality_evaluator(model_name: str | None = None) -> SemanticQualityEvaluator:
    resolved_model = str(model_name or DEFAULT_QUALITY_EVAL_EMBED_MODEL).strip() or DEFAULT_QUALITY_EVAL_EMBED_MODEL
    return SemanticQualityEvaluator(SentenceTransformerEmbedder(resolved_model), resolved_model)


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


def split_sentences(text: str) -> list[str]:
    cleaned = str(text or "").replace("\r", "\n")
    parts = re.split(r"(?:\n{2,}|[\.\?\!]\s+|\n- |\n\* )", cleaned)
    items = [part.strip(" -\n\t") for part in parts if part and part.strip(" -\n\t")]
    return items


def split_paragraphs(text: str) -> list[str]:
    cleaned = str(text or "").replace("\r", "\n").strip()
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]


def extract_alpha_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", str(text or ""))


def count_bullet_items(text: str) -> int:
    return len(re.findall(r"(?m)^\s*[-*•]\s+", str(text or "")))


def count_numbered_items(text: str) -> int:
    return len(re.findall(r"(?m)^\s*\d+[\.\)]\s+", str(text or "")))


def count_sequence_markers(text: str) -> int:
    markers = re.findall(
        r"\b(first|second|third|next|then|finally|step\s+\d+)\b",
        str(text or "").lower(),
    )
    return len(markers)


def quantize_tenth(value: float) -> float:
    clipped = max(0.0, min(1.0, float(value)))
    return round(math.floor((clipped * 10.0) + 0.5) / 10.0, 1)


def gold_point_coverage_status(similarity: float) -> str:
    score = float(similarity or 0.0)
    if score >= SEMANTIC_FULL_COVERAGE_THRESHOLD:
        return "full"
    if score >= SEMANTIC_PARTIAL_COVERAGE_THRESHOLD:
        return "partial"
    return "miss"


def gold_point_coverage_credit(similarity: float) -> float:
    status = gold_point_coverage_status(similarity)
    if status == "full":
        return 1.0
    if status == "partial":
        return SEMANTIC_PARTIAL_COVERAGE_CREDIT
    return 0.0


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


def detect_style_requirements(question: str) -> dict[str, object]:
    lowered = normalize_text(question)
    requirements: dict[str, object] = {}

    bullet_match = re.search(r"\bexactly\s+(\d+)\s+bullet points?\b", lowered)
    if bullet_match:
        requirements["exact_bullet_count"] = int(bullet_match.group(1))
    elif "bullet points" in lowered:
        requirements["require_bullets"] = True

    paragraph_match = re.search(r"\b(\d+)\s+short paragraphs?\b", lowered)
    if paragraph_match:
        requirements["exact_paragraph_count"] = int(paragraph_match.group(1))
        requirements["require_short_paragraphs"] = True
    elif re.search(r"\b(\d+)\s+paragraphs?\b", lowered):
        generic_match = re.search(r"\b(\d+)\s+paragraphs?\b", lowered)
        if generic_match:
            requirements["exact_paragraph_count"] = int(generic_match.group(1))

    if "numbered list" in lowered:
        requirements["require_numbered_list"] = True

    if lowered.startswith("list "):
        requirements["require_list_format"] = True

    if "answer briefly" in lowered or "briefly" in lowered or "brief answer" in lowered:
        requirements["require_brief"] = True

    if "simple language" in lowered or "plain language" in lowered or "to a beginner" in lowered or "for a beginner" in lowered:
        requirements["require_simple_language"] = True

    if "step-by-step" in lowered or "step by step" in lowered:
        requirements["require_step_by_step"] = True

    return requirements


def get_format_constraint(spec: dict) -> str | None:
    raw = str(spec.get("format_constraint") or "").strip().lower()
    if raw in {"", "none", "null", "n/a", "na"}:
        return None
    return raw


def score_format_compliance(answer: str, format_constraint: str | None) -> float | None:
    if not format_constraint:
        return None

    text = str(answer or "")
    bullet_count = count_bullet_items(text)
    numbered_count = count_numbered_items(text)
    paragraph_count = len(split_paragraphs(text))

    if format_constraint == "bullet_3":
        return 1.0 if bullet_count == 3 else 0.0
    if format_constraint == "numbered_list":
        return 1.0 if numbered_count >= 2 else 0.0
    if format_constraint == "paragraph_2":
        return 1.0 if paragraph_count == 2 else 0.0
    if format_constraint == "bullet_list":
        return 1.0 if bullet_count >= 1 else 0.0

    return None


def score_instruction_compliance(question: str, answer: str) -> float:
    text = str(answer or "")
    if not text.strip():
        return 0.0

    score = 1.0
    if "<think>" in text.lower():
        score -= 0.4
    if "?" in text and len(text.split()) < 30:
        score -= 0.2

    requirements = detect_style_requirements(question)
    bullet_count = count_bullet_items(text)
    numbered_count = count_numbered_items(text)
    paragraph_count = len(split_paragraphs(text))
    word_count = len(text.split())
    alpha_words = extract_alpha_words(text)
    sentence_count = max(1, len(split_sentences(text)))
    average_sentence_words = len(alpha_words) / sentence_count if alpha_words else 0.0
    long_word_ratio = (
        sum(1 for word in alpha_words if len(word) >= 10) / len(alpha_words)
        if alpha_words
        else 0.0
    )

    exact_bullet_count = requirements.get("exact_bullet_count")
    if isinstance(exact_bullet_count, int):
        if bullet_count != exact_bullet_count:
            score -= 0.5
    elif requirements.get("require_bullets") and bullet_count <= 0:
        score -= 0.4

    if requirements.get("require_numbered_list"):
        if numbered_count <= 0:
            score -= 0.5 if bullet_count <= 0 else 0.3

    if requirements.get("require_list_format"):
        if max(bullet_count, numbered_count) <= 0:
            score -= 0.4

    exact_paragraph_count = requirements.get("exact_paragraph_count")
    if isinstance(exact_paragraph_count, int) and paragraph_count != exact_paragraph_count:
        score -= 0.4

    if requirements.get("require_short_paragraphs"):
        paragraphs = split_paragraphs(text)
        if any(len(paragraph.split()) > 90 for paragraph in paragraphs):
            score -= 0.2

    if requirements.get("require_brief"):
        if word_count > 90:
            score -= 0.4
        elif word_count > 60:
            score -= 0.2

    if requirements.get("require_simple_language"):
        if average_sentence_words > 24:
            score -= 0.2
        if long_word_ratio > 0.22:
            score -= 0.2

    if requirements.get("require_step_by_step"):
        sequence_count = count_sequence_markers(text)
        has_step_structure = numbered_count >= 2 or bullet_count >= 2 or sequence_count >= 2
        if not has_step_structure:
            score -= 0.5

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


def build_question_lookup(dataset: dict) -> dict[str, dict]:
    return {
        str(item.get("id") or "").strip(): item
        for item in list(dataset.get("questions") or [])
        if str(item.get("id") or "").strip()
    }


def judge_row(run: dict, spec: dict, default_scope: str, *, evaluator: SemanticQualityEvaluator | None = None) -> dict:
    if evaluator is None:
        raise ValueError("Semantic quality evaluator is required; legacy keyword evaluation is not supported.")

    question_id = str(run.get("question_id") or "").strip()
    question = str(run.get("question") or spec.get("question") or "").strip()
    answer = str(run.get("model_answer") or "").strip()
    gold_points = list(spec.get("gold_points") or [])
    retrieved_context = list(run.get("retrieved_context") or [])
    mode = str(run.get("mode") or "").strip()
    groundedness_applicable = mode != "llm_only"
    scope = str(spec.get("scope") or default_scope or "unknown").strip()

    covered, coverage_rate, per_point = evaluator.compute_gold_coverage(answer, gold_points)
    gold_point_similarities = [
        {
            "gold_point_index": index,
            "gold_point": str(point or "").strip(),
            "similarity": round(float(score), 4),
            "coverage_status": gold_point_coverage_status(float(score)),
            "coverage_credit": gold_point_coverage_credit(float(score)),
        }
        for index, (point, score) in enumerate(zip(gold_points, per_point), start=1)
    ]
    question_focus = evaluator.compute_question_focus(question, answer)
    context_details = evaluator.compute_context_support_details(answer, retrieved_context, gold_points)
    context_support = float(context_details["context_support_raw"])
    core_context_support = float(context_details["core_context_support"])
    unsupported_sentences = int(context_details["unsupported_sentence_count"])
    unsupported_extra_claims = int(context_details["unsupported_extra_claim_count"])
    # Correctness is intentionally source-agnostic here:
    # score only by gold-point coverage and question focus.
    correctness_raw = (coverage_rate * 0.85) + (question_focus * 0.15)

    completeness_raw = min(1.0, coverage_rate + (0.1 if count_distinct_content_sentences(answer) >= max(1, len(gold_points)) else 0.0))
    if len(gold_points) >= 3 and covered == 1 and coverage_rate < 0.5:
        completeness_raw = min(completeness_raw, 0.45)

    groundedness_raw = None
    if groundedness_applicable:
        if retrieved_context:
            groundedness_raw = (core_context_support * 0.7) + (context_support * 0.3)
        else:
            groundedness_raw = max(0.15, coverage_rate * 0.45)
        if retrieved_context and unsupported_extra_claims > 0:
            groundedness_raw -= min(0.2, unsupported_extra_claims * 0.05)
        if not retrieved_context and unsupported_sentences > 0:
            groundedness_raw -= min(0.15, unsupported_sentences * 0.04)

    relevance_raw = min(1.0, (question_focus * 0.55) + (coverage_rate * 0.35) + 0.15)
    if len(answer.split()) <= 3:
        relevance_raw -= 0.1

    answer_correctness = quantize_tenth(correctness_raw)
    answer_completeness = quantize_tenth(completeness_raw)
    answer_groundedness = quantize_tenth(groundedness_raw) if groundedness_raw is not None else None
    answer_relevance = quantize_tenth(relevance_raw)
    question_type = detect_question_type(question)
    format_constraint = get_format_constraint(spec)
    format_compliance = score_format_compliance(answer, format_constraint)
    answer_clarity = score_clarity(answer)
    instruction_compliance = score_instruction_compliance(question, answer)
    scaffolding_quality = score_scaffolding(answer, question_type)
    pedagogical_actionability = score_actionability(answer)
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

    quality_components = [
        (0.4, answer_correctness),
        (0.2, answer_completeness),
        (0.25, answer_groundedness),
        (0.15, answer_relevance),
    ]
    usable_components = [(weight, value) for weight, value in quality_components if value is not None]
    total_weight = sum(weight for weight, _ in usable_components)
    quality_score = quantize_tenth(
        sum(weight * value for weight, value in usable_components) / total_weight
    ) if total_weight > 0 else 0.0

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
        f"groundedness {answer_groundedness:.1f}" if answer_groundedness is not None else "groundedness N/A",
    ]
    if strongest_text:
        judge_reason_parts.append(f"strongest match: {strongest_text[:80]}")
    judge_reason = "; ".join(judge_reason_parts) + "."

    return {
        "question_id": question_id,
        "question": question,
        "mode": mode,
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
        "format_constraint": format_constraint,
        "format_compliance": format_compliance,
        "instruction_compliance": instruction_compliance,
        "need_alignment": need_alignment,
        "scaffolding_quality": scaffolding_quality,
        "pedagogical_actionability": pedagogical_actionability,
        "key_points_total": len(gold_points),
        "key_points_covered": covered,
        "gold_point_similarities": gold_point_similarities,
        "gold_point_similarity_thresholds": {
            "full": SEMANTIC_FULL_COVERAGE_THRESHOLD,
            "partial": SEMANTIC_PARTIAL_COVERAGE_THRESHOLD,
            "partial_credit": SEMANTIC_PARTIAL_COVERAGE_CREDIT,
        },
        "unsupported_claim_count": unsupported_claim_count,
        "must_not_claim_violations": must_not_claim_violations,
        "judge_label": judge_label,
        "judge_reason": judge_reason,
        "context_support_raw": round(context_support, 4),
        "core_context_support": round(core_context_support, 4),
        "supported_sentence_ratio": round(float(context_details["supported_sentence_ratio"]), 4),
        "unsupported_sentence_count": unsupported_sentences,
        "unsupported_extra_claim_count": unsupported_extra_claims,
        "quality_eval_method": evaluator.method,
        "quality_eval_embedding_model": evaluator.model_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-fill answer quality judgments from answer runs and gold points.")
    parser.add_argument("--questions", required=True, help="Path to question dataset JSON.")
    parser.add_argument("--answer-runs", required=True, help="Path to answer runs JSONL.")
    parser.add_argument("--output", default="", help="Optional output JSONL path.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "quality_eval_inputs"

    questions_path = Path(args.questions).resolve()
    answer_runs_path = Path(args.answer_runs).resolve()
    output_path = Path(args.output).resolve() if str(args.output).strip() else create_output_path(output_dir, "judged_answer_runs_auto", "jsonl")

    dataset = load_json(questions_path)
    answer_runs = load_jsonl(answer_runs_path)
    question_lookup = build_question_lookup(dataset)
    default_scope = str(dataset.get("scope") or "unknown").strip() or "unknown"
    evaluator = load_semantic_quality_evaluator(os.getenv("QUALITY_EVAL_EMBED_MODEL"))

    judged_rows = [
        judge_row(run, question_lookup.get(str(run.get("question_id") or "").strip(), {}), default_scope, evaluator=evaluator)
        for run in answer_runs
    ]
    write_jsonl(output_path, judged_rows)
    print("Auto answer-quality judging completed.")
    print(f"- output: {output_path}")
    print(f"- total_rows: {len(judged_rows)}")


if __name__ == "__main__":
    main()
