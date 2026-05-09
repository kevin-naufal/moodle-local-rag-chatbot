import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
import traceback
from typing import Any
import urllib.error
import urllib.request
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""Moodle RAG Runner.
Digunakan plugin Moodle untuk menjalankan retrieval + jawaban model dan mengembalikan JSON.
"""


EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"
RELEVANCE_THRESHOLD = 0.2
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
EMPTY_ANSWER_FALLBACK = "Sorry, I cannot provide an answer for that question yet."
CHAT_NUM_PREDICT = int(os.getenv("CHAT_NUM_PREDICT", "2048"))
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "auto").strip().lower()
BERT_MODEL = os.getenv("BERT_MODEL", "bert-base-uncased").strip()
BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "256"))
BERT_BATCH_SIZE = int(os.getenv("BERT_BATCH_SIZE", "16"))
INDEX_COLLECTION_NAME = "moodle_chatbot_docs"
INDEX_DIR_NAME = ".rag_chroma"
INDEX_MANIFEST_NAME = ".rag_index_manifest.json"
TRACE_TEXT_MAX_CHARS = int(os.getenv("TRACE_TEXT_MAX_CHARS", "8000"))

PROMPT_TEMPLATE = """You are a careful assistant. Use ONLY the following context to answer the question.
If the answer is not in the context, say "Not found in context."
Answer directly and concisely. Do not start with "Based on the context".
Never output internal reasoning tags like <think>.
Do not claim you cannot access files; file content is already provided in context.
Answer only the concept asked in the question; ignore unrelated examples in the context.
If context coverage is thin, say that briefly instead of expanding with outside details.
Return the final answer in Markdown format.
Use bullet points only when they improve readability.

Context:
{context}

Question: {question}
"""

GENERAL_PROMPT_TEMPLATE = """You are a helpful assistant.
Answer the user's question directly and concisely.
Never output internal reasoning tags like <think>.
Return the final answer in Markdown format.
Do not ask follow-up or clarification questions.
If details are missing, give a best-effort answer and clearly state assumptions.
Use bullet points only when they improve readability.

Question: {question}
"""


def now_ms() -> int:
    return int(round(time.time() * 1000))


def trim_error(error: str, max_len: int = 4000) -> str:
    text = str(error or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def truncate_text(text: str, max_len: int = 8000) -> tuple[str, bool]:
    value = str(text or "")
    if len(value) <= max_len:
        return value, False
    return value[:max_len] + "...(truncated)", True


class TraceLogger:
    def __init__(
        self,
        log_path: str | None,
        request_id: str = "",
        question_number: int = 0,
        attempt: int = 0,
    ) -> None:
        self._path = Path(log_path) if log_path else None
        self._request_id = request_id.strip()
        self._question_number = int(question_number or 0)
        self._attempt = int(attempt or 0)
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, level: str = "info", **fields: Any) -> None:
        if self._path is None:
            return
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ts_ms": now_ms(),
            "layer": "python",
            "event": event,
            "level": level,
            "request_id": self._request_id,
            "question_number": self._question_number,
            "attempt": self._attempt,
        }
        payload.update(fields)
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # Tracing must never break normal response path.
            pass


class BertEmbeddings(Embeddings):
    """Hugging Face BERT embeddings with mean pooling."""

    def __init__(self, model_name: str, max_length: int = 256, batch_size: int = 16):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "BERT embedding backend requires `transformers` and `torch`. "
                "Install with: pip install transformers torch"
            ) from exc

        model_name = model_name.strip() or "bert-base-uncased"
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()
        self._max_length = max(32, min(int(max_length), 512))
        self._batch_size = max(1, min(int(batch_size), 128))

    def _mean_pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        torch = self._torch
        expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * expanded_mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        torch = self._torch
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = self._model(**encoded)

            pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(normalized.cpu().tolist())
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned = [str(text or "") for text in texts]
        return self._embed_batch(cleaned)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed_batch([str(text or "")])
        return vectors[0] if vectors else []


def build_embeddings() -> tuple[Embeddings, str]:
    backend = EMBED_BACKEND
    if backend not in {"auto", "bert", "ollama"}:
        backend = "auto"

    if backend in {"auto", "bert"}:
        try:
            return BertEmbeddings(
                model_name=BERT_MODEL,
                max_length=BERT_MAX_LENGTH,
                batch_size=BERT_BATCH_SIZE,
            ), "bert"
        except Exception as exc:
            if backend == "bert":
                raise RuntimeError(f"BERT embedding initialization failed: {exc}") from exc

    # Fallback/default: Ollama embeddings.
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL), "ollama"


def source_label(doc) -> str:
    source = Path(str(doc.metadata.get("source", "unknown"))).name
    page = doc.metadata.get("page")
    if page is None:
        return source
    return f"{source} p.{int(page) + 1}"


def load_docs(data_dir: Path):
    docs = []
    for file_path in sorted(data_dir.iterdir(), key=lambda p: p.name.lower()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(file_path), autodetect_encoding=True).load())
        elif file_path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(file_path)).load())
    return docs


def normalize_page_range(page_start: int, page_end: int) -> tuple[int | None, int | None]:
    start = int(page_start or 0)
    end = int(page_end or 0)
    if start <= 0 and end <= 0:
        return None, None
    if start <= 0 and end > 0:
        start = end
    if end <= 0 and start > 0:
        end = start
    if end < start:
        start, end = end, start
    return start, end


def build_page_metadata_filter(page_start: int | None, page_end: int | None) -> dict[str, Any] | None:
    if page_start is None or page_end is None:
        return None
    # PyPDFLoader metadata page is 0-based; UI range is 1-based.
    start0 = max(0, int(page_start) - 1)
    end0 = max(0, int(page_end) - 1)
    return {"$and": [{"page": {"$gte": start0}}, {"page": {"$lte": end0}}]}


def filter_docs_by_page_range(docs, page_start: int | None, page_end: int | None):
    if page_start is None or page_end is None:
        return docs
    start0 = max(0, int(page_start) - 1)
    end0 = max(0, int(page_end) - 1)
    filtered = []
    for doc in docs:
        page = doc.metadata.get("page")
        if page is None:
            continue
        try:
            page_num = int(page)
        except (TypeError, ValueError):
            continue
        if start0 <= page_num <= end0:
            filtered.append(doc)
    return filtered


def smalltalk_response(query: str) -> str | None:
    normalized = query.strip().lower()
    if normalized in {"tes", "test", "ping"}:
        return "System is active. Ask a specific question about your documents."

    # Only treat as smalltalk when greeting appears as a standalone word
    # and the full query is short. This prevents false positives such as
    # "Ethics" containing "hi" as a substring.
    if "how are you" in normalized:
        return (
            "Hello. I can help with questions about your uploaded documents. "
            "Try asking about a specific topic or page."
        )

    if len(normalized) <= 40 and re.fullmatch(
        r"\s*(hello|hi|halo|hey)(\s+[a-z]+)?\s*[.!?]?\s*",
        normalized,
        flags=re.IGNORECASE,
    ):
        return (
            "Hello. I can help with questions about your uploaded documents. "
            "Try asking about a specific topic or page."
        )

    return None


def split_chat_style_and_question(query: str) -> tuple[str, str]:
    text = str(query or "").strip()
    if not text:
        return "", ""

    lowered = text.lower()
    marker = "\nquestion:"
    if lowered.startswith("chat mode:") and marker in lowered:
        idx = lowered.rfind(marker)
        style = text[:idx].strip()
        user_question = text[idx + len(marker):].strip()
        if user_question:
            return style, user_question
    return "", text


def clean_answer(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip() or EMPTY_ANSWER_FALLBACK


def is_unusable_answer(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    if normalized == EMPTY_ANSWER_FALLBACK.lower():
        return True
    if normalized.startswith(EMPTY_ANSWER_FALLBACK.lower()):
        return True
    if normalized in {"## answer", "answer:"}:
        return True
    return False


def is_assignment_generation_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = [
        "judul tugas:",
        "tujuan pembelajaran:",
        "instruksi untuk siswa:",
        "daftar soal:",
        "kunci jawaban:",
        "rubrik penilaian:",
        "assignment title:",
        "learning objectives:",
        "instructions for students:",
        "question list:",
        "answer key:",
        "grading rubric:",
    ]
    return sum(1 for marker in markers if marker in lowered) >= 4


def is_practice_generation_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    practice_markers = [
        "additional notes: practice mode",
        "generate practice questions",
        "practice questions in english",
        "practice mode, designed for self-learning",
        "practice phase: question-bank-only",
    ]
    return any(marker in lowered for marker in practice_markers)


def is_practice_question_bank_only_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = [
        "practice phase: question-bank-only",
        "question-bank-only",
        "question list:",
        "answer key:",
        "no explanations. no introductions. no extra sections.",
    ]
    return "question-bank-only" in lowered or (
        ("question list:" in lowered and "answer key:" in lowered)
        and any(marker in lowered for marker in markers)
    )


def get_section_text(answer: str, starts: list[str], ends: list[str]) -> str:
    lowered = answer.lower()
    start_index = -1
    for marker in starts:
        idx = lowered.find(marker)
        if idx >= 0:
            start_index = idx
            break
    if start_index < 0:
        return ""
    end_index = len(answer)
    for marker in ends:
        idx = lowered.find(marker, start_index + 1)
        if idx >= 0:
            end_index = min(end_index, idx)
    return answer[start_index:end_index]


def extract_expected_count_from_prompt(prompt: str) -> int:
    patterns = [
        r"(?:number of questions/components|jumlah soal/komponen)\s*(?::|=)?\s*(\d+)",
        r"(?:create exactly|buat tepat)\s*(\d+)\s*(?:multiple-choice|essay|case-study|soal|pertanyaan)",
        r"(?:continue until)\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1))
            return max(0, min(50, value))
    return 0


def detect_assignment_type(prompt: str) -> str:
    lowered = prompt.lower()
    if re.search(r"(?:assignment type|jenis tugas)\s*(?::|=)?\s*multiple[\s-]?choice", lowered):
        return "multiple-choice"
    if re.search(r"(?:assignment type|jenis tugas)\s*(?::|=)?\s*pilihan\s+ganda", lowered):
        return "multiple-choice"
    if re.search(r"(?:assignment type|jenis tugas)\s*(?::|=)?\s*essay", lowered):
        return "essay"
    if re.search(r"(?:assignment type|jenis tugas)\s*(?::|=)?\s*case[\s-]?study", lowered):
        return "case-study"
    if "multiple-choice questions" in lowered:
        return "multiple-choice"
    return "unknown"


def is_multiple_choice_assignment_prompt(prompt: str) -> bool:
    return detect_assignment_type(prompt) == "multiple-choice"


def build_assignment_format_guardrails(prompt: str) -> str:
    expected_count = extract_expected_count_from_prompt(prompt) or 5
    assignment_type = detect_assignment_type(prompt)
    lowered_prompt = prompt.lower()
    is_presentation_assignment = "presentation assignment" in lowered_prompt
    base_rules = (
        "\n\nSTRICT OUTPUT RULES:\n"
        "- Return Markdown only.\n"
        "- Keep section order exactly: Assignment Title, Learning Objectives, "
        "Instructions for Students, Question List, Answer Key, Grading Rubric.\n"
        "- Do not use placeholders like [due date], [insert], or [tbd].\n"
        "- Ensure all numbering is sequential and starts at 1.\n"
    )
    if assignment_type == "multiple-choice":
        return base_rules + (
            f"- Create exactly {expected_count} multiple-choice questions.\n"
            "- For EACH question, include exactly 4 options: A., B., C., D.\n"
            "- Use numbered questions in this format: `1. Question text`.\n"
            "- Answer Key MUST use this exact line format only: `1. A`.\n"
            "- Never use mapping/cross-reference format like `1. 2 (D)`.\n"
            "- The Answer Key must contain one line per question and only A/B/C/D letters.\n"
            "- Keep question numbers and answer-key numbers aligned one-to-one.\n"
        )

    if assignment_type == "essay":
        if is_presentation_assignment:
            return base_rules + (
                f"- Create exactly {expected_count} presentation topics/components.\n"
                "- Use numbered lines in this format: `1. Topic/subtopic for slides`.\n"
                "- Do NOT include options A/B/C/D.\n"
                "- Do NOT put inline answers inside Question List (never write `Answer:` in Question List).\n"
                "- Answer Key MUST use this exact numbered format: `1. Key points: ...`.\n"
                "- Keep Answer Key concise as internal teacher guidance, not full model answers.\n"
                "- Keep question numbers and answer-key numbers aligned one-to-one.\n"
            )
        return base_rules + (
            f"- Create exactly {expected_count} essay questions.\n"
            "- Use numbered questions in this format: `1. Question text`.\n"
            "- Do NOT include options A/B/C/D.\n"
            "- Do NOT put inline answers inside Question List (never write `Answer:` in Question List).\n"
            "- Answer Key MUST use this exact numbered format: `1. Key points: ...`.\n"
            "- Provide concise expected key points for each essay answer.\n"
            "- Keep question numbers and answer-key numbers aligned one-to-one.\n"
        )

    if assignment_type == "case-study":
        return base_rules + (
            f"- Create exactly {expected_count} case-study questions/components.\n"
            "- Use numbered questions in this format: `1. Case prompt + task`.\n"
            "- Do NOT include options A/B/C/D unless explicitly requested.\n"
            "- Answer Key MUST use numbered lines: `1. Expected analysis points: ...`.\n"
            "- Keep question numbers and answer-key numbers aligned one-to-one.\n"
        )

    return base_rules


def build_practice_format_guardrails(prompt: str) -> str:
    expected_count = extract_expected_count_from_prompt(prompt) or 5
    question_bank_only = is_practice_question_bank_only_prompt(prompt)
    section_order = (
        "Question List, Answer Key"
        if question_bank_only
        else "Assignment Title, Learning Objectives, Instructions for Students, Question List, Answer Key, Grading Rubric"
    )
    extra_rule = (
        "- Do not include Assignment Title, Learning Objectives, Instructions for Students, or Grading Rubric.\n"
        if question_bank_only
        else ""
    )
    return (
        "\n\nSTRICT PRACTICE OUTPUT RULES:\n"
        "- Return Markdown only.\n"
        f"- Keep section order exactly: {section_order}.\n"
        f"- Create exactly {expected_count} multiple-choice practice questions.\n"
        "- In Question List, each question must include exactly 4 options: A), B), C), D).\n"
        "- Answer Key MUST use this exact numbered format: `1. A`.\n"
        "- Do not include explanations inside Answer Key.\n"
        f"{extra_rule}"
        "- Keep question numbers and answer-key numbers aligned one-to-one.\n"
        "- Do not use placeholders like [due date], [insert], or [tbd].\n"
    )


def has_core_assignment_sections(answer: str) -> bool:
    lowered = answer.lower()
    required_markers_id = [
        "judul tugas",
        "tujuan pembelajaran",
        "instruksi untuk siswa",
        "daftar soal",
        "kunci jawaban",
        "rubrik penilaian",
    ]
    required_markers_en = [
        "assignment title",
        "learning objectives",
        "instructions for students",
        "question list",
        "answer key",
        "grading rubric",
    ]
    has_all_required = all(marker in lowered for marker in required_markers_id) or all(
        marker in lowered for marker in required_markers_en
    )
    return has_all_required and len(answer.strip()) >= 250


def has_min_practice_sections(answer: str, prompt: str = "") -> bool:
    question_bank_only = is_practice_question_bank_only_prompt(prompt)
    lowered = answer.lower()
    if not question_bank_only and "assignment title" not in lowered and "judul tugas" not in lowered:
        return False

    question_section = get_section_text(
        answer,
        ["question list", "questions", "daftar soal", "soal"],
        ["answer key", "kunci jawaban", "correct answer", "jawaban benar"],
    )
    answer_key_section = get_section_text(
        answer,
        ["answer key", "kunci jawaban", "correct answer", "jawaban benar"],
        ["grading rubric", "rubrik penilaian"],
    )
    if not question_section or not answer_key_section:
        return False

    expected_count = extract_expected_count_from_prompt(prompt)
    if expected_count <= 0:
        expected_count = count_question_items(question_section)
    if expected_count <= 0:
        return False

    question_count = count_question_items(question_section)
    if question_count != expected_count:
        return False
    numbered_questions = re.findall(
        r"(?mi)^\s*(?:question\s*)?(\d+)\s*[.)]\s+",
        question_section,
    )
    if len(numbered_questions) != expected_count:
        return False
    if [int(item) for item in numbered_questions] != list(range(1, expected_count + 1)):
        return False
    if not has_strict_multiple_choice_answer_key(answer_key_section, expected_count):
        return False

    for option in ["A", "B", "C", "D"]:
        option_count = len(
            re.findall(rf"(?mi)^\s*(?:[-*]\s*)?{option}\s*[.):]\s*", question_section)
        )
        if option_count < expected_count:
            return False

    if re.search(r"\[(?:due date|insert|tbd)\]", answer, flags=re.IGNORECASE):
        return False

    return True


def count_question_items(question_section: str) -> int:
    marker_count = len(
        re.findall(
            r"(?mi)^\s*(?:question\s*\d+\s*[:.)]|pertanyaan\s*\d+\s*[:.)]|q\s*\d+\s*[:.)]|\d+\s*[.)]\s+)",
            question_section,
        )
    )
    # Fallback for outputs that omit numbering but keep one line per question.
    unnumbered_question_like_count = len(re.findall(r"(?mi)^\s*[^\n]{4,}\?\s*$", question_section))
    return max(marker_count, unnumbered_question_like_count)


def has_strict_multiple_choice_answer_key(answer_key_section: str, expected_count: int) -> bool:
    if expected_count <= 0:
        return False
    # Reject malformed cross-reference style, e.g. "1. 2 (D)".
    if re.search(r"(?mi)^\s*\d+\s*[.)\-:]\s*\d+\s*\([A-D]\)\s*$", answer_key_section):
        return False

    matches = re.findall(
        r"(?mi)^\s*(\d+)\s*[.)\-:]\s*([A-D])(?:\s*[\).:\-].*)?\s*$",
        answer_key_section,
    )
    if len(matches) != expected_count:
        return False
    expected_numbers = list(range(1, expected_count + 1))
    actual_numbers = [int(number) for number, _ in matches]
    return actual_numbers == expected_numbers


def has_min_assignment_sections(answer: str, prompt: str = "") -> bool:
    if not has_core_assignment_sections(answer) or len(answer.strip()) < 350:
        return False

    question_section = get_section_text(
        answer,
        ["question list", "daftar soal"],
        ["answer key", "kunci jawaban"],
    )
    answer_key_section = get_section_text(
        answer,
        ["answer key", "kunci jawaban"],
        ["grading rubric", "rubrik penilaian"],
    )
    if not question_section or not answer_key_section:
        return False

    assignment_type = detect_assignment_type(prompt)
    expected_count = extract_expected_count_from_prompt(prompt)
    if expected_count <= 0:
        expected_count = count_question_items(question_section)
    if expected_count <= 0:
        return True

    question_count = count_question_items(question_section)
    if question_count != expected_count:
        return False

    if assignment_type == "multiple-choice":
        if not has_strict_multiple_choice_answer_key(answer_key_section, expected_count):
            return False

        for option in ["A", "B", "C", "D"]:
            option_count = len(
                re.findall(rf"(?mi)^\s*(?:[-*]\s*)?{option}\s*[.)]\s*", question_section)
            )
            if option_count < expected_count:
                return False
    elif assignment_type in {"essay", "case-study"}:
        # For non-MC assignments, accept numbered lines OR bullets as key-points.
        numbered_keys = len(
            re.findall(r"(?mi)^\s*(?:question\s*\d+\s*[:.)]|\d+\s*[.)-]\s+)", answer_key_section)
        )
        bullet_keys = len(re.findall(r"(?mi)^\s*[-*]\s+", answer_key_section))
        if max(numbered_keys, bullet_keys) < expected_count:
            return False
    else:
        key_markers = re.findall(r"(?mi)^\s*\d+\s*[.):-]\s*", answer_key_section)
        if len(key_markers) < expected_count:
            return False

    if re.search(r"\[(?:due date|insert|tbd)\]", answer, flags=re.IGNORECASE):
        return False

    return True


def strip_leading_boilerplate(answer: str) -> str:
    stripped = answer.strip()
    patterns = [
        r"^\s*however,\s*(?:based on|from)\s+the\s+provided\s+context[:,]?\s*",
        r"^\s*based on\s+the\s+provided\s+context[:,]?\s*",
        r"^\s*however[:,]?\s*",
    ]
    for pattern in patterns:
        stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def ensure_markdown_answer(answer: str) -> str:
    stripped = strip_leading_boilerplate(answer)
    if not stripped:
        return "## Answer\n\nSorry, I cannot provide an answer for that question yet."

    has_markdown_block = re.search(
        r"(?m)^\s{0,3}(#{1,6}\s+\S|[-*+]\s+\S|\d+\.\s+\S|>\s+\S|```)",
        stripped,
    )
    if has_markdown_block:
        return stripped
    return f"## Answer\n\n{stripped}"


def ensure_plain_answer(answer: str) -> str:
    stripped = strip_leading_boilerplate(answer)
    if not stripped:
        return EMPTY_ANSWER_FALLBACK

    # Remove markdown heading wrapper that may be introduced upstream.
    stripped = re.sub(r"(?mi)^\s*##\s*answer\s*$", "", stripped).strip()
    # Remove one fenced-code wrapper if model returns JSON in a code block.
    match = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", stripped, flags=re.IGNORECASE | re.DOTALL)
    if match:
        stripped = match.group(1).strip()
    return stripped or EMPTY_ANSWER_FALLBACK


def list_source_files(data_dir: Path) -> list[Path]:
    files = []
    for file_path in sorted(data_dir.iterdir(), key=lambda p: p.name.lower()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() in {".txt", ".pdf"}:
            files.append(file_path)
    return files


def build_data_signature(files: list[Path]) -> str:
    parts: list[str] = []
    for file_path in files:
        try:
            stat = file_path.stat()
        except OSError:
            continue
        parts.append(f"{file_path.name}:{stat.st_size}:{int(stat.st_mtime)}")
    return "|".join(parts)


def read_index_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_index_manifest(manifest_path: Path, signature: str, chunk_count: int) -> None:
    payload = {
        "signature": signature,
        "chunk_count": int(max(0, chunk_count)),
        "updated_at": int(time.time()),
    }
    try:
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        # Best effort only. Index still usable even if manifest write fails.
        return


def build_vectorstore_from_docs(docs, embeddings: Embeddings) -> Chroma | None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    if not splits:
        return None
    return Chroma.from_documents(documents=splits, embedding=embeddings)


def load_or_build_cached_vectorstore(
    data_dir: Path,
    docs,
    embeddings: Embeddings,
) -> tuple[Chroma | None, bool]:
    source_files = list_source_files(data_dir)
    signature = build_data_signature(source_files)
    if not signature:
        return None, False

    index_dir = data_dir / INDEX_DIR_NAME
    manifest_path = data_dir / INDEX_MANIFEST_NAME
    manifest = read_index_manifest(manifest_path)
    is_cache_fresh = (
        index_dir.exists()
        and index_dir.is_dir()
        and manifest.get("signature") == signature
    )

    if is_cache_fresh:
        try:
            cached = Chroma(
                persist_directory=str(index_dir),
                collection_name=INDEX_COLLECTION_NAME,
                embedding_function=embeddings,
            )
            return cached, False
        except Exception:
            # Fallback to rebuild when cache directory is corrupted/incompatible.
            pass

    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    if not splits:
        return None, False

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(index_dir),
        collection_name=INDEX_COLLECTION_NAME,
    )
    write_index_manifest(manifest_path, signature, len(splits))
    return vectorstore, True


def normalize_lookup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_single_source(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return TextLoader(str(file_path), autodetect_encoding=True).load()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    return []


def query_mentions_file(query: str) -> bool:
    return re.search(r"\b[\w.\-]+\.(txt|pdf)\b", query, flags=re.IGNORECASE) is not None


def find_explicit_source(query: str, files: list[Path]) -> Path | None:
    lowered = query.lower()
    normalized_query = normalize_lookup_key(query)
    candidates: list[tuple[int, Path]] = []

    for file_path in files:
        file_name = file_path.name.lower()
        file_stem = file_path.stem.lower()
        if file_name in lowered:
            candidates.append((len(file_name), file_path))
            continue
        # Allow users to reference a file without extension, e.g. "info1" -> "info1.txt".
        if file_stem and re.search(rf"\b{re.escape(file_stem)}\b", lowered):
            candidates.append((len(file_stem), file_path))
            continue
        normalized_file_name = normalize_lookup_key(file_name)
        if normalized_file_name and normalized_file_name in normalized_query:
            candidates.append((len(normalized_file_name), file_path))
            continue
        normalized_file_stem = normalize_lookup_key(file_stem)
        if normalized_file_stem and normalized_file_stem in normalized_query:
            candidates.append((len(normalized_file_stem), file_path))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def format_context(docs) -> str:
    chunks = []
    for doc in docs:
        chunks.append(f"[Source: {source_label(doc)}]\n{doc.page_content}")
    return "\n\n".join(chunks)


def extract_focus_terms(query: str) -> list[str]:
    lowered = query.lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9\-]{1,}", lowered)
    stopwords = {
        "apa", "itu", "yang", "dan", "atau", "untuk", "dengan", "dari", "pada",
        "jelaskan", "tolong", "dong", "gimana", "bagaimana", "adalah",
        "what", "is", "the", "a", "an", "of", "to", "in", "on", "for", "about",
        "explain", "define", "please", "me",
    }
    focus = []
    seen = set()
    for token in tokens:
        if token in stopwords:
            continue
        if token in seen:
            continue
        seen.add(token)
        focus.append(token)
    return focus[:8]


def _normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", text.lower())


def token_soft_match(term: str, token: str) -> bool:
    if term == token:
        return True
    if len(term) >= 5 and token.startswith(term[:5]):
        return True
    if len(token) >= 5 and term.startswith(token[:5]):
        return True
    return False


def keyword_focus_score(query: str, text: str) -> float:
    terms = extract_focus_terms(query)
    if not terms:
        return 0.0

    normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    text_tokens = _normalize_tokens(normalized_text)
    match_count = 0
    for term in terms:
        if any(token_soft_match(term, token) for token in text_tokens):
            match_count += 1
    coverage = match_count / max(1, len(terms))

    phrase_bonus = 0.0
    if len(terms) >= 2:
        phrase = " ".join(terms[:2])
        if phrase in normalized_text:
            phrase_bonus = 0.35

    return coverage + phrase_bonus


def answer_focus_coverage(query: str, answer: str) -> float:
    terms = extract_focus_terms(query)
    if not terms:
        return 1.0
    tokens = _normalize_tokens(answer)
    if not tokens:
        return 0.0
    hits = 0
    for term in terms:
        if any(token_soft_match(term, token) for token in tokens):
            hits += 1
    return hits / max(1, len(terms))


def get_relevant_docs(
    vectorstore: Chroma,
    query: str,
    metadata_filter: dict[str, Any] | None = None,
):
    # Score is expected in range [0, 1], where larger means more relevant.
    pairs = vectorstore.similarity_search_with_relevance_scores(query, k=8, filter=metadata_filter)
    if not pairs:
        return []

    filtered = [(doc, score) for doc, score in pairs if score >= RELEVANCE_THRESHOLD]
    if not filtered:
        filtered = pairs[:4]

    scored = []
    for doc, score in filtered:
        focus = keyword_focus_score(query, doc.page_content)
        combined = float(score) + (focus * 0.65)
        scored.append((combined, float(score), doc))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [doc for _, _, doc in scored[:4]]


def emit(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))


def check_ollama(base_url: str, timeout: float = 3.0, trace: TraceLogger | None = None) -> tuple[bool, str]:
    url = f"{base_url}/api/tags"
    request = urllib.request.Request(url=url, method="GET")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 0)
            if 200 <= status < 300:
                if trace is not None:
                    trace.log(
                        "ollama_healthcheck_success",
                        url=url,
                        status=int(status),
                        duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    )
                return True, ""
            if trace is not None:
                trace.log(
                    "ollama_healthcheck_error",
                    level="error",
                    url=url,
                    status=int(status),
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    error=f"Ollama health check returned HTTP {status}.",
                )
            return False, f"Ollama health check returned HTTP {status}."
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if trace is not None:
            trace.log(
                "ollama_healthcheck_error",
                level="error",
                url=url,
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                error=trim_error(str(exc)),
            )
        return False, str(exc)


def ollama_unreachable_message(base_url: str, details: str) -> str:
    return (
        f"RAG backend error: Failed to connect to Ollama at {base_url}. "
        f"Start Ollama with `ollama serve`, then verify with "
        f"`ollama list`. Details: {details}"
    )


def ask_general(llm: ChatOllama, query: str, markdown: bool = True, trace: TraceLogger | None = None) -> str:
    prompt = GENERAL_PROMPT_TEMPLATE.format(question=query)
    answer = invoke_llm_with_retry(llm, prompt, retries=1, trace=trace)
    if markdown:
        return ensure_markdown_answer(answer)
    return ensure_plain_answer(answer)


def build_rag_prompt(context: str, question: str, style_instruction: str = "") -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    style = style_instruction.strip()
    if not style:
        return prompt
    return (
        prompt
        + "\n\nResponse style instruction (apply silently):\n"
        + style
        + "\nDo not restate or explain the style instruction. "
        + "Answer only the user question content."
    )


def invoke_llm_with_retry(
    llm: ChatOllama,
    prompt: str,
    retries: int = 1,
    trace: TraceLogger | None = None,
) -> str:
    practice_mode = is_practice_generation_prompt(prompt)
    assignment_mode = is_assignment_generation_prompt(prompt) and not practice_mode
    structured_mode = assignment_mode or practice_mode
    assignment_type = detect_assignment_type(prompt)
    assignment_guardrails = build_assignment_format_guardrails(prompt) if assignment_mode else ""
    practice_guardrails = build_practice_format_guardrails(prompt) if practice_mode else ""
    last_answer = EMPTY_ANSWER_FALLBACK
    max_retries = 2 if structured_mode else max(0, retries)
    max_llm_calls = max_retries + 1
    llm_calls = 0

    def invoke_once(current_prompt: str) -> str | None:
        nonlocal llm_calls
        if llm_calls >= max_llm_calls:
            return None
        llm_calls += 1
        started = time.perf_counter()
        try:
            response = llm.invoke(current_prompt)
            rawanswer = response.content if hasattr(response, "content") else str(response)
            cleaned = clean_answer(str(rawanswer))
            if trace is not None:
                trace.log(
                    "ollama_llm_invoke_success",
                    llm_call=llm_calls,
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    prompt_chars=len(current_prompt),
                    answer_chars=len(cleaned),
                )
            return cleaned
        except Exception as exc:
            if trace is not None:
                trace.log(
                    "ollama_llm_invoke_error",
                    level="error",
                    llm_call=llm_calls,
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    prompt_chars=len(current_prompt),
                    error=trim_error(str(exc)),
                )
            raise

    attempts = max_llm_calls
    for attempt in range(attempts):
        current_prompt = prompt + assignment_guardrails + practice_guardrails
        if attempt > 0:
            current_prompt = (
                prompt
                + assignment_guardrails
                + practice_guardrails
                + "\n\nIMPORTANT: Your previous answer was empty or unusable. "
                + "Return the final answer now in Markdown only, without <think> tags."
            )
            if assignment_mode:
                current_prompt += (
                    "\nThe answer MUST include these sections with real content: "
                    "Assignment Title, Learning Objectives, Instructions for Students, "
                    "Question List, Answer Key, Grading Rubric."
                )
            if practice_mode:
                if is_practice_question_bank_only_prompt(prompt):
                    current_prompt += (
                        "\nThe answer MUST include these sections with real content: "
                        "Assignment Title, Question List, Answer Key."
                    )
                else:
                    current_prompt += (
                        "\nThe answer MUST include these sections with real content: "
                        "Assignment Title, Learning Objectives, Instructions for Students, "
                        "Question List, Answer Key, Grading Rubric."
                    )
        candidate = invoke_once(current_prompt)
        if candidate is None:
            break
        last_answer = candidate
        if is_unusable_answer(last_answer):
            continue
        if assignment_mode and not has_min_assignment_sections(last_answer, prompt):
            continue
        if practice_mode and not has_min_practice_sections(last_answer, prompt):
            continue
        return last_answer

    if assignment_mode and not is_unusable_answer(last_answer) and llm_calls < max_llm_calls:
        repair_prompt = (
            prompt
            + assignment_guardrails
            + "\n\nYour previous draft is incomplete.\n"
            + "Previous draft:\n"
            + last_answer
            + "\n\nRewrite from scratch in Markdown with complete sections: "
            + "Assignment Title, Learning Objectives, Instructions for Students, "
            + "Question List, Answer Key, Grading Rubric."
        )
        repaired_answer = invoke_once(repair_prompt)
        if repaired_answer is not None and not is_unusable_answer(repaired_answer) and has_min_assignment_sections(repaired_answer, prompt):
            return repaired_answer

    if practice_mode and not is_unusable_answer(last_answer) and llm_calls < max_llm_calls:
        practice_question_bank_only = is_practice_question_bank_only_prompt(prompt)
        required_sections = (
            "Question List, Answer Key"
            if practice_question_bank_only
            else "Assignment Title, Learning Objectives, Instructions for Students, Question List, Answer Key, Grading Rubric"
        )
        repair_prompt = (
            prompt
            + practice_guardrails
            + "\n\nYour previous draft is incomplete.\n"
            + "Previous draft:\n"
            + last_answer
            + "\n\nRewrite from scratch in Markdown with complete sections: "
            + required_sections
        )
        repaired_answer = invoke_once(repair_prompt)
        if repaired_answer is not None and not is_unusable_answer(repaired_answer) and has_min_practice_sections(repaired_answer, prompt):
            return repaired_answer

    if assignment_mode and llm_calls < max_llm_calls:
        expected_count = extract_expected_count_from_prompt(prompt) or 5
        rescue_header = (
            "Create a complete Moodle assignment draft in English.\n"
            "Use this exact structure only:\n"
            "Assignment Title:\n"
            "Learning Objectives:\n"
            "Instructions for Students:\n"
            "Question List:\n"
            "Answer Key:\n"
            "Grading Rubric:\n"
        )
        if assignment_type == "multiple-choice":
            rescue_rules = (
                f"Create exactly {expected_count} multiple-choice questions.\n"
                "Each question must include A), B), C), D).\n"
                "Answer Key format must be concise: 1. A\n"
                "Never use format like: 1. 2 (D)\n"
            )
        elif assignment_type == "essay":
            rescue_rules = (
                f"Create exactly {expected_count} essay questions.\n"
                "Do not include A/B/C/D options.\n"
                "Answer Key format must be concise: 1. Key points: ...\n"
            )
        elif assignment_type == "case-study":
            rescue_rules = (
                f"Create exactly {expected_count} case-study questions/components.\n"
                "Do not include A/B/C/D options unless explicitly requested.\n"
                "Answer Key format must be concise: 1. Expected analysis points: ...\n"
            )
        else:
            rescue_rules = (
                f"Create exactly {expected_count} numbered questions/components.\n"
                "Answer Key must use numbered lines aligned with Question List.\n"
            )

        rescue_prompt = (
            rescue_header
            + rescue_rules
            + "Do not use placeholders like [due date].\n\n"
            + "Reference request:\n"
            + prompt
            + assignment_guardrails
        )
        rescued_answer = invoke_once(rescue_prompt)
        if rescued_answer is not None and not is_unusable_answer(rescued_answer) and has_min_assignment_sections(rescued_answer, prompt):
            return rescued_answer

    if practice_mode and llm_calls < max_llm_calls:
        expected_count = extract_expected_count_from_prompt(prompt) or 5
        if is_practice_question_bank_only_prompt(prompt):
            rescue_prompt = (
                "Create a complete Moodle practice question bank draft in English.\n"
                "Use this exact structure only:\n"
                "Question List:\n"
                "Answer Key:\n"
                f"Create exactly {expected_count} multiple-choice questions.\n"
                "Each question must include A), B), C), D).\n"
                "Answer Key format must be concise: 1. A\n"
                "Do not include explanations inside Answer Key.\n\n"
                "Reference request:\n"
                + prompt
                + practice_guardrails
            )
        else:
            rescue_prompt = (
                "Create a complete Moodle practice quiz draft in English.\n"
                "Use this exact structure only:\n"
                "Assignment Title:\n"
                "Learning Objectives:\n"
                "Instructions for Students:\n"
                "Question List:\n"
                "Answer Key:\n"
                "Grading Rubric:\n"
                f"Create exactly {expected_count} multiple-choice questions.\n"
                "Each question must include A), B), C), D).\n"
                "Answer Key format must be concise: 1. A\n"
                "Do not include explanations inside Answer Key.\n\n"
                "Reference request:\n"
                + prompt
                + practice_guardrails
            )
        rescued_answer = invoke_once(rescue_prompt)
        if rescued_answer is not None and not is_unusable_answer(rescued_answer) and has_min_practice_sections(rescued_answer, prompt):
            return rescued_answer

    if assignment_mode and not has_min_assignment_sections(last_answer, prompt):
        if has_core_assignment_sections(last_answer):
            return (
                last_answer
                + "\n\nNote: This draft may be incomplete. You can click Regenerate for a cleaner structure."
            )
        return "Assignment draft is incomplete after retries. Please click Regenerate to try again."
    if practice_mode and not has_min_practice_sections(last_answer, prompt):
        if not is_unusable_answer(last_answer):
            return (
                last_answer
                + "\n\nNote: Practice draft may be incomplete. You can click Regenerate for a cleaner structure."
            )
        return "Practice draft is incomplete after retries. Please click Regenerate to try again."
    return last_answer


def main() -> None:
    # Ambil parameter dari CLI (dipanggil oleh Moodle plugin).
    parser = argparse.ArgumentParser(description="RAG runner for Moodle local_chatbot plugin")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--query")
    parser.add_argument("--query-b64")
    parser.add_argument("--mode", choices=["auto", "general", "general_raw"], default="auto")
    parser.add_argument("--preparse", action="store_true")
    parser.add_argument("--request-id", default="")
    parser.add_argument("--question-number", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--page-start", type=int, default=0)
    parser.add_argument("--page-end", type=int, default=0)
    parser.add_argument("--trace-log", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    trace = TraceLogger(
        log_path=str(args.trace_log or "").strip() or None,
        request_id=str(args.request_id or "").strip(),
        question_number=int(args.question_number or 0),
        attempt=int(args.attempt or 0),
    )

    try:
        data_dir = Path(args.data_dir)
        page_start, page_end = normalize_page_range(args.page_start, args.page_end)
        page_filter = build_page_metadata_filter(page_start, page_end)
        trace.log(
            "python_request_start",
            mode=str(args.mode),
            preparse=bool(args.preparse),
            data_dir=str(data_dir),
            page_start=page_start,
            page_end=page_end,
        )

        def emit_answer_payload(answer_text: str, sources_list: list[str], emit_mode: str = "") -> None:
            safe_sources = list(sources_list or [])
            final_answer = str(answer_text or "")
            answer_text_log, answer_truncated = truncate_text(final_answer, TRACE_TEXT_MAX_CHARS)
            trace.log(
                "python_response_emit",
                answer_chars=len(final_answer),
                answer_text=answer_text_log,
                answer_truncated=bool(answer_truncated),
                sources_count=len(safe_sources),
                mode=emit_mode,
            )
            emit({"answer": final_answer, "sources": safe_sources})

        if args.preparse:
            if not data_dir.exists():
                trace.log(
                    "python_request_success",
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    preparse=True,
                    sources=0,
                )
                emit({"ok": True, "preparsed": False, "rebuilt": False, "sources": 0})
                return
            docs = load_docs(data_dir)
            if not docs:
                trace.log(
                    "python_request_success",
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    preparse=True,
                    sources=0,
                )
                emit({"ok": True, "preparsed": False, "rebuilt": False, "sources": 0})
                return
            embeddings, embed_backend = build_embeddings()
            vectorstore, rebuilt = load_or_build_cached_vectorstore(data_dir, docs, embeddings)
            if vectorstore is None:
                trace.log(
                    "python_request_success",
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    preparse=True,
                    sources=0,
                )
                emit({"ok": True, "preparsed": False, "rebuilt": False, "sources": 0})
                return
            source_count = len(list_source_files(data_dir))
            trace.log(
                "python_request_success",
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                preparse=True,
                sources=source_count,
                embedding_backend=embed_backend,
                rebuilt=bool(rebuilt),
            )
            emit(
                {
                    "ok": True,
                    "preparsed": True,
                    "rebuilt": bool(rebuilt),
                    "sources": source_count,
                    "embedding_backend": embed_backend,
                }
            )
            return

        # Bangun query final, lalu validasi input kosong.
        query = args.query or ""
        if args.query_b64:
            query = base64.b64decode(args.query_b64).decode("utf-8", errors="ignore")
        trace.log(
            "python_query_ready",
            query_chars=len(query),
        )
        if not query.strip():
            trace.log(
                "python_request_success",
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                answer_chars=len("Question is empty."),
                sources=0,
            )
            emit_answer_payload("Question is empty.", [], "empty_query")
            return
        style_instruction, semantic_query = split_chat_style_and_question(query)
        query_for_answer = semantic_query if semantic_query else query
        query_log, query_truncated = truncate_text(query_for_answer, TRACE_TEXT_MAX_CHARS)
        trace.log(
            "python_query_text",
            query_text=query_log,
            query_truncated=bool(query_truncated),
        )

        # Shortcut smalltalk supaya response cepat tanpa proses RAG.
        smalltalk = smalltalk_response(query_for_answer)
        if smalltalk is not None:
            trace.log(
                "python_request_success",
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                answer_chars=len(smalltalk),
                sources=0,
                mode="smalltalk",
            )
            emit_answer_payload(smalltalk, [], "smalltalk")
            return

        ollama_ok, ollama_details = check_ollama(OLLAMA_BASE_URL, trace=trace)
        if not ollama_ok:
            message = ollama_unreachable_message(OLLAMA_BASE_URL, ollama_details)
            trace.log(
                "python_request_error",
                level="error",
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                error=trim_error(message),
            )
            emit_answer_payload(message, [], "ollama_unreachable")
            return

        llm = ChatOllama(
            model=CHAT_MODEL,
            temperature=0,
            base_url=OLLAMA_BASE_URL,
            num_predict=CHAT_NUM_PREDICT,
        )

        if args.mode == "general":
            answer = ask_general(llm, query_for_answer, markdown=True, trace=trace)
            trace.log(
                "python_request_success",
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                answer_chars=len(answer),
                sources=0,
                mode="general",
            )
            emit_answer_payload(answer, [], "general")
            return
        if args.mode == "general_raw":
            answer = ask_general(llm, query_for_answer, markdown=False, trace=trace)
            trace.log(
                "python_request_success",
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                answer_chars=len(answer),
                sources=0,
                mode="general_raw",
            )
            emit_answer_payload(answer, [], "general_raw")
            return

        # Jika data source belum ada/kosong, fallback ke mode general QA.
        if not data_dir.exists():
            answer = ask_general(llm, query_for_answer, trace=trace)
            trace.log(
                "python_request_success",
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                answer_chars=len(answer),
                sources=0,
                mode="fallback_general_no_data_dir",
            )
            emit_answer_payload(answer, [], "fallback_general_no_data_dir")
            return

        docs_started = time.perf_counter()
        docs = load_docs(data_dir)
        trace.log(
            "rag_docs_loaded",
            duration_ms=int(round((time.perf_counter() - docs_started) * 1000)),
            docs_count=len(docs),
        )
        if not docs:
            answer = ask_general(llm, query_for_answer, trace=trace)
            trace.log(
                "python_request_success",
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                answer_chars=len(answer),
                sources=0,
                mode="fallback_general_no_docs",
            )
            emit_answer_payload(answer, [], "fallback_general_no_docs")
            return

        embeddings_started = time.perf_counter()
        embeddings, _ = build_embeddings()
        trace.log(
            "rag_embeddings_ready",
            duration_ms=int(round((time.perf_counter() - embeddings_started) * 1000)),
        )

        # Pipeline retrieval: split -> embed -> vectorstore -> filter relevance.
        source_files_started = time.perf_counter()
        source_files = list_source_files(data_dir)
        trace.log(
            "rag_source_files_listed",
            duration_ms=int(round((time.perf_counter() - source_files_started) * 1000)),
            source_files_count=len(source_files),
        )
        if page_start is not None and page_end is not None:
            trace.log(
                "rag_page_range_applied",
                page_start=page_start,
                page_end=page_end,
            )
        explicit_source = find_explicit_source(query_for_answer, source_files)
        assignment_mode = is_assignment_generation_prompt(query_for_answer)
        if explicit_source is None and query_mentions_file(query_for_answer) and not assignment_mode:
            trace.log(
                "python_request_success",
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                answer_chars=len("Not found in context."),
                sources=0,
                mode="explicit_source_not_found",
            )
            emit_answer_payload(ensure_markdown_answer("Not found in context."), [], "explicit_source_not_found")
            return
        use_similarity_threshold = True
        vectorstore = None
        if explicit_source is not None:
            # Jika query menyebut nama file, fokus retrieval ke file itu.
            single_source_started = time.perf_counter()
            retrieval_docs = load_single_source(explicit_source)
            retrieval_docs = filter_docs_by_page_range(retrieval_docs, page_start, page_end)
            trace.log(
                "rag_single_source_loaded",
                duration_ms=int(round((time.perf_counter() - single_source_started) * 1000)),
                explicit_source=str(explicit_source),
                docs_count=len(retrieval_docs),
            )
            if not retrieval_docs:
                trace.log(
                    "python_request_success",
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    answer_chars=len("Not found in context."),
                    sources=0,
                    mode="single_source_empty",
                )
                emit_answer_payload(ensure_markdown_answer("Not found in context."), [], "single_source_empty")
                return
            build_vector_started = time.perf_counter()
            vectorstore = build_vectorstore_from_docs(retrieval_docs, embeddings)
            trace.log(
                "rag_vectorstore_built_single_source",
                duration_ms=int(round((time.perf_counter() - build_vector_started) * 1000)),
                explicit_source=str(explicit_source),
            )
            if vectorstore is None:
                trace.log(
                    "python_request_success",
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    answer_chars=len("Not found in context."),
                    sources=0,
                    mode="single_source_vectorstore_failed",
                )
                emit_answer_payload(ensure_markdown_answer("Not found in context."), [], "single_source_vectorstore_failed")
                return
            use_similarity_threshold = False
        else:
            cache_vector_started = time.perf_counter()
            vectorstore, _ = load_or_build_cached_vectorstore(data_dir, docs, embeddings)
            trace.log(
                "rag_vectorstore_ready",
                duration_ms=int(round((time.perf_counter() - cache_vector_started) * 1000)),
            )
            if vectorstore is None:
                answer = ask_general(llm, query_for_answer, trace=trace)
                trace.log(
                    "python_request_success",
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    answer_chars=len(answer),
                    sources=0,
                    mode="fallback_general_vectorstore_failed",
                )
                emit_answer_payload(answer, [], "fallback_general_vectorstore_failed")
                return
        retrieval_started = time.perf_counter()
        if use_similarity_threshold:
            context_docs = get_relevant_docs(vectorstore, query_for_answer, metadata_filter=page_filter)
        else:
            search_kwargs: dict[str, Any] = {"k": 4}
            if page_filter is not None:
                search_kwargs["filter"] = page_filter
            context_docs = vectorstore.as_retriever(search_kwargs=search_kwargs).invoke(query_for_answer)
        trace.log(
            "rag_context_retrieved",
            duration_ms=int(round((time.perf_counter() - retrieval_started) * 1000)),
            context_docs_count=len(context_docs) if context_docs is not None else 0,
            use_similarity_threshold=bool(use_similarity_threshold),
        )

        if not context_docs:
            not_found_message = "Not found in context."
            if page_start is not None and page_end is not None:
                not_found_message = (
                    f"Not found in selected page range ({page_start}-{page_end}). "
                    "Try widening page range."
                )
            if (explicit_source is not None or query_mentions_file(query_for_answer)) and not assignment_mode:
                trace.log(
                    "python_request_success",
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    answer_chars=len(not_found_message),
                    sources=0,
                    mode="no_context_docs",
                )
                emit_answer_payload(not_found_message, [], "no_context_docs")
            else:
                answer = ask_general(llm, query_for_answer, trace=trace)
                trace.log(
                    "python_request_success",
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    answer_chars=len(answer),
                    sources=0,
                    mode="fallback_general_no_context_docs",
                )
                emit_answer_payload(answer, [], "fallback_general_no_context_docs")
            return

        # Bentuk prompt RAG dan minta jawaban dari model.
        context = format_context(context_docs)
        prompt = build_rag_prompt(context=context, question=query_for_answer, style_instruction=style_instruction)

        answer = invoke_llm_with_retry(llm, prompt, retries=1, trace=trace)
        answer = ensure_markdown_answer(answer)
        focus_coverage = answer_focus_coverage(query_for_answer, answer)
        if focus_coverage < 0.34:
            focusterms = extract_focus_terms(query_for_answer)
            termsline = ", ".join(focusterms[:4]) if focusterms else query_for_answer.strip()
            strict_prompt = (
                build_rag_prompt(context=context, question=query_for_answer, style_instruction=style_instruction)
                + "\n\nSTRICT FOCUS RULE:\n"
                + f"- Target concept from question: {termsline}\n"
                + "- Your answer MUST stay on that target concept only.\n"
                + "- If context only gives limited detail, say that briefly.\n"
                + "- Do not switch to other example topics.\n"
                + "- Do not output meta templates with headings like Task/Context/Constraints/Format.\n"
            )
            strict_answer = invoke_llm_with_retry(llm, strict_prompt, retries=1, trace=trace)
            strict_answer = ensure_markdown_answer(strict_answer)
            if answer_focus_coverage(query_for_answer, strict_answer) >= focus_coverage:
                answer = strict_answer

        seen = set()
        sources = []
        for doc in context_docs:
            label = source_label(doc)
            if label not in seen:
                seen.add(label)
                sources.append(label)

        lowered = answer.lower()
        # Untuk pertanyaan berbasis file, jangan fallback ke general agar tidak muncul jawaban halusinasi.
        if lowered.startswith("not found in context"):
            if assignment_mode or (explicit_source is None and not query_mentions_file(query_for_answer)):
                answer = ask_general(llm, query_for_answer, trace=trace)
                sources = []
        elif "cannot access" in lowered and "file" in lowered:
            if assignment_mode:
                answer = ask_general(llm, query_for_answer, trace=trace)
                sources = []
            else:
                answer = "Not found in context."
                sources = []

        trace.log(
            "python_request_success",
            duration_ms=int(round((time.perf_counter() - started) * 1000)),
            answer_chars=len(str(answer)),
            sources=len(sources),
            mode="rag",
        )
        emit_answer_payload(str(answer), sources, "rag")
    except Exception as exc:
        trace.log(
            "python_request_error",
            level="error",
            duration_ms=int(round((time.perf_counter() - started) * 1000)),
            error=trim_error(str(exc)),
            traceback=trim_error(traceback.format_exc(), 12000),
        )
        emit_answer_payload(f"RAG backend error: {exc}", [], "error")


if __name__ == "__main__":
    main()
