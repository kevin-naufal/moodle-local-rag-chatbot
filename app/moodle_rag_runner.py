import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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

PROMPT_TEMPLATE = """You are a careful assistant. Use ONLY the following context to answer the question.
If the answer is not in the context, say "Not found in context."
Answer directly and concisely. Do not start with "Based on the context".
Never output internal reasoning tags like <think>.
Do not claim you cannot access files; file content is already provided in context.
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
        return base_rules + (
            f"- Create exactly {expected_count} essay questions.\n"
            "- Use numbered questions in this format: `1. Question text`.\n"
            "- Do NOT include options A/B/C/D.\n"
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


def count_question_items(question_section: str) -> int:
    marker_count = len(
        re.findall(
            r"(?mi)^\s*(?:question\s*\d+\s*[:.)]|pertanyaan\s*\d+\s*[:.)]|\d+\s*[.)]\s+)",
            question_section,
        )
    )
    if marker_count > 0:
        return marker_count
    # Fallback for outputs that omit numbering but keep one line per question.
    return len(re.findall(r"(?mi)^\s*[^\n]{8,}\?\s*$", question_section))


def has_strict_multiple_choice_answer_key(answer_key_section: str, expected_count: int) -> bool:
    if expected_count <= 0:
        return False
    # Reject malformed cross-reference style, e.g. "1. 2 (D)".
    if re.search(r"(?mi)^\s*\d+\.\s*\d+\s*\([A-D]\)\s*$", answer_key_section):
        return False

    matches = re.findall(r"(?mi)^\s*(\d+)\.\s*([A-D])\s*$", answer_key_section)
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


def get_relevant_docs(vectorstore: Chroma, query: str):
    # Score is expected in range [0, 1], where larger means more relevant.
    pairs = vectorstore.similarity_search_with_relevance_scores(query, k=4)
    docs = [doc for doc, score in pairs if score >= RELEVANCE_THRESHOLD]
    return docs


def emit(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))


def check_ollama(base_url: str, timeout: float = 3.0) -> tuple[bool, str]:
    url = f"{base_url}/api/tags"
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 0)
            if 200 <= status < 300:
                return True, ""
            return False, f"Ollama health check returned HTTP {status}."
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def ollama_unreachable_message(base_url: str, details: str) -> str:
    return (
        f"RAG backend error: Failed to connect to Ollama at {base_url}. "
        f"Start Ollama with `ollama serve`, then verify with "
        f"`ollama list`. Details: {details}"
    )


def ask_general(llm: ChatOllama, query: str, markdown: bool = True) -> str:
    prompt = GENERAL_PROMPT_TEMPLATE.format(question=query)
    answer = invoke_llm_with_retry(llm, prompt, retries=1)
    if markdown:
        return ensure_markdown_answer(answer)
    return ensure_plain_answer(answer)


def invoke_llm_with_retry(llm: ChatOllama, prompt: str, retries: int = 1) -> str:
    assignment_mode = is_assignment_generation_prompt(prompt)
    assignment_type = detect_assignment_type(prompt)
    assignment_guardrails = build_assignment_format_guardrails(prompt) if assignment_mode else ""
    last_answer = EMPTY_ANSWER_FALLBACK
    extra_retries = 2 if assignment_mode else 0
    attempts = max(0, retries) + 1 + extra_retries
    for attempt in range(attempts):
        current_prompt = prompt + assignment_guardrails
        if attempt > 0:
            current_prompt = (
                prompt
                + assignment_guardrails
                + "\n\nIMPORTANT: Your previous answer was empty or unusable. "
                + "Return the final answer now in Markdown only, without <think> tags."
            )
            if assignment_mode:
                current_prompt += (
                    "\nThe answer MUST include these sections with real content: "
                    "Assignment Title, Learning Objectives, Instructions for Students, "
                    "Question List, Answer Key, Grading Rubric."
                )
        response = llm.invoke(current_prompt)
        rawanswer = response.content if hasattr(response, "content") else str(response)
        last_answer = clean_answer(str(rawanswer))
        if is_unusable_answer(last_answer):
            continue
        if assignment_mode and not has_min_assignment_sections(last_answer, prompt):
            continue
        return last_answer

    if assignment_mode and not is_unusable_answer(last_answer):
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
        repair_response = llm.invoke(repair_prompt)
        repair_raw = repair_response.content if hasattr(repair_response, "content") else str(repair_response)
        repaired_answer = clean_answer(str(repair_raw))
        if not is_unusable_answer(repaired_answer) and has_min_assignment_sections(repaired_answer, prompt):
            return repaired_answer

    if assignment_mode:
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
        rescue_response = llm.invoke(rescue_prompt)
        rescue_raw = rescue_response.content if hasattr(rescue_response, "content") else str(rescue_response)
        rescued_answer = clean_answer(str(rescue_raw))
        if not is_unusable_answer(rescued_answer) and has_min_assignment_sections(rescued_answer, prompt):
            return rescued_answer

    if assignment_mode and not has_min_assignment_sections(last_answer, prompt):
        if has_core_assignment_sections(last_answer):
            return (
                last_answer
                + "\n\nNote: This draft may be incomplete. You can click Regenerate for a cleaner structure."
            )
        return "Assignment draft is incomplete after retries. Please click Regenerate to try again."
    return last_answer


def main() -> None:
    # Ambil parameter dari CLI (dipanggil oleh Moodle plugin).
    parser = argparse.ArgumentParser(description="RAG runner for Moodle local_chatbot plugin")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--query")
    parser.add_argument("--query-b64")
    parser.add_argument("--mode", choices=["auto", "general", "general_raw"], default="auto")
    args = parser.parse_args()

    try:
        # Bangun query final, lalu validasi input kosong.
        query = args.query or ""
        if args.query_b64:
            query = base64.b64decode(args.query_b64).decode("utf-8", errors="ignore")
        if not query.strip():
            emit({"answer": "Question is empty.", "sources": []})
            return

        # Shortcut smalltalk supaya response cepat tanpa proses RAG.
        smalltalk = smalltalk_response(query)
        if smalltalk is not None:
            emit({"answer": smalltalk, "sources": []})
            return

        ollama_ok, ollama_details = check_ollama(OLLAMA_BASE_URL)
        if not ollama_ok:
            emit(
                {
                    "answer": ollama_unreachable_message(
                        OLLAMA_BASE_URL, ollama_details
                    ),
                    "sources": [],
                }
            )
            return

        llm = ChatOllama(
            model=CHAT_MODEL,
            temperature=0,
            base_url=OLLAMA_BASE_URL,
            num_predict=CHAT_NUM_PREDICT,
        )

        if args.mode == "general":
            emit({"answer": ask_general(llm, query, markdown=True), "sources": []})
            return
        if args.mode == "general_raw":
            emit({"answer": ask_general(llm, query, markdown=False), "sources": []})
            return

        # Jika data source belum ada/kosong, fallback ke mode general QA.
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            emit({"answer": ask_general(llm, query), "sources": []})
            return

        docs = load_docs(data_dir)
        if not docs:
            emit({"answer": ask_general(llm, query), "sources": []})
            return

        # Pipeline retrieval: split -> embed -> vectorstore -> filter relevance.
        source_files = list_source_files(data_dir)
        explicit_source = find_explicit_source(query, source_files)
        assignment_mode = is_assignment_generation_prompt(query)
        if explicit_source is None and query_mentions_file(query) and not assignment_mode:
            emit({"answer": ensure_markdown_answer("Not found in context."), "sources": []})
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        retrieval_docs = docs
        use_similarity_threshold = True
        if explicit_source is not None:
            # Jika query menyebut nama file, fokus retrieval ke file itu.
            retrieval_docs = load_single_source(explicit_source)
            if not retrieval_docs:
                emit({"answer": ensure_markdown_answer("Not found in context."), "sources": []})
                return
            use_similarity_threshold = False

        splits = splitter.split_documents(retrieval_docs)
        embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        if use_similarity_threshold:
            context_docs = get_relevant_docs(vectorstore, query)
        else:
            context_docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(query)

        if not context_docs:
            if (explicit_source is not None or query_mentions_file(query)) and not assignment_mode:
                emit({"answer": "Not found in context.", "sources": []})
            else:
                emit({"answer": ask_general(llm, query), "sources": []})
            return

        # Bentuk prompt RAG dan minta jawaban dari model.
        context = format_context(context_docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)

        answer = invoke_llm_with_retry(llm, prompt, retries=1)
        answer = ensure_markdown_answer(answer)

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
            if assignment_mode or (explicit_source is None and not query_mentions_file(query)):
                answer = ask_general(llm, query)
                sources = []
        elif "cannot access" in lowered and "file" in lowered:
            if assignment_mode:
                answer = ask_general(llm, query)
                sources = []
            else:
                answer = "Not found in context."
                sources = []

        emit({"answer": str(answer), "sources": sources})
    except Exception as exc:
        emit({"answer": f"RAG backend error: {exc}", "sources": []})


if __name__ == "__main__":
    main()
