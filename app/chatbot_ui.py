from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4
from datetime import datetime

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from eval_logger import append_jsonl
from eval_schema import build_raw_result_payload
from moodle_rag_runner import BertEmbeddings

"""ALUR UTAMA (Streamlit RAG UI)
1) Setup halaman + folder + chat id.
2) Muat histori chat user.
3) Proses upload dan daftar dokumen.
4) Render chat dan terima pertanyaan user.
5) Jalankan RAG untuk menghasilkan jawaban.
6) Tampilkan jawaban dan simpan histori terbaru.
"""


DATA_DIR = Path("data")
CHAT_STORE_DIR = Path(".chat_store")
ANSWER_RUNS_DIR = DATA_DIR / "answer_runs"
ANSWER_RUNS_PATH = ANSWER_RUNS_DIR / "llm_answer_results.jsonl"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MAX_STORED_MESSAGES = 200

PROMPT_TEMPLATE = """You are a careful assistant. Use ONLY the following context to answer the question.
If the answer is not in the context, say "Not found in context."
Answer directly and concisely. Do not start with "Based on the context".
Do not claim you cannot access files; the extracted file content is already provided in context.
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

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": "Hello. Upload PDF/TXT in the left panel, then ask about your documents here.",
    "sources": [],
}


def ensure_data_dir() -> None:
    # Pastikan folder `data/` ada sebelum proses upload/read dokumen.
    # `parents=True` membuat parent folder jika belum ada.
    # `exist_ok=True` mencegah error kalau folder sudah ada.
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_chat_store_dir() -> None:
    # Pastikan folder `.chat_store/` ada untuk menyimpan histori chat per chat_id.
    # Dengan opsi yang sama, fungsi aman dipanggil berulang kali saat startup app.
    CHAT_STORE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_answer_runs_dir() -> None:
    ANSWER_RUNS_DIR.mkdir(parents=True, exist_ok=True)


def list_source_files() -> list[Path]:
    # Ambil semua file PDF/TXT dari folder data sebagai sumber RAG.
    files: list[Path] = []
    for ext in ("*.pdf", "*.txt"):
        files.extend(DATA_DIR.glob(ext))
    files.sort(key=lambda item: item.name.lower())
    return files


def normalize_lookup_key(value: str) -> str:
    # Normalisasi untuk pencocokan nama file yang lebih toleran terhadap tanda baca/spasi.
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def file_fingerprint(files: Iterable[Path]) -> str:
    # Fingerprint dipakai sebagai "signature" agar cache retriever ikut berubah saat file berubah.
    parts: list[str] = []
    for file_path in files:
        stat = file_path.stat()
        parts.append(f"{file_path.name}:{stat.st_size}:{int(stat.st_mtime)}")
    return "|".join(parts)


def load_documents(files: Iterable[Path]) -> list[Document]:
    # Loader berbeda dipakai sesuai ekstensi file.
    docs: list[Document] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            docs.extend(TextLoader(str(file_path), autodetect_encoding=True).load())
        elif suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file_path)).load())
    return docs


def check_ollama(base_url: str, timeout: float = 2.0) -> tuple[bool, str]:
    request = urllib.request.Request(url=f"{base_url}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 0)
            if 200 <= status < 300:
                return True, ""
            return False, f"Ollama health check returned HTTP {status}."
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def ollama_help_message() -> str:
    return (
        "## Ollama is not reachable\n\n"
        f"- URL: `{OLLAMA_BASE_URL}`\n"
        "- Start service: `ollama serve`\n"
        "- Verify models: `ollama list`\n"
        f"- Expected chat model: `{CHAT_MODEL}`\n"
        f"- Expected embedding model: `{EMBED_MODEL}`"
    )


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOllama:
    # Cache model chat supaya tidak inisialisasi ulang tiap pertanyaan.
    return ChatOllama(model=CHAT_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)


@st.cache_resource(show_spinner=False)
def get_retriever(signature: str, backend: str = "ollama"):
    # Signature hanya untuk invalidasi cache ketika dokumen berubah.
    _ = (signature, backend)
    files = list_source_files()
    docs = load_documents(files)
    if not docs:
        return None

    # Dokumen dipecah jadi chunk lalu diubah jadi embedding untuk retrieval.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    if str(backend).strip().lower() == "bert":
        embeddings = BertEmbeddings()
    else:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def build_sources(docs: list[Document]) -> list[str]:
    # Format sumber dibuat unik agar tidak ada label duplikat di UI.
    seen: set[str] = set()
    sources: list[str] = []
    for doc in docs:
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        page = doc.metadata.get("page")
        if page is None:
            label = source
        else:
            label = f"{source} p.{int(page) + 1}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


def serialize_retrieved_context(docs: list[Document]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for doc in docs:
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        page = doc.metadata.get("page")
        items.append(
            {
                "text": str(doc.page_content or ""),
                "source": source,
                "page": (int(page) + 1) if page is not None else None,
            }
        )
    return items


def find_explicit_source(question: str, files: list[Path]) -> Path | None:
    # Jika user menyebut nama file secara eksplisit, prioritaskan file tersebut.
    # Gunakan beberapa strategi agar tetap match walau query punya backtick/tanda baca tambahan.
    lowered = question.lower()
    normalized_question = normalize_lookup_key(question)
    candidates: list[tuple[int, Path]] = []

    for file_path in files:
        file_name = file_path.name.lower()
        file_stem = file_path.stem.lower()
        if file_name in lowered:
            candidates.append((len(file_name), file_path))
            continue
        # Dukung referensi nama file tanpa ekstensi, contoh "info1" untuk "info1.txt".
        if file_stem and re.search(rf"\b{re.escape(file_stem)}\b", lowered):
            candidates.append((len(file_stem), file_path))
            continue

        normalized_file_name = normalize_lookup_key(file_name)
        if normalized_file_name and normalized_file_name in normalized_question:
            candidates.append((len(normalized_file_name), file_path))
            continue
        normalized_file_stem = normalize_lookup_key(file_stem)
        if normalized_file_stem and normalized_file_stem in normalized_question:
            candidates.append((len(normalized_file_stem), file_path))

    if not candidates:
        return None
    # Pilih kandidat terpanjang untuk menghindari match parsial yang terlalu umum.
    return max(candidates, key=lambda item: item[0])[1]


def query_mentions_file(question: str) -> bool:
    # Deteksi jika user secara eksplisit menyebut nama file (contoh: moon.txt / modul.pdf).
    return re.search(r"\b[\w.\- ]+\.(txt|pdf)\b", question, flags=re.IGNORECASE) is not None


def format_context(docs: list[Document]) -> str:
    # Sisipkan label sumber agar model tahu potongan teks berasal dari file mana.
    chunks: list[str] = []
    for doc in docs:
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        page = doc.metadata.get("page")
        if page is None:
            label = source
        else:
            label = f"{source} p.{int(page) + 1}"
        chunks.append(f"[Source: {label}]\n{doc.page_content}")
    return "\n\n".join(chunks)


def clean_answer(text: str) -> str:
    # Hilangkan internal reasoning tags agar tidak tampil di chat.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip() or "Sorry, I cannot provide an answer for that question yet."


def strip_leading_boilerplate(answer: str) -> str:
    # Hapus frasa pembuka generik yang sering diulang model di awal jawaban.
    stripped = answer.strip()
    patterns = [
        r"^\s*however,\s*(?:based on|from)\s+the\s+provided\s+context[:,]?\s*",
        r"^\s*based on\s+the\s+provided\s+context[:,]?\s*",
        r"^\s*however[:,]?\s*",
    ]
    for pattern in patterns:
        stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def normalize_not_found_prefix(answer: str) -> tuple[str, bool]:
    # Jika jawaban diawali "Not found in context." tapi masih punya isi, buang prefix saja.
    # Jika jawabannya hanya kalimat itu, tandai agar fallback ke LLM general.
    stripped = answer.strip()
    only_not_found = re.fullmatch(
        r"(?:[*_`>#\-\s]*)not found in context\.?(?:[*_`>#\-\s]*)",
        stripped,
        flags=re.IGNORECASE,
    )
    if only_not_found is not None:
        return "", True

    without_prefix = re.sub(
        r"^\s*(?:[*_`>#\-\s]*)not found in context\.?\s*",
        "",
        stripped,
        count=1,
        flags=re.IGNORECASE,
    )
    return without_prefix.strip(), False


def ensure_markdown_answer(answer: str) -> str:
    # Pastikan output akhir valid Markdown, tanpa memaksa bullet list.
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


def ask_general(question: str) -> str:
    # Fallback umum ketika context dokumen tidak tersedia/tidak relevan.
    prompt = GENERAL_PROMPT_TEMPLATE.format(question=question)
    response = get_llm().invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return ensure_markdown_answer(clean_answer(str(content)))


def build_embeddings_for_backend(backend: str):
    if str(backend).strip().lower() == "bert":
        return BertEmbeddings()
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def run_llm_only(question: str) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = GENERAL_PROMPT_TEMPLATE.format(question=question)
    response = get_llm().invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    answer = ensure_markdown_answer(clean_answer(str(content)))
    latency_generation_ms = int(round((time.perf_counter() - started) * 1000))
    return {
        "answer": answer,
        "sources": [],
        "retrieved_context": [],
        "latency_total_ms": latency_generation_ms,
        "latency_retrieval_ms": 0,
        "latency_generation_ms": latency_generation_ms,
        "embedding_backend": "none",
        "status": "success",
        "error_message": None,
    }


def run_rag(question: str, signature: str, backend: str = "ollama") -> dict[str, Any]:
    # step 5: Ambil context relevan dari retriever.
    total_started = time.perf_counter()
    files = list_source_files()
    mentions_file = query_mentions_file(question)
    explicit_source = find_explicit_source(question, files)
    backend_name = str(backend).strip().lower() or "ollama"
    latency_retrieval_ms = 0
    if explicit_source is not None:
        # Pertanyaan bernama-file: retrieval dipersempit ke satu file agar jawaban lebih akurat.
        retrieval_started = time.perf_counter()
        docs = load_documents([explicit_source])
        if not docs:
            latency_retrieval_ms = int(round((time.perf_counter() - retrieval_started) * 1000))
            answer = ensure_markdown_answer("Not found in context.")
            return {
                "answer": answer,
                "sources": [],
                "retrieved_context": [],
                "latency_total_ms": int(round((time.perf_counter() - total_started) * 1000)),
                "latency_retrieval_ms": latency_retrieval_ms,
                "latency_generation_ms": 0,
                "embedding_backend": backend_name,
                "status": "success",
                "error_message": None,
            }
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        embeddings = build_embeddings_for_backend(backend_name)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        context_docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)
        latency_retrieval_ms = int(round((time.perf_counter() - retrieval_started) * 1000))
    elif mentions_file:
        # Jika user menyebut file tapi tidak ada match, jangan tarik context acak dari dokumen lain.
        answer = ensure_markdown_answer("Not found in context.")
        return {
            "answer": answer,
            "sources": [],
            "retrieved_context": [],
            "latency_total_ms": int(round((time.perf_counter() - total_started) * 1000)),
            "latency_retrieval_ms": 0,
            "latency_generation_ms": 0,
            "embedding_backend": backend_name,
            "status": "success",
            "error_message": None,
        }
    else:
        retrieval_started = time.perf_counter()
        retriever = get_retriever(signature, backend_name)
        if retriever is None:
            fallback = run_llm_only(question)
            fallback["latency_total_ms"] = int(round((time.perf_counter() - total_started) * 1000))
            fallback["embedding_backend"] = backend_name
            return fallback
        context_docs = retriever.invoke(question)
        latency_retrieval_ms = int(round((time.perf_counter() - retrieval_started) * 1000))

    if not context_docs:
        if mentions_file:
            answer = ensure_markdown_answer("Not found in context.")
            return {
                "answer": answer,
                "sources": [],
                "retrieved_context": [],
                "latency_total_ms": int(round((time.perf_counter() - total_started) * 1000)),
                "latency_retrieval_ms": latency_retrieval_ms,
                "latency_generation_ms": 0,
                "embedding_backend": backend_name,
                "status": "success",
                "error_message": None,
            }
        fallback = run_llm_only(question)
        fallback["latency_total_ms"] = int(round((time.perf_counter() - total_started) * 1000))
        fallback["latency_retrieval_ms"] = latency_retrieval_ms
        fallback["embedding_backend"] = backend_name
        return fallback

    # Lanjut: gabungkan context + question ke prompt, lalu panggil LLM.
    context = format_context(context_docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    generation_started = time.perf_counter()
    response = get_llm().invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    answer = clean_answer(str(content))
    lowered = answer.lower()
    if "cannot access" in lowered and "file" in lowered:
        if mentions_file:
            answer = ensure_markdown_answer("Not found in context.")
            return {
                "answer": answer,
                "sources": [],
                "retrieved_context": serialize_retrieved_context(context_docs),
                "latency_total_ms": int(round((time.perf_counter() - total_started) * 1000)),
                "latency_retrieval_ms": latency_retrieval_ms,
                "latency_generation_ms": int(round((time.perf_counter() - generation_started) * 1000)),
                "embedding_backend": backend_name,
                "status": "success",
                "error_message": None,
            }
        fallback = run_llm_only(question)
        fallback["latency_total_ms"] = int(round((time.perf_counter() - total_started) * 1000))
        fallback["latency_retrieval_ms"] = latency_retrieval_ms
        fallback["embedding_backend"] = backend_name
        return fallback

    normalized_answer, is_only_not_found = normalize_not_found_prefix(answer)
    if is_only_not_found:
        if mentions_file:
            answer = ensure_markdown_answer("Not found in context.")
            return {
                "answer": answer,
                "sources": [],
                "retrieved_context": serialize_retrieved_context(context_docs),
                "latency_total_ms": int(round((time.perf_counter() - total_started) * 1000)),
                "latency_retrieval_ms": latency_retrieval_ms,
                "latency_generation_ms": int(round((time.perf_counter() - generation_started) * 1000)),
                "embedding_backend": backend_name,
                "status": "success",
                "error_message": None,
            }
        fallback = run_llm_only(question)
        fallback["latency_total_ms"] = int(round((time.perf_counter() - total_started) * 1000))
        fallback["latency_retrieval_ms"] = latency_retrieval_ms
        fallback["embedding_backend"] = backend_name
        return fallback
    latency_generation_ms = int(round((time.perf_counter() - generation_started) * 1000))
    return {
        "answer": ensure_markdown_answer(normalized_answer),
        "sources": build_sources(context_docs),
        "retrieved_context": serialize_retrieved_context(context_docs),
        "latency_total_ms": int(round((time.perf_counter() - total_started) * 1000)),
        "latency_retrieval_ms": latency_retrieval_ms,
        "latency_generation_ms": latency_generation_ms,
        "embedding_backend": backend_name,
        "status": "success",
        "error_message": None,
    }


def get_or_create_chat_id() -> str:
    # Gunakan chat_id dari URL jika ada; jika tidak, buat id baru.
    raw_chat_id = st.query_params.get("chat_id")
    chat_id = str(raw_chat_id).strip() if raw_chat_id else ""
    if not chat_id:
        chat_id = uuid4().hex
        st.query_params["chat_id"] = chat_id
    return chat_id


def chat_store_path(chat_id: str) -> Path:
    # Sanitasi id agar aman dipakai sebagai nama file JSON.
    safe_id = "".join(ch for ch in chat_id if ch.isalnum() or ch in ("-", "_"))
    if not safe_id:
        safe_id = "default"
    return CHAT_STORE_DIR / f"{safe_id}.json"


def load_messages(chat_id: str) -> list[dict]:
    # Kalau file histori belum ada, mulai dari welcome message.
    path = chat_store_path(chat_id)
    if not path.exists():
        return [WELCOME_MESSAGE]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return [WELCOME_MESSAGE]

        # Filter data rusak/tidak valid agar render chat tidak error.
        valid_messages: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            sources = item.get("sources", [])
            if role in ("assistant", "user") and isinstance(content, str):
                if not isinstance(sources, list):
                    sources = []
                valid_messages.append(
                    {"role": role, "content": content, "sources": sources}
                )

        return valid_messages or [WELCOME_MESSAGE]
    except Exception:
        return [WELCOME_MESSAGE]


def save_messages(chat_id: str, messages: list[dict]) -> None:
    # Batasi histori yang disimpan agar file tidak tumbuh tanpa batas.
    path = chat_store_path(chat_id)
    payload = messages[-MAX_STORED_MESSAGES:]
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # Best-effort persistence; do not break chat flow if disk write fails.
        return


def render_file_list(files: list[Path]) -> None:
    # Tampilkan dokumen aktif beserta ukuran file.
    st.markdown("### Uploaded Documents")
    if not files:
        st.caption("Belum ada file di-upload.")
        return
    for file_path in files:
        size_kb = file_path.stat().st_size / 1024
        st.markdown(
            f"- **{file_path.name}**  \n`{size_kb:.1f} KB`",
        )


def create_eval_output_path(prefix: str = "eval") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:6]
    return ANSWER_RUNS_DIR / f"{prefix}_{stamp}_{suffix}.jsonl"


def get_or_create_manual_eval_output_path() -> Path:
    raw = str(st.session_state.get("manual_eval_output_path", "") or "").strip()
    if raw:
        return Path(raw)
    path = create_eval_output_path("answer_runs_manual")
    st.session_state.manual_eval_output_path = str(path)
    return path


def start_new_manual_eval_session() -> Path:
    path = create_eval_output_path("answer_runs_manual")
    st.session_state.manual_eval_output_path = str(path)
    return path


def auto_manual_question_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"manual-{stamp}-{uuid4().hex[:4]}"


def load_eval_questions_from_text(raw_text: str) -> list[dict[str, Any]]:
    payload = json.loads(str(raw_text or ""))
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        items = payload["questions"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("JSON must be an array or an object with a 'questions' list.")

    questions: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        question_text = str(item.get("question", "")).strip()
        if not question_text:
            continue
        question_id = str(item.get("id") or item.get("question_id") or f"auto-q{idx:03d}").strip()
        normalized = dict(item)
        normalized["question"] = question_text
        normalized["question_id"] = question_id
        questions.append(normalized)
    if not questions:
        raise ValueError("No valid questions found in the uploaded JSON.")
    return questions


def execute_selected_mode(question: str, signature: str, mode: str) -> dict[str, Any]:
    if mode == "llm_only":
        return run_llm_only(question)
    if mode == "rag_bert":
        return run_rag(question, signature, backend="bert")
    return run_rag(question, signature, backend="ollama")


def append_eval_run(
    output_path: Path,
    *,
    question_id: str,
    question: str,
    mode: str,
    run_id: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    source_files = list_source_files()
    payload = build_raw_result_payload(
        question_id=question_id,
        question=question,
        mode=mode,
        run_id=int(run_id),
        model_name=CHAT_MODEL,
        embedding_backend=result.get("embedding_backend"),
        model_answer=str(result.get("answer", "")),
        retrieved_context=result.get("retrieved_context") or [],
        latency_total_ms=result.get("latency_total_ms", 0),
        latency_retrieval_ms=result.get("latency_retrieval_ms", 0),
        latency_generation_ms=result.get("latency_generation_ms", 0),
        status=str(result.get("status", "success")),
        error_message=result.get("error_message"),
    )
    payload.update(
        {
            "corpus_sources": [file_path.name for file_path in source_files],
            "corpus_signature": file_fingerprint(source_files),
            "corpus_data_dir": str(DATA_DIR),
        }
    )
    append_jsonl(output_path, payload)
    return payload


def main() -> None:
    # step 1: Setup halaman app, folder kerja, dan chat_id aktif.
    st.set_page_config(page_title="Campus RAG Assistant", page_icon=":books:", layout="wide")
    ensure_data_dir()
    ensure_chat_store_dir()
    ensure_answer_runs_dir()
    chat_id = get_or_create_chat_id()

    # step 2: Muat histori chat dari file ke session_state.
    if st.session_state.get("chat_id") != chat_id:
        st.session_state.chat_id = chat_id
        st.session_state.messages = load_messages(chat_id)
    elif "messages" not in st.session_state:
        st.session_state.messages = load_messages(chat_id)

    st.markdown(
        """
        <style>
          .block-container { padding-top: 1rem; padding-bottom: 1rem; }
          .stChatMessage { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # step 3: Panel upload dokumen dipindah ke sidebar agar area chat tetap penuh.
    with st.sidebar:
        st.markdown("## Insert PDF/TXT")
        st.caption("Upload source documents for the chatbot. Supported: .pdf and .txt")

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        # Lanjut: simpan file ke data/, clear cache, lalu rerun agar list terbarui.
        if st.button("Upload selected files", use_container_width=True):
            if not uploaded_files:
                st.warning("Please choose at least one file.")
            else:
                saved = 0
                for file in uploaded_files:
                    target = DATA_DIR / Path(file.name).name
                    target.write_bytes(file.getbuffer())
                    saved += 1
                st.cache_resource.clear()
                st.success(f"{saved} file(s) uploaded successfully.")
                st.rerun()

        files = list_source_files()
        render_file_list(files)

        eval_mode = st.checkbox("Enable evaluation mode", value=False)
        selected_mode = "rag_ollama"
        eval_question_id = ""
        eval_run_id = 1

        if eval_mode:
            st.markdown("### Evaluation mode")
            selected_mode = st.selectbox(
                "Mode",
                options=["llm_only", "rag_ollama", "rag_bert"],
                index=1,
            )
            eval_question_id = st.text_input("Question ID", value="", help="Leave blank to auto-generate for manual chat runs.")
            eval_run_id = st.number_input("Run ID", min_value=1, step=1, value=1)
            current_manual_eval_path = get_or_create_manual_eval_output_path()
            st.caption(f"Manual answer-run file: `{current_manual_eval_path.name}`")
            if st.button("Start new manual answer-run session", use_container_width=True):
                new_path = start_new_manual_eval_session()
                st.success(f"Started new manual answer-run session: {new_path.name}")
                st.rerun()

            st.markdown("### Answer-Run Dataset")
            eval_dataset_file = st.file_uploader(
                "Upload dataset JSON",
                type=["json"],
                accept_multiple_files=False,
                key="eval_dataset_uploader",
            )
            dataset_runs = st.number_input(
                "Runs per question",
                min_value=1,
                max_value=10,
                step=1,
                value=1,
                key="eval_dataset_runs",
            )
            if eval_dataset_file is not None:
                st.caption(f"Dataset file: `{eval_dataset_file.name}`")
            run_dataset_clicked = st.button(
                "Run uploaded answer-run dataset",
                use_container_width=True,
                disabled=eval_dataset_file is None,
            )
        else:
            eval_dataset_file = None
            dataset_runs = 1
            run_dataset_clicked = False

    # step 4: Render histori chat dan tunggu pertanyaan user.
    files = list_source_files()
    signature = file_fingerprint(files)
    # Status singkat dipakai untuk indikasi apakah basis dokumen sudah siap.
    ollama_ok, _ = check_ollama(OLLAMA_BASE_URL)
    if files and ollama_ok:
        ready = "RAG ready"
    elif files:
        ready = "Documents ready, Ollama offline"
    elif ollama_ok:
        ready = "No documents yet"
    else:
        ready = "No documents, Ollama offline"
    st.markdown(f"## Chat with your documents  \n`{ready} | mode: {selected_mode}`")

    if run_dataset_clicked and eval_dataset_file is not None:
        try:
            dataset_questions = load_eval_questions_from_text(eval_dataset_file.getvalue().decode("utf-8"))
        except Exception as exc:
            st.error(f"Failed to parse answer-run dataset: {exc}")
            dataset_questions = []

        if dataset_questions:
            output_path = create_eval_output_path("answer_runs_dataset")
            progress = st.progress(0.0, text="Starting answer-run dataset...")
            success_count = 0
            total_jobs = len(dataset_questions) * int(dataset_runs)
            completed_jobs = 0
            with st.spinner("Running answer-run dataset..."):
                for item in dataset_questions:
                    question_text = str(item["question"])
                    question_id = str(item["question_id"])
                    for run_number in range(1, int(dataset_runs) + 1):
                        completed_jobs += 1
                        progress.progress(
                            completed_jobs / max(1, total_jobs),
                            text=f"Running {question_id} ({completed_jobs}/{total_jobs})",
                        )
                        try:
                            result = execute_selected_mode(question_text, signature, selected_mode)
                            payload = append_eval_run(
                                output_path,
                                question_id=question_id,
                                question=question_text,
                                mode=selected_mode,
                                run_id=run_number,
                                result=result,
                            )
                            if payload.get("status") == "success":
                                success_count += 1
                        except Exception as exc:
                            error_result = {
                                "answer": f"Failed to process question: {exc}",
                                "retrieved_context": [],
                                "latency_total_ms": 0,
                                "latency_retrieval_ms": 0,
                                "latency_generation_ms": 0,
                                "embedding_backend": "none" if selected_mode == "llm_only" else ("bert" if selected_mode == "rag_bert" else "ollama"),
                                "status": "error",
                                "error_message": str(exc),
                            }
                            append_eval_run(
                                output_path,
                                question_id=question_id,
                                question=question_text,
                                mode=selected_mode,
                                run_id=run_number,
                                result=error_result,
                            )
            progress.progress(1.0, text="Answer-run dataset completed.")
            st.success(
                f"Answer-run dataset finished. Saved {total_jobs} run(s) to `{output_path}` with {success_count} successful run(s)."
            )

    # Render semua pesan histori sebelum menerima input baru.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
            else:
                st.write(message["content"])
            if message.get("sources"):
                st.caption("source: " + ", ".join(message["sources"]))

    question = st.chat_input("Ask a question about uploaded files...")
    if question:
        # step 6: Simpan pertanyaan user terlebih dahulu.
        st.session_state.messages.append(
            {"role": "user", "content": question, "sources": []}
        )
        save_messages(chat_id, st.session_state.messages)
        with st.chat_message("user"):
            st.write(question)

        # Lanjut: minta jawaban ke step 5 (ask_rag), lalu tampilkan ke chat.
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ollama_ok, ollama_error = check_ollama(OLLAMA_BASE_URL)
                if not ollama_ok:
                    answer = (
                        f"{ollama_help_message()}\n\n"
                        f"Error details: `{ollama_error}`"
                    )
                    sources = []
                else:
                    try:
                        result = execute_selected_mode(question, signature, selected_mode)
                        answer = str(result["answer"])
                        sources = list(result.get("sources") or [])
                        if eval_mode:
                            question_id = str(eval_question_id or "").strip() or auto_manual_question_id()
                            output_path = get_or_create_manual_eval_output_path()
                            append_eval_run(
                                output_path,
                                question_id=question_id,
                                question=question,
                                mode=selected_mode,
                                run_id=int(eval_run_id),
                                result=result,
                            )
                    except Exception as exc:
                        answer = f"Failed to process question: {exc}"
                        sources = []
            st.markdown(answer)
            if sources:
                st.caption("source: " + ", ".join(sources))

        # Lanjut: simpan jawaban assistant ke histori.
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        save_messages(chat_id, st.session_state.messages)


if __name__ == "__main__":
    main()
