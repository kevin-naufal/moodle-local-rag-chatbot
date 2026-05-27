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
from eval_logger import append_jsonl
from eval_schema import build_raw_result_payload

"""Moodle RAG Runner.
Digunakan plugin Moodle untuk menjalankan retrieval + jawaban model dan mengembalikan JSON.
"""


def load_local_dotenv(start_dir: Path | None = None) -> None:
    """Load a simple .env file without requiring python-dotenv."""
    current = start_dir or Path(__file__).resolve().parent
    candidates = [current / ".env", current.parent / ".env"]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                    value = value[1:-1]
                os.environ.setdefault(key, value)
        except Exception:
            # Env loading must not break normal execution.
            pass
        break


load_local_dotenv(Path(__file__).resolve().parent)


EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text").strip()
CHAT_MODEL = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"
RELEVANCE_THRESHOLD = 0.2
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
EMPTY_ANSWER_FALLBACK = "Sorry, I cannot provide an answer for that question yet."
CHAT_NUM_PREDICT = int(os.getenv("CHAT_NUM_PREDICT", "2048"))
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "auto").strip().lower()
BERT_MODEL = os.getenv("BERT_MODEL", "sentence-transformers/msmarco-bert-base-dot-v5").strip()
BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "256"))
BERT_BATCH_SIZE = int(os.getenv("BERT_BATCH_SIZE", "16"))
RAG_TOP_K = max(1, int(os.getenv("RAG_TOP_K", "4")))
RAG_CANDIDATE_K = max(RAG_TOP_K, int(os.getenv("RAG_CANDIDATE_K", "8")))
RAG_CHUNK_SIZE = max(100, int(os.getenv("RAG_CHUNK_SIZE", "1000")))
RAG_CHUNK_OVERLAP = max(0, min(RAG_CHUNK_SIZE - 1, int(os.getenv("RAG_CHUNK_OVERLAP", "200"))))
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
Do not add examples, benefits, causes, or implications unless they are explicitly supported by the context.
Use the Primary context first.
Answer the exact question before adding supporting details.
If the context contains multiple related topics, prioritize the part that directly answers the question.
Do not shift to a neighboring topic unless the question asks for it.
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

        model_name = model_name.strip() or "sentence-transformers/msmarco-bert-base-dot-v5"
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


def build_embeddings(
    backend_override: str | None = None,
    model_override: str | None = None,
) -> tuple[Embeddings, str, str]:
    backend = str(backend_override or EMBED_BACKEND).strip().lower()
    if backend not in {"auto", "bert", "ollama"}:
        backend = "auto"

    if backend in {"auto", "bert"}:
        bert_model_name = str(model_override or BERT_MODEL).strip() or BERT_MODEL
        try:
            return BertEmbeddings(
                model_name=bert_model_name,
                max_length=BERT_MAX_LENGTH,
                batch_size=BERT_BATCH_SIZE,
            ), "bert", bert_model_name
        except Exception as exc:
            if backend == "bert":
                raise RuntimeError(f"BERT embedding initialization failed: {exc}") from exc

    # Fallback/default: Ollama embeddings.
    ollama_model_name = str(model_override or EMBED_MODEL).strip() or EMBED_MODEL
    return OllamaEmbeddings(model=ollama_model_name, base_url=OLLAMA_BASE_URL), "ollama", ollama_model_name


def source_label(doc) -> str:
    source = Path(str(doc.metadata.get("source", "unknown"))).name
    page = doc.metadata.get("page")
    if page is None:
        return source
    return f"{source} p.{int(page) + 1}"


def serialize_retrieved_context(docs) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for doc in docs or []:
        page = doc.metadata.get("page")
        item = {
            "text": clean_document_text(str(getattr(doc, "page_content", "") or "")),
            "source": Path(str(doc.metadata.get("source", "unknown"))).name,
            "page": (int(page) + 1) if page is not None else None,
        }
        for key in ("retrieval_rank", "retrieval_score", "focus_score", "combined_score", "context_role"):
            if key in doc.metadata:
                item[key] = doc.metadata.get(key)
        items.append(item)
    return items


def clean_document_text(text: str) -> str:
    replacements = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u00ad": "",
    }
    cleaned = str(text or "")
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    return cleaned


def clean_loaded_docs(docs):
    for doc in docs or []:
        doc.page_content = clean_document_text(getattr(doc, "page_content", "") or "")
    return docs


def load_docs(data_dir: Path):
    docs = []
    for file_path in sorted(data_dir.iterdir(), key=lambda p: p.name.lower()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(file_path), autodetect_encoding=True).load())
        elif file_path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(file_path)).load())
    return clean_loaded_docs(docs)


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


def extract_current_user_question(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    marker = "current user question:"
    if marker in lowered:
        start = lowered.rfind(marker) + len(marker)
        tail = text[start:].strip()
        end_marker = "answer the current user question."
        tail_lower = tail.lower()
        if end_marker in tail_lower:
            tail = tail[:tail_lower.find(end_marker)].strip()
        if tail:
            return tail
    return text


def clean_answer(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() or EMPTY_ANSWER_FALLBACK


def strip_forbidden_meta_sections(answer: str) -> str:
    stripped = str(answer or "").strip()
    if not stripped:
        return ""

    forbidden_section_patterns = [
        r"(?is)\n+\s*\*{0,2}\s*why not other answers\??\s*\*{0,2}\s*:?\s*.*$",
        r"(?is)\n+\s*\*{0,2}\s*why this answer\??\s*\*{0,2}\s*:?\s*.*$",
        r"(?is)\n+\s*\*{0,2}\s*task\s*\*{0,2}\s*:?\s*.*$",
        r"(?is)\n+\s*\*{0,2}\s*context\s*\*{0,2}\s*:?\s*.*$",
        r"(?is)\n+\s*\*{0,2}\s*constraints\s*\*{0,2}\s*:?\s*.*$",
        r"(?is)\n+\s*\*{0,2}\s*format\s*\*{0,2}\s*:?\s*.*$",
    ]
    for pattern in forbidden_section_patterns:
        stripped = re.sub(pattern, "", stripped).strip()
    return stripped


def normalize_final_answer(answer: str) -> str:
    cleaned = clean_answer(answer)
    cleaned = strip_forbidden_meta_sections(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or EMPTY_ANSWER_FALLBACK


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
    stripped = strip_leading_boilerplate(normalize_final_answer(answer))
    if not stripped:
        return EMPTY_ANSWER_FALLBACK

    # Remove common answer wrappers that only add visual noise in plain-text chat UI.
    stripped = re.sub(r"(?mi)^\s*##\s*answer\s*$", "", stripped).strip()
    stripped = re.sub(r"(?mi)^\s*\*\*\s*answer\s*:\s*\*\*\s*", "", stripped).strip()
    stripped = re.sub(r"(?mi)^\s*answer\s*:\s*", "", stripped).strip()
    stripped = re.sub(r"(?mi)^\s*final\s+answer\s*:\s*", "", stripped).strip()
    # Remove one fenced-code wrapper if model returns JSON in a code block.
    match = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", stripped, flags=re.IGNORECASE | re.DOTALL)
    if match:
        stripped = match.group(1).strip()
    stripped = strip_forbidden_meta_sections(stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
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


def build_file_signature(file_path: Path) -> str:
    try:
        stat = file_path.stat()
    except OSError:
        return ""
    return f"{file_path.name}:{int(stat.st_size)}:{int(stat.st_mtime)}"


def build_file_manifest_entries(files: list[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    entries: list[dict[str, Any]] = []
    signatures: dict[str, str] = {}
    for file_path in files:
        signature = build_file_signature(file_path)
        if not signature:
            continue
        try:
            stat = file_path.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": file_path.name,
                "size": int(stat.st_size),
                "mtime": int(stat.st_mtime),
                "signature": signature,
            }
        )
        signatures[file_path.name] = signature
    return entries, signatures


def read_index_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_index_manifest(
    manifest_path: Path,
    signature: str,
    chunk_count: int,
    embedding_backend: str,
    embedding_model: str,
    cache_namespace: str,
    source_files: list[Path],
) -> None:
    file_entries, file_signatures = build_file_manifest_entries(source_files)
    now_ts = int(time.time())
    payload = {
        "signature": signature,
        "chunk_count": int(max(0, chunk_count)),
        "updated_at": now_ts,
        "embedded_at": now_ts,
        "embedding_backend": str(embedding_backend or "").strip().lower(),
        "embedding_model": str(embedding_model or "").strip(),
        "chunk_size": RAG_CHUNK_SIZE,
        "chunk_overlap": RAG_CHUNK_OVERLAP,
        "cache_namespace": str(cache_namespace or "").strip(),
        "collection_name": INDEX_COLLECTION_NAME,
        "source_count": len(file_entries),
        "file_signatures": file_signatures,
        "files": file_entries,
    }
    try:
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        # Best effort only. Index still usable even if manifest write fails.
        return


def resolve_embedding_model_name(backend: str | None, model_override: str | None = None) -> str | None:
    normalized = str(backend or "").strip().lower()
    explicit_model = str(model_override or "").strip()
    if explicit_model:
        return explicit_model
    if normalized == "bert":
        return BERT_MODEL
    if normalized == "ollama":
        return EMBED_MODEL
    return None


def resolve_embedding_cache_namespace(backend: str | None, model_name: str | None = None) -> str:
    normalized = str(backend or "").strip().lower()
    if not normalized:
        return ""
    resolved_model_name = str(resolve_embedding_model_name(normalized, model_name) or "").strip().lower()
    if not resolved_model_name:
        return normalized
    safe_model = re.sub(r"[^a-z0-9._-]+", "_", resolved_model_name)
    return f"{normalized}_{safe_model}"


def get_index_paths(data_dir: Path, cache_namespace: str = "") -> tuple[Path, Path]:
    suffix = str(cache_namespace or "").strip().lower()
    if not suffix:
        return data_dir / INDEX_DIR_NAME, data_dir / INDEX_MANIFEST_NAME
    safe_suffix = re.sub(r"[^a-z0-9._-]+", "_", suffix)
    return (
        data_dir / f"{INDEX_DIR_NAME}_{safe_suffix}",
        data_dir / f".rag_index_manifest_{safe_suffix}.json",
    )


def build_vectorstore_from_docs(docs, embeddings: Embeddings) -> Chroma | None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)
    if not splits:
        return None
    return Chroma.from_documents(documents=splits, embedding=embeddings)


def load_or_build_cached_vectorstore(
    data_dir: Path,
    docs,
    embeddings: Embeddings,
    embedding_backend: str,
    embedding_model: str,
    cache_namespace: str = "",
) -> tuple[Chroma | None, bool]:
    source_files = list_source_files(data_dir)
    signature = build_data_signature(source_files)
    if not signature:
        return None, False

    index_dir, manifest_path = get_index_paths(data_dir, cache_namespace)
    manifest = read_index_manifest(manifest_path)
    is_cache_fresh = (
        index_dir.exists()
        and index_dir.is_dir()
        and manifest.get("signature") == signature
        and int(manifest.get("chunk_size") or 0) == RAG_CHUNK_SIZE
        and int(manifest.get("chunk_overlap") or -1) == RAG_CHUNK_OVERLAP
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)
    if not splits:
        return None, False

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(index_dir),
        collection_name=INDEX_COLLECTION_NAME,
    )
    write_index_manifest(
        manifest_path,
        signature,
        len(splits),
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        cache_namespace=cache_namespace,
        source_files=source_files,
    )
    return vectorstore, True


def normalize_lookup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_single_source(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return clean_loaded_docs(TextLoader(str(file_path), autodetect_encoding=True).load())
    if suffix == ".pdf":
        return clean_loaded_docs(PyPDFLoader(str(file_path)).load())
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
    for index, doc in enumerate(docs, start=1):
        role = str(doc.metadata.get("context_role") or "").strip().lower()
        if index == 1 or role == "primary":
            label = "Primary context"
        else:
            label = f"Supporting context {index}"
        chunks.append(f"[{label} | Source: {source_label(doc)}]\n{doc.page_content}")
    return "\n\n".join(chunks)


def annotate_retrieved_docs(docs):
    annotated = []
    for rank, doc in enumerate(docs or [], start=1):
        doc.metadata.setdefault("retrieval_rank", rank)
        doc.metadata.setdefault("context_role", "primary" if rank == 1 else "supporting")
        annotated.append(doc)
    return annotated


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
    pairs = vectorstore.similarity_search_with_relevance_scores(query, k=RAG_CANDIDATE_K, filter=metadata_filter)
    if not pairs:
        return []

    filtered = [(doc, score) for doc, score in pairs if score >= RELEVANCE_THRESHOLD]
    if not filtered:
        filtered = pairs[:RAG_TOP_K]

    scored = []
    for doc, score in filtered:
        focus = keyword_focus_score(query, doc.page_content)
        combined = float(score) + (focus * 0.65)
        scored.append((combined, float(score), float(focus), doc))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    docs = []
    for rank, (combined, score, focus, doc) in enumerate(scored[:RAG_TOP_K], start=1):
        doc.metadata["retrieval_rank"] = rank
        doc.metadata["retrieval_score"] = round(score, 4)
        doc.metadata["focus_score"] = round(focus, 4)
        doc.metadata["combined_score"] = round(combined, 4)
        doc.metadata["context_role"] = "primary" if rank == 1 else "supporting"
        docs.append(doc)
    return docs


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


def ask_general(llm: ChatOllama, query: str, markdown: bool = False, trace: TraceLogger | None = None) -> str:
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
    last_answer = EMPTY_ANSWER_FALLBACK
    attempts = max(1, max(0, retries) + 1)

    def invoke_once(current_prompt: str, llm_call: int) -> str:
        started = time.perf_counter()
        try:
            response = llm.invoke(current_prompt)
            rawanswer = response.content if hasattr(response, "content") else str(response)
            cleaned = clean_answer(str(rawanswer))
            if trace is not None:
                trace.log(
                    "ollama_llm_invoke_success",
                    llm_call=llm_call,
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
                    llm_call=llm_call,
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    prompt_chars=len(current_prompt),
                    error=trim_error(str(exc)),
                )
            raise

    for attempt in range(attempts):
        current_prompt = prompt
        if attempt > 0:
            current_prompt = (
                prompt
                + "\n\nIMPORTANT: Your previous answer was empty or unusable. "
                + "Return the final answer now in Markdown only, without <think> tags."
            )
        candidate = invoke_once(current_prompt, attempt + 1)
        last_answer = candidate
        if is_unusable_answer(last_answer):
            continue
        return last_answer

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
    parser.add_argument("--embed-backend", default="")
    parser.add_argument("--embed-model", default="")
    parser.add_argument("--eval-mode", action="store_true")
    parser.add_argument("--question-id", default="")
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--raw-results-path", default="")
    parser.add_argument("--eval-mode-name", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    eval_enabled = bool(args.eval_mode)
    eval_mode_name = str(args.eval_mode_name or "").strip()
    eval_question_id = str(args.question_id or "").strip()
    eval_run_id = int(args.run_id or 0)
    raw_results_path = str(args.raw_results_path or "").strip()
    trace = TraceLogger(
        log_path=str(args.trace_log or "").strip() or None,
        request_id=str(args.request_id or "").strip(),
        question_number=int(args.question_number or 0),
        attempt=int(args.attempt or 0),
    )
    eval_question_text = ""
    eval_embedding_backend: str | None = None
    eval_embedding_model: str | None = None
    eval_retrieved_context: list[dict[str, Any]] = []
    latency_retrieval_ms = 0
    latency_generation_ms = 0
    if eval_mode_name == "llm_only":
        eval_embedding_backend = "none"
    elif eval_mode_name == "rag_ollama":
        eval_embedding_backend = "ollama"
    elif eval_mode_name in {"rag_bert", "rag_msmarco"}:
        eval_embedding_backend = "bert"

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

        def resolve_effective_eval_mode() -> str:
            if eval_mode_name:
                return eval_mode_name
            if args.mode in {"general", "general_raw"}:
                return "llm_only"
            if eval_embedding_backend == "bert":
                return "rag_bert"
            if eval_embedding_backend == "ollama":
                return "rag_ollama"
            if args.mode == "auto":
                return "rag"
            return str(args.mode)

        def resolve_eval_corpus_metadata() -> dict[str, Any]:
            source_files = list_source_files(data_dir)
            return {
                "corpus_sources": [file_path.name for file_path in source_files],
                "corpus_signature": build_data_signature(source_files),
                "corpus_data_dir": str(data_dir),
            }

        def emit_answer_payload(
            answer_text: str,
            sources_list: list[str],
            emit_mode: str = "",
            status: str = "success",
            error_message: str | None = None,
        ) -> None:
            safe_sources = list(sources_list or [])
            final_answer = normalize_final_answer(str(answer_text or ""))
            effective_eval_mode = resolve_effective_eval_mode()
            answer_text_log, answer_truncated = truncate_text(final_answer, TRACE_TEXT_MAX_CHARS)
            trace.log(
                "python_response_emit",
                answer_chars=len(final_answer),
                answer_text=answer_text_log,
                answer_truncated=bool(answer_truncated),
                sources_count=len(safe_sources),
                mode=emit_mode,
            )
            raw_payload = build_raw_result_payload(
                question_id=eval_question_id,
                question=eval_question_text,
                mode=effective_eval_mode,
                run_id=eval_run_id,
                model_name=CHAT_MODEL,
                embedding_backend=eval_embedding_backend,
                embedding_model_name=eval_embedding_model or resolve_embedding_model_name(eval_embedding_backend),
                model_answer=final_answer,
                retrieved_context=eval_retrieved_context,
                latency_total_ms=int(round((time.perf_counter() - started) * 1000)),
                latency_retrieval_ms=latency_retrieval_ms,
                latency_generation_ms=latency_generation_ms,
                status=status,
                error_message=error_message,
            )
            payload: dict[str, Any] = {
                "answer": final_answer,
                "sources": safe_sources,
                "mode": effective_eval_mode,
                "question_id": str(raw_payload.get("question_id") or "").strip(),
                "run_id": int(raw_payload.get("run_id") or 0),
                "model_name": str(raw_payload.get("model_name") or "").strip(),
                "embedding_backend": raw_payload.get("embedding_backend"),
                "embedding_model_name": raw_payload.get("embedding_model_name"),
                "latency_total": float(raw_payload.get("latency_total") or 0.0),
                "latency_retrieval": float(raw_payload.get("latency_retrieval") or 0.0),
                "latency_generation": float(raw_payload.get("latency_generation") or 0.0),
                "retrieved_context_count": len(eval_retrieved_context),
                "status": str(raw_payload.get("status") or status).strip().lower(),
                "error_message": raw_payload.get("error_message"),
            }
            if eval_enabled:
                raw_payload.update(resolve_eval_corpus_metadata())
                payload.update(raw_payload)
                if raw_results_path:
                    append_jsonl(raw_results_path, raw_payload)
            emit(payload)

        def run_general_answer(llm: ChatOllama, prompt_text: str, markdown: bool = False) -> str:
            nonlocal latency_generation_ms
            generation_started = time.perf_counter()
            answer = ask_general(llm, prompt_text, markdown=markdown, trace=trace)
            latency_generation_ms = int(round((time.perf_counter() - generation_started) * 1000))
            return answer

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
            embeddings, embed_backend, embed_model = build_embeddings(args.embed_backend, args.embed_model)
            eval_embedding_backend = embed_backend
            eval_embedding_model = embed_model
            vectorstore, rebuilt = load_or_build_cached_vectorstore(
                data_dir,
                docs,
                embeddings,
                embedding_backend=embed_backend,
                embedding_model=embed_model,
                cache_namespace=resolve_embedding_cache_namespace(embed_backend, embed_model),
            )
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
                embedding_model=embed_model,
                rebuilt=bool(rebuilt),
            )
            emit(
                {
                    "ok": True,
                    "preparsed": True,
                    "rebuilt": bool(rebuilt),
                    "sources": source_count,
                    "embedding_backend": embed_backend,
                    "embedding_model": embed_model,
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
        query_for_answer = extract_current_user_question(query_for_answer)
        eval_question_text = str(query_for_answer or "").strip()
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
            emit_answer_payload(message, [], "ollama_unreachable", status="error", error_message=message)
            return

        llm = ChatOllama(
            model=CHAT_MODEL,
            temperature=0,
            base_url=OLLAMA_BASE_URL,
            num_predict=CHAT_NUM_PREDICT,
        )

        if args.mode == "general":
            answer = run_general_answer(llm, query_for_answer, markdown=False)
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
            answer = run_general_answer(llm, query_for_answer, markdown=False)
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
            answer = run_general_answer(llm, query_for_answer)
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
            answer = run_general_answer(llm, query_for_answer)
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
        embeddings, embed_backend, embed_model = build_embeddings(args.embed_backend, args.embed_model)
        eval_embedding_backend = embed_backend
        eval_embedding_model = embed_model
        trace.log(
            "rag_embeddings_ready",
            duration_ms=int(round((time.perf_counter() - embeddings_started) * 1000)),
            embedding_backend=embed_backend,
            embedding_model=embed_model,
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
        if explicit_source is None and query_mentions_file(query_for_answer):
            trace.log(
                "python_request_success",
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                answer_chars=len("Not found in context."),
                sources=0,
                mode="explicit_source_not_found",
            )
            emit_answer_payload(ensure_plain_answer("Not found in context."), [], "explicit_source_not_found")
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
                emit_answer_payload(ensure_plain_answer("Not found in context."), [], "single_source_empty")
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
                emit_answer_payload(ensure_plain_answer("Not found in context."), [], "single_source_vectorstore_failed")
                return
            use_similarity_threshold = False
        else:
            cache_vector_started = time.perf_counter()
            vectorstore, _ = load_or_build_cached_vectorstore(
                data_dir,
                docs,
                embeddings,
                embedding_backend=embed_backend,
                embedding_model=embed_model,
                cache_namespace=resolve_embedding_cache_namespace(embed_backend, embed_model),
            )
            trace.log(
                "rag_vectorstore_ready",
                duration_ms=int(round((time.perf_counter() - cache_vector_started) * 1000)),
                embedding_backend=embed_backend,
                embedding_model=embed_model,
            )
            if vectorstore is None:
                answer = run_general_answer(llm, query_for_answer)
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
            search_kwargs: dict[str, Any] = {"k": RAG_TOP_K}
            if page_filter is not None:
                search_kwargs["filter"] = page_filter
            context_docs = vectorstore.as_retriever(search_kwargs=search_kwargs).invoke(query_for_answer)
            context_docs = annotate_retrieved_docs(context_docs)
        trace.log(
            "rag_context_retrieved",
            duration_ms=int(round((time.perf_counter() - retrieval_started) * 1000)),
            context_docs_count=len(context_docs) if context_docs is not None else 0,
            use_similarity_threshold=bool(use_similarity_threshold),
        )
        latency_retrieval_ms = int(round((time.perf_counter() - retrieval_started) * 1000))
        eval_retrieved_context = serialize_retrieved_context(context_docs)

        if not context_docs:
            not_found_message = "Not found in context."
            if page_start is not None and page_end is not None:
                not_found_message = (
                    f"Not found in selected page range ({page_start}-{page_end}). "
                    "Try widening page range."
                )
            if explicit_source is not None or query_mentions_file(query_for_answer):
                trace.log(
                    "python_request_success",
                    duration_ms=int(round((time.perf_counter() - started) * 1000)),
                    answer_chars=len(not_found_message),
                    sources=0,
                    mode="no_context_docs",
                )
                emit_answer_payload(not_found_message, [], "no_context_docs")
            else:
                answer = run_general_answer(llm, query_for_answer)
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

        generation_started = time.perf_counter()
        answer = invoke_llm_with_retry(llm, prompt, retries=1, trace=trace)
        answer = ensure_plain_answer(answer)
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
            strict_answer = ensure_plain_answer(strict_answer)
            if answer_focus_coverage(query_for_answer, strict_answer) >= focus_coverage:
                answer = strict_answer
        latency_generation_ms = int(round((time.perf_counter() - generation_started) * 1000))

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
            if explicit_source is None and not query_mentions_file(query_for_answer):
                answer = run_general_answer(llm, query_for_answer)
                sources = []
        elif "cannot access" in lowered and "file" in lowered:
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
        error_message = f"RAG backend error: {exc}"
        trace.log(
            "python_request_error",
            level="error",
            duration_ms=int(round((time.perf_counter() - started) * 1000)),
            error=trim_error(str(exc)),
            traceback=trim_error(traceback.format_exc(), 12000),
        )
        emit_answer_payload(error_message, [], "error", status="error", error_message=error_message)


if __name__ == "__main__":
    main()
