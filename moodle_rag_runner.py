import argparse
import base64
import json
import re
import sys
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

PROMPT_TEMPLATE = """You are a careful assistant. Use ONLY the following context to answer the question.
If the answer is not in the context, say "Not found in context."
Answer directly and concisely. Do not start with "Based on the context".
Never output internal reasoning tags like <think>.

Context:
{context}

Question: {question}
"""

GENERAL_PROMPT_TEMPLATE = """You are a helpful assistant.
Answer the user's question directly and concisely.
Never output internal reasoning tags like <think>.

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

    greetings = ("hello", "hi", "halo", "hey")
    if any(token in normalized for token in greetings) or "how are you" in normalized:
        return (
            "Hello. I can help with questions about your uploaded documents. "
            "Try asking about a specific topic or page."
        )

    return None


def clean_answer(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip() or "Sorry, I cannot provide an answer for that question yet."


def get_relevant_docs(vectorstore: Chroma, query: str):
    # Score is expected in range [0, 1], where larger means more relevant.
    pairs = vectorstore.similarity_search_with_relevance_scores(query, k=4)
    docs = [doc for doc, score in pairs if score >= RELEVANCE_THRESHOLD]
    return docs


def emit(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))


def ask_general(llm: ChatOllama, query: str) -> str:
    prompt = GENERAL_PROMPT_TEMPLATE.format(question=query)
    response = llm.invoke(prompt)
    rawanswer = response.content if hasattr(response, "content") else str(response)
    return clean_answer(str(rawanswer))


def main() -> None:
    # Ambil parameter dari CLI (dipanggil oleh Moodle plugin).
    parser = argparse.ArgumentParser(description="RAG runner for Moodle local_chatbot plugin")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--query")
    parser.add_argument("--query-b64")
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

        llm = ChatOllama(model=CHAT_MODEL, temperature=0)

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
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        context_docs = get_relevant_docs(vectorstore, query)
        if not context_docs:
            emit({"answer": ask_general(llm, query), "sources": []})
            return

        # Bentuk prompt RAG dan minta jawaban dari model.
        context = "\n\n".join(doc.page_content for doc in context_docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)

        response = llm.invoke(prompt)
        rawanswer = response.content if hasattr(response, "content") else str(response)
        answer = clean_answer(str(rawanswer))

        seen = set()
        sources = []
        for doc in context_docs:
            label = source_label(doc)
            if label not in seen:
                seen.add(label)
                sources.append(label)

        # Jika model bilang context tidak cukup, fallback ke jawaban umum.
        if answer.lower().startswith("not found in context"):
            answer = ask_general(llm, query)
            sources = []

        emit({"answer": str(answer), "sources": sources})
    except Exception as exc:
        emit({"answer": f"RAG backend error: {exc}", "sources": []})


if __name__ == "__main__":
    main()
