from __future__ import annotations

from pathlib import Path
from typing import Iterable

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"

PROMPT_TEMPLATE = """You are a careful assistant. Use ONLY the following context to answer the question.
If the answer is not in the context, say "Not found in context."
Answer directly and concisely. Do not start with "Based on the context".

Context:
{context}

Question: {question}
"""


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def list_source_files() -> list[Path]:
    files: list[Path] = []
    for ext in ("*.pdf", "*.txt"):
        files.extend(DATA_DIR.glob(ext))
    files.sort(key=lambda item: item.name.lower())
    return files


def file_fingerprint(files: Iterable[Path]) -> str:
    parts: list[str] = []
    for file_path in files:
        stat = file_path.stat()
        parts.append(f"{file_path.name}:{stat.st_size}:{int(stat.st_mtime)}")
    return "|".join(parts)


def load_documents(files: Iterable[Path]) -> list[Document]:
    docs: list[Document] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            docs.extend(TextLoader(str(file_path), autodetect_encoding=True).load())
        elif suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file_path)).load())
    return docs


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL, temperature=0)


@st.cache_resource(show_spinner=False)
def get_retriever(signature: str):
    _ = signature
    files = list_source_files()
    docs = load_documents(files)
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def build_sources(docs: list[Document]) -> list[str]:
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


def ask_rag(question: str, signature: str) -> tuple[str, list[str]]:
    retriever = get_retriever(signature)
    if retriever is None:
        return "No documents yet. Upload PDF/TXT files first.", []

    context_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = get_llm().invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return str(content), build_sources(context_docs)


def render_file_list(files: list[Path]) -> None:
    st.markdown("### Uploaded Documents")
    if not files:
        st.caption("Belum ada file di-upload.")
        return
    for file_path in files:
        size_kb = file_path.stat().st_size / 1024
        st.markdown(
            f"- **{file_path.name}**  \n`{size_kb:.1f} KB`",
        )


def main() -> None:
    st.set_page_config(page_title="Campus RAG Assistant", page_icon=":books:", layout="wide")
    ensure_data_dir()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello. Upload PDF/TXT in the left panel, then ask about your documents here.",
                "sources": [],
            }
        ]

    st.markdown(
        """
        <style>
          .block-container { padding-top: 1rem; padding-bottom: 1rem; }
          .stChatMessage { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown("## Insert PDF/TXT")
        st.caption("Upload source documents for the chatbot. Supported: .pdf and .txt")

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
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

    with right_col:
        files = list_source_files()
        signature = file_fingerprint(files)
        ready = "RAG ready" if files else "No documents yet"
        st.markdown(f"## Chat with your documents  \n`{ready}`")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("sources"):
                    st.caption("source: " + ", ".join(message["sources"]))

        question = st.chat_input("Ask a question about uploaded files...")
        if question:
            st.session_state.messages.append(
                {"role": "user", "content": question, "sources": []}
            )
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer, sources = ask_rag(question, signature)
                    except Exception as exc:
                        answer = f"Failed to process question: {exc}"
                        sources = []
                st.write(answer)
                if sources:
                    st.caption("source: " + ", ".join(sources))

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )


if __name__ == "__main__":
    main()
