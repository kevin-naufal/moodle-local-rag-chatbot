from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"
MAX_STORED_MESSAGES = 200

PROMPT_TEMPLATE = """You are a careful assistant. Use ONLY the following context to answer the question.
If the answer is not in the context, say "Not found in context."
Answer directly and concisely. Do not start with "Based on the context".

Context:
{context}

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


def list_source_files() -> list[Path]:
    # Ambil semua file PDF/TXT dari folder data sebagai sumber RAG.
    files: list[Path] = []
    for ext in ("*.pdf", "*.txt"):
        files.extend(DATA_DIR.glob(ext))
    files.sort(key=lambda item: item.name.lower())
    return files


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


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOllama:
    # Cache model chat supaya tidak inisialisasi ulang tiap pertanyaan.
    return ChatOllama(model=CHAT_MODEL, temperature=0)


@st.cache_resource(show_spinner=False)
def get_retriever(signature: str):
    # Signature hanya untuk invalidasi cache ketika dokumen berubah.
    _ = signature
    files = list_source_files()
    docs = load_documents(files)
    if not docs:
        return None

    # Dokumen dipecah jadi chunk lalu diubah jadi embedding untuk retrieval.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
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


def ask_rag(question: str, signature: str) -> tuple[str, list[str]]:
    # step 5: Ambil context relevan dari retriever.
    retriever = get_retriever(signature)
    if retriever is None:
        return "No documents yet. Upload PDF/TXT files first.", []

    # Lanjut: gabungkan context + question ke prompt, lalu panggil LLM.
    context_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = get_llm().invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return str(content), build_sources(context_docs)


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


def main() -> None:
    # step 1: Setup halaman app, folder kerja, dan chat_id aktif.
    st.set_page_config(page_title="Campus RAG Assistant", page_icon=":books:", layout="wide")
    ensure_data_dir()
    ensure_chat_store_dir()
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

    # Dua kolom utama: kiri untuk upload dokumen, kanan untuk chat.
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        # step 3: Panel upload dokumen sumber (PDF/TXT).
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

    with right_col:
        # step 4: Render histori chat dan tunggu pertanyaan user.
        files = list_source_files()
        signature = file_fingerprint(files)
        # Status singkat dipakai untuk indikasi apakah basis dokumen sudah siap.
        ready = "RAG ready" if files else "No documents yet"
        st.markdown(f"## Chat with your documents  \n`{ready}`")

        # Render semua pesan histori sebelum menerima input baru.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
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
                    try:
                        answer, sources = ask_rag(question, signature)
                    except Exception as exc:
                        answer = f"Failed to process question: {exc}"
                        sources = []
                st.write(answer)
                if sources:
                    st.caption("source: " + ", ".join(sources))

            # Lanjut: simpan jawaban assistant ke histori.
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            save_messages(chat_id, st.session_state.messages)


if __name__ == "__main__":
    main()
