import os
import sys
import argparse
import re

parser = argparse.ArgumentParser(
    description="Simple RAG demo over files in ./data"
)
parser.add_argument(
    "query",
    nargs="*",
    help="Question to ask (leave empty to use default)",
)
parser.add_argument(
    "--file",
    dest="pdf_file",
    help="PDF file name (inside ./data) or full path",
)
parser.add_argument(
    "--page",
    dest="pdf_page",
    type=int,
    help="1-based page number to target from the PDF",
)
parser.add_argument(
    "--show-page",
    action="store_true",
    help="Print the extracted page text and exit",
)
parser.add_argument(
    "--find",
    dest="find_text",
    help="Find text in a PDF and list matching page numbers",
)
args = parser.parse_args()

# Check standard input for queries if provided, otherwise use default
query = None
if args.query:
    query = " ".join(args.query)

# If query mentions a page number, use it automatically
if query and args.pdf_page is None:
    match = re.search(r"\bpage\s+(\d+)\b", query, flags=re.IGNORECASE)
    if match:
        args.pdf_page = int(match.group(1))

try:
    from langchain_community.document_loaders import (
        DirectoryLoader,
        TextLoader,
        PyPDFLoader,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# 1. Load Documents
print("Loading documents from ./data...")
docs = []
if args.find_text:
    if not args.pdf_file:
        print("Error: --find requires --file to be provided.")
        sys.exit(1)
    pdf_path = args.pdf_file
    if not os.path.exists(pdf_path):
        pdf_path = os.path.join("./data", args.pdf_file)
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {args.pdf_file}")
        sys.exit(1)
    pdf_pages = PyPDFLoader(pdf_path).load()
    needle = args.find_text.lower()
    matches = []
    for idx, page in enumerate(pdf_pages):
        if needle in page.page_content.lower():
            matches.append(idx + 1)
    if matches:
        print(f"Found '{args.find_text}' on pages: {', '.join(map(str, matches))}")
    else:
        print(f"No matches found for '{args.find_text}'.")
    sys.exit(0)
elif args.pdf_page is not None:
    if not args.pdf_file:
        print("Error: --page requires --file to be provided.")
        sys.exit(1)
    pdf_path = args.pdf_file
    if not os.path.exists(pdf_path):
        pdf_path = os.path.join("./data", args.pdf_file)
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {args.pdf_file}")
        sys.exit(1)
    pdf_pages = PyPDFLoader(pdf_path).load()
    page_index = args.pdf_page - 1
    if page_index < 0 or page_index >= len(pdf_pages):
        print(f"Error: page {args.pdf_page} is out of range (1-{len(pdf_pages)}).")
        sys.exit(1)
    docs = [pdf_pages[page_index]]
    if args.show_page:
        print("\n--- Extracted Page Text ---\n")
        print(docs[0].page_content)
        sys.exit(0)
    if not args.query:
        query = f"Summarize page {args.pdf_page} of {os.path.basename(pdf_path)}."
else:
    if args.pdf_file:
        pdf_path = args.pdf_file
        if not os.path.exists(pdf_path):
            pdf_path = os.path.join("./data", args.pdf_file)
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found: {args.pdf_file}")
            sys.exit(1)
        docs = PyPDFLoader(pdf_path).load()
    else:
        txt_loader = DirectoryLoader("./data", glob="**/*.txt", loader_cls=TextLoader)
        pdf_loader = DirectoryLoader("./data", glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs = txt_loader.load() + pdf_loader.load()

print(f"Loaded {len(docs)} documents.")

if query is None:
    query = "What is the capital of France?"

print(f"Querying: {query}...")

# 2. Split (even though files are small, this is best practice)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Embed and Store (skip when targeting a specific page)
retriever = None
page_context = None
if args.pdf_page is not None:
    # Use the full page content to avoid missing later items.
    page_context = "\n\n".join(doc.page_content for doc in docs)
else:
    print("Initializing Vector Store (this may take a moment)...")
    # Ensure you have 'nomic-embed-text' pulled in Ollama: `ollama pull nomic-embed-text`
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")

    try:
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        print("Ensure you have a valid embedding model (e.g., 'ollama pull nomic-embed-text')")
        sys.exit(1)

# 4. RAG Chain
print("Initializing LLM...")
llm = ChatOllama(
    model="hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M",
    temperature=0,
)

template = """You are a careful assistant. Use ONLY the following context to answer the question.
If the answer is not in the context, say "Not found in context."
Answer directly and concisely. Do not start with "Based on the context".

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if page_context is not None:
    rag_chain = (
        {"context": lambda _: page_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# 5. Execute
print("\n--- Answer ---")
try:
    response = rag_chain.invoke(query)
    print(response)
except Exception as e:
    print(f"Error during query: {e}")
