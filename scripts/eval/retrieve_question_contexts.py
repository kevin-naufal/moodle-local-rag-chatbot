from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_output_path(dataset_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "data" / "eval_results" / f"retrieved_contexts_{dataset_path.stem}_{stamp}.jsonl"


def build_retrieval_rows(
    dataset: dict[str, Any],
    retrieve_context: Callable[[str], list[dict[str, Any]]],
    *,
    embedding_backend: str,
    embedding_model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(list(dataset.get("questions") or []), start=1):
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        question_id = str(item.get("id") or item.get("question_id") or f"q{index:03d}").strip()
        retrieved_context = retrieve_context(question)
        rows.append(
            {
                "question_id": question_id,
                "question": question,
                "embedding_backend": str(embedding_backend or "").strip(),
                "embedding_model": str(embedding_model or "").strip(),
                "retrieved_context_count": len(retrieved_context),
                "retrieved_context": retrieved_context,
            }
        )
    return rows


def retrieve_contexts(
    dataset: dict[str, Any],
    *,
    data_dir: Path,
    embed_backend: str,
    embed_model: str,
) -> tuple[list[dict[str, Any]], str, str, bool]:
    if str(PROJECT_ROOT / "app") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "app"))

    from moodle_rag_runner import (  # pylint: disable=import-outside-toplevel
        build_embeddings,
        get_relevant_docs,
        load_docs,
        load_or_build_cached_vectorstore,
        resolve_embedding_cache_namespace,
        serialize_retrieved_context,
    )

    docs = load_docs(data_dir)
    if not docs:
        raise ValueError(f"No PDF/TXT documents found in data dir: {data_dir}")

    embeddings, resolved_backend, resolved_model = build_embeddings(embed_backend, embed_model)
    vectorstore, rebuilt = load_or_build_cached_vectorstore(
        data_dir,
        docs,
        embeddings,
        embedding_backend=resolved_backend,
        embedding_model=resolved_model,
        cache_namespace=resolve_embedding_cache_namespace(resolved_backend, resolved_model),
    )
    if vectorstore is None:
        raise RuntimeError(f"Unable to build or load vectorstore for data dir: {data_dir}")

    def retrieve(question: str) -> list[dict[str, Any]]:
        docs_for_question = get_relevant_docs(vectorstore, question)
        return serialize_retrieved_context(docs_for_question)

    rows = build_retrieval_rows(
        dataset,
        retrieve,
        embedding_backend=resolved_backend,
        embedding_model=resolved_model,
    )
    return rows, resolved_backend, resolved_model, bool(rebuilt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve RAG context for every question in an eval dataset.")
    parser.add_argument("--questions", required=True, help="Path to question dataset JSON.")
    parser.add_argument("--data-dir", default="data/eval_ch03_only", help="Path to corpus directory used by RAG.")
    parser.add_argument("--output", default="", help="Optional JSONL output path.")
    parser.add_argument("--embed-backend", default="auto", help="Embedding backend: auto, bert, or ollama.")
    parser.add_argument("--embed-model", default="", help="Optional embedding model override.")
    args = parser.parse_args()

    questions_path = Path(args.questions).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve() if str(args.output).strip() else create_output_path(questions_path)

    dataset = load_json(questions_path)
    rows, resolved_backend, resolved_model, rebuilt = retrieve_contexts(
        dataset,
        data_dir=data_dir,
        embed_backend=str(args.embed_backend or "").strip(),
        embed_model=str(args.embed_model or "").strip(),
    )
    write_jsonl(output_path, rows)

    print("Question context retrieval completed.")
    print(f"- questions: {questions_path}")
    print(f"- data_dir: {data_dir}")
    print(f"- output: {output_path}")
    print(f"- total_questions: {len(rows)}")
    print(f"- embedding_backend: {resolved_backend}")
    print(f"- embedding_model: {resolved_model}")
    print(f"- vectorstore_rebuilt: {rebuilt}")


if __name__ == "__main__":
    main()
