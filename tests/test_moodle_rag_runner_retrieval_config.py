import importlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"


class MoodleRagRunnerRetrievalConfigTest(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(APP_DIR))
        sys.modules.pop("moodle_rag_runner", None)

    def tearDown(self):
        if sys.path and sys.path[0] == str(APP_DIR):
            sys.path.pop(0)
        sys.modules.pop("moodle_rag_runner", None)
        for name in (
            "RAG_TOP_K",
            "RAG_CANDIDATE_K",
            "RAG_CHUNK_SIZE",
            "RAG_CHUNK_OVERLAP",
        ):
            os.environ.pop(name, None)

    def test_retrieval_config_reads_environment(self):
        os.environ["RAG_TOP_K"] = "6"
        os.environ["RAG_CANDIDATE_K"] = "12"
        os.environ["RAG_CHUNK_SIZE"] = "800"
        os.environ["RAG_CHUNK_OVERLAP"] = "120"

        runner = importlib.import_module("moodle_rag_runner")

        self.assertEqual(runner.RAG_TOP_K, 6)
        self.assertEqual(runner.RAG_CANDIDATE_K, 12)
        self.assertEqual(runner.RAG_CHUNK_SIZE, 800)
        self.assertEqual(runner.RAG_CHUNK_OVERLAP, 120)

    def test_prompt_blocks_unsupported_elaboration(self):
        runner = importlib.import_module("moodle_rag_runner")

        self.assertIn("Do not add examples, benefits, causes, or implications", runner.PROMPT_TEMPLATE)
        self.assertIn("explicitly supported by the context", runner.PROMPT_TEMPLATE)
        self.assertIn("Use the Primary context first", runner.PROMPT_TEMPLATE)
        self.assertIn("prioritize the part that directly answers the question", runner.PROMPT_TEMPLATE)

    def test_document_text_cleanup_normalizes_pdf_ligatures(self):
        runner = importlib.import_module("moodle_rag_runner")

        self.assertEqual(runner.clean_document_text("e\ufb03cient \ufb02ow"), "efficient flow")

    def test_format_context_marks_primary_and_supporting_chunks(self):
        runner = importlib.import_module("moodle_rag_runner")
        docs = [
            SimpleNamespace(page_content="direct answer", metadata={"source": "book.pdf", "page": 0, "context_role": "primary"}),
            SimpleNamespace(page_content="extra detail", metadata={"source": "book.pdf", "page": 1, "context_role": "supporting"}),
        ]

        context = runner.format_context(docs)

        self.assertIn("[Primary context | Source: book.pdf p.1]", context)
        self.assertIn("[Supporting context 2 | Source: book.pdf p.2]", context)

    def test_build_focused_excerpt_prefers_question_matching_sentence(self):
        runner = importlib.import_module("moodle_rag_runner")
        text = (
            "Running time matters for large programs. "
            "As a general rule, choose an algorithm that is easy to understand, implement, and document. "
            "Big-oh notation is discussed later."
        )

        excerpt = runner.build_focused_excerpt(
            text,
            "What general rule does the chapter give for choosing an algorithm?",
            max_sentences=1,
        )

        self.assertIn("general rule", excerpt.lower())
        self.assertIn("easy to understand", excerpt)
        self.assertNotIn("Big-oh", excerpt)

    def test_format_context_uses_focused_excerpt_when_query_is_available(self):
        runner = importlib.import_module("moodle_rag_runner")
        doc = SimpleNamespace(
            page_content=(
                "Running time matters for large programs. "
                "As a general rule, choose an algorithm that is easy to understand, implement, and document. "
                "Big-oh notation is discussed later."
            ),
            metadata={"source": "book.pdf", "page": 0, "context_role": "primary"},
        )

        context = runner.format_context(
            [doc],
            query="What general rule does the chapter give for choosing an algorithm?",
        )
        [item] = runner.serialize_retrieved_context([doc])

        self.assertIn("Focused excerpt:", context)
        self.assertIn("easy to understand", context)
        self.assertNotIn("Big-oh notation is discussed later", context)
        self.assertIn("focused_excerpt", item)
        self.assertNotIn("Big-oh notation is discussed later", item["focused_excerpt"])

    def test_serialize_retrieved_context_includes_retrieval_metadata(self):
        runner = importlib.import_module("moodle_rag_runner")
        doc = SimpleNamespace(
            page_content="direct answer",
            metadata={
                "source": "book.pdf",
                "page": 0,
                "retrieval_rank": 1,
                "retrieval_score": 0.75,
                "focus_score": 0.9,
                "combined_score": 1.335,
                "context_role": "primary",
            },
        )

        [item] = runner.serialize_retrieved_context([doc])

        self.assertEqual(item["retrieval_rank"], 1)
        self.assertEqual(item["retrieval_score"], 0.75)
        self.assertEqual(item["focus_score"], 0.9)
        self.assertEqual(item["combined_score"], 1.335)
        self.assertEqual(item["context_role"], "primary")

    def test_get_relevant_docs_reranks_and_annotates_metadata(self):
        runner = importlib.import_module("moodle_rag_runner")
        direct = SimpleNamespace(page_content="general rule choose easy understand implement document", metadata={})
        noisy = SimpleNamespace(page_content="running time performance efficient", metadata={})
        vectorstore = SimpleNamespace(
            similarity_search_with_relevance_scores=lambda query, k, filter=None: [
                (noisy, 0.8),
                (direct, 0.6),
            ]
        )

        docs = runner.get_relevant_docs(vectorstore, "What general rule does the chapter give for choosing an algorithm?")

        self.assertIs(docs[0], direct)
        self.assertEqual(docs[0].metadata["retrieval_rank"], 1)
        self.assertEqual(docs[0].metadata["context_role"], "primary")
        self.assertGreater(docs[0].metadata["focus_score"], docs[1].metadata["focus_score"])
        self.assertEqual(docs[1].metadata["context_role"], "supporting")

    def test_annotate_retrieved_docs_adds_rank_and_role_to_raw_docs(self):
        runner = importlib.import_module("moodle_rag_runner")
        docs = [
            SimpleNamespace(page_content="first", metadata={}),
            SimpleNamespace(page_content="second", metadata={}),
        ]

        annotated = runner.annotate_retrieved_docs(docs)

        self.assertEqual(annotated[0].metadata["retrieval_rank"], 1)
        self.assertEqual(annotated[0].metadata["context_role"], "primary")
        self.assertEqual(annotated[1].metadata["retrieval_rank"], 2)
        self.assertEqual(annotated[1].metadata["context_role"], "supporting")


if __name__ == "__main__":
    unittest.main()
