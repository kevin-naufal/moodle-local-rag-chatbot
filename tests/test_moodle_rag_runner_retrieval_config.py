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

    def test_plain_answer_strips_leaked_strict_focus_retry_instructions(self):
        runner = importlib.import_module("moodle_rag_runner")

        answer = runner.ensure_plain_answer(
            "STRICT FOCUS RULE:\n"
            "- Target concept: general rule\n"
            "- Specific answer: choose a simple algorithm\n\n"
            "Explanation: We should pick an algorithm that is easy to understand, implement, and document."
        )

        self.assertNotIn("STRICT FOCUS RULE", answer)
        self.assertNotIn("Target concept", answer)
        self.assertNotIn("Explanation:", answer)
        self.assertEqual(answer, "We should pick an algorithm that is easy to understand, implement, and document.")

    def test_plain_answer_strips_single_line_strict_focus_retry_instruction(self):
        runner = importlib.import_module("moodle_rag_runner")

        answer = runner.ensure_plain_answer(
            "STRICT FOCUS RULE: general rule, does, chapter\n\n"
            "The chapter says to choose an algorithm that is easy to understand, implement, and document."
        )

        self.assertNotIn("STRICT FOCUS RULE", answer)
        self.assertNotIn("general rule, does, chapter", answer)
        self.assertEqual(
            answer,
            "The chapter says to choose an algorithm that is easy to understand, implement, and document.",
        )

    def test_context_evidence_detects_answerable_not_found_case(self):
        runner = importlib.import_module("moodle_rag_runner")
        docs = [
            SimpleNamespace(
                page_content=(
                    "Simple algorithms are desirable because they are easier to implement correctly "
                    "than complex ones. They are less likely to have subtle bugs after unexpected input "
                    "and are easier to describe and maintain by other people over time."
                ),
                metadata={"context_role": "primary"},
            )
        ]

        self.assertTrue(
            runner.has_context_answer_evidence(
                "Why is simplicity important when choosing an algorithm for long-term use?",
                docs,
            )
        )

    def test_context_evidence_rejects_unrelated_context(self):
        runner = importlib.import_module("moodle_rag_runner")
        docs = [
            SimpleNamespace(
                page_content="Merge sort divides a list and has running time proportional to n log n.",
                metadata={"context_role": "primary"},
            )
        ]

        self.assertFalse(
            runner.has_context_answer_evidence(
                "Why is simplicity important when choosing an algorithm for long-term use?",
                docs,
            )
        )

    def test_extractive_fallback_preserves_one_time_program_answer(self):
        runner = importlib.import_module("moodle_rag_runner")
        docs = [
            SimpleNamespace(
                page_content=(
                    "If you need to write a program that will be used once on small amounts of data "
                    "and then discarded, then you should select the easiest-to-implement algorithm "
                    "you know, get the program written and debugged, and move on to something else."
                ),
                metadata={"context_role": "primary"},
            )
        ]

        answer = runner.build_extractive_context_answer(
            "According to Section 3.2, what should you do if a program will be used once on small amounts of data and then discarded?",
            docs,
        )

        self.assertIn("easiest-to-implement algorithm", answer)
        self.assertIn("written and debugged", answer)
        self.assertIn("move on", answer)

    def test_extractive_fallback_preserves_numbered_resource_list(self):
        runner = importlib.import_module("moodle_rag_runner")
        docs = [
            SimpleNamespace(
                page_content=(
                    "Generally, we associate efficiency with the time it takes a program to run, "
                    "although there are other resources that a program sometimes must conserve, such as "
                    "1. The amount of storage space taken by its variables. "
                    "2. The amount of traffic it generates on a network of computers. "
                    "3. The amount of data that must be moved to and from disks."
                ),
                metadata={"context_role": "primary"},
            )
        ]

        answer = runner.build_extractive_context_answer(
            "Besides running time, what three other resources does the chapter say a program may sometimes need to conserve?",
            docs,
        )

        self.assertIn("storage space", answer)
        self.assertIn("network", answer)
        self.assertIn("disks", answer)

    def test_extractive_fallback_preserves_ocr_split_numbered_resource_list(self):
        runner = importlib.import_module("moodle_rag_runner")
        docs = [
            SimpleNamespace(
                page_content=(
                    "Generally, we associate efficiency with the time it takes a program to run, "
                    "although there are other resources th at a program sometimes\n"
                    "must conserve, such as\n"
                    "1. The amount of storage space taken by its variables.\n"
                    "2. The amount of traffic it generates on a network of computers.\n"
                    "3. The amount of data that must be moved to and from disks."
                ),
                metadata={"context_role": "primary"},
            )
        ]

        answer = runner.build_extractive_context_answer(
            "Besides running time, what three other resources does the chapter say a program may sometimes need to conserve?",
            docs,
        )

        self.assertIn("storage space", answer)
        self.assertIn("network", answer)
        self.assertIn("disks", answer)

    def test_extractive_fallback_preserves_ocr_split_principal_approaches_list(self):
        runner = importlib.import_module("moodle_rag_runner")
        docs = [
            SimpleNamespace(
                page_content=(
                    "The two principal approaches to summarizing the running time ar e\n"
                    "1. Benchmarking\n"
                    "2. Analysis\n"
                    "We shall consider each in turn."
                ),
                metadata={"context_role": "primary"},
            )
        ]

        answer = runner.build_extractive_context_answer(
            "What two principal approaches does Section 3.3 give for summarizing the running time of a program?",
            docs,
        )

        self.assertIn("Benchmarking", answer)
        self.assertIn("Analysis", answer)

    def test_list_answer_gap_detects_short_answer_missing_context_list_items(self):
        runner = importlib.import_module("moodle_rag_runner")
        docs = [
            SimpleNamespace(
                page_content=(
                    "There are other resources th at a program sometimes must conserve, such as\n"
                    "1. The amount of storage space taken by its variables.\n"
                    "2. The amount of traffic it generates on a network of computers.\n"
                    "3. The amount of data that must be moved to and from disks."
                ),
                metadata={"context_role": "primary"},
            )
        ]

        self.assertTrue(
            runner.should_use_extractive_list_fallback(
                "Besides running time, what three other resources does the chapter say a program may sometimes need to conserve?",
                "resources",
                docs,
            )
        )

    def test_list_answer_gap_accepts_answer_covering_context_list_items(self):
        runner = importlib.import_module("moodle_rag_runner")
        docs = [
            SimpleNamespace(
                page_content=(
                    "There are other resources th at a program sometimes must conserve, such as\n"
                    "1. The amount of storage space taken by its variables.\n"
                    "2. The amount of traffic it generates on a network of computers.\n"
                    "3. The amount of data that must be moved to and from disks."
                ),
                metadata={"context_role": "primary"},
            )
        ]

        self.assertFalse(
            runner.should_use_extractive_list_fallback(
                "Besides running time, what three other resources does the chapter say a program may sometimes need to conserve?",
                "The three resources are storage space, network traffic, and data moved to and from disks.",
                docs,
            )
        )

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

    def test_focus_terms_prioritize_content_words_over_question_boilerplate(self):
        runner = importlib.import_module("moodle_rag_runner")

        terms = runner.extract_focus_terms(
            "According to Section 3.2, what should you do if a program will be used once on small amounts of data and then discarded?"
        )

        self.assertNotIn("according", terms)
        self.assertNotIn("should", terms)
        self.assertIn("once", terms)
        self.assertIn("small", terms)
        self.assertIn("discarded", terms)

    def test_get_relevant_docs_boosts_explicit_section_answer_chunk(self):
        runner = importlib.import_module("moodle_rag_runner")
        maintainability = SimpleNamespace(
            page_content=(
                "people over a long period of time, other issues arise. One is the understandability, "
                "or simplicity, of the underlying algorithm. Simple algorithms are easier to implement "
                "correctly and less likely to have subtle bugs. Programs should be written clearly and "
                "documented carefully so that they can be maintained by others."
            ),
            metadata={},
        )
        answer_chunk = SimpleNamespace(
            page_content=(
                "3.2 Choosing an Algorithm\n"
                "If you need to write a program that will be used once on small amounts of data "
                "and then discarded, then you should select the easiest-to-implement algorithm you "
                "know, get the program written and debugged, and move on to something else."
            ),
            metadata={},
        )
        analysis = SimpleNamespace(
            page_content=(
                "Analysis of a Program. To analyze a program, we begin by grouping inputs according "
                "to size. What we choose to call the size of an input can vary from program to program."
            ),
            metadata={},
        )
        vectorstore = SimpleNamespace(
            similarity_search_with_relevance_scores=lambda query, k, filter=None: [
                (maintainability, 0.8578),
                (analysis, 0.8329),
                (answer_chunk, 0.8354),
            ]
        )

        docs = runner.get_relevant_docs(
            vectorstore,
            "According to Section 3.2, what should you do if a program will be used once on small amounts of data and then discarded?",
        )

        self.assertIs(docs[0], answer_chunk)
        self.assertEqual(docs[0].metadata["context_role"], "primary")
        self.assertGreater(docs[0].metadata["section_match_score"], 0)
        self.assertGreater(docs[0].metadata["direct_answer_score"], docs[1].metadata["direct_answer_score"])

    def test_focused_excerpt_keeps_full_one_time_program_answer(self):
        runner = importlib.import_module("moodle_rag_runner")
        text = (
            "90 THE RUNNING TIME OF PROGRAMS with no function calls. "
            "Section 3.8 extends our capability to programs with calls to nonrecursive functions. "
            "3.2 Choosing an Algorithm\n"
            "If you need to write a program that will be used once on small amounts of data "
            "and then discarded, then you should select the easiest-to-implement algorithm you "
            "know, get the program written and debugged, and move on to something else. "
            "When a program is to be used and maintained by many people, other issues arise."
        )

        excerpt = runner.build_focused_excerpt(
            text,
            "According to Section 3.2, what should you do if a program will be used once on small amounts of data and then discarded?",
            max_sentences=2,
        )

        self.assertIn("used once", excerpt)
        self.assertIn("easiest-to-implement algorithm", excerpt)
        self.assertIn("written and debugged", excerpt)
        self.assertIn("move on", excerpt)

    def test_focused_excerpt_joins_pdf_line_wrapped_answer_sentence(self):
        runner = importlib.import_module("moodle_rag_runner")
        text = (
            "3.2 Choosing an Algorithm\n"
            "If you need to write a program that will be used once on small am ounts of data\n"
            "and then discarded, then you should select the easiest-to-i mplement algorithm you\n"
            "know, get the program written and debugged, and move on to som ething else. How-\n"
            "ever, when you need to write a program that is to be used and maintained by many\n"
            "people over a long period of time, other issues arise."
        )

        excerpt = runner.build_focused_excerpt(
            text,
            "According to Section 3.2, what should you do if a program will be used once on small amounts of data and then discarded?",
            max_sentences=1,
        )

        self.assertIn("easiest-to-i mplement algorithm you know", excerpt)
        self.assertIn("written and debugged", excerpt)
        self.assertIn("move on", excerpt)
        self.assertNotIn("How-ever", excerpt)

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
