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
        for name in (
            "RAG_TOP_K",
            "RAG_CANDIDATE_K",
            "RAG_CHUNK_SIZE",
            "RAG_CHUNK_OVERLAP",
        ):
            os.environ.pop(name, None)

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
        self.assertEqual(runner.RAG_CANDIDATE_K, 16)
        self.assertEqual(runner.RAG_CHUNK_SIZE, 800)
        self.assertEqual(runner.RAG_CHUNK_OVERLAP, 120)

    def test_prompt_blocks_unsupported_elaboration(self):
        runner = importlib.import_module("moodle_rag_runner")

        self.assertIn("Do not add examples, benefits, causes, or implications", runner.PROMPT_TEMPLATE)
        self.assertIn("explicitly supported by the context", runner.PROMPT_TEMPLATE)
        self.assertIn("Use the Primary context first", runner.PROMPT_TEMPLATE)
        self.assertIn("prioritize the part that directly answers the question", runner.PROMPT_TEMPLATE)
        self.assertIn("specific example", runner.PROMPT_TEMPLATE)
        self.assertIn("numeric comparison", runner.PROMPT_TEMPLATE)
        self.assertIn("why/how questions", runner.PROMPT_TEMPLATE)

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

    def test_focused_excerpt_prefers_grade_transcript_example_details(self):
        runner = importlib.import_module("moodle_rag_runner")
        text = (
            "SEC. 3.3 MEASURING RUNNING TIME 95\n"
            "Never Mind Algorithm Efficiency; Just W ait a F ew Y ears\n"
            "Frequently, one hears the argument that there is no need to im prove the running\n"
            "time of algorithms or to select efficient algorithms, because computer speeds are\n"
            "doubling every few years and it will not be long before any alg orithm, however\n"
            "inefficient, will take so little time that one will not care. Pe ople have made this claim\n"
            "for many decades, yet there is no limit in sight to the demand f or computational\n"
            "resources. Thus, we generally reject the view that hardware improvements will\n"
            "make the study of efficient algorithms superfluous.\n"
            "There are situations, however, when we need not be overly con cerned with\n"
            "efficiency. For example, a school may, at the end of each term, t ranscribe grades\n"
            "reported on electronically readable grade sheets to studen t transcripts, all of which\n"
            "are stored in a computer. "
            "The time this operation takes is pro bably linear in the number of grades reported, "
            "like the hypothetical algorithm A. "
            "If the school replaces its computer by one 10 times as fast, it can do the job in one-te nth the time. "
            "It is very unlikely, however, that the school will therefore en roll 10 times as many "
            "students, or require each student to take 10 times as many cla sses. "
            "The computer speedup will not affect the size of the input to the transcript program, "
            "because that size is limited by other factors."
        )

        excerpt = runner.build_focused_excerpt(
            text,
            "In the chapter's grade-transcript example, why would a ten-times-faster computer not necessarily make algorithm efficiency important?",
            max_sentences=3,
        )

        self.assertIn("one-te nth", excerpt)
        self.assertIn("students", excerpt)
        self.assertIn("cla sses", excerpt)
        self.assertIn("limited by other factors", excerpt)
        self.assertNotIn("computer speeds are doubling", excerpt)

    def test_eval_chunking_keeps_grade_transcript_example_answer_together(self):
        os.environ["RAG_CHUNK_SIZE"] = "1600"
        os.environ["RAG_CHUNK_OVERLAP"] = "300"
        runner = importlib.import_module("moodle_rag_runner")
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        page_text = (
            "SEC. 3.3 MEASURING RUNNING TIME 95\n"
            "Never Mind Algorithm Efficiency; Just W ait a F ew Y ears\n"
            "Frequently, one hears the argument that there is no need to im prove the running\n"
            "time of algorithms or to select efficient algorithms, because computer speeds are\n"
            "doubling every few years and it will not be long before any alg orithm, however\n"
            "inefficient, will take so little time that one will not care. Pe ople have made this claim\n"
            "for many decades, yet there is no limit in sight to the demand f or computational\n"
            "resources. Thus, we generally reject the view that hardware improvements will\n"
            "make the study of efficient algorithms superfluous.\n"
            "There are situations, however, when we need not be overly con cerned with\n"
            "efficiency. For example, a school may, at the end of each term, t ranscribe grades\n"
            "reported on electronically readable grade sheets to studen t transcripts, all of which\n"
            "are stored in a computer. The time this operation takes is pro bably linear in the\n"
            "number of grades reported, like the hypothetical algorithm A. If the school replaces\n"
            "its computer by one 10 times as fast, it can do the job in one-te nth the time. It is very\n"
            "unlikely, however, that the school will therefore en roll 10 times as many students,\n"
            "or require each student to take 10 times as many cla sses. The computer speedup\n"
            "will not affect the size of the input to the transcript program, because that size is\n"
            "limited by other factors."
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=runner.RAG_CHUNK_SIZE,
            chunk_overlap=runner.RAG_CHUNK_OVERLAP,
        )
        [first_chunk, *_] = splitter.split_text(page_text)

        self.assertIn("one-te nth", first_chunk)
        self.assertIn("cla sses", first_chunk)
        self.assertIn("limited by other factors", first_chunk)

    def test_get_relevant_docs_promotes_direct_grade_transcript_answer_over_generic_speedup(self):
        runner = importlib.import_module("moodle_rag_runner")
        generic_speedup = SimpleNamespace(
            page_content=(
                "For example, suppose we can afford 100 seconds of computer time. "
                "If computers become 10 times as fast, then in 100 seconds we can handle "
                "problems of the size that used to require 1000 seconds. "
                "With algorithm A, we can now solve problems 10 times as large."
            ),
            metadata={},
        )
        partial_school_context = SimpleNamespace(
            page_content=(
                "There are situations when we need not be overly concerned with efficiency. "
                "For example, a school may transcribe grades reported on grade sheets to "
                "student transcripts, all of which are stored in a computer."
            ),
            metadata={},
        )
        direct_school_answer = SimpleNamespace(
            page_content=(
                "If the school replaces its computer by one 10 times as fast, it can do the job "
                "in one-tenth the time. It is very unlikely, however, that the school will "
                "therefore enroll 10 times as many students, or require each student to take "
                "10 times as many classes. The computer speedup will not affect the size of "
                "the input to the transcript program, because that size is limited by other factors."
            ),
            metadata={},
        )
        vectorstore = SimpleNamespace(
            similarity_search_with_relevance_scores=lambda query, k, filter=None: [
                (generic_speedup, 0.8970),
                (partial_school_context, 0.9039),
                (direct_school_answer, 0.8609),
            ]
        )

        docs = runner.get_relevant_docs(
            vectorstore,
            "In the chapter's grade-transcript example, why would a ten-times-faster computer not necessarily make algorithm efficiency important?",
        )

        self.assertIs(docs[0], direct_school_answer)
        self.assertEqual(docs[0].metadata["context_role"], "primary")
        self.assertGreater(docs[0].metadata["direct_answer_score"], docs[1].metadata["direct_answer_score"])

    def test_extractive_explanation_fallback_uses_primary_context_sentences_in_order(self):
        runner = importlib.import_module("moodle_rag_runner")
        primary = SimpleNamespace(
            page_content=(
                "For example, a school may, at the end of each term, t ranscribe grades reported "
                "on electronically readable grade sheets to studen t transcripts, all of which are "
                "stored in a computer. The time this operation takes is pro bably linear in the "
                "number of grades reported, like the hypothetical algorithm A. If the school replaces "
                "its computer by one 10 times as fast, it can do the job in one-te nth the time. "
                "It is very unlikely, however, that the school will therefore en roll 10 times as many "
                "students, or require each student to take 10 times as many cla sses. The computer "
                "speedup will not affect the size of the input to the transcript program, because "
                "that size is limited by other factors."
            ),
            metadata={"context_role": "primary"},
        )
        supporting = SimpleNamespace(
            page_content=(
                "If computers become 10 times as fast, then in 100 seconds we can handle problems "
                "of the size that used to require 1000 seconds."
            ),
            metadata={"context_role": "supporting"},
        )
        query = (
            "In the chapter's grade-transcript example, why would a ten-times-faster computer "
            "not necessarily make algorithm efficiency important?"
        )
        answer = "The input size is limited by other factors."

        self.assertTrue(runner.should_use_extractive_explanation_fallback(query, answer, [primary, supporting]))

        extracted = runner.build_extractive_context_answer(query, [primary, supporting])

        self.assertIn("transcribe grades", extracted)
        self.assertIn("student transcripts", extracted)
        self.assertIn("one-tenth", extracted)
        self.assertIn("students", extracted)
        self.assertIn("classes", extracted)
        self.assertIn("limited by other factors", extracted)
        self.assertNotIn("100 seconds", extracted)

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

    def test_get_relevant_docs_prefers_actual_section_heading_over_incidental_reference(self):
        runner = importlib.import_module("moodle_rag_runner")
        incidental_reference = SimpleNamespace(
            page_content=(
                "3.1 What This Chapter Is About. Big-oh notation is introduced in Sections 3.4 "
                "and 3.5. It lets us avoid constants when discussing the running time of programs."
            ),
            metadata={},
        )
        section_answer = SimpleNamespace(
            page_content=(
                "3.4 Big-Oh and Approximate Running Time. The running time of a C program depends "
                "on the computer used and on the compiler. We could count generated machine "
                "instructions, but big-oh notation lets us ignore constant factors."
            ),
            metadata={},
        )
        vectorstore = SimpleNamespace(
            similarity_search_with_relevance_scores=lambda query, k, filter=None: [
                (incidental_reference, 0.91),
                (section_answer, 0.86),
            ]
        )

        docs = runner.get_relevant_docs(
            vectorstore,
            "According to Section 3.4, why is big-oh notation useful when discussing the running time of a C program?",
        )

        self.assertIs(docs[0], section_answer)
        self.assertGreater(docs[0].metadata["section_match_score"], docs[1].metadata["section_match_score"])

    def test_get_relevant_docs_prefers_big_oh_c_program_reason_over_later_section_heading(self):
        runner = importlib.import_module("moodle_rag_runner")
        later_section_heading = SimpleNamespace(
            page_content=(
                "SEC. 3.4 BIG-OH AND APPROXIMATE RUNNING TIME. The notation O(m) enables us "
                "to make statements without getting involved in unknowable or meaningless constants. "
                "It says the time to execute the fragment on progressively larger arrays grows linearly."
            ),
            metadata={},
        )
        c_program_reason = SimpleNamespace(
            page_content=(
                "The running time of a C program depends on the computer on which it is run and "
                "the particular C compiler used to generate the executable program. Even when the "
                "program, input, machine, and compiler are known, predicting the exact number of "
                "machine instructions executed is usually too complex. Big-oh notation hides constant "
                "factors such as instruction counts and machine instruction speed."
            ),
            metadata={},
        )
        vectorstore = SimpleNamespace(
            similarity_search_with_relevance_scores=lambda query, k, filter=None: [
                (later_section_heading, 0.7909),
                (c_program_reason, 0.7724),
            ]
        )

        docs = runner.get_relevant_docs(
            vectorstore,
            "Why does Section 3.4 say big-oh notation is useful when discussing the running time of a C program?",
        )

        self.assertIs(docs[0], c_program_reason)
        self.assertGreater(docs[0].metadata["direct_answer_score"], docs[1].metadata["direct_answer_score"])

    def test_get_relevant_docs_prefers_for_statement_rule_over_summary_table(self):
        runner = importlib.import_module("moodle_rag_runner")
        summary_table = SimpleNamespace(
            page_content=(
                "3.7 A Recursive Rule for Bounding Running Time. Fig. 3.13 Construct Test Body "
                "O(1) O(f(n)) while-statement O(g(n)f(n)) for-statement O(g(n)f(n))."
            ),
            metadata={},
        )
        rule_text = SimpleNamespace(
            page_content=(
                "For-statement. If O(f(n)) is an upper bound on the running time of the body, "
                "and g(n) is an upper bound on the number of times around the loop, then an "
                "upper bound on the running time of the for-statement is O(1 + (f(n)+1)g(n)). "
                "The term f(n)+1 represents the body, the test, and the reinitialization; the "
                "leading 1 represents the initialization and the possibility that the first test is negative."
            ),
            metadata={},
        )
        vectorstore = SimpleNamespace(
            similarity_search_with_relevance_scores=lambda query, k, filter=None: [
                (summary_table, 0.90),
                (rule_text, 0.86),
            ]
        )

        docs = runner.get_relevant_docs(
            vectorstore,
            "What upper bound does the chapter give for the running time of a for-statement, and what do the terms mean?",
        )

        self.assertIs(docs[0], rule_text)
        self.assertGreater(docs[0].metadata["direct_answer_score"], docs[1].metadata["direct_answer_score"])

    def test_get_relevant_docs_prefers_for_statement_rule_over_section_figure_when_scores_are_close(self):
        runner = importlib.import_module("moodle_rag_runner")
        section_figure = SimpleNamespace(
            page_content=(
                "SEC. 3.7 A RECURSIVE RULE FOR BOUNDING RUNNING TIME 121 Test Body O(1) "
                "O(f(n)) O(g(n)f(n)) g(n) times around At most. Test Body O(1) O(1) "
                "O(f(n)) Initialize O(g(n)f(n)) g(n) times around O(1) Reinitialize "
                "(c) For-statement. Fig. 3.13. Computing the running time of loop statements."
            ),
            metadata={},
        )
        rule_text = SimpleNamespace(
            page_content=(
                "122 THE RUNNING TIME OF PROGRAMS 3. For-statement. If O(f(n)) is an upper bound "
                "on the running time of the body, and g(n) is an upper bound on the number of times "
                "around the loop, then an upper bound on the time of a for-statement is "
                "O(1 + (f(n) + 1)g(n)). The factor f(n) + 1 represents the cost of going around "
                "once, including the body, the test, and the reinitialization. The 1+ at the "
                "beginning represents the first initialization and the possibility that the first "
                "test is negative, resulting in zero iterations of the loop. In the common case "
                "the running time of the for-statement is O(f(n)g(n))."
            ),
            metadata={},
        )
        vectorstore = SimpleNamespace(
            similarity_search_with_relevance_scores=lambda query, k, filter=None: [
                (section_figure, 0.9163),
                (rule_text, 0.9166),
            ]
        )

        docs = runner.get_relevant_docs(
            vectorstore,
            "According to Section 3.7, how is the running time of a for-statement bounded when the body takes O(f(n)) time and the loop runs at most g(n) times?",
        )

        self.assertIs(docs[0], rule_text)
        self.assertLess(docs[0].metadata["section_match_score"], docs[1].metadata["section_match_score"])
        self.assertGreater(docs[0].metadata["direct_answer_score"], docs[1].metadata["direct_answer_score"])

    def test_get_relevant_docs_prefers_mergesort_recurrence_over_generic_recursion(self):
        runner = importlib.import_module("moodle_rag_runner")
        generic_recursion = SimpleNamespace(
            page_content=(
                "A Common Form of Recursion. Many recursive functions take time O(1) and then "
                "call themselves on a subproblem of size n-1, giving T(n)=O(1)+T(n-1)."
            ),
            metadata={},
        )
        mergesort_recurrence = SimpleNamespace(
            page_content=(
                "MergeSort. For n > 1 the recurrence is T(n)=2T(n/2)+g(n), with T(1)=a. "
                "For MergeSort, g(n)=bn because splitting and merging take linear time. "
                "The solution is an + bn log n, so the running time is O(n log n)."
            ),
            metadata={},
        )
        vectorstore = SimpleNamespace(
            similarity_search_with_relevance_scores=lambda query, k, filter=None: [
                (generic_recursion, 0.91),
                (mergesort_recurrence, 0.86),
            ]
        )

        docs = runner.get_relevant_docs(
            vectorstore,
            "What recurrence relation does the chapter derive for MergeSort, and what running time does it imply?",
        )

        self.assertIs(docs[0], mergesort_recurrence)
        self.assertGreater(docs[0].metadata["direct_answer_score"], docs[1].metadata["direct_answer_score"])

    def test_get_relevant_docs_prefers_mergesort_recurrence_details_over_section_intro(self):
        runner = importlib.import_module("moodle_rag_runner")
        section_intro = SimpleNamespace(
            page_content=(
                "This problem requires that you estimate the running times sufficiently precisely. "
                "3.11 Solving Recurrence Relations. There are many techniques for solving recurrence "
                "relations. This follows the analysis of MergeSort and mentions that merge sort is an "
                "O(n log n)-time algorithm."
            ),
            metadata={},
        )
        recurrence_details = SimpleNamespace(
            page_content=(
                "Another common form of recurrence generalizes the recurrence we derived for MergeSort. "
                "Basis: T(1)=a. Induction: T(n)=2T(n/2)+g(n), for n a power of 2 and greater than 1. "
                "For MergeSort, g(n) is bn because the work outside recursive calls is O(n), principally "
                "for split and merge. There are log2 n levels with bn work at each non-basis level, "
                "and the basis calls contribute an. Therefore T(n)=an+bn log n, so MergeSort is O(n log n)."
            ),
            metadata={},
        )
        vectorstore = SimpleNamespace(
            similarity_search_with_relevance_scores=lambda query, k, filter=None: [
                (section_intro, 0.8877),
                (recurrence_details, 0.8720),
            ]
        )

        docs = runner.get_relevant_docs(
            vectorstore,
            "In Section 3.11's MergeSort recurrence, why does the chapter conclude that MergeSort has running time O(n log n)?",
        )

        self.assertIs(docs[0], recurrence_details)
        self.assertGreater(docs[0].metadata["direct_answer_score"], docs[1].metadata["direct_answer_score"])

    def test_default_candidate_pool_is_large_enough_for_formula_reranking(self):
        runner = importlib.import_module("moodle_rag_runner")

        self.assertGreaterEqual(runner.RAG_CANDIDATE_K, 16)

    def test_get_relevant_docs_expands_mergesort_recurrence_query_for_retrieval(self):
        runner = importlib.import_module("moodle_rag_runner")
        seen_queries = []
        recurrence_details = SimpleNamespace(
            page_content=(
                "Basis: T(1)=a. Induction: T(n)=2T(n/2)+g(n). For MergeSort, g(n)=bn. "
                "The solution is an+bn log n, so MergeSort is O(n log n)."
            ),
            metadata={},
        )
        generic_mergesort = SimpleNamespace(
            page_content="Merge sort splits lists and eventually has running time O(n log n).",
            metadata={},
        )

        def fake_search(query, k, filter=None):
            seen_queries.append(query)
            if "T(n)=2T(n/2)+g(n)" in query and "bn log n" in query:
                return [(recurrence_details, 0.82), (generic_mergesort, 0.90)]
            return [(generic_mergesort, 0.90)]

        vectorstore = SimpleNamespace(similarity_search_with_relevance_scores=fake_search)

        docs = runner.get_relevant_docs(
            vectorstore,
            "In Section 3.11's MergeSort recurrence, why does the chapter conclude that MergeSort has running time O(n log n)?",
        )

        self.assertIn("T(n)=2T(n/2)+g(n)", seen_queries[0])
        self.assertIn("bn log n", seen_queries[0])
        self.assertIs(docs[0], recurrence_details)

    def test_recurrence_query_expansion_is_not_specific_to_mergesort(self):
        runner = importlib.import_module("moodle_rag_runner")

        expanded = runner.build_retrieval_search_query(
            "How does the recurrence relation imply the final running time?"
        )

        self.assertIn("T(n)=2T(n/2)+g(n)", expanded)
        self.assertIn("T(n)=T(n-1)+g(n)", expanded)
        self.assertIn("basis", expanded.lower())
        self.assertNotIn("MergeSort", expanded)

    def test_recurrence_scoring_prefers_formula_evidence_for_any_algorithm(self):
        runner = importlib.import_module("moodle_rag_runner")
        generic_heading = (
            "Section 5.4 Solving Recurrences. This section discusses recurrence relations "
            "and says that Algorithm Z has running time O(n log n)."
        )
        formula_evidence = (
            "For Algorithm Z, the basis is T(1)=a and the induction step is "
            "T(n)=2T(n/2)+g(n). The nonrecursive work is g(n)=bn, so the solution "
            "is an+bn log n and therefore O(n log n)."
        )

        self.assertGreater(
            runner.technical_answer_cue_score(
                "How does Algorithm Z's recurrence relation imply its final running time?",
                formula_evidence,
            ),
            runner.technical_answer_cue_score(
                "How does Algorithm Z's recurrence relation imply its final running time?",
                generic_heading,
            ),
        )

    def test_focused_excerpt_prefers_mergesort_recurrence_over_prior_selection_sort_text(self):
        runner = importlib.import_module("moodle_rag_runner")
        text = (
            "SEC. 3.11 SOLVING RECURRENCE RELATIONS 147 T (m) = a + 2b + 3b + ... + mb. "
            "Thus, T (m) is O(m2). Since we are interested in SelectionSort, T (n) is O(n2). "
            "Thus, the recursive version of selection sort is quadratic. "
            "Another common form of recurrence generalizes the recurrence we derived for MergeSort. "
            "Basis: T (1) = a. Induction: T (n) = 2 T (n/2) + g(n), for n a power of 2 and greater than 1. "
            "For MergeSort, g(n) = bn because split and merge take O(n) time outside recursive calls. "
            "The solution is T (n) = an + bn log n, so MergeSort is O(n log n)."
        )

        excerpt = runner.build_focused_excerpt(
            text,
            "In Section 3.11's MergeSort recurrence, why does the chapter conclude that MergeSort has running time O(n log n)?",
            max_sentences=3,
        )

        self.assertIn("MergeSort", excerpt)
        self.assertIn("T (n) = 2 T (n/2) + g(n)", excerpt)
        self.assertIn("T (1) = a", excerpt)
        self.assertNotIn("SelectionSort", excerpt)

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
