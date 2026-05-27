import importlib
import os
import sys
import unittest
from pathlib import Path


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

    def test_document_text_cleanup_normalizes_pdf_ligatures(self):
        runner = importlib.import_module("moodle_rag_runner")

        self.assertEqual(runner.clean_document_text("e\ufb03cient \ufb02ow"), "efficient flow")


if __name__ == "__main__":
    unittest.main()
