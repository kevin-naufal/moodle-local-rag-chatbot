import unittest
import tempfile
import json
from pathlib import Path

from scripts.eval.run_eval import (
    dedupe_answer_runs_file,
    get_mode_config,
    normalize_chat_models,
    load_existing_completed_keys,
    normalize_modes,
)


class RunEvalModeConfigTest(unittest.TestCase):
    def test_rag_bert_and_msmarco_use_distinct_bert_models(self):
        rag_bert = get_mode_config("rag_bert")
        rag_msmarco = get_mode_config("rag_msmarco")

        self.assertEqual(rag_bert["embed_backend"], "bert")
        self.assertEqual(rag_bert["bert_model"], "bert-base-uncased")
        self.assertEqual(rag_msmarco["bert_model"], "sentence-transformers/msmarco-bert-base-dot-v5")
        self.assertNotEqual(rag_bert["bert_model"], rag_msmarco["bert_model"])

    def test_default_modes_include_msmarco(self):
        self.assertEqual(
            normalize_modes(""),
            ["llm_only", "rag_bert", "rag_msmarco"],
        )

    def test_default_chat_models_use_qwen_size_family(self):
        self.assertEqual(
            normalize_chat_models(""),
            ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"],
        )

    def test_resume_only_skips_successful_rows_per_chat_model(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            path = handle.name
            handle.write(json.dumps({"question_id": "q1", "mode": "llm_only", "model_name": "qwen2.5:0.5b", "run_id": 1, "status": "error"}) + "\n")
            handle.write(json.dumps({"question_id": "q1", "mode": "rag_bert", "model_name": "qwen2.5:0.5b", "run_id": 1, "status": "success"}) + "\n")
            handle.write(json.dumps({"question_id": "q1", "mode": "rag_bert", "model_name": "qwen2.5:1.5b", "run_id": 1, "status": "success"}) + "\n")

        completed = load_existing_completed_keys(Path(path))

        self.assertNotIn(("q1", "llm_only", "qwen2.5:0.5b", 1), completed)
        self.assertIn(("q1", "rag_bert", "qwen2.5:0.5b", 1), completed)
        self.assertIn(("q1", "rag_bert", "qwen2.5:1.5b", 1), completed)

    def test_dedupe_answer_runs_keeps_latest_row_per_planned_model_job(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            path = Path(handle.name)
            handle.write(json.dumps({"question_id": "q1", "mode": "llm_only", "model_name": "qwen2.5:0.5b", "run_id": 1, "status": "error"}) + "\n")
            handle.write(json.dumps({"question_id": "q1", "mode": "llm_only", "model_name": "qwen2.5:0.5b", "run_id": 1, "status": "success"}) + "\n")
            handle.write(json.dumps({"question_id": "q1", "mode": "llm_only", "model_name": "qwen2.5:1.5b", "run_id": 1, "status": "success"}) + "\n")
            handle.write(json.dumps({"question_id": "q1", "mode": "rag_bert", "model_name": "qwen2.5:0.5b", "run_id": 1, "status": "success"}) + "\n")
            handle.write(json.dumps({"question_id": "q99", "mode": "llm_only", "model_name": "qwen2.5:0.5b", "run_id": 1, "status": "success"}) + "\n")

        planned_keys = {
            ("q1", "llm_only", "qwen2.5:0.5b", 1),
            ("q1", "llm_only", "qwen2.5:1.5b", 1),
            ("q1", "rag_bert", "qwen2.5:0.5b", 1),
        }
        dedupe_answer_runs_file(path, planned_keys)

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [(row["question_id"], row["mode"], row["model_name"], row["run_id"]) for row in rows],
            [
                ("q1", "llm_only", "qwen2.5:0.5b", 1),
                ("q1", "llm_only", "qwen2.5:1.5b", 1),
                ("q1", "rag_bert", "qwen2.5:0.5b", 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
