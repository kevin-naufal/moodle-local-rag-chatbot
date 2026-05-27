import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from quality_eval import build_quality_eval_summary
from system_eval import build_objective_eval_summary
from scripts.eval.plot_quality_eval import build_answer_quality_table
from scripts.eval.plot_system_eval import build_mode_metric_table


class EvalModelGroupingTest(unittest.TestCase):
    def test_quality_summary_and_table_group_by_model_and_mode(self):
        rows = [
            {
                "question_id": "q1",
                "mode": "rag_bert",
                "model_name": "qwen2.5:0.5b",
                "run_id": 1,
                "scope": "in-scope",
                "expected_behavior": "answer",
                "answer_correctness": 0.4,
                "answer_completeness": 0.4,
                "answer_groundedness": 0.5,
                "answer_relevance": 0.6,
            },
            {
                "question_id": "q1",
                "mode": "rag_bert",
                "model_name": "qwen2.5:1.5b",
                "run_id": 1,
                "scope": "in-scope",
                "expected_behavior": "answer",
                "answer_correctness": 0.8,
                "answer_completeness": 0.8,
                "answer_groundedness": 0.7,
                "answer_relevance": 0.9,
            },
        ]

        summary = build_quality_eval_summary(rows)
        columns, table_rows = build_answer_quality_table(summary)

        self.assertIn("by_model_mode", summary)
        self.assertEqual(len(summary["by_model_mode"]), 2)
        self.assertIn("model_name", columns)
        self.assertEqual(
            [(row["model_name"], row["mode"], row["correctness"]) for row in table_rows],
            [("qwen2.5:0.5b", "rag_bert", 0.4), ("qwen2.5:1.5b", "rag_bert", 0.8)],
        )

    def test_system_summary_and_table_group_by_model_and_mode(self):
        rows = [
            {
                "question_id": "q1",
                "mode": "rag_bert",
                "model_name": "qwen2.5:0.5b",
                "run_id": 1,
                "scope": "in-scope",
                "status": "success",
                "success_score": 1,
                "latency_total": 10.0,
                "latency_retrieval": 1.0,
                "latency_generation": 9.0,
                "source_hit_at_k": 1,
                "source_recall_at_k": 1.0,
                "rank_of_gold_source": 1,
                "mrr": 1.0,
            },
            {
                "question_id": "q1",
                "mode": "rag_bert",
                "model_name": "qwen2.5:1.5b",
                "run_id": 1,
                "scope": "in-scope",
                "status": "success",
                "success_score": 1,
                "latency_total": 20.0,
                "latency_retrieval": 1.0,
                "latency_generation": 19.0,
                "source_hit_at_k": 1,
                "source_recall_at_k": 1.0,
                "rank_of_gold_source": 1,
                "mrr": 1.0,
            },
        ]

        summary = build_objective_eval_summary(rows, top_k=4)
        columns, table_rows = build_mode_metric_table(summary)

        self.assertIn("by_model_mode", summary)
        self.assertEqual(len(summary["by_model_mode"]), 2)
        self.assertIn("model_name", columns)
        self.assertEqual(
            [(row["model_name"], row["mode"], row["latency"]) for row in table_rows],
            [("qwen2.5:0.5b", "rag_bert", 10.0), ("qwen2.5:1.5b", "rag_bert", 20.0)],
        )


if __name__ == "__main__":
    unittest.main()
