import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from quality_eval import build_quality_eval_summary
from system_eval import build_objective_eval_summary
from scripts.eval.plot_quality_eval import build_answer_quality_table, build_quality_heatmap_matrix
from scripts.eval.plot_system_eval import build_mode_metric_table, build_system_heatmap_matrix


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
                "coverage_at_k": 0.5,
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
                "coverage_at_k": 1.0,
                "rank_of_gold_source": 1,
                "mrr": 1.0,
            },
        ]

        summary = build_objective_eval_summary(rows, top_k=4)
        columns, table_rows = build_mode_metric_table(summary)

        self.assertIn("by_model_mode", summary)
        self.assertEqual(len(summary["by_model_mode"]), 2)
        self.assertIn("model_name", columns)
        self.assertIn("coverage_at_k", columns)
        self.assertEqual(
            [(row["model_name"], row["mode"], row["latency"]) for row in table_rows],
            [("qwen2.5:0.5b", "rag_bert", 10.0), ("qwen2.5:1.5b", "rag_bert", 20.0)],
        )
        self.assertEqual(
            [(row["model_name"], row["coverage_at_k"]) for row in table_rows],
            [("qwen2.5:0.5b", 0.5), ("qwen2.5:1.5b", 1.0)],
        )

    def test_quality_heatmap_matrix_uses_models_as_rows_and_modes_as_columns(self):
        summary = {
            "by_model_mode": [
                {
                    "model_name": "qwen2.5:0.5b",
                    "mode": "llm_only",
                    "answer_quality": {"correctness": 0.2, "groundedness": None},
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "mode": "rag_bert",
                    "answer_quality": {"correctness": 0.7, "groundedness": 0.8},
                },
                {
                    "model_name": "qwen2.5:1.5b",
                    "mode": "rag_bert",
                    "answer_quality": {"correctness": 0.9, "groundedness": 0.85},
                },
            ]
        }

        matrix = build_quality_heatmap_matrix(summary, "correctness", "avg_answer_correctness")

        self.assertEqual(matrix.models, ["qwen2.5:0.5b", "qwen2.5:1.5b"])
        self.assertEqual(matrix.modes, ["llm_only", "rag_bert"])
        self.assertEqual(matrix.values, [[0.2, 0.7], [None, 0.9]])

    def test_system_heatmap_matrix_includes_latency_retrieval(self):
        summary = {
            "by_model_mode": [
                {"model_name": "qwen2.5:0.5b", "mode": "llm_only", "avg_latency_retrieval": 0.0},
                {"model_name": "qwen2.5:0.5b", "mode": "rag_bert", "avg_latency_retrieval": 0.1517},
                {"model_name": "qwen2.5:1.5b", "mode": "rag_msmarco", "avg_latency_retrieval": 0.1504},
            ]
        }

        matrix = build_system_heatmap_matrix(summary, "avg_latency_retrieval")

        self.assertEqual(matrix.models, ["qwen2.5:0.5b", "qwen2.5:1.5b"])
        self.assertEqual(matrix.modes, ["llm_only", "rag_bert", "rag_msmarco"])
        self.assertEqual(matrix.values, [[0.0, 0.1517, None], [None, None, 0.1504]])


if __name__ == "__main__":
    unittest.main()
