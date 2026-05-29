import tempfile
import unittest
from pathlib import Path

from scripts.eval.evaluate_role_model_answers import (
    build_role_model_answer_runs,
    load_retrieved_context_lookup,
    summarize_by_question,
)


class RoleModelAnswerEvalTest(unittest.TestCase):
    def test_build_role_model_answer_runs_repeats_each_role_model_answer(self):
        dataset = {
            "scope": "in-scope",
            "questions": [
                {
                    "id": "q1",
                    "question": "Why?",
                    "role_model_answer": "Because it matches the source.",
                    "gold_points": ["It matches the source.", "It answers the question."],
                },
                {
                    "id": "q2",
                    "question": "How?",
                    "role_model_answer": "By using documented evidence.",
                    "gold_points": ["Use documented evidence."],
                },
            ],
        }

        retrieved_context_lookup = {
            "q1": {
                "embedding_backend": "bert",
                "embedding_model": "sentence-transformers/msmarco-bert-base-dot-v5",
                "retrieved_context": [
                    {"text": "Retrieved context for q1.", "source": "doc.pdf"},
                ],
            },
            "q2": {
                "embedding_backend": "bert",
                "embedding_model": "sentence-transformers/msmarco-bert-base-dot-v5",
                "retrieved_context": [
                    {"text": "Retrieved context for q2.", "source": "doc.pdf"},
                ],
            },
        }

        rows = build_role_model_answer_runs(dataset, runs=3, retrieved_context_lookup=retrieved_context_lookup)

        self.assertEqual(len(rows), 6)
        self.assertEqual([row["run_id"] for row in rows[:3]], [1, 2, 3])
        self.assertEqual(rows[0]["question_id"], "q1")
        self.assertEqual(rows[0]["mode"], "role_model_semantic")
        self.assertEqual(rows[0]["model_name"], "role_model_answer")
        self.assertEqual(rows[0]["model_answer"], "Because it matches the source.")
        self.assertEqual(
            [item["text"] for item in rows[0]["retrieved_context"]],
            ["Retrieved context for q1."],
        )
        self.assertEqual(rows[0]["embedding_backend"], "bert")
        self.assertEqual(rows[0]["embedding_model_name"], "sentence-transformers/msmarco-bert-base-dot-v5")
        self.assertEqual(rows[3]["question_id"], "q2")
        self.assertEqual(rows[3]["run_id"], 1)

    def test_build_role_model_answer_runs_without_retrieval_marks_groundedness_not_applicable(self):
        dataset = {
            "questions": [
                {
                    "id": "q1",
                    "question": "Why?",
                    "role_model_answer": "Because it matches the source.",
                    "gold_points": ["It matches the source."],
                },
            ],
        }

        rows = build_role_model_answer_runs(dataset, runs=1)

        self.assertEqual(rows[0]["mode"], "llm_only")
        self.assertEqual(rows[0]["retrieved_context"], [])

    def test_load_retrieved_context_lookup_reads_jsonl_by_question_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retrieved_contexts.jsonl"
            path.write_text(
                '{"question_id":"q1","embedding_backend":"bert","embedding_model":"msmarco","retrieved_context":[{"text":"ctx"}]}\n',
                encoding="utf-8",
            )

            lookup = load_retrieved_context_lookup(path)

        self.assertEqual(lookup["q1"]["embedding_backend"], "bert")
        self.assertEqual(lookup["q1"]["embedding_model"], "msmarco")
        self.assertEqual(lookup["q1"]["retrieved_context"][0]["text"], "ctx")

    def test_summarize_by_question_reports_pass_rate(self):
        rows = [
            {
                "question_id": "q1",
                "question": "Why?",
                "answer_correctness": 1.0,
                "answer_groundedness": 0.9,
                "answer_relevance": 0.8,
                "quality_score": 0.9,
                "judge_label": "high_quality",
            },
            {
                "question_id": "q1",
                "question": "Why?",
                "answer_correctness": 0.8,
                "answer_groundedness": 0.9,
                "answer_relevance": 0.8,
                "quality_score": 0.8,
                "judge_label": "high_quality",
            },
            {
                "question_id": "q2",
                "question": "How?",
                "answer_correctness": 0.4,
                "answer_groundedness": 0.5,
                "answer_relevance": 0.7,
                "quality_score": 0.5,
                "judge_label": "medium_quality",
            },
        ]

        summary = summarize_by_question(rows)

        self.assertEqual(summary["total_questions"], 2)
        self.assertEqual(summary["questions"][0]["question_id"], "q1")
        self.assertEqual(summary["questions"][0]["pass_rate"], 1.0)
        self.assertEqual(summary["questions"][0]["final_label"], "strong_pass")
        self.assertEqual(summary["questions"][1]["pass_rate"], 0.0)
        self.assertEqual(summary["questions"][1]["final_label"], "fail")


if __name__ == "__main__":
    unittest.main()
