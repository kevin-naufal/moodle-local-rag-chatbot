import unittest

from scripts.eval.auto_judge_answer_quality import SemanticQualityEvaluator, judge_row


class FakeSemanticEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return [self.vectors.get(text, [0.0, 0.0, 1.0]) for text in texts]


class SemanticAnswerQualityEvalTest(unittest.TestCase):
    def test_semantic_gold_coverage_credits_paraphrase(self):
        evaluator = SemanticQualityEvaluator(
            FakeSemanticEmbedder(
                {
                    "Simple algorithms are easier to maintain.": [1.0, 0.0, 0.0],
                    "Other developers can modify the code more easily later.": [0.95, 0.05, 0.0],
                }
            ),
            model_name="fake-semantic",
        )

        covered, coverage_rate, per_point = evaluator.compute_gold_coverage(
            "Other developers can modify the code more easily later.",
            ["Simple algorithms are easier to maintain."],
        )

        self.assertEqual(covered, 1)
        self.assertEqual(coverage_rate, 1.0)
        self.assertGreaterEqual(per_point[0], 0.72)

    def test_judge_row_uses_semantic_metadata_and_no_personalization_scores(self):
        evaluator = SemanticQualityEvaluator(
            FakeSemanticEmbedder(
                {
                    "Why is simplicity important?": [0.0, 1.0, 0.0],
                    "Other developers can modify the code more easily later.": [0.95, 0.05, 0.0],
                    "Simple algorithms are easier to maintain.": [1.0, 0.0, 0.0],
                    "Simple algorithms are easier to maintain and modify.": [0.98, 0.02, 0.0],
                }
            ),
            model_name="fake-semantic",
        )
        run = {
            "question_id": "q1",
            "question": "Why is simplicity important?",
            "mode": "rag_bert",
            "run_id": 1,
            "model_answer": "Other developers can modify the code more easily later.",
            "retrieved_context": [{"text": "Simple algorithms are easier to maintain and modify."}],
        }
        spec = {
            "question": run["question"],
            "gold_points": ["Simple algorithms are easier to maintain."],
        }

        judged = judge_row(run, spec, "in-scope", evaluator=evaluator)

        self.assertEqual(judged["quality_eval_method"], "semantic_embedding_v1")
        self.assertEqual(judged["quality_eval_embedding_model"], "fake-semantic")
        self.assertGreaterEqual(judged["answer_completeness"], 0.9)
        self.assertGreaterEqual(judged["answer_groundedness"], 0.9)
        self.assertIsNone(judged["answer_clarity"])
        self.assertIsNone(judged["need_alignment"])

    def test_groundedness_reports_core_support_and_extra_claim_diagnostics(self):
        evaluator = SemanticQualityEvaluator(
            FakeSemanticEmbedder(
                {
                    "Why is simplicity important?": [1.0, 0.0, 0.0],
                    "Simple algorithms are easier to maintain. Extra unsupported benefit appears.": [1.0, 0.0, 0.0],
                    "Simple algorithms are easier to maintain.": [1.0, 0.0, 0.0],
                    "Simple algorithms are easier to maintain": [1.0, 0.0, 0.0],
                    "Extra unsupported benefit appears.": [0.0, 1.0, 0.0],
                    "Simple algorithms are easier to understand and maintain.": [1.0, 0.0, 0.0],
                }
            ),
            model_name="fake-semantic",
        )
        run = {
            "question_id": "q1",
            "question": "Why is simplicity important?",
            "mode": "rag_bert",
            "run_id": 1,
            "model_answer": "Simple algorithms are easier to maintain. Extra unsupported benefit appears.",
            "retrieved_context": [{"text": "Simple algorithms are easier to understand and maintain."}],
        }
        spec = {
            "question": run["question"],
            "gold_points": ["Simple algorithms are easier to maintain."],
        }

        judged = judge_row(run, spec, "in-scope", evaluator=evaluator)

        self.assertEqual(judged["context_support_raw"], 0.5)
        self.assertEqual(judged["core_context_support"], 1.0)
        self.assertEqual(judged["supported_sentence_ratio"], 0.5)
        self.assertEqual(judged["unsupported_sentence_count"], 1)
        self.assertEqual(judged["unsupported_extra_claim_count"], 1)
        self.assertEqual(judged["answer_groundedness"], 0.8)

    def test_judge_row_requires_semantic_evaluator(self):
        with self.assertRaisesRegex(ValueError, "Semantic quality evaluator is required"):
            judge_row(
                {
                    "question_id": "q1",
                    "question": "Why?",
                    "mode": "llm_only",
                    "run_id": 1,
                    "model_answer": "Because it helps.",
                },
                {"gold_points": ["It helps."]},
                "in-scope",
            )


if __name__ == "__main__":
    unittest.main()
