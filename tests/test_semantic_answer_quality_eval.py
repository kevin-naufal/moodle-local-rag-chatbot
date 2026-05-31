import unittest

from scripts.eval.auto_judge_answer_quality import SemanticQualityEvaluator, judge_row


class FakeSemanticEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.encode_calls = []

    def encode(self, texts):
        self.encode_calls.append(list(texts))
        return [self.vectors.get(text, [0.0, 0.0, 1.0]) for text in texts]


class SemanticAnswerQualityEvalTest(unittest.TestCase):
    def test_semantic_evaluator_reuses_cached_embeddings(self):
        embedder = FakeSemanticEmbedder(
            {
                "Simple algorithms are easier to maintain.": [1.0, 0.0, 0.0],
                "They are easier to maintain.": [1.0, 0.0, 0.0],
            }
        )
        evaluator = SemanticQualityEvaluator(embedder, model_name="fake-semantic")

        evaluator._best_similarity(
            "Simple algorithms are easier to maintain.",
            ["They are easier to maintain."],
        )
        evaluator._best_similarity(
            "Simple algorithms are easier to maintain.",
            ["They are easier to maintain."],
        )

        encoded_texts = [text for call in embedder.encode_calls for text in call]
        self.assertEqual(
            encoded_texts.count("Simple algorithms are easier to maintain."),
            1,
        )
        self.assertEqual(encoded_texts.count("They are easier to maintain."), 1)

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

    def test_semantic_gold_coverage_gives_strong_partial_credit(self):
        evaluator = SemanticQualityEvaluator(
            FakeSemanticEmbedder(
                {
                    "A simple algorithm is easier to implement correctly than a complex one.": [1.0, 0.0, 0.0],
                    "A simple algorithm is easier to implement correctly.": [0.65, 0.7607, 0.0],
                }
            ),
            model_name="fake-semantic",
        )

        covered, coverage_rate, per_point = evaluator.compute_gold_coverage(
            "A simple algorithm is easier to implement correctly.",
            ["A simple algorithm is easier to implement correctly than a complex one."],
        )

        self.assertEqual(covered, 0)
        self.assertEqual(coverage_rate, 0.75)
        self.assertGreaterEqual(per_point[0], 0.60)
        self.assertLess(per_point[0], 0.72)

    def test_completeness_uses_effective_coverage_not_only_full_covered_count(self):
        evaluator = SemanticQualityEvaluator(
            FakeSemanticEmbedder(
                {
                    "Why is simplicity important?": [0.0, 1.0, 0.0],
                    "A simple algorithm is easier to implement correctly.": [1.0, 0.0, 0.0],
                    "A simple algorithm is less likely to have subtle bugs after unexpected input.": [0.65, 0.7607, 0.0],
                    "It is less likely to have subtle bugs after unexpected input.": [1.0, 0.0, 0.0],
                    "A simple algorithm is easier to maintain by other people.": [0.65, 0.7607, 0.0],
                    "It is easier to maintain by other people.": [1.0, 0.0, 0.0],
                    "A simple algorithm is easier to implement correctly. It is less likely to have subtle bugs after unexpected input. It is easier to maintain by other people.": [1.0, 0.0, 0.0],
                }
            ),
            model_name="fake-semantic",
        )
        run = {
            "question_id": "q1",
            "question": "Why is simplicity important?",
            "mode": "llm_only",
            "run_id": 1,
            "model_answer": (
                "A simple algorithm is easier to implement correctly. "
                "It is less likely to have subtle bugs after unexpected input. "
                "It is easier to maintain by other people."
            ),
        }
        spec = {
            "question": run["question"],
            "gold_points": [
                "A simple algorithm is easier to implement correctly.",
                "A simple algorithm is less likely to have subtle bugs after unexpected input.",
                "A simple algorithm is easier to maintain by other people.",
            ],
        }

        judged = judge_row(run, spec, "in-scope", evaluator=evaluator)

        self.assertEqual(judged["key_points_covered"], 1)
        self.assertGreaterEqual(judged["answer_completeness"], 0.8)

    def test_judge_row_logs_per_gold_point_similarity(self):
        evaluator = SemanticQualityEvaluator(
            FakeSemanticEmbedder(
                {
                    "Why is simplicity important?": [0.0, 1.0, 0.0],
                    "Simple algorithms are easier to implement correctly.": [1.0, 0.0, 0.0],
                    "They are easier to implement correctly.": [1.0, 0.0, 0.0],
                    "Simple algorithms are easier to maintain.": [0.0, 1.0, 0.0],
                    "Unrelated sentence.": [0.0, 0.0, 1.0],
                }
            ),
            model_name="fake-semantic",
        )
        run = {
            "question_id": "q1",
            "question": "Why is simplicity important?",
            "mode": "llm_only",
            "run_id": 1,
            "model_answer": "They are easier to implement correctly.",
        }
        spec = {
            "question": run["question"],
            "gold_points": [
                "Simple algorithms are easier to implement correctly.",
                "Simple algorithms are easier to maintain.",
            ],
        }

        judged = judge_row(run, spec, "in-scope", evaluator=evaluator)

        self.assertEqual(
            judged["gold_point_similarities"],
            [
                {
                    "gold_point_index": 1,
                    "gold_point": "Simple algorithms are easier to implement correctly.",
                    "similarity": 1.0,
                    "coverage_status": "full",
                    "coverage_credit": 1.0,
                },
                {
                    "gold_point_index": 2,
                    "gold_point": "Simple algorithms are easier to maintain.",
                    "similarity": 0.0,
                    "coverage_status": "miss",
                    "coverage_credit": 0.0,
                },
            ],
        )

    def test_judge_row_requires_anchor_terms_when_gold_point_defines_them(self):
        evaluator = SemanticQualityEvaluator(
            FakeSemanticEmbedder(
                {
                    "What two approaches?": [0.0, 1.0, 0.0],
                    "Section 3.3 identifies benchmarking as one principal approach.": [1.0, 0.0, 0.0],
                    "Section 3.3 identifies analysis as the other principal approach.": [1.0, 0.0, 0.0],
                    "The two approaches are measuring directly and Big-Oh notation.": [1.0, 0.0, 0.0],
                }
            ),
            model_name="fake-semantic",
        )
        run = {
            "question_id": "q1",
            "question": "What two approaches?",
            "mode": "llm_only",
            "run_id": 1,
            "model_answer": "The two approaches are measuring directly and Big-Oh notation.",
        }
        spec = {
            "question": run["question"],
            "gold_points": [
                "Section 3.3 identifies benchmarking as one principal approach.",
                "Section 3.3 identifies analysis as the other principal approach.",
            ],
            "gold_point_anchor_terms": [
                ["benchmarking"],
                ["analysis"],
            ],
        }

        judged = judge_row(run, spec, "in-scope", evaluator=evaluator)

        self.assertEqual(judged["key_points_covered"], 0)
        self.assertEqual(judged["gold_point_similarities"][0]["coverage_status"], "miss")
        self.assertEqual(judged["gold_point_similarities"][0]["anchor_terms"], ["benchmarking"])
        self.assertFalse(judged["gold_point_similarities"][0]["anchor_terms_matched"])

    def test_judge_row_uses_semantic_metadata_and_quality_scores(self):
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
        self.assertIsNotNone(judged["answer_clarity"])
        self.assertIsNotNone(judged["need_alignment"])

    def test_groundedness_matches_against_context_sentences_not_only_full_chunks(self):
        evaluator = SemanticQualityEvaluator(
            FakeSemanticEmbedder(
                {
                    "Simple algorithms are easier to implement correctly.": [1.0, 0.0, 0.0],
                    "Irrelevant preface. Simple algorithms are easier to implement correctly. Unrelated tail.": [0.0, 1.0, 0.0],
                    "Irrelevant preface": [0.0, 1.0, 0.0],
                    "Simple algorithms are easier to implement correctly": [1.0, 0.0, 0.0],
                    "Unrelated tail.": [0.0, 1.0, 0.0],
                }
            ),
            model_name="fake-semantic",
        )

        details = evaluator.compute_context_support_details(
            "Simple algorithms are easier to implement correctly.",
            [
                {
                    "text": "Irrelevant preface. Simple algorithms are easier to implement correctly. Unrelated tail.",
                }
            ],
            ["Simple algorithms are easier to implement correctly."],
        )

        self.assertEqual(details["context_support_raw"], 1.0)
        self.assertEqual(details["unsupported_sentence_count"], 0)

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
