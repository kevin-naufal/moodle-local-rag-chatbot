import unittest

from scripts.eval.retrieve_question_contexts import build_retrieval_rows


class RetrieveQuestionContextsTest(unittest.TestCase):
    def test_build_retrieval_rows_uses_each_dataset_question(self):
        dataset = {
            "questions": [
                {"id": "q1", "question": "What is A?"},
                {"id": "q2", "question": "Why B?"},
            ]
        }

        rows = build_retrieval_rows(
            dataset,
            retrieve_context=lambda question: [{"text": f"context for {question}", "source": "doc.pdf"}],
            embedding_backend="bert",
            embedding_model="sentence-transformers/msmarco-bert-base-dot-v5",
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["question_id"], "q1")
        self.assertEqual(rows[0]["question"], "What is A?")
        self.assertEqual(rows[0]["retrieved_context_count"], 1)
        self.assertEqual(rows[0]["retrieved_context"][0]["text"], "context for What is A?")
        self.assertEqual(rows[0]["embedding_backend"], "bert")
        self.assertEqual(rows[1]["question_id"], "q2")


if __name__ == "__main__":
    unittest.main()
