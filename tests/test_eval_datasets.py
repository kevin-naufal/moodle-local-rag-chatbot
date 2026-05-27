import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EvalDatasetTest(unittest.TestCase):
    def test_dev_mini_dataset_has_three_questions(self):
        dataset_path = PROJECT_ROOT / "data" / "answer_run_questions" / "ch03_running_time_in_scope_3q.json"

        payload = json.loads(dataset_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["batch"], "dev_3")
        self.assertEqual(payload["question_count"], 3)
        self.assertEqual(len(payload["questions"]), 3)
        self.assertEqual(
            [question["id"] for question in payload["questions"]],
            ["ch03-full-q01", "ch03-full-q02", "ch03-full-q03"],
        )


if __name__ == "__main__":
    unittest.main()
