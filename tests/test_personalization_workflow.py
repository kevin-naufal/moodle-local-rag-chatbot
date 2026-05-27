import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.eval.auto_judge_answer_quality import SemanticQualityEvaluator, judge_row
from scripts.eval.plot_quality_eval import build_plots, build_tables


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSONALIZATION_FIELDS = (
    "answer_clarity",
    "instruction_compliance",
    "need_alignment",
    "scaffolding_quality",
    "pedagogical_actionability",
)


class FakeSemanticEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class PersonalizationWorkflowTest(unittest.TestCase):
    def test_auto_judge_leaves_personalization_fields_empty(self):
        run = {
            "question_id": "q1",
            "question": "Why is simplicity important?",
            "mode": "rag_bert",
            "run_id": 1,
            "model_answer": "Simplicity helps people understand and maintain the algorithm.",
            "retrieved_context": [{"text": "Simple algorithms are easier to understand and maintain."}],
        }
        spec = {
            "question": run["question"],
            "gold_points": ["Simple algorithms are easier to understand and maintain."],
        }

        judged = judge_row(
            run,
            spec,
            "in-scope",
            evaluator=SemanticQualityEvaluator(FakeSemanticEmbedder(), "fake-semantic"),
        )

        for field in PERSONALIZATION_FIELDS:
            self.assertIsNone(judged[field])

    def test_default_quality_plots_skip_personalization_artifacts(self):
        summary = {
            "by_mode": [
                {
                    "mode": "rag_bert",
                    "total_runs": 1,
                    "answer_quality": {"correctness": 0.8, "groundedness": 0.7, "relevance": 0.9},
                    "answer_personalization": {
                        "instruction_compliance": None,
                        "need_alignment": None,
                        "scaffolding_quality": None,
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            files = build_plots(summary, output_dir)
            files.extend(build_tables(summary, output_dir))

            names = {Path(file_path).name for file_path in files}
            self.assertIn("answer_quality_core.png", names)
            self.assertIn("mode_vs_answer_quality.md", names)
            self.assertNotIn("answer_personalization_core.png", names)
            self.assertNotIn("mode_vs_answer_personalization.md", names)

    def test_personalization_plot_script_fails_when_fields_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged_path = Path(tmp) / "judged_runs.jsonl"
            judged_path.write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "mode": "rag_bert",
                        "run_id": 1,
                        "scope": "in-scope",
                        "answer_correctness": 0.8,
                        "answer_completeness": 0.8,
                        "answer_groundedness": 0.8,
                        "answer_relevance": 0.8,
                        "answer_clarity": None,
                        "instruction_compliance": None,
                        "need_alignment": None,
                        "scaffolding_quality": None,
                        "pedagogical_actionability": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
                    str(PROJECT_ROOT / "scripts" / "eval" / "plot_answer_personalization.py"),
                    "--judged-runs",
                    str(judged_path),
                    "--output-dir",
                    str(Path(tmp) / "plots"),
                ],
                cwd=str(PROJECT_ROOT),
                text=True,
                capture_output=True,
                encoding="utf-8",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Run the LLM personalization judge first", result.stderr + result.stdout)

    def test_personalization_plot_script_generates_artifacts_when_fields_are_filled(self):
        with tempfile.TemporaryDirectory() as tmp:
            judged_path = Path(tmp) / "judged_runs.jsonl"
            output_dir = Path(tmp) / "plots"
            summary_path = Path(tmp) / "quality_eval_summary.json"
            row = {
                "question_id": "q1",
                "mode": "rag_bert",
                "run_id": 1,
                "scope": "in-scope",
                "answer_correctness": 0.8,
                "answer_completeness": 0.8,
                "answer_groundedness": 0.8,
                "answer_relevance": 0.8,
                "answer_clarity": 0.7,
                "instruction_compliance": 0.8,
                "need_alignment": 0.9,
                "scaffolding_quality": 0.6,
                "pedagogical_actionability": 0.5,
            }
            judged_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
                    str(PROJECT_ROOT / "scripts" / "eval" / "plot_answer_personalization.py"),
                    "--judged-runs",
                    str(judged_path),
                    "--output-dir",
                    str(output_dir),
                    "--output-summary",
                    str(summary_path),
                ],
                cwd=str(PROJECT_ROOT),
                text=True,
                capture_output=True,
                encoding="utf-8",
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue((output_dir / "answer_personalization_core.png").exists())
            self.assertTrue((output_dir / "mode_vs_answer_personalization.md").exists())
            self.assertTrue(summary_path.exists())


if __name__ == "__main__":
    unittest.main()
