import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RunDemoEvalScriptTest(unittest.TestCase):
    def test_ollama_startup_reuses_reachable_server(self):
        script = (PROJECT_ROOT / "scripts" / "run_demo_eval.ps1").read_text(encoding="utf-8")

        function_start = script.index("function Start-OllamaDebugServer")
        process_lookup = script.index("$running = Get-Process ollama", function_start)
        reachable_guard = script.index("if (Test-OllamaReachable", function_start)

        self.assertLess(reachable_guard, process_lookup)
        self.assertIn("Ollama sudah berjalan", script[function_start:process_lookup])
        self.assertIn("return", script[function_start:process_lookup])

    def test_force_new_answer_runs_env_disables_resume(self):
        script = (PROJECT_ROOT / "scripts" / "run_demo_eval.ps1").read_text(encoding="utf-8")

        self.assertIn("DEMO_EVAL_FORCE_NEW_ANSWER_RUNS", script)
        self.assertIn("if (-not $ForceNewAnswerRuns)", script)
        self.assertIn('$runEvalArgs += "--resume"', script)

    def test_demo_eval_supports_chat_model_matrix(self):
        script = (PROJECT_ROOT / "scripts" / "run_demo_eval.ps1").read_text(encoding="utf-8")

        self.assertIn("DEMO_EVAL_CHAT_MODELS", script)
        self.assertIn("--chat-models", script)
        self.assertIn("qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b", script)


if __name__ == "__main__":
    unittest.main()
