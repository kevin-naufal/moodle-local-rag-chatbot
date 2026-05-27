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


if __name__ == "__main__":
    unittest.main()
