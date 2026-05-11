# Evaluation Step 3: Raw Result Schema

The selected raw result schema for each run is:

```json
{
  "question_id": "ch03-q01",
  "question": "What is the purpose of benchmarking when measuring program running time?",
  "mode": "rag_ollama",
  "run_id": 1,
  "model_name": "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M",
  "embedding_backend": "ollama",
  "model_answer": "Benchmarking uses a set of representative inputs to evaluate how well a program performs on typical workloads.",
  "retrieved_context": [
    {
      "text": "When comparing two or more programs designed to do the same set of tasks, it is customary to develop a small collection of typical inputs that can serve as benchmarks.",
      "source": "ch03_running time of algorithm (1).pdf",
      "page": 91
    }
  ],
  "latency_total": 1.95,
  "latency_retrieval": 0.42,
  "latency_generation": 1.53,
  "status": "success",
  "error_message": null,
  "timestamp": "2026-05-10T20:15:00Z"
}
```

## Field rules

- `question_id` must match the ID used in the evaluation dataset
- `mode` must be one of:
  - `llm_only`
  - `rag_ollama`
  - `rag_bert`
- `run_id` should represent repetition number, such as `1`, `2`, or `3`
- `embedding_backend` should be:
  - `null` or `"none"` for `llm_only`
  - `"ollama"` for `rag_ollama`
  - `"bert"` for `rag_bert`
- `retrieved_context` should be an empty list for `llm_only`
- `latency_retrieval` should be `0` for `llm_only`
- `status` should be either `"success"` or `"error"`
- `error_message` should be `null` when the run succeeds

## Storage recommendation

Store the raw results in:

- `data/eval_results/raw_results.jsonl`

Using JSONL is recommended because:

- one run can be appended as one JSON line
- it is easier to process in later scoring steps
- it is safer for long-running experiments than rewriting one large JSON file
