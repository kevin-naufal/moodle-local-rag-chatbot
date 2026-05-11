# Evaluation Step 2: Run Each Question Across 3 Modes

This step means:

Each question is executed in 3 modes, and each mode is repeated 3 times.

- `llm_only` -> 3 runs
- `rag_ollama` -> 3 runs
- `rag_bert` -> 3 runs

If there are 10 questions:

- `10 x 3 modes x 3 runs = 90 results`

## Why repeat each mode 3 times?

The repetition is used to:

- observe output stability
- see whether the answer changes across runs
- measure latency variation
- capture non-deterministic behavior from the model or retrieval pipeline

## Suggested result structure

Example table:

| question_id | mode | run_id | model_answer | retrieved_context | latency_total | status |
|---|---|---|---|---|---|---|
| q01 | llm_only | 1 | ... | null | ... | success |
| q01 | llm_only | 2 | ... | null | ... | success |
| q01 | llm_only | 3 | ... | null | ... | success |
| q01 | rag_ollama | 1 | ... | ... | ... | success |
| q01 | rag_ollama | 2 | ... | ... | ... | success |
| q01 | rag_ollama | 3 | ... | ... | ... | success |

## Experimental decisions to fix

Before running the evaluation, decide:

1. whether `temperature` is fixed
2. whether the prompt is identical across runs
3. whether the order of modes is fixed or randomized
4. whether final reporting uses:
   - the average of 3 runs
   - or each run reported separately

## Recommended setup

- keep the prompt fixed
- keep model parameters fixed
- run each mode 3 times
- use the average score per mode as the final reported result
- still keep all raw run outputs for traceability

## Core points to define in Step 2

1. Modes being tested:
   - `llm_only`
   - `rag_ollama`
   - `rag_bert`

2. Number of repetitions:
   - each mode is run 3 times for each question

3. Parameters must stay fixed:
   - prompt
   - model
   - temperature
   - maximum output length

4. Data stored for each run:
   - `question_id`
   - `mode`
   - `run_id`
   - `model_answer`
   - `retrieved_context`
   - `latency_total`
   - `latency_retrieval`
   - `latency_generation`
   - `status`

5. Purpose of repeating each run 3 times:
   - to observe answer stability
   - to observe latency variation
   - to reduce the risk of drawing conclusions from a single accidental output

## Short formal wording

Each question is executed in three modes: `llm_only`, `rag_ollama`, and `rag_bert`. Each mode is run three times using the same parameter settings so that the results can be compared consistently and to observe answer stability as well as response-time variation.
