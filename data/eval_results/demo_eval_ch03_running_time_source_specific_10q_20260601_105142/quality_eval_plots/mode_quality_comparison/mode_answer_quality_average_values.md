| mode | total_runs | correctness | completeness | relevance | groundedness |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM-only | 90 | 0.3133 | 0.3244 | 0.6344 | N/A |
| RAG-BERT | 90 | 0.6411 | 0.6689 | 0.7556 | 0.7800 |
| RAG-MSMARCO | 90 | 0.6711 | 0.7111 | 0.7533 | 0.7944 |

Catatan: `groundedness` untuk LLM-only adalah N/A karena seluruh run LLM-only tidak memiliki skor groundedness pada data evaluasi ini.
