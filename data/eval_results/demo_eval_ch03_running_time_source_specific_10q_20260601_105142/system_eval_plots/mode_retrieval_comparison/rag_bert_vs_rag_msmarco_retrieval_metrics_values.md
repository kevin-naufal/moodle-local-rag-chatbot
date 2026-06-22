| mode | total_runs | hit@k | mrr | coverage@k |
| --- | ---: | ---: | ---: | ---: |
| RAG-BERT | 90 | 1.0000 | 1.0000 | 0.7829 |
| RAG-MSMARCO | 90 | 1.0000 | 1.0000 | 0.7979 |

Catatan: nilai `coverage@k` menggunakan koreksi nilai Coverage@K: RAG-BERT = 0.7829 dan RAG-MSMARCO = 0.7979. Nilai ini menggantikan pemakaian keliru `avg_source_recall_at_k` pada plot sebelumnya.
