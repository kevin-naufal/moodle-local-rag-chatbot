# Answer Quality Evaluation

Dokumen ini menjelaskan metrik yang dipakai untuk menilai kualitas substansi jawaban chatbot.

## Fokus Evaluasi

Answer Quality digunakan untuk menilai apakah jawaban:

- benar secara isi
- cukup lengkap
- tetap grounded pada materi
- tetap relevan terhadap pertanyaan

## Metrik

- `answer_correctness`
  Menilai ketepatan isi jawaban terhadap reference answer atau materi sumber.

- `answer_completeness`
  Menilai apakah jawaban sudah mencakup poin-poin penting yang seharusnya ada.

- `answer_groundedness`
  Menilai apakah jawaban didukung oleh retrieved context, bukan halusinasi.

- `answer_relevance`
  Menilai apakah jawaban tetap fokus pada pertanyaan yang diajukan.

- `refusal_appropriateness`
  Menilai apakah sistem menolak dengan benar ketika pertanyaan memang tidak seharusnya dijawab.

- `key_point_coverage_rate`
  Menilai proporsi key points yang berhasil tercakup dalam jawaban.

- `quality_score`
  Skor gabungan utama untuk kualitas jawaban. Saat ini dihitung dari correctness, completeness, groundedness, dan relevance.

- `consistency_score`
  Menilai kestabilan kualitas antar-run untuk pertanyaan dan mode yang sama.

- `unsupported_claim_count`
  Menghitung rata-rata jumlah klaim yang tidak didukung konteks.

- `must_not_claim_violations`
  Menghitung pelanggaran terhadap klaim yang tidak boleh disebutkan.

## Output Ringkasan

Summary JSON menyimpan grup ini di:

```json
{
  "answer_quality": {
    "correctness": 0.84,
    "completeness": 0.79,
    "groundedness": 0.91,
    "relevance": 0.87,
    "refusal_appropriateness": 0.90,
    "key_point_coverage_rate": 0.81,
    "quality_score": 0.85,
    "consistency_score": 0.86,
    "unsupported_claim_count": 0.10,
    "must_not_claim_violations": 0.00
  }
}
```

## Output Visual

Saat `python scripts/eval/evaluate_answer_quality.py --plot` dijalankan, hasil visual untuk grup ini dipisah menjadi:

- `answer_quality_core.png`
- `answer_quality_grounding.png`
- `answer_quality_detail.png`
- `answer_quality_risks.png`
- `mode_vs_answer_quality.md`
