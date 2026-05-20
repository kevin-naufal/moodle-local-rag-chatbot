# Answer Personalization Evaluation

Dokumen ini menjelaskan metrik yang dipakai untuk menilai query-based personalization pada jawaban chatbot.

## Fokus Evaluasi

Answer Personalization digunakan untuk menilai apakah jawaban:

- mengikuti instruksi yang tersurat dalam query
- selaras dengan kebutuhan langsung pengguna
- cukup jelas untuk pembelajar
- memberi dukungan belajar yang sesuai

## Metrik

- `instruction_compliance`
  Menilai apakah jawaban mengikuti instruksi atau batasan yang dinyatakan dalam query.

- `need_alignment`
  Menilai apakah jawaban selaras dengan kebutuhan langsung pengguna pada pertanyaan tersebut.

- `answer_clarity`
  Menilai apakah jawaban disampaikan dengan jelas dan mudah dipahami.

- `scaffolding_quality`
  Menilai apakah jawaban membimbing pemahaman pengguna secara bertahap, bukan hanya memberi hasil akhir.

- `pedagogical_actionability`
  Menilai apakah jawaban memberi arahan belajar yang bisa ditindaklanjuti, misalnya contoh, langkah lanjutan, atau fokus belajar berikutnya.

## Output Ringkasan

Summary JSON menyimpan grup ini di:

```json
{
  "answer_personalization": {
    "instruction_compliance": 0.94,
    "need_alignment": 0.88,
    "answer_clarity": 0.83,
    "scaffolding_quality": 0.61,
    "pedagogical_actionability": 0.34
  }
}
```

## Catatan Ruang Lingkup

Evaluasi ini menilai personalization pada level query, bukan profile-based personalization. Artinya yang dinilai adalah penyesuaian jawaban terhadap kebutuhan yang dinyatakan dalam pertanyaan, bukan terhadap profil pengguna jangka panjang.

## Output Visual

Saat `python scripts/eval/evaluate_answer_quality.py --plot` dijalankan, hasil visual untuk grup ini dipisah menjadi:

- `answer_personalization_core.png`
- `answer_personalization_learning_support.png`
- `mode_vs_answer_personalization.md`
