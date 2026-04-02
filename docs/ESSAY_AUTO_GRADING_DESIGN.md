# Essay Auto-Grading Design (v1)

Dokumen ini mendefinisikan rubrik penilaian essay otomatis untuk sistem `local_chatbot` yang sudah ada.
Fokus v1: penilaian per-jawaban essay, output JSON terstruktur, dan flag untuk review manual.

## 1) Tujuan

- Menilai jawaban essay secara konsisten berdasarkan rubrik berbobot.
- Menyediakan feedback yang jelas untuk mahasiswa.
- Menyediakan struktur output yang bisa langsung disimpan ke DB / ditampilkan di UI.
- Menekan risiko salah nilai dengan mekanisme `needs_manual_review`.

## 2) Ruang Lingkup v1

- Input dinilai per soal essay (`question_number`).
- Rubrik default total `100` poin.
- Output utama: skor total, skor per kriteria, alasan singkat, kekuatan, saran perbaikan, dan flags.
- Belum mencakup plagiarism checker eksternal (bisa ditambah di v2).

## 3) Rubrik Default v1 (100 poin)

| Key | Kriteria | Bobot | Deskripsi Penilaian |
| --- | --- | ---: | --- |
| `content_accuracy` | Ketepatan konten dan relevansi jawaban terhadap soal | 35 | Menilai apakah jawaban menjawab pertanyaan dan tidak keluar topik |
| `coverage_of_key_points` | Kelengkapan terhadap poin kunci pada `answer_key` | 30 | Menilai seberapa banyak poin penting yang tercakup |
| `reasoning_quality` | Kualitas argumentasi dan penalaran | 20 | Menilai alur logika, sebab-akibat, justifikasi |
| `organization_clarity` | Struktur, koherensi, keterbacaan | 10 | Menilai urutan ide, transisi, dan kejelasan |
| `language_mechanics` | Tata bahasa, ejaan, pilihan kata | 5 | Menilai akurasi bahasa tanpa mendominasi nilai substansi |

### Skala level per kriteria

- `0`: Tidak ada bukti kompetensi / tidak menjawab.
- `1`: Sangat lemah, banyak kesalahan mendasar.
- `2`: Cukup, sebagian benar namun ada gap penting.
- `3`: Baik, mayoritas tepat dan jelas, ada gap kecil.
- `4`: Sangat baik, tepat, lengkap, konsisten.

### Formula nilai

- `criterion_score = (level / 4) * weight`
- `overall_score = round(sum(criterion_score), 2)`
- `max_score = 100`

## 4) Aturan Review Manual

Set `flags.needs_manual_review = true` bila salah satu kondisi terjadi:

- Jawaban terlalu singkat (mis. `< 40` kata) untuk soal yang butuh elaborasi.
- Jawaban tidak relevan / indikasi copy-paste prompt / teks acak.
- Confidence model rendah (mis. `< 0.55`).
- Kontradiksi tinggi: menyebut banyak konsep tapi berlawanan dengan `answer_key`.
- Konten sensitif atau tidak pantas.

## 5) Kontrak Input Penilaian (disarankan)

Contoh struktur input ke service penilaian:

```json
{
  "version": "essay_autograde_input_v1",
  "courseid": 12,
  "assignmentid": 101,
  "studentid": 2001,
  "question_number": 1,
  "question_text": "Jelaskan prinsip fairness pada AI dan contohnya.",
  "expected_key_points": "Definisi fairness, jenis bias, mitigasi bias, contoh penerapan.",
  "grading_rubric": {
    "rubric_id": "essay_default_v1",
    "max_score": 100
  },
  "student_answer": "..."
}
```

Mapping ke draft kamu saat ini:

- `question_text` dari `draft_json.questions[n].stem`
- `expected_key_points` dari `draft_json.answer_key[n]`
- `grading_rubric` dari rubrik default atau custom per tugas

## 6) Kontrak Output JSON (wajib)

Output harus valid terhadap schema: `docs/essay_autograde_output.schema.json`.
Field inti:

- `overall_score` (0..100)
- `criterion_scores[]` (level, score, reason per kriteria)
- `strengths[]`
- `improvement_suggestions[]`
- `missing_key_points[]`
- `flags.needs_manual_review`
- `confidence` (0..1)

## 7) Prompting Guidelines (untuk backend nanti)

- Paksa model output JSON murni, tanpa markdown/prosa tambahan.
- Nilai per kriteria dulu, baru hitung total.
- Larang model membuat fakta di luar jawaban siswa dan konteks soal.
- Jika data tidak cukup, turunkan confidence dan aktifkan review manual.

Template ringkas:

```text
You are an academic essay grader.
Score the student's answer using the provided rubric.
Return JSON only, following the required schema exactly.
Do not add markdown fences.
```

## 8) Catatan Integrasi v2 (opsional)

- Simpan raw output + normalized output untuk audit.
- Tambahkan `grader_model`, `latency_ms`, `token_usage` untuk observabilitas.
- Tambahkan kalibrasi antar mata kuliah (rubrik custom per class/topic).