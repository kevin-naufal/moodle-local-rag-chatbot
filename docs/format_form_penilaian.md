# Format Form Penilaian Jawaban Chatbot

Format ini ditujukan untuk **evaluasi per jawaban** di Moodle, bukan untuk Google Form panjang.

Form hanya muncul saat **evaluation mode aktif** dan ditampilkan **setelah chatbot memberi jawaban**.

## Skala Penilaian

- 1 = Sangat tidak setuju
- 2 = Tidak setuju
- 3 = Netral
- 4 = Setuju
- 5 = Sangat setuju

## Pertanyaan Inti

Gunakan **1 pertanyaan untuk tiap metrik** agar user tidak kelelahan.

- **Correctness**
  Jawaban chatbot benar sesuai materi pembelajaran.

- **Groundedness**
  Jawaban chatbot didukung oleh materi atau konteks yang relevan.

- **Relevance**
  Jawaban chatbot sesuai dengan pertanyaan yang diajukan.

- **Instruction Compliance**
  Jawaban chatbot mengikuti instruksi yang saya berikan.

- **Need Alignment**
  Jawaban chatbot membantu kebutuhan saya saat bertanya.

- **Scaffolding Quality**
  Jawaban chatbot membantu saya memahami materi secara bertahap.

## Pertanyaan Tambahan Opsional

- Jawaban chatbot mudah dipahami.

- Komentar:
  Bagian apa yang paling membantu atau masih perlu diperbaiki?

## Saran Tampilan Di Moodle

- Tampilkan form tepat di bawah jawaban chatbot.
- Satu form hanya untuk satu jawaban chatbot.
- Setelah user submit, form untuk jawaban itu dikunci atau disembunyikan.
- Komentar dibuat opsional agar proses evaluasi tetap cepat.

## Data Yang Sebaiknya Disimpan

- `userid`
- `question_id`
- `run_id`
- `chat_mode`
- `user_question`
- `chatbot_answer`
- skor untuk tiap metrik
- komentar opsional
- `timestamp`

## Catatan

Kalau tujuan evaluasi adalah cepat dan konsisten, form ini lebih cocok daripada versi panjang 3 pertanyaan per metrik.
