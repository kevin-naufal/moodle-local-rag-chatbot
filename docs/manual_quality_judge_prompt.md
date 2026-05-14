# Manual Answer-Quality Judge Prompt

Gunakan salah satu prompt berikut saat kamu ingin saya menilai kualitas jawaban sistem secara manual.

## Prompt Ringkas

```text
Saya mau mulai penilaian answer quality.
Tolong bertindak sebagai human evaluator untuk sistem LLM saya.

File yang dinilai:
<path ke answer_runs.jsonl>

Tolong:
1. baca semua jawaban di file itu,
2. nilai manual setiap jawaban,
3. buat file judged runs untuk evaluator kualitas,
4. jalankan sistem evaluasinya sampai keluar summary, tabel, dan grafik.
```

## Prompt Yang Disarankan

```text
Saya mau mulai penilaian answer quality untuk sistem LLM saya.
Tolong anggap kamu sebagai evaluator manusia, bukan sebagai sistem yang diuji.

File input:
<path ke answer_runs.jsonl>

Gunakan penilaian manual untuk setiap jawaban dengan metrik:
- answer_correctness
- answer_completeness
- answer_groundedness
- answer_relevance
- refusal_appropriateness jika perlu

Gunakan hanya skor diskret:
- 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

Tolong kerjakan langkah berikut:
1. baca file answer runs,
2. nilai manual tiap row,
3. simpan hasilnya ke judged_runs.jsonl,
4. jalankan evaluator kualitas,
5. tampilkan ringkasan hasil akhirnya.

Kalau ada rubric atau aturan penilaian khusus, saya akan kirim setelah ini. Kalau belum ada, pakai rubric default yang selama ini kita gunakan.
```

## Prompt Dengan Rubric Khusus

```text
Saya mau mulai penilaian answer quality.
Tolong bertindak sebagai evaluator manusia untuk output sistem LLM saya.

File input:
<path ke answer_runs.jsonl>

Rubric penilaian:
- correctness: <aturan>
- completeness: <aturan>
- groundedness: <aturan>
- relevance: <aturan>
- refusal: <aturan jika ada>

Gunakan hanya skor diskret 0.0 sampai 1.0 dengan kenaikan 0.1.

Tolong nilai manual semua jawaban di file tersebut lalu:
1. buat file judged runs,
2. jalankan evaluator kualitas,
3. berikan summary hasil per mode.
```

## Output Yang Akan Saya Kerjakan

Jika kamu mengirim prompt di atas, saya akan:

1. membaca file `answer_runs`
2. memberi nilai manual per jawaban
3. membuat file input judged runs di `data/quality_eval_inputs/`
4. menjalankan:

```bash
python scripts/evaluate_answer_quality.py --judged-runs <judged_runs.jsonl> --plot
```

5. memberi kamu:
   - path file judged runs
   - path summary JSON
   - path tabel markdown
   - ringkasan hasil utama
