# LLM Mastery Level Mapping

## Tujuan

Dokumen ini mendefinisikan mapping **mastery group -> level output LLM** untuk dua use case:

1. `chatbot`
2. `task_generation`

Target user utama adalah **siswa/pelajar**, jadi leveling tidak boleh dimaknai sebagai "semakin rendah semakin miskin jawaban", tetapi sebagai **perbedaan tingkat scaffolding, kompleksitas bahasa, dan tantangan belajar**.

## Prinsip Desain

1. Leveling diletakkan pada **output behavior**, bukan pada model yang dipakai.
2. Semakin rendah mastery, semakin tinggi bantuan yang diberikan.
3. Semakin tinggi mastery, semakin tinggi tantangan, kemandirian, dan transfer pengetahuan.
4. Untuk siswa, `task_generation` tidak boleh hanya membuat soal lebih mudah; ia harus tetap mendorong naik level.
5. Karena policy mastery saat ini di plugin masih berbasis **minimum threshold per topik**, mastery group ini diposisikan sebagai layer tambahan di atas score mastery 0-100 yang sudah ada.

## Rekomendasi Mastery Group

Mapping 3 tier yang paling aman untuk sistem sekarang:

| Group | Rentang mastery | Makna pedagogis |
| --- | --- | --- |
| `low` | `0-69` | Belum tuntas, butuh remedial dan banyak scaffolding |
| `mid` | `70-84` | Sudah tuntas dasar, siap latihan terarah |
| `high` | `85-100` | Sudah kuat, siap tantangan analitis dan mandiri |

Catatan:

- Angka `70` dipakai sebagai batas bawah `mid` agar selaras dengan mastery policy saat ini yang berorientasi pada ambang kelulusan topik.
- Angka `85` dipakai sebagai awal `high` agar tier atas benar-benar merepresentasikan siswa yang sudah relatif stabil.

## Mapping Untuk Chatbot

### `low`

Tujuan output:

- Menjelaskan konsep dengan bahasa sederhana
- Memberi jawaban langsung terlebih dahulu
- Mengurangi beban kognitif
- Menutup jawaban dengan satu langkah belajar berikutnya

Ciri output:

- 1 ide utama per jawaban
- kalimat pendek
- istilah teknis dibatasi dan langsung dijelaskan
- gunakan contoh konkret
- boleh beri format: `konsep -> contoh -> latihan mini`

Contoh perilaku:

- langsung jawab inti pertanyaan
- beri analogi sederhana
- jika konsep sulit, pecah menjadi langkah kecil
- tutup dengan 1 pertanyaan cek pemahaman atau 1 latihan singkat

### `mid`

Tujuan output:

- Menjelaskan konsep dan alasan
- Melatih aplikasi sederhana
- Mulai mengurangi ketergantungan pada jawaban jadi

Ciri output:

- jawaban tetap jelas, tetapi lebih lengkap
- boleh memakai istilah mapan dengan penjelasan singkat
- berikan langkah penyelesaian
- berikan 1 variasi contoh atau 1 kesalahan umum

Contoh perilaku:

- `jawaban inti -> alasan -> contoh -> tips`
- bila user minta bantuan, berikan hint bertahap
- dorong user membandingkan dua konsep yang mirip

### `high`

Tujuan output:

- Mendorong penalaran mandiri
- Menguji transfer konsep
- Mengembangkan kemampuan analisis dan evaluasi

Ciri output:

- tidak selalu memberi jawaban penuh di awal bila konteksnya latihan
- lebih banyak prompt reflektif
- boleh memberi alternatif pendekatan
- tambahkan extension question

Contoh perilaku:

- `ringkasan inti -> alasan -> alternatif -> tantangan lanjutan`
- minta user mempertimbangkan trade-off
- hubungkan konsep ke konteks baru atau kasus nyata

## Mapping Untuk Task Generation

### `low`

Tujuan task:

- membangun fondasi
- memastikan pengenalan konsep inti
- fokus pada recall, comprehension, dan aplikasi satu langkah

Karakter task:

- 70% recall/comprehension
- 30% aplikasi dasar
- konteks singkat
- instruksi eksplisit
- satu kompetensi utama per soal

Format yang cocok:

- multiple choice
- matching
- isian singkat
- essay sangat pendek

Aturan generasi:

- gunakan kata kerja seperti `identify`, `mention`, `choose`, `complete`, `explain briefly`
- hindari soal multi-langkah
- distractor dibuat jelas dan tidak terlalu menjebak

### `mid`

Tujuan task:

- melatih penerapan konsep
- membangun koneksi antar konsep
- menyiapkan transisi ke soal analitis

Karakter task:

- 40% comprehension
- 40% application
- 20% analysis ringan
- mulai pakai konteks atau skenario singkat
- bisa menggabungkan 2 konsep terkait

Format yang cocok:

- multiple choice dengan distractor lebih dekat
- short essay
- problem solving sederhana
- case mini

Aturan generasi:

- gunakan kata kerja seperti `apply`, `compare`, `classify`, `solve`, `justify`
- instruksi tetap jelas, tetapi tidak terlalu diarahkan

### `high`

Tujuan task:

- menguji analisis, evaluasi, dan transfer
- memfasilitasi pemikiran mandiri
- memberi tantangan yang tetap relevan dengan topik

Karakter task:

- 20% application
- 50% analysis
- 30% evaluation/create
- konteks lebih kaya
- bisa multi-step
- bisa ada lebih dari satu jawaban masuk akal jika rubrik jelas

Format yang cocok:

- case-based question
- analytical essay
- open response
- project/presentation prompt

Aturan generasi:

- gunakan kata kerja seperti `analyze`, `evaluate`, `design`, `argue`, `defend`, `synthesize`
- distractor lebih halus untuk multiple choice
- rubrik perlu menilai kualitas alasan, bukan hanya benar/salah

## Rule Yang Disarankan

### Untuk chatbot

Pilih group berdasarkan:

1. `mastery topic terbaru` bila tersedia
2. jika tidak ada, pakai `mastery rata-rata course`
3. jika belum ada data sama sekali, default ke `mid` untuk aman

Alasan default `mid`:

- tidak terlalu meremehkan siswa baru
- tidak terlalu menuntut bagi siswa yang belum punya histori

### Untuk task generation

Jika task dipakai untuk **latihan siswa**, jangan 100% mengikuti tier saat ini.

Gunakan komposisi:

- `70%` sesuai tier saat ini
- `30%` satu tingkat di atasnya

Contoh:

- siswa `low` mendapat task dominan `low`, tetapi 30% item `mid`
- siswa `mid` mendapat task dominan `mid`, tetapi 30% item `high`
- siswa `high` tetap dominan `high`, dengan variasi transfer konteks

Ini penting agar sistem tidak mengunci siswa di level yang sama.

## Prompt Strategy Yang Disarankan

Tambahkan blok instruksi khusus sesuai group ke prompt builder.

### Chatbot prompt modifier

#### `low`

`Explain in simple language for a student. Give the direct answer first, then one short example, then one small follow-up practice question. Avoid long paragraphs and heavy jargon.`

#### `mid`

`Explain for a student with basic mastery. Give the answer, the reason, one example, and one common mistake or tip. Keep the explanation structured and moderately detailed.`

#### `high`

`Explain for a student with strong mastery. Prioritize reasoning, comparison, and transfer. Add one extension question or challenge that encourages independent thinking.`

### Task generation prompt modifier

#### `low`

`Generate beginner-friendly items with explicit wording, one concept per item, and mostly recall/comprehension plus a small amount of basic application.`

#### `mid`

`Generate intermediate items that balance comprehension and application, with a few light analysis tasks and realistic but still concise context.`

#### `high`

`Generate advanced student tasks that emphasize analysis, justification, transfer, and independent reasoning. Use richer context and less scaffolding.`

## Titik Integrasi Teknis

Dengan struktur repo saat ini, titik integrasi paling masuk akal:

1. Prompt builder frontend Moodle:
   - `local/chatbot/amd/src/widget.js`
2. General prompt runner:
   - `app/moodle_rag_runner.py`
3. Policy resolver berbasis mastery:
   - helper baru di `local/chatbot/locallib.php`

Urutan implementasi yang disarankan:

1. Tambah resolver `mastery -> group`
2. Tambah config mapping per group
3. Sisipkan modifier ke prompt `chatbot`
4. Sisipkan modifier ke prompt `practice/assignment generation`
5. Uji sample siswa `low`, `mid`, `high`

## Keputusan Utama

Kalau ingin cepat jalan, gunakan keputusan ini:

- `low = 0-69`
- `mid = 70-84`
- `high = 85-100`
- chatbot dibedakan lewat `scaffolding depth`
- task generation dibedakan lewat `cognitive complexity`
- default tanpa data = `mid`
- latihan siswa = `70% current tier + 30% next tier`

Itu biasanya cukup stabil untuk MVP dan masih mudah dikembangkan nanti jika ingin jadi 4 atau 5 level.
