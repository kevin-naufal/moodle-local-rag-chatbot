	aku ingin membuat mapping dimana tiap level mastery memiliki output llm yang berbeda. tier rendah, tier tengah, dan tier atas memiliki level output llm yang berbeda. masalahnya sekarang aku bingung levelling nya gimana. levelling ini untuk 2 grup output llm, yaitu untuk chatbot dan untuk task generation. user yang di expect menggunakan fitur ini adalah siswa/pelajar

### Mapping final

- low = mastery topik 0-69
- mid = mastery topik 70-84
- high = mastery topik 85-100
### Chatbot
- chatbot low: jawab langsung, bahasa sederhana, 1 contoh, 1 latihan mini
- chatbot mid: jawab + alasan + contoh + tips/kesalahan umum
- chatbot high: fokus reasoning, comparison, transfer, extension question

### Task
- task low: dominan recall/comprehension, 1 langkah, instruksi eksplisit
- task mid: seimbang comprehension/application, mulai ada analysis ringan
- task high: dominan analysis/evaluation/create, konteks lebih kaya, bisa multi-step

###  Next Step?

**Ticket 1: Define Mastery Source**  
Tujuan:

- Menetapkan sumber data mastery per topik yang akan dipakai fitur

Scope:

- Tentukan field minimal: userid, courseid, topic, mastery
- Tentukan aturan baca data terbaru
- Tentukan fallback kalau data belum ada

Acceptance criteria:

- Ada definisi jelas source of truth mastery topik
- Ada rule fallback mid saat data kosong
- Ada contoh input/output data

**Ticket 2: Build Mastery Group Resolver**  
Tujuan:

- Membuat helper untuk mapping mastery -> group

Rule:

- low = 0-69
- mid = 70-84
- high = 85-100

Acceptance criteria:

- Helper bisa menerima nilai mastery dan mengembalikan low|mid|high
- Nilai batas 69/70/84/85 ter-handle dengan benar
- Ada unit test atau test case manual

**Ticket 3: Build Topic Context Resolver**  
Tujuan:

- Menentukan topik mana yang dipakai saat user chat atau generate task

Rule awal:

- 1 topik -> pakai topik itu
- multi-topik -> pakai mastery topik terendah
- topik tidak jelas -> fallback mid

Acceptance criteria:

- Sistem bisa resolve active topic
- Ada rule untuk kasus multi-topik
- Ada fallback yang konsisten

**Ticket 4: Integrate Mapping into Chatbot**  
Tujuan:

- Menyesuaikan prompt chatbot berdasarkan mastery group topik

Behavior:

- low: jawab langsung, bahasa sederhana, 1 contoh, 1 latihan mini
- mid: jawab + alasan + contoh + tips
- high: reasoning, comparison, transfer, extension question

Acceptance criteria:

- Prompt chatbot berubah sesuai group
- Output untuk low, mid, high terlihat berbeda
- Tidak merusak flow chatbot existing

**Ticket 5: Integrate Mapping into Task Generation**  
Tujuan:

- Menyesuaikan prompt task generation berdasarkan mastery group topik

Behavior:

- low: recall/comprehension, 1 langkah, instruksi eksplisit
- mid: comprehension/application, analysis ringan
- high: analysis/evaluation/create, richer context, multi-step

Acceptance criteria:

- Prompt task generation berubah sesuai group
- Hasil task berbeda jelas antar tier
- Format output existing tetap valid

**Ticket 6: Add Config/Policy Layer**  
Tujuan:

- Menyimpan mapping ini sebagai policy yang bisa diubah nanti tanpa ubah kode besar

Scope:

- Simpan threshold mastery
- Simpan descriptor output chatbot/task per group
- Versi awal boleh hardcoded dulu, asal struktur jelas

Acceptance criteria:

- Mapping policy punya satu source yang konsisten
- Threshold dan behavior mudah di-maintain

**Ticket 7: Add Logging and Debug Info**  
Tujuan:

- Membuat feature ini mudah dicek saat testing

Log minimal:

- userid
- courseid
- topic
- mastery
- group
- mode = chatbot/task

Acceptance criteria:

- Dev bisa melihat group apa yang dipilih sistem
- Mudah debug kalau output terasa tidak sesuai

**Ticket 8: QA and Scenario Validation**  
Tujuan:

- Memastikan feature benar secara logika dan terasa benar secara pedagogis

Test scenario:

- siswa low di topik A
- siswa mid di topik A
- siswa high di topik A
- siswa high di topik B tapi low di topik C

Acceptance criteria:

- Output berbeda sesuai tier
- Tidak ada case fallback yang membingungkan
- Guru bisa memahami kenapa output tertentu muncul

**Urutan Pengerjaan**

1. Ticket 1
2. Ticket 2
3. Ticket 3
4. Ticket 4
5. Ticket 5
6. Ticket 7
7. Ticket 8
8. Ticket 6 bisa paralel atau setelah resolver stabil