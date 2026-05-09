# Ticket 06 - Student Mastery View

## Summary

Tambahkan tampilan khusus siswa untuk melihat `mastery per topik` dan `group low/mid/high` pada kelas yang diikuti.

## Objective

Siswa bisa melakukan self-check kemampuan per bab/topik tanpa harus meminta guru membuka `Teacher Report` atau `Mastery Policy`.

## Scope

- Halaman baru untuk siswa, contoh: `/local/chatbot/student_mastery.php`
- Data source: `local_chatbot_std_profile`
- Mapping group:
  - `0-69 => low`
  - `70-84 => mid`
  - `85-100 => high`
- Tampilkan daftar topik per kelas + status mastery
- Tampilkan fallback status jika belum ada data topik

## User Story

Sebagai siswa, saya ingin melihat mastery saya per topik agar tahu bab mana yang perlu saya perbaiki sebelum ujian.

## Functional Requirements

1. Siswa hanya melihat data miliknya sendiri (`$USER->id`).
2. Siswa dapat memilih kelas (course filter) jika ikut lebih dari satu kelas.
3. Sistem menampilkan kolom minimum:
   - `Topic`
   - `Mastery (%)`
   - `Level (low/mid/high)`
   - `Status` (`Ready/Need Practice/No Data`)
   - `Last Update`
4. Jika topik belum memiliki row profile:
   - tampilkan `No Data`
   - level default ditampilkan sebagai `mid (fallback)` atau label khusus fallback.
5. Halaman tidak menampilkan data siswa lain.

## Non-Functional Requirements

- Akses cepat, tidak melakukan query berat berulang.
- Konsisten dengan UI plugin existing (report layout Moodle).
- Aman secara permission.

## Permission & Access

- Role siswa: boleh akses halaman student mastery miliknya.
- Role teacher/admin: boleh akses juga, tapi default tetap menampilkan data user yang sedang login, kecuali nanti ditambah mode impersonasi (out of scope ticket ini).

## Suggested Technical Design

1. Tambah helper resolver dataset siswa:
   - gunakan `local_chatbot_get_student_course_topic_mastery_status_rows($userid, $courseid)`
   - atau helper baru tipis yang menambahkan `group` berdasarkan mastery
2. Tambah mapper presentation:
   - `mastery -> group`
   - `group/status -> badge label`
3. Render table di `student_mastery.php`
4. Tambah link navigasi dari halaman `index.php` chatbot

## API/Helper Reuse

- `local_chatbot_map_mastery_to_group(...)`
- `local_chatbot_resolve_topic_mastery_group(...)`
- `local_chatbot_get_student_course_topic_mastery_status_rows(...)`

## Acceptance Criteria

1. Siswa dapat membuka halaman Student Mastery tanpa error.
2. Data yang tampil hanya milik siswa login.
3. Tiap topik menampilkan mastery + group dengan mapping benar.
4. Topik tanpa data menampilkan status fallback yang jelas.
5. Guru/admin tidak kehilangan akses ke halaman report existing.

## Test Scenarios

1. Siswa dengan mastery campuran:
   - topik A `65` -> `low`
   - topik B `75` -> `mid`
   - topik C `90` -> `high`
2. Siswa tanpa data topik:
   - tampil `No Data`
3. Siswa pindah course filter:
   - data topik mengikuti kelas terpilih
4. Permission check:
   - user lain tidak bisa melihat data siswa berbeda

## Out of Scope

- Grafik tren mingguan detail.
- Rekomendasi AI otomatis di halaman ini.
- Edit policy dari sisi siswa.
- Perubahan prompt runtime LLM.

## Definition of Done

- Halaman student mastery tersedia dan bisa diakses role siswa.
- Mapping level tampil benar untuk semua boundary.
- UI/permission lolos smoke test.
