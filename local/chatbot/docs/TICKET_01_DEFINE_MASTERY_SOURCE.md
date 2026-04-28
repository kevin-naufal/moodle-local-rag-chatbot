# Ticket 01 - Define Mastery Source

## Summary

Menetapkan **source of truth** untuk `mastery topik siswa per kelas` yang akan dipakai fitur:

- `chatbot output mapping by mastery group`
- `task generation mapping by mastery group`

Ticket ini tidak mengubah output LLM dulu. Fokus ticket ini adalah **kontrak data dan aturan pembacaan mastery**.

## Objective

Memastikan sistem punya satu definisi resmi untuk:

1. dari tabel mana `mastery topik` dibaca,
2. field apa saja yang dianggap valid,
3. bagaimana memilih record yang dipakai,
4. apa fallback jika data tidak tersedia.

## Decision

### Source of truth

Gunakan tabel:

- `local_chatbot_std_profile`

Alasan:

- tabel ini memang menyimpan profil siswa per `user-course-topic`,
- sudah berisi nilai `mastery` final hasil agregasi event belajar,
- sudah punya unique key `userid, courseid, topic`,
- sudah dipakai helper mastery existing.

Referensi:

- schema tabel: `local/chatbot/db/install.xml`
- helper existing: `local_chatbot_get_user_topic_mastery_map($userid, $courseid)`

Catatan scope:

- Untuk V1 flow `mastery topik -> llm level`, hanya `local_chatbot_std_profile` yang dipakai.
- Monitoring aktivitas, histori attempt, dan analytics tidak masuk flow runtime mapping pada ticket ini.

## Source Schema

### Primary table

Tabel: `local_chatbot_std_profile`

Field minimum yang dipakai:

- `userid`
- `courseid`
- `topic`
- `mastery`
- `attempt_count`
- `last_event_time`
- `timemodified`

Field berguna tambahan:

- `accuracy_avg`
- `last_score`
- `trend`

## Read Rules

### Rule 1 - Lookup key

Lookup mastery topik dilakukan dengan key:

- `userid`
- `courseid`
- `topic`

Karena fitur ini berbasis `mastery per bab/topik di dalam kelas`.

### Rule 2 - Topic identity

Topic identity mengikuti nama topik yang sudah dinormalisasi oleh flow existing plugin.

Sumber topik harus konsisten dengan:

- section/topic course yang dipakai di Moodle,
- hasil resolver topic di learning profile service,
- helper list topic existing.

### Rule 3 - Returned mastery value

Nilai yang dipakai adalah:

- `mastery` dari `local_chatbot_std_profile`

Nilai dibaca sebagai persentase `0..100`.

### Rule 4 - Freshness

Record dianggap usable jika:

- row ditemukan untuk `userid + courseid + topic`

Freshness tambahan untuk observability:

- `timemodified` dan/atau `last_event_time` disimpan sebagai metadata pembacaan

Pada Ticket 1 ini, data lama **masih boleh dipakai**. Kebijakan stale threshold bisa ditambahkan di ticket lanjutan.

## Fallback Rules

### Case A - Row topic ditemukan

Gunakan langsung:

- `mastery` dari `local_chatbot_std_profile`

### Case B - Topic ada di course, tapi siswa belum punya row profile

Status:

- `no_topic_mastery_data`

Behavior:

- jangan pakai `0` sebagai mastery implisit untuk mapping LLM,
- gunakan fallback group default: `mid`

Catatan:

- `0` di UI mastery debt masih bisa dipakai sebagai visual "belum ada capaian",
- tetapi untuk adaptive LLM, `no data` harus dibedakan dari `low mastery`.

### Case C - Tabel profile belum siap atau kosong

Fallback:

- group default `mid`
- status source `profile_table_unavailable`

### Case D - Topic tidak match karena naming mismatch

Fallback:

- treat as `topic_not_resolved`
- group default `mid`

Catatan:

- mismatch naming harus ditangani di Ticket 3 `Topic Context Resolver`

## Output Contract For Next Tickets

Task 1 menghasilkan kontrak konseptual berikut:

```json
{
  "userid": 123,
  "courseid": 45,
  "topic": "Persamaan Linear",
  "source": "local_chatbot_std_profile",
  "mastery": 72.5,
  "attempt_count": 4,
  "last_event_time": 1776230400,
  "timemodified": 1776230500,
  "status": "ok"
}
```

Jika data tidak ada:

```json
{
  "userid": 123,
  "courseid": 45,
  "topic": "Persamaan Linear",
  "source": "local_chatbot_std_profile",
  "mastery": null,
  "attempt_count": 0,
  "last_event_time": null,
  "timemodified": null,
  "status": "no_topic_mastery_data",
  "fallback_group": "mid"
}
```

## Existing Functions To Reuse

Helper existing yang relevan:

- `local_chatbot_get_user_topic_mastery_map(int $userid, int $courseid)`
- `local_chatbot_get_student_mastery_rows(int $userid)`
- `local_chatbot_get_student_course_topic_mastery_status_rows(int $userid, int $courseid)`

Catatan implementasi:

- Ticket 2 sebaiknya tidak query tabel langsung dari banyak tempat.
- Buat helper baru yang membungkus resolver source ini dalam satu fungsi pusat.

## Proposed Helper For Next Ticket

Disiapkan untuk Ticket 2/3:

`local_chatbot_get_topic_mastery_source_row(int $userid, int $courseid, string $topic): array`

Return minimum:

- `status`
- `mastery`
- `topic`
- `courseid`
- `userid`
- `attempt_count`
- `last_event_time`
- `timemodified`

## Acceptance Criteria

- Source of truth mastery topik ditetapkan sebagai `local_chatbot_std_profile`
- Ada aturan eksplisit untuk `no data` vs `low mastery`
- Ada output contract yang bisa dipakai Ticket 2 dan Ticket 3
- Tidak ada ambiguitas antara topik level dan course level

## Out of Scope

Ticket ini belum mencakup:

- mapping `mastery -> low|mid|high`
- resolver topik aktif dari prompt/chat context
- integrasi ke prompt chatbot
- integrasi ke prompt task generation
- UI konfigurasi policy
- monitoring aktivitas siswa
- analytics berbasis histori attempt

## Notes

Keputusan penting di ticket ini:

- `no data` tidak sama dengan `0 mastery`
- adaptive LLM sebaiknya fallback ke `mid`, bukan ke `low`

Ini penting agar siswa baru tidak otomatis dianggap lemah hanya karena histori belajar belum terbentuk.
