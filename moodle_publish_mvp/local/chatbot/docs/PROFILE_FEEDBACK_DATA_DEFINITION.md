# Profile Feedback Data Definition (v1)

Dokumen ini mendefinisikan data yang dipakai untuk fitur:
`Berikan feedback untuk siswa pada profile report menggunakan LLM`.

Fokus dokumen ini: kontrak data dulu (input, output, mapping sumber data, dan data yang perlu disimpan), sebelum implementasi UI/service.

## 1) Tujuan

- Menghasilkan feedback belajar yang personal, ringkas, dan actionable untuk siswa.
- Menggunakan data yang sudah ada di plugin `local_chatbot` (tanpa menambah event baru di v1).
- Menetapkan kontrak JSON agar integrasi Moodle <-> `my-llm` konsisten.

## 2) Sumber Data Existing (Sudah Ada)

### 2.1 Ringkasan siswa (`local_chatbot_std_profile`)

Diambil via helper:
- `local_chatbot_get_student_overall_mastery($userid)`
- `local_chatbot_get_student_class_mastery_rows($userid)`
- `local_chatbot_get_student_mastery_rows($userid)`
- `local_chatbot_get_student_topic_progress_rows($userid, 75.0)`

Field utama:
- `overallmastery`, `overallaccuracy`, `classcount`, `topiccount`, `attemptsum`, `lastupdate`
- Per kelas: `courseid`, `fullname`, `shortname`, `classmastery`, `classaccuracy`, `topiccount`, `attemptsum`, `lastupdate`
- Per topik: `courseid`, `topic`, `mastery`, `accuracy_avg`, `attempt_count`, `timemodified`
- Progress topik: `mastery_change`, `first_attempt_accuracy`, `target_reached`, `time_to_target_seconds`, `trend_points`

### 2.2 Event belajar (`local_chatbot_learn_events`)

Dipakai untuk sinyal aktivitas terbaru:
- `event_type`, `score_topic`, `duration_seconds`, `submitted_at`, `topic`, `courseid`

Contoh agregat untuk feedback:
- `recent_attempts_14d`
- `avg_score_14d`
- `avg_duration_seconds_14d`
- `last_activity_at`

### 2.3 Snapshot mingguan (`local_chatbot_weekly_snap`)

Dipakai untuk tren stabil lintas minggu:
- `week_start`, `mastery`, `accuracy_avg`, `attempt_count`

### 2.4 Sinyal essay (opsional, jika tersedia)

Sumber: `local_chatbot_essay_grades`
- `overall_score`, `confidence`, `needs_manual_review`, `timecreated`

Agregat opsional:
- `essay_count`
- `avg_essay_score`
- `manual_review_ratio`

## 3) Kontrak Input Ke LLM

Versi kontrak: `profile_feedback_input_v1`

Field minimum (wajib):
- `version`
- `generated_at`
- `student.id`
- `summary` (overall)
- `topic_metrics` (minimal topik terlemah, max 5)
- `feedback_constraints` (language, tone, max_words)

Contoh payload:

```json
{
  "version": "profile_feedback_input_v1",
  "generated_at": 1776230400,
  "student": {
    "id": 2001,
    "display_name": "Student",
    "locale": "id"
  },
  "viewer": {
    "id": 101,
    "role": "teacher"
  },
  "summary": {
    "overall_mastery": 68.4,
    "overall_accuracy": 70.1,
    "class_count": 3,
    "topic_count": 12,
    "attempt_sum": 41,
    "last_update": 1776144000
  },
  "class_metrics": [
    {
      "courseid": 12,
      "course_name": "Matematika Dasar",
      "mastery": 65.2,
      "accuracy": 67.8,
      "topic_count": 5,
      "attempt_sum": 18,
      "last_update": 1776144000
    }
  ],
  "topic_metrics": [
    {
      "courseid": 12,
      "course_name": "Matematika Dasar",
      "topic": "Persamaan Linear",
      "mastery": 52.7,
      "accuracy_avg": 58.4,
      "attempt_count": 6,
      "mastery_change": 4.2,
      "first_attempt_accuracy": 40.0,
      "target_reached": false,
      "target_mastery": 75.0,
      "time_to_target_seconds": null,
      "trend_points": [40.0, 45.0, 50.0, 52.7],
      "last_update": 1776144000
    }
  ],
  "behavior_signals": {
    "recent_attempts_14d": 7,
    "avg_score_14d": 66.3,
    "avg_duration_seconds_14d": 830,
    "last_activity_at": 1776144000
  },
  "essay_signals": {
    "essay_count": 2,
    "avg_essay_score": 74.5,
    "manual_review_ratio": 0.0
  },
  "feedback_constraints": {
    "language": "id",
    "tone": "supportive",
    "max_words": 180,
    "avoid_sensitive_labels": true
  }
}
```

## 4) Kontrak Output Dari LLM

Versi kontrak: `profile_feedback_output_v1`

Field wajib:
- `version`
- `headline`
- `overall_feedback`
- `strengths[]` (1..3 item)
- `focus_areas[]` (1..3 item)
- `next_actions[]` (2..4 item)
- `motivation`
- `confidence` (0..1)

Contoh payload:

```json
{
  "version": "profile_feedback_output_v1",
  "headline": "Performa kamu meningkat, tapi 2 topik masih perlu dikejar",
  "overall_feedback": "Mastery keseluruhan sudah naik dan konsisten. Fokus berikutnya adalah memperbaiki akurasi pada topik dengan attempt tinggi.",
  "strengths": [
    "Akurasi rata-rata sudah stabil di atas 70% pada sebagian besar topik",
    "Ada tren peningkatan pada topik yang sebelumnya rendah"
  ],
  "focus_areas": [
    {
      "topic": "Persamaan Linear",
      "issue": "Mastery masih di bawah target 75%",
      "evidence": "Mastery 52.7% dari 6 attempt"
    }
  ],
  "next_actions": [
    {
      "action": "Kerjakan 5 soal latihan Persamaan Linear dengan pembahasan",
      "success_metric": "Akurasi latihan >= 75% dalam 7 hari"
    },
    {
      "action": "Review kesalahan pada 2 attempt terakhir sebelum mencoba ulang",
      "success_metric": "Turunkan error berulang pada tipe soal yang sama"
    }
  ],
  "motivation": "Progress kamu sudah terlihat. Lanjutkan ritme latihan singkat tapi rutin.",
  "risk_flags": {
    "low_activity": false,
    "stale_data": false
  },
  "confidence": 0.82
}
```

## 5) Aturan Data & Guardrails

- Jangan kirim PII sensitif (email, phone, alamat, NIK, dsb) ke prompt LLM.
- `topic_metrics` diurutkan dari mastery terendah, kirim maksimal 5 item agar prompt tetap fokus.
- Jika `last_update` lebih lama dari 30 hari, set `risk_flags.stale_data = true`.
- Jika data sangat minim (mis. `attempt_sum < 3`), feedback harus menyatakan keterbatasan data.
- Bahasa output mengikuti `feedback_constraints.language` (`id` untuk Indonesia).

## 6) Data Penyimpanan Feedback (Disarankan)

Untuk cache/audit hasil feedback, tambahkan tabel baru (v1.1):
`local_chatbot_profile_feedback`

Field minimum yang disarankan:
- `id` (PK)
- `userid` (student)
- `viewerid`
- `courseid` (nullable, jika feedback lintas kelas)
- `input_json`
- `output_json`
- `headline`
- `confidence`
- `model_name`
- `status` (`success|fallback|error`)
- `timecreated`
- `timemodified`

Tujuan penyimpanan:
- menghindari call LLM berulang untuk data yang sama,
- audit kualitas feedback,
- baseline evaluasi sebelum optimasi prompt/model.

## 7) Implementasi MVP (Data Minimum)

Untuk MVP profile report, data minimum yang dipakai:
- `summary` dari `local_chatbot_get_student_overall_mastery()`
- `topic_metrics` (top 5 mastery terendah) dari `local_chatbot_get_student_topic_progress_rows()`
- `behavior_signals.last_activity_at` + `recent_attempts_14d` dari `local_chatbot_learn_events`

Data essay dan snapshot mingguan bisa menyusul di iterasi berikutnya.
