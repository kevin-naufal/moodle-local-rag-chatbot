# Publish-to-Class MVP Skeleton (Moodle)

Folder ini berisi skeleton siap tempel untuk fitur `Publish` dari draft LLM ke course Moodle.

## Target Path di Moodle

Copy isi folder `local/chatbot` dari paket ini ke plugin kamu:

- `local/chatbot/save_draft.php`
- `local/chatbot/publish.php`
- `local/chatbot/classes/service/draft_repository.php`
- `local/chatbot/classes/service/draft_validator.php`
- `local/chatbot/classes/service/markdown_draft_parser.php`
- `local/chatbot/classes/service/publisher.php`
- `local/chatbot/db/access.php`
- `local/chatbot/db/install.xml`
- `local/chatbot/db/upgrade.php`
- `local/chatbot/lang/en/local_chatbot.php`

## Data Contract `draft_json` (minimal)

```json
{
  "assignment_title": "AI Ethics and Security Quiz",
  "learning_objectives": [
    "Understand key principles of AI ethics",
    "Identify AI security risks"
  ],
  "instructions": "Answer all questions.",
  "questions": [
    {
      "number": 1,
      "stem": "What is fairness in AI?",
      "options": {
        "A": "No bias",
        "B": "Fast model",
        "C": "No logs",
        "D": "No human review"
      }
    }
  ],
  "answer_key": {
    "1": "A"
  },
  "grading_rubric": [
    "Correct: 20 points",
    "Incorrect: 0 points"
  ]
}
```

## Cara Pakai Singkat

1. Dari UI, kirim POST ke `local/chatbot/save_draft.php` untuk simpan draft.
2. Endpoint bisa menerima:
   - `draft_json` (sudah terstruktur), atau
   - `draft_text` (hasil markdown mentah dari LLM, akan di-parse otomatis).
3. Simpan `draftid` dari response.
4. Saat user klik Publish, kirim POST ke `local/chatbot/publish.php` dengan `draftid`, `courseid`, dan `sesskey`.
5. Endpoint akan validasi draft, buat activity `mod_assign`, lalu update status ke `published` / `failed`.

## Contoh Integrasi UI (Fetch)

```javascript
// Save draft.
const saveResp = await fetch(M.cfg.wwwroot + '/local/chatbot/save_draft.php', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({
    sesskey: M.cfg.sesskey,
    courseid: String(courseId),
    assignment_type: 'multiple_choice',
    question_count: '5',
    draft_text: llmDraftText // atau pakai draft_json
  })
}).then(r => r.json());

if (!saveResp.success) {
  throw new Error(saveResp.message);
}

// Publish draft.
const publishResp = await fetch(M.cfg.wwwroot + '/local/chatbot/publish.php', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({
    sesskey: M.cfg.sesskey,
    courseid: String(courseId),
    draftid: String(saveResp.draftid)
  })
}).then(r => r.json());
```

## Catatan

- Publish behavior:
  - `assignment_type = multiple-choice` -> publish ke **Quiz** (auto-graded MC).
  - selain itu (mis. `essay`) -> publish ke **Assignment**.
- `Answer Key` tidak dipublish ke siswa (tetap tersimpan di DB draft).
- Jalankan upgrade plugin setelah menambah `db/install.xml` dan `db/access.php`.
- Untuk plugin existing, tambahkan `db/upgrade.php` lalu bump `$plugin->version` (mis. `2026032900`) di `version.php`.
- Setelah bump versi, jalankan `Site administration -> Notifications` untuk eksekusi migration.
- Capability yang dipakai:
  - `local/chatbot:managedrafts` untuk save/edit draft.
  - `local/chatbot:publish` untuk publish ke kelas.
