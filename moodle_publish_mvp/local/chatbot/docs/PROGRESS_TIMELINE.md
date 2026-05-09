# Weekly Sprint Progress Timeline (Combined)

Timeline ini digenerate murni dari commit history gabungan dua repo: `my-llm` + `moodle`.
Format sprint mingguan menggunakan rentang **Senin-Minggu**, termasuk minggu tanpa commit.

Periode: **2026-02-12** s.d. **2026-04-02**

## Sprint Overview

| Sprint | Rentang Minggu | Total Commit | Fokus Progress |
|---|---|---:|---|
| Sprint 1 | 09 Feb 2026 - 15 Feb 2026 | 3 | Inisialisasi repositori dan fondasi awal chatbot. Penguatan core chat interface dan alur interaksi pengguna. |
| Sprint 2 | 16 Feb 2026 - 22 Feb 2026 | 0 | Tidak ada commit (buffer, sinkronisasi, atau validasi). |
| Sprint 3 | 23 Feb 2026 - 01 Mar 2026 | 0 | Tidak ada commit (buffer, sinkronisasi, atau validasi). |
| Sprint 4 | 02 Mar 2026 - 08 Mar 2026 | 2 | Inisialisasi repositori dan fondasi awal chatbot. Penguatan core chat interface dan alur interaksi pengguna. |
| Sprint 5 | 09 Mar 2026 - 15 Mar 2026 | 6 | Penguatan core chat interface dan alur interaksi pengguna. Peningkatan kualitas retrieval, formatting, dan reliability sistem. |
| Sprint 6 | 16 Mar 2026 - 22 Mar 2026 | 0 | Tidak ada commit (buffer, sinkronisasi, atau validasi). |
| Sprint 7 | 23 Mar 2026 - 29 Mar 2026 | 7 | Penguatan core chat interface dan alur interaksi pengguna. Peningkatan kualitas retrieval, formatting, dan reliability sistem. Pengembangan alur draft-to-publish untuk assignment/practice. |
| Sprint 8 | 30 Mar 2026 - 05 Apr 2026 | 5 | Penguatan core chat interface dan alur interaksi pengguna. Implementasi penilaian essay otomatis. Penguatan learning mastery report dan analytics progression. |

## Sprint Details

### Sprint 1 (09 Feb 2026 - 15 Feb 2026)
- Fokus: Inisialisasi repositori dan fondasi awal chatbot. Penguatan core chat interface dan alur interaksi pengguna.
- Total commit: **3**
- Commit list:
  - [moodle] `e9196915` - 2026-02-12 15:08 - Initial commit
  - [moodle] `0670834d` - 2026-02-13 14:46 - add create course script file
  - [moodle] `5a363058` - 2026-02-13 15:32 - Create chatbot interface (demo)

### Sprint 2 (16 Feb 2026 - 22 Feb 2026)
- Fokus: Tidak ada commit (buffer, sinkronisasi, atau validasi).
- Total commit: **0**
- Commit list: Tidak ada commit pada minggu ini.

### Sprint 3 (23 Feb 2026 - 01 Mar 2026)
- Fokus: Tidak ada commit (buffer, sinkronisasi, atau validasi).
- Total commit: **0**
- Commit list: Tidak ada commit pada minggu ini.

### Sprint 4 (02 Mar 2026 - 08 Mar 2026)
- Fokus: Inisialisasi repositori dan fondasi awal chatbot. Penguatan core chat interface dan alur interaksi pengguna.
- Total commit: **2**
- Commit list:
  - [my-llm] `3d84cb1c` - 2026-03-05 17:06 - Initial commit
  - [moodle] `bc9b6977` - 2026-03-05 17:06 - Implement chatbot interface with file upload and chat functionalities

### Sprint 5 (09 Mar 2026 - 15 Mar 2026)
- Fokus: Penguatan core chat interface dan alur interaksi pengguna. Peningkatan kualitas retrieval, formatting, dan reliability sistem.
- Total commit: **6**
- Commit list:
  - [my-llm] `8d9abea3` - 2026-03-09 13:29 - Add chat message persistence and chat ID management
  - [moodle] `fe56ad16` - 2026-03-09 13:30 - Enhance chatbot interface with user message limit and usage display
  - [my-llm] `890f6491` - 2026-03-09 22:34 - Add .gitignore and enhance documentation in chatbot_ui.py, moodle_rag_runner.py, and rag.py
  - [my-llm] `4e6449ef` - 2026-03-10 04:56 - Add step-by-step guide for activating LLM and enhance file handling in chatbot_ui.py and moodle_rag_runner.py
  - [moodle] `af5f9ac7` - 2026-03-10 05:06 - feat(chatbot): add interface view and update widget integration
  - [my-llm] `2c3938e8` - 2026-03-10 11:29 - Add new source PDFs and refresh model list

### Sprint 6 (16 Mar 2026 - 22 Mar 2026)
- Fokus: Tidak ada commit (buffer, sinkronisasi, atau validasi).
- Total commit: **0**
- Commit list: Tidak ada commit pada minggu ini.

### Sprint 7 (23 Mar 2026 - 29 Mar 2026)
- Fokus: Penguatan core chat interface dan alur interaksi pengguna. Peningkatan kualitas retrieval, formatting, dan reliability sistem. Pengembangan alur draft-to-publish untuk assignment/practice.
- Total commit: **7**
- Commit list:
  - [my-llm] `82f78fe4` - 2026-03-25 13:16 - Improve chat UX, markdown formatting, and file-specific RAG routing
  - [my-llm] `69566f59` - 2026-03-28 15:06 - Improve Ollama connectivity checks and file-name matching
  - [moodle] `48372ab6` - 2026-03-28 18:01 - Enhance LLM Tutor assignment flow and course material selection
  - [my-llm] `794254de` - 2026-03-29 17:16 - feat: add Moodle publish MVP with quiz publishing and topic-based materials
  - [moodle] `01e503f3` - 2026-03-29 17:18 - feat(local/chatbot): add draft publish flow and topic-based material sync
  - [my-llm] `7ef86706` - 2026-03-29 18:54 - feat(chatbot): add practice publish flow and restore assignment type selection
  - [moodle] `15e36a1d` - 2026-03-29 18:54 - feat(local/chatbot): add practice publish flow and mode-aware quiz publishing

### Sprint 8 (30 Mar 2026 - 05 Apr 2026)
- Fokus: Penguatan core chat interface dan alur interaksi pengguna. Implementasi penilaian essay otomatis. Penguatan learning mastery report dan analytics progression.
- Total commit: **5**
- Commit list:
  - [my-llm] `0625bac7` - 2026-03-30 11:05 - Refactor project layout into app/scripts/docs and update run guides
  - [my-llm] `b9384706` - 2026-04-02 14:27 - feat: implement essay auto-grading flow and teacher mastery reporting
  - [moodle] `42200940` - 2026-04-02 14:31 - feat(local/chatbot): add essay autograding pipeline and teacher mastery reporting
  - [my-llm] `dfe326f6` - 2026-04-02 15:49 - feat: add snapshot-based mastery analytics and unify topic progress view
  - [moodle] `64a8ec07` - 2026-04-02 15:49 - feat(local/chatbot): add snapshot analytics and unify topic mastery/progress report

## Summary

- Total sprint (mingguan): **8**
- Total commit gabungan: **23**
- Commit my-llm: **12**
- Commit moodle: **11**
