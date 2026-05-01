# Ticket 07 - Chat Output Cleanup

## Summary

Rapikan output jawaban chat agar lebih konsisten dibaca user (tanpa duplikasi heading, tanpa format campur aduk, dan tanpa pengulangan konten).

## Problem Statement

Saat ini output chat kadang muncul seperti:

- `## Answer` + `**Answer:**` dobel
- struktur jawaban berulang (isi sama muncul 2 kali)
- format campur (paragraf + bullet + label) yang terasa tidak rapi

Akibatnya user merasa hasil LLM "berantakan" walau isinya benar.

## Objective

Jawaban chat menjadi:

- bersih (clean markdown)
- ringkas dan konsisten
- tetap mempertahankan isi utama + sumber (jika mode RAG)

## Scope

- Berlaku untuk output `action=chat` di tab Chat.
- Berlaku untuk mode:
  - `rag` (dengan class/topic valid)
  - `general` (tanpa retrieval)
- Tetap menampilkan `sources` jika mode `rag`.
- Tidak mengubah format draft generator Assignment/Practice.

## Out of Scope

- Redesign UI besar.
- Perubahan model LLM.
- Perubahan struktur output untuk endpoint `save_draft.php` / `publish.php`.

## Functional Requirements

1. Hapus heading/label duplikat seperti `## Answer` + `**Answer:**` jika berurutan.
2. Cegah pengulangan blok isi yang sama dalam satu jawaban.
3. Normalisasi markdown chat:
   - maksimal satu heading pembuka (opsional),
   - paragraf terstruktur,
   - bullet hanya jika perlu.
4. Pertahankan style adaptif low/mid/high saat topic valid.
5. Saat topik kosong/invalid, tetap fallback group `mid`.
6. Sumber tidak hilang di mode `rag`, dan tetap kosong di mode `general`.

## Suggested Technical Design

1. Tambah sanitizer output khusus chat, contoh:
   - `local_chatbot_normalize_chat_answer(string $answer): string`
2. Panggil sanitizer setelah hasil dari:
   - `local_chatbot_run_rag(...)`
   - `local_chatbot_run_llm_general(...)`
3. Gunakan rule berbasis regex ringan:
   - collapse heading dobel,
   - deduplicate paragraf identik berurutan,
   - trim whitespace berlebih.
4. Batasi sanitizer hanya untuk action `chat` agar flow assignment/practice tidak berubah.

## Acceptance Criteria

1. Jawaban chat tidak lagi menampilkan format dobel `## Answer` + `**Answer:**` beruntun.
2. Jawaban chat tidak menampilkan blok isi yang terduplikasi.
3. Pada mode valid topic (`rag`), `sources` tetap muncul normal.
4. Pada mode tanpa/invalid topic (`general`), `sources = []`.
5. Tidak ada regresi pada generate assignment dan generate practice.

## Test Scenarios

1. Chat umum tanpa class/topic:
   - input: `What is AI ethics?`
   - expected: jawaban rapi, `chat_mode=general`, `sources=[]`
2. Chat dengan class/topic valid:
   - input: `What is AI ethics?`
   - expected: jawaban rapi, `chat_mode=rag`, `sources` berisi PDF terkait
3. Chat dengan class valid + topic invalid:
   - expected: `chat_mode=general`, `llm_group=mid`, output tetap rapi
4. Prompt panjang yang biasanya memicu format berulang:
   - expected: tidak ada duplikasi paragraf
5. Smoke test Assignment/Practice:
   - generate tetap berhasil, format draft tidak berubah

## Definition of Done

- Ticket cleanup output chat selesai di backend.
- Hasil manual test 5 skenario di atas pass.
- Tidak ada regresi pada assignment/practice flow.
