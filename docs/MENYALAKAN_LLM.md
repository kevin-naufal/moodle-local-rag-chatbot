# Menyalakan LLM Dengan Cara Paling Mudah

## Opsi A (paling praktis, disarankan)
### Nyalakan
1. Buka folder project: `c:\Users\Kevin\Downloads\my-llm`
2. Jalankan salah satu:
   - Double-click `scripts\llm\nyalakan_llm.bat`
   - Atau lewat terminal:
```powershell
cd c:\Users\Kevin\Downloads\my-llm
powershell -ExecutionPolicy Bypass -File .\scripts\llm\nyalakan_llm.ps1
```

Script ini otomatis:
- Membuka terminal baru untuk `OLLAMA_DEBUG=1 ollama serve`
- Restart proses Ollama lama jika perlu agar log HTTP/API tampil di terminal debug
- Membuat `.venv` jika belum ada
- Install dependency dari `requirements.txt`
- Download model jika belum ada:
  - `hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M`
  - `nomic-embed-text`
- Menyalakan chat model agar siap dipakai evaluasi
- Menyiapkan environment LLM untuk evaluasi/backend

## Mode Embedding BERT (opsional)
Runner Moodle sekarang mendukung backend embedding BERT.

Variabel yang bisa dipakai:
- `EMBED_BACKEND=bert` -> pakai BERT
- `EMBED_BACKEND=ollama` -> pakai `nomic-embed-text`
- `EMBED_BACKEND=auto` -> coba BERT dulu, fallback ke Ollama
- `BERT_MODEL=sentence-transformers/msmarco-bert-base-dot-v5` (default)

Contoh PowerShell:
```powershell
$env:EMBED_BACKEND='bert'
$env:BERT_MODEL='sentence-transformers/msmarco-bert-base-dot-v5'
```

### Matikan
1. Jalankan:
   - Double-click `scripts\llm\matikan_llm.bat`
   - Atau:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\llm\matikan_llm.ps1
```

## Opsi B (manual, kalau mau kontrol penuh)
```powershell
cd c:\Users\Kevin\Downloads\my-llm
ollama serve
ollama pull hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M
ollama pull nomic-embed-text
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Cek cepat LLM aktif
Di terminal baru:
```powershell
ollama ps
```
Kalau model muncul di daftar, berarti LLM sudah aktif.

Saat `scripts\run_demo_eval.bat` dijalankan, script juga akan:
- Menampilkan status model aktif dari `ollama ps`
- Membuka terminal Ollama debug server untuk melihat request HTTP/API ke model
- Membuka terminal monitoring trace Python untuk melihat komunikasi backend dengan LLM
- Mengirim trace ke `C:\xampp\moodledata\local_chatbot\logs\e2e_trace_python.jsonl`
