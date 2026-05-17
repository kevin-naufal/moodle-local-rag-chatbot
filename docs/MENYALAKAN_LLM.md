# Menyalakan LLM Dengan Cara Paling Mudah

## Opsi A (paling praktis, disarankan)
### Nyalakan
1. Buka folder project: `c:\Users\Kevin\Downloads\my-llm`
2. Jalankan salah satu:
   - Double-click `scripts\nyalakan_llm.bat`
   - Atau lewat terminal:
```powershell
cd c:\Users\Kevin\Downloads\my-llm
powershell -ExecutionPolicy Bypass -File .\scripts\nyalakan_llm.ps1
```

Script ini otomatis:
- Menyalakan `ollama serve` jika belum aktif
- Membuat `.venv` jika belum ada
- Install dependency dari `requirements.txt`
- Download model jika belum ada:
  - `hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M`
  - `nomic-embed-text`
- Menjalankan Streamlit di `http://127.0.0.1:8501`

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
1. Di terminal Streamlit, tekan `Ctrl + C`
2. Jalankan:
   - Double-click `scripts\matikan_llm.bat`
   - Atau:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\matikan_llm.ps1
```

## Opsi B (manual, kalau mau kontrol penuh)
```powershell
cd c:\Users\Kevin\Downloads\my-llm
ollama serve
ollama pull hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M
ollama pull nomic-embed-text
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app/chatbot_ui.py --server.headless true --server.port 8501
```

## Cek cepat LLM aktif
Di terminal baru:
```powershell
ollama ps
```
Kalau model muncul di daftar, berarti LLM sudah aktif.
