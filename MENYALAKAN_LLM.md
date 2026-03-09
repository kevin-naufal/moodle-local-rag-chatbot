# Menyalakan LLM (Step by Step)

## 1. Masuk ke folder project
```powershell
cd c:\Users\Kevin\Downloads\my-llm
```

## 2. Cek Ollama sudah terpasang
```powershell
ollama --version
```

## 3. Pastikan model yang dipakai project tersedia
```powershell
ollama pull hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M
ollama pull nomic-embed-text
```

## 4. Aktifkan virtual environment Python
```powershell
.venv\Scripts\activate
```

## 5. Jalankan UI chatbot (Streamlit)
```powershell
python -m streamlit run chatbot_ui.py --server.headless true --server.port 8501
```

## 6. Buka aplikasi di browser
```text
http://127.0.0.1:8501
```

## 7. Verifikasi LLM sudah aktif
Di terminal baru:
```powershell
ollama ps
```
Jika ada model `hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M`, berarti LLM aktif.

## 8. Cara mematikan
1. Di terminal Streamlit, tekan `Ctrl + C`.
2. Hentikan model yang sedang loaded:
```powershell
ollama stop hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M
```
