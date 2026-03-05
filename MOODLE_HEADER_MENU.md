# Moodle Integrated LLM Chat (Local XAMPP)

Halaman chatbot sekarang terintegrasi di Moodle plugin:

- URL: `http://localhost/moodle/local/chatbot/index.php`

## 1) Jalankan upgrade plugin
1. Login Moodle sebagai admin.
2. Buka `Site administration` -> `Notifications`.
3. Lanjutkan upgrade sampai selesai (karena versi plugin naik).

## 2) Atur path backend Python
1. Buka `Site administration` -> `Plugins` -> `Local plugins` -> `LLM Chat`.
2. Pastikan:
   - `Project path`: `C:\Users\Kevin\Downloads\my-llm`
   - `Python executable path`: `C:\Users\Kevin\Downloads\my-llm\.venv\Scripts\python.exe`
   - `Runner file name`: `moodle_rag_runner.py`

## 3) Tampilkan di header Moodle
Masuk ke:
`Site administration` -> `Appearance` -> `Themes` -> `Theme settings`

Pada `Custom menu items`, tambahkan:

```text
LLM Chat|http://localhost/moodle/local/chatbot/index.php|Buka LLM Chat terintegrasi Moodle
```

## 4) Jalankan service lokal yang dibutuhkan
- Pastikan `Ollama` running.
- Pastikan model sudah ada:

```powershell
ollama pull nomic-embed-text
ollama pull hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M
```

Setelah itu klik menu header `LLM Chat`, dan UI upload+chat muncul di halaman Moodle.
