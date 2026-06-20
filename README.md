# NutriMom Web

Aplikasi Flask untuk analisis risiko maternal dan rekomendasi makanan menggunakan
XGBoost, fuzzy logic, dan SAW.

## Struktur project

```text
app.py                 Entry point Flask dan API prediksi
index.html             Halaman landing
predict.html           Form input data maternal
result.html            Halaman hasil rekomendasi
static/                CSS, JavaScript, dan asset gambar
model/                 File model, imputer, konfigurasi, dan knowledge base makanan
render.yaml            Blueprint deploy Render
Procfile               Start command alternatif untuk web service
requirements.txt       Dependency Python
```

## Menjalankan aplikasi

```powershell
python -m pip install -r requirements.txt
python app.py
```

Buka aplikasi di:

```text
http://127.0.0.1:5000
```

Cara paling mudah di Windows, klik dua kali:

```text
start-nutrimom.bat
```

Alternatif melalui PowerShell:

```powershell
.\start-nutrimom.ps1
```

Biarkan jendela terminal tetap terbuka selama aplikasi digunakan. Jika terminal
ditutup, browser akan menampilkan `ERR_CONNECTION_REFUSED`.

## Deploy ke Render

Project ini sudah disiapkan sebagai Python Web Service Render.

1. Push seluruh project ke repository GitHub, termasuk folder `model/`.
2. Di Render, pilih **New > Web Service** lalu hubungkan repository.
3. Jika Render membaca `render.yaml`, gunakan konfigurasi Blueprint yang tersedia.
4. Jika setup manual, gunakan:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT app:app
Health Check Path: /healthz
```

Render akan membuat `SECRET_KEY` dari `render.yaml`. Jika setup manual, tambahkan
environment variable `SECRET_KEY` di dashboard Render.
