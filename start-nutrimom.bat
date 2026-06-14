@echo off
cd /d "%~dp0"
echo Menjalankan NutriMom di http://127.0.0.1:5000
echo Biarkan jendela ini terbuka. Tekan Ctrl+C untuk menghentikan server.
start "" http://127.0.0.1:5000
python app.py
echo.
echo Server berhenti atau gagal dijalankan.
pause
