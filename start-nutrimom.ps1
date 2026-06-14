$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Get-Command python -ErrorAction Stop).Source

$listener = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    Write-Host "NutriMom sudah berjalan di http://127.0.0.1:5000"
    Start-Process "http://127.0.0.1:5000"
    exit 0
}

Set-Location $project
Write-Host "Menjalankan NutriMom di http://127.0.0.1:5000"
Write-Host "Biarkan terminal ini terbuka. Tekan Ctrl+C untuk menghentikan server."
Start-Process "http://127.0.0.1:5000"
& $python "app.py"
