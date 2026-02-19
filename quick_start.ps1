# Скрипт для быстрого запуска RAG-системы (Windows PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  RAG-чат: Бенгальские кошки v2.5.0" -ForegroundColor Cyan
Write-Host "  Авторы: Line_GV, Koda, Алиса" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Проверка Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
}

if (-not $pythonCmd) {
    Write-Host "[ERROR] Python не найден. Пожалуйста, установите Python 3.8+" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Проверка зависимостей..." -ForegroundColor Cyan

# Проверка и установка зависимостей
try {
    python -c "import colorama" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] colorama не установлен. Установка..." -ForegroundColor Yellow
        pip install colorama
    }
} catch {
    Write-Host "[WARN] colorama не установлен. Установка..." -ForegroundColor Yellow
    pip install colorama
}

try {
    python -c "import openai" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] openai не установлен. Установка..." -ForegroundColor Yellow
        pip install openai
    }
} catch {
    Write-Host "[WARN] openai не установлен. Установка..." -ForegroundColor Yellow
    pip install openai
}

try {
    python -c "import faiss" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] faiss не установлен. Установка..." -ForegroundColor Yellow
        pip install faiss-cpu
    }
} catch {
    Write-Host "[WARN] faiss не установлен. Установка..." -ForegroundColor Yellow
    pip install faiss-cpu
}

Write-Host ""
Write-Host "[INFO] Запуск интерактивного интерфейса..." -ForegroundColor Cyan
Write-Host ""

# Установка кодировки UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Запуск
python rag_chat_interactive.py
