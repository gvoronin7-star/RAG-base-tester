#!/bin/bash
# Скрипт для быстрого запуска RAG-системы

echo "=========================================="
echo "  RAG-чат: Бенгальские кошки v2.5.0"
echo "  Авторы: Line_GV, Koda, Алиса"
echo "=========================================="
echo ""

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 не найден. Пожалуйста, установите Python 3.8+"
    exit 1
fi

echo "[INFO] Проверка зависимостей..."

# Проверка и установка зависимостей
if ! python3 -c "import colorama" 2>/dev/null; then
    echo "[WARN] colorama не установлен. Установка..."
    pip3 install colorama
fi

if ! python3 -c "import openai" 2>/dev/null; then
    echo "[WARN] openai не установлен. Установка..."
    pip3 install openai
fi

if ! python3 -c "import faiss" 2>/dev/null; then
    echo "[WARN] faiss не установлен. Установка..."
    pip3 install faiss-cpu
fi

echo ""
echo "[INFO] Запуск интерактивного интерфейса..."
echo ""

# Запуск
python3 rag_chat_interactive.py
