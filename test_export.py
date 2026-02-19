#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки функции экспорта истории
"""

from rag_chat_interactive import AnswerHistory
from datetime import datetime

def test_export():
    """Тестирование экспорта истории"""
    print("=" * 70)
    print("[TEST] Тестирование экспорта истории")
    print("=" * 70)

    # Создание истории
    history = AnswerHistory(reports_dir="reports")

    # Добавление тестовых данных
    print("\n[ADD] Добавление тестовых данных...")
    history.add(
        question="Тестовый вопрос 1",
        answer="Тестовый ответ 1",
        mode="basic",
        metadata={"chunks": 3, "time": 2.5}
    )

    history.add(
        question="Тестовый вопрос 2",
        answer="Тестовый ответ 2",
        mode="relevance",
        metadata={"threshold": 0.5, "chunks": 2, "avg_score": 0.68}
    )

    history.add(
        question="Тестовый вопрос 3",
        answer="Тестовый ответ 3",
        mode="basic",
        metadata={"chunks": 5, "time": 3.1}
    )

    print(f"[OK] Добавлено {len(history.get_all())} записей")

    # Тест экспорта в Markdown
    print("\n[EXPORT] Тест экспорта в Markdown...")
    try:
        md_filename = history.export_to_markdown()
        if md_filename:
            print(f"[OK] Успешно экспортировано в: {md_filename}")
        else:
            print("[ERROR] Ошибка при экспорте")
    except Exception as e:
        print(f"[ERROR] Исключение: {e}")

    # Тест экспорта в JSON
    print("\n[EXPORT] Тест экспорта в JSON...")
    try:
        json_filename = history.export_to_json()
        if json_filename:
            print(f"[OK] Успешно экспортировано в: {json_filename}")
        else:
            print("[ERROR] Ошибка при экспорте")
    except Exception as e:
        print(f"[ERROR] Исключение: {e}")

    # Проверка файлов
    print("\n[CHECK] Проверка созданных файлов...")
    import os

    if os.path.exists(md_filename):
        size = os.path.getsize(md_filename)
        print(f"[OK] Файл Markdown существует ({size} байт)")
    else:
        print("[ERROR] Файл Markdown не найден")

    if os.path.exists(json_filename):
        size = os.path.getsize(json_filename)
        print(f"[OK] Файл JSON существует ({size} байт)")
    else:
        print("[ERROR] Файл JSON не найден")

    print("\n" + "=" * 70)
    print("[SUCCESS] Тестирование завершено")
    print("=" * 70)


if __name__ == "__main__":
    test_export()
