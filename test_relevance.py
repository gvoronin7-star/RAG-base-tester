#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки работы RAG-системы с оценкой релевантности.
"""

from rag_chat import RAGChatSystem

def test_rag_with_relevance():
    """Тестирование RAG-системы с оценкой релевантности чанков"""
    print("=" * 70)
    print("[TEST] Тестирование RAG с оценкой релевантности")
    print("=" * 70)

    # Инициализация системы с оценкой релевантности
    print("\n[INIT] Инициализация RAG-системы с оценкой релевантности...")
    try:
        rag_system = RAGChatSystem(use_relevance_scorer=True)
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации: {e}")
        return

    # Тестовые вопросы
    test_questions = [
        ("Кто основатель породы бенгальских кошек?", 0.4),
        ("Какие окрасы бывают у бенгалов?", 0.5),
        ("Какой характер у бенгальских кошек?", 0.5),
        ("Что такое квантовая физика?", 0.6)  # Нерелевантный вопрос
    ]

    for i, (question, threshold) in enumerate(test_questions, 1):
        print(f"\n{'=' * 70}")
        print(f"Тест {i}/{len(test_questions)}")
        print(f"Вопрос: {question}")
        print(f"Порог релевантности: {threshold}")
        print(f"{'=' * 70}")

        try:
            answer = rag_system.ask_with_relevance(
                question=question,
                threshold=threshold,
                top_k=3,
                show_scores=True
            )
            print(f"\n{answer}")
        except Exception as e:
            print(f"[ERROR] Ошибка: {e}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Тестирование завершено")
    print("=" * 70)


if __name__ == "__main__":
    test_rag_with_relevance()
