#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки работы RAG-системы
"""

from rag_chat import RAGChatSystem

def test_rag_system():
    """Тестирование RAG-системы с примерами вопросов"""
    print("=" * 60)
    print("[TEST] Тестирование RAG-системы")
    print("=" * 60)

    # Инициализация
    try:
        rag_system = RAGChatSystem()
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации: {e}")
        return
    
    # Тестовые вопросы
    test_questions = [
        "Кто основатель породы бенгальских кошек?",
        "Какие окрасы бывают у бенгалов?",
        "Какой характер у бенгальских кошек?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 60}")
        print(f"Тест {i}/{len(test_questions)}")
        print(f"Вопрос: {question}")
        print(f"{'=' * 60}")
        
        try:
            answer = rag_system.ask(question, top_k=3)
            print(f"\n{answer}")
        except Exception as e:
            print(f"[ERROR] Ошибка: {e}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Тестирование завершено")
    print("=" * 60)

if __name__ == "__main__":
    test_rag_system()
