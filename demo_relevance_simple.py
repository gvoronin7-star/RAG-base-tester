#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстрационный скрипт для оценки релевантности с простыми тестовыми данными.
Пример из задания: документы об ИИ и Python.
"""

from relevance_scorer import RelevanceScorer


def main():
    """
    Демонстрация использования RelevanceScorer с тестовыми данными из задания.
    """
    print("=" * 70)
    print("[DEMO] Демонстрация оценки релевантности (простой пример)")
    print("=" * 70)

    # Входные данные из задания
    docs = [
        "Искусственный интеллект меняет мир технологий.",
        "Машинное обучение — подраздел ИИ, связанный с обучением моделей.",
        "Python — популярный язык для разработки ИИ-приложений."
    ]

    query = "Как Python связан с ИИ?"
    threshold = 0.5
    k = 2

    print(f"\nДокументы в базе ({len(docs)} шт.):")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc}")

    print(f"\nЗапрос: «{query}»")
    print(f"Порог релевантности: {threshold}")
    print(f"Макс. количество документов: {k}")
    print("\n" + "-" * 70)

    # Инициализация оценщика релевантности
    print("\n[STEP 1] Инициализация модели эмбеддингов...")
    scorer = RelevanceScorer(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Создание FAISS индекса
    print("\n[STEP 2] Создание FAISS индекса...")
    faiss_index = scorer.build_faiss_index(docs)
    print(f"[OK] Индекс создан: {faiss_index.ntotal} векторов")

    # Поиск релевантных документов
    print("\n[STEP 3] Поиск релевантных документов с оценкой...")
    results = scorer.search_relevant_docs(
        query=query,
        docs=docs,
        faiss_index=faiss_index,
        threshold=threshold,
        k=k
    )

    # Вывод результатов
    print("\n[STEP 4] Результаты:")
    print("-" * 70)

    if results:
        print(f"Найдено {len(results)} релевантных документов:\n")
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. Score: {score:.4f}")
            print(f"   Текст: {doc}\n")
    else:
        print("Не найдено документов, удовлетворяющих порогу релевантности.\n")

    # Ожидаемый формат вывода
    print("-" * 70)
    print("[RESULT] Формат вывода (список кортежей):")
    print("-" * 70)
    print(results)
    print("-" * 70)

    # Дополнительные тесты
    print("\n" + "=" * 70)
    print("[EXTRA TESTS] Дополнительные тесты")
    print("=" * 70)

    # Тест 1: Высокий порог
    print("\n[TEST 1] Высокий порог релевантности (threshold = 0.9)...")
    results_strict = scorer.search_relevant_docs(
        query=query,
        docs=docs,
        faiss_index=faiss_index,
        threshold=0.9,
        k=k
    )

    if results_strict:
        print(f"Найдено {len(results_strict)} документов с score >= 0.9:")
        for doc, score in results_strict:
            print(f"  - {score:.4f}: {doc}")
    else:
        print("Нет документов с score >= 0.9")

    # Тест 2: Другой запрос
    print("\n[TEST 2] Другой запрос: «Что такое машинное обучение?»...")
    query2 = "Что такое машинное обучение?"
    results2 = scorer.search_relevant_docs(
        query=query2,
        docs=docs,
        faiss_index=faiss_index,
        threshold=0.3,
        k=3
    )

    if results2:
        print(f"Найдено {len(results2)} релевантных документов:")
        for doc, score in results2:
            print(f"  - {score:.4f}: {doc}")
    else:
        print("Нет релевантных документов")

    # Тест 3: Нерелевантный запрос
    print("\n[TEST 3] Нерелевантный запрос: «Как готовить пасту?»...")
    query3 = "Как готовить пасту?"
    results3 = scorer.search_relevant_docs(
        query=query3,
        docs=docs,
        faiss_index=faiss_index,
        threshold=0.4,
        k=3
    )

    if results3:
        print(f"Найдено {len(results3)} документов:")
        for doc, score in results3:
            print(f"  - {score:.4f}: {doc}")
    else:
        print("Нет документов, удовлетворяющих порогу релевантности (как и ожидалось)")

    # Тест 4: Пустой запрос
    print("\n[TEST 4] Пустой запрос (крайний случай)...")
    results4 = scorer.search_relevant_docs(
        query="",
        docs=docs,
        faiss_index=faiss_index,
        threshold=0.3,
        k=3
    )
    print(f"Результат: {results4} (пустой список)")

    # Тест 5: Пустая база документов
    print("\n[TEST 5] Пустая база документов (крайний случай)...")
    try:
        results5 = scorer.search_relevant_docs(
            query=query,
            docs=[],
            faiss_index=None,
            threshold=0.3,
            k=3
        )
        print(f"Результат: {results5} (пустой список)")
    except Exception as e:
        print(f"Ожидаемая ошибка: {e}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Все тесты завершены")
    print("=" * 70)


if __name__ == "__main__":
    main()
