#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для оценки релевантности документов на основе косинусного сходства.
Использует OpenAI API для эмбеддингов (более стабильная альтернатива PyTorch).
"""

import numpy as np
from typing import List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import os

# Загрузка переменных окружения
load_dotenv()


class RelevanceScorerOpenAI:
    """
    Класс для оценки релевантности документов с использованием OpenAI API.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Инициализация модели эмбеддингов OpenAI.

        Args:
            model_name: Название модели OpenAI для эмбеддингов.
        """
        print(f"[LOAD] Инициализация OpenAI API для эмбеддингов: {model_name}")

        # Инициализация клиента OpenAI с прокси
        self.client = OpenAI(
            api_key=os.getenv("PROXI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )

        self.model_name = model_name
        # Размерность для text-embedding-3-small = 1536
        self.embedding_dim = 1536 if model_name == "text-embedding-3-small" else 1536

        print(f"[OK] OpenAI API клиент инициализирован")

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Вычисление косинусного сходства между двумя векторами.

        Args:
            vec1: Первый вектор
            vec2: Второй вектор

        Returns:
            Значение косинусного сходства в диапазоне [-1, 1]
        """
        # Нормализация векторов
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)

        # Косинусное сходство = скалярное произведение нормализованных векторов
        return float(np.dot(vec1_norm.flatten(), vec2_norm.flatten()))

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Создание эмбеддингов для списка текстов через OpenAI API.

        Args:
            texts: Список текстов

        Returns:
            Матрица эмбеддингов (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )

            # Извлечение эмбеддингов из ответа
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Ошибка при создании эмбеддингов: {e}")
            return np.array([])

    def build_faiss_index(self, docs: List[str]) -> faiss.IndexFlatIP:
        """
        Построение FAISS индекса для документов.
        Используем IndexFlatIP для скалярного произведения (эквивалентно косинусному сходству для нормализованных векторов).

        Args:
            docs: Список документов

        Returns:
            FAISS индекс
        """
        if not docs:
            raise ValueError("Список документов пуст")

        # Создание эмбеддингов для документов
        print(f"[EMBED] Создание эмбеддингов для {len(docs)} документов...")
        doc_embeddings = self.create_embeddings(docs)

        if doc_embeddings.size == 0:
            raise ValueError("Не удалось создать эмбеддинги")

        # Нормализация эмбеддингов для корректного косинусного сходства
        faiss.normalize_L2(doc_embeddings)

        # Создание индекса (Inner Product = косинусное сходство для нормализованных векторов)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(doc_embeddings)

        print(f"[OK] Индекс создан: {index.ntotal} векторов")

        return index

    def search_relevant_docs(
        self,
        query: str,
        docs: List[str],
        faiss_index: Optional[faiss.Index] = None,
        threshold: float = 0.5,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Поиск релевантных документов с оценкой релевантности.

        Args:
            query: Текстовый запрос пользователя
            docs: Список документов для поиска
            faiss_index: Предварительно созданный FAISS индекс (если None, создается заново)
            threshold: Порог релевантности (документы с score < threshold отбрасываются)
            k: Максимальное количество возвращаемых документов

        Returns:
            Список кортежей (документ, score), отсортированный по убыванию score
        """
        # Обработка крайних случаев
        if not query or not query.strip():
            print("[WARN] Пустой запрос")
            return []

        if not docs:
            print("[WARN] База документов пуста")
            return []

        query = query.strip()

        # Создание или использование существующего индекса
        if faiss_index is None:
            try:
                faiss_index = self.build_faiss_index(docs)
            except Exception as e:
                print(f"[ERROR] Ошибка при создании индекса: {e}")
                return []

        # Создание эмбеддинга для запроса
        print(f"[EMBED] Создание эмбеддинга для запроса...")
        query_embedding = self.create_embeddings([query])

        if query_embedding.size == 0:
            print("[ERROR] Не удалось создать эмбеддинг для запроса")
            return []

        # Нормализация эмбеддинга запроса
        faiss.normalize_L2(query_embedding)

        # Поиск ближайших векторов
        # k не может быть больше количества документов
        k = min(k, len(docs))
        scores, indices = faiss_index.search(query_embedding, k)

        # Подготовка результатов
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(docs):  # Проверка валидности индекса
                doc_score = float(score)
                # Фильтрация по порогу
                if doc_score >= threshold:
                    results.append((docs[idx], doc_score))

        # Сортировка по убыванию score (FAISS уже возвращает отсортированные результаты,
        # но для надежности сортируем явно)
        results.sort(key=lambda x: x[1], reverse=True)

        return results


def demo_usage():
    """
    Демонстрация использования RelevanceScorerOpenAI с тестовыми данными.
    """
    print("=" * 70)
    print("[DEMO] Демонстрация оценки релевантности (OpenAI API)")
    print("=" * 70)

    # Входные данные
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
    scorer = RelevanceScorerOpenAI()

    # Создание индекса
    print("\n[STEP 1] Создание FAISS индекса...")
    faiss_index = scorer.build_faiss_index(docs)

    # Поиск релевантных документов
    print("\n[STEP 2] Поиск релевантных документов...")
    results = scorer.search_relevant_docs(
        query=query,
        docs=docs,
        faiss_index=faiss_index,
        threshold=threshold,
        k=k
    )

    # Вывод результатов
    print("\n[STEP 3] Результаты:")
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

    # Дополнительный тест: запрос с высоким порогом
    print("\n[EXTRA TEST] Тест с высоким порогом (threshold = 0.9)...")
    results_strict = scorer.search_relevant_docs(
        query=query,
        docs=docs,
        faiss_index=faiss_index,
        threshold=0.9,
        k=k
    )

    if results_strict:
        print(f"Найдено {len(results_strict)} документов с score >= 0.9")
        for doc, score in results_strict:
            print(f"  - {score:.4f}: {doc}")
    else:
        print("Нет документов с score >= 0.9")

    print("\n" + "=" * 70)
    print("[SUCCESS] Демонстрация завершена")
    print("=" * 70)


if __name__ == "__main__":
    demo_usage()
