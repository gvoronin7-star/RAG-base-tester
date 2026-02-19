#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для оценки релевантности документов на основе косинусного сходства.
Использует sentence-transformers для эмбеддингов и FAISS для поиска.
"""

import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss


class RelevanceScorer:
    """
    Класс для оценки релевантности документов с использованием косинусного сходства.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Инициализация модели эмбеддингов.
        
        Args:
            model_name: Название модели sentence-transformers.
                       Используем многоязычную модель для русского языка.
        """
        print(f"[LOAD] Загрузка модели эмбеддингов: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[OK] Модель загружена. Размерность: {self.embedding_dim}")
    
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
        Создание эмбеддингов для списка текстов.
        
        Args:
            texts: Список текстов
            
        Returns:
            Матрица эмбеддингов (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)
    
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
        doc_embeddings = self.create_embeddings(docs)
        
        # Нормализация эмбеддингов для корректного косинусного сходства
        faiss.normalize_L2(doc_embeddings)
        
        # Создание индекса (Inner Product = косинусное сходство для нормализованных векторов)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(doc_embeddings)
        
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
        query_embedding = self.create_embeddings([query])
        
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
    Демонстрация использования RelevanceScorer с тестовыми данными.
    """
    print("=" * 70)
    print("[DEMO] Демонстрация оценки релевантности")
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
    scorer = RelevanceScorer()
    
    # Создание индекса
    print("\n[STEP 1] Создание FAISS индекса...")
    faiss_index = scorer.build_faiss_index(docs)
    print(f"[OK] Индекс создан: {faiss_index.ntotal} векторов")
    
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
