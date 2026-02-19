#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простая RAG-система для ответов на вопросы о бенгальских кошках
с использованием существующей базы данных FAISS и прокси API.
С интеграцией оценки релевантности найденных чанков.
"""

import os
import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Tuple, Optional
from relevance_scorer_openai import RelevanceScorerOpenAI

# Загрузка переменных окружения
load_dotenv()

class RAGChatSystem:
    def __init__(self, rag_db_path: str = "бенгальские кошки RAG", use_relevance_scorer: bool = False):
        """
        Инициализация RAG-системы
        
        Args:
            rag_db_path: Путь к директории с базой данных RAG
            use_relevance_scorer: Использовать оценку релевантности чанков через sentence-transformers
        """
        # Используем абсолютный путь для решения проблем с кодировкой
        self.rag_db_path = os.path.abspath(rag_db_path)
        self.index_path = os.path.join(self.rag_db_path, "index.faiss")
        self.dataset_path = os.path.join(self.rag_db_path, "dataset.json")
        self.metadata_path = os.path.join(self.rag_db_path, "metadata.json")
        
        # Инициализация клиента OpenAI с прокси
        self.client = OpenAI(
            api_key=os.getenv("PROXI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        # Инициализация оценщика релевантности
        self.use_relevance_scorer = use_relevance_scorer
        self.relevance_scorer = None

        # Загрузка базы данных
        self.index = None
        self.dataset = None
        self.metadata = None
        
        self._load_rag_database()
    
    def _load_rag_database(self):
        """Загрузка FAISS индекса и датасета"""
        print("[LOAD] Загрузка базы данных RAG...")
        
        # Загрузка метаданных
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        print(f"   [OK] Модель эмбеддингов: {self.metadata['model_name']}")
        print(f"   [OK] Размерность: {self.metadata['embedding_dim']}")
        print(f"   [OK] Количество векторов: {self.metadata['total_vectors']}")
        
        # Загрузка индекса FAISS - обход проблемы с кириллицей в пути
        try:
            # Сначала пробуем обычный способ
            self.index = faiss.read_index(self.index_path)
        except Exception as e:
            # Если не получилось, пробуем через временный файл
            import shutil
            import tempfile
            print(f"   [WARN] Проблема с чтением FAISS индекса напрямую, используем обходной путь...")
            temp_dir = tempfile.mkdtemp()
            temp_index_path = os.path.join(temp_dir, "temp_index.faiss")
            try:
                shutil.copy2(self.index_path, temp_index_path)
                self.index = faiss.read_index(temp_index_path)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"   [OK] Индекс FAISS загружен: {self.index.ntotal} векторов")

        # Загрузка датасета
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        print(f"   [OK] Датасет загружен: {len(self.dataset)} чанков")

        # Инициализация оценщика релевантности (если включено)
        if self.use_relevance_scorer:
            print("\n[RELEVANCE] Инициализация оценщика релевантности...")
            try:
                self.relevance_scorer = RelevanceScorerOpenAI()
                print("   [OK] Оценщик релевантности готов\n")
            except Exception as e:
                print(f"   [WARN] Ошибка при инициализации оценщика: {e}")
                print(f"   [INFO] Продолжаем без оценки релевантности\n")
                self.use_relevance_scorer = False
        else:
            print()
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Создание эмбеддинга для текста запроса
        
        Args:
            text: Текст запроса
            
        Returns:
            Вектор эмбеддинга
        """
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array([response.data[0].embedding], dtype=np.float32)
    
    def _search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, dict]]:
        """
        Поиск релевантных чанков по запросу
        
        Args:
            query: Текст запроса
            top_k: Количество возвращаемых чанков
            
        Returns:
            Список кортежей (текст_чанка, метаданные)
        """
        # Создание эмбеддинга запроса
        query_embedding = self._create_embedding(query)
        
        # Поиск ближайших векторов
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Сбор результатов
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.dataset[idx]
            results.append((chunk["text"], chunk["metadata"], distance))
        
        return results
    
    def _search_relevant_chunks_with_score(
        self,
        query: str,
        threshold: float = 0.5,
        top_k: int = 5
    ) -> List[Tuple[str, dict, float]]:
        """
        Поиск релевантных чанков с оценкой релевантности (score).
        Использует sentence-transformers для вычисления косинусного сходства.

        Args:
            query: Текст запроса
            threshold: Порог релевантности (чанки с score < threshold отбрасываются)
            top_k: Максимальное количество возвращаемых чанков

        Returns:
            Список кортежей (текст_чанка, метаданные, score), отсортированный по убыванию score
        """
        if not self.relevance_scorer:
            # Если оценщик не инициализирован, используем стандартный поиск
            print("[WARN] Оценщик релевантности не доступен, используем стандартный поиск")
            standard_results = self._search_relevant_chunks(query, top_k)
            # Преобразуем формат и нормализуем distance в score (0-1)
            results = []
            for text, metadata, distance in standard_results:
                # Преобразуем евклидово расстояние в подобный score (меньше расстояние = выше score)
                # Простое преобразование: score = 1 / (1 + distance)
                score = 1.0 / (1.0 + float(distance))
                results.append((text, metadata, score))
            return results

        # Подготовка текстов чанков
        chunk_texts = [chunk["text"] for chunk in self.dataset]

        # Поиск с оценкой релевантности
        scored_results = self.relevance_scorer.search_relevant_docs(
            query=query,
            docs=chunk_texts,
            threshold=threshold,
            k=top_k
        )

        # Добавляем метаданные к результатам
        results = []
        for doc_text, score in scored_results:
            # Находим соответствующий чанк с метаданными
            for chunk in self.dataset:
                if chunk["text"] == doc_text:
                    results.append((chunk["text"], chunk["metadata"], score))
                    break

        return results
    
    def _build_prompt(self, question: str, context: List[str]) -> str:
        """
        Формирование промпта для LLM с контекстом
        
        Args:
            question: Вопрос пользователя
            context: Список релевантных чанков
            
        Returns:
            Сформированный промпт
        """
        context_text = "\n\n".join([f"--- Чанк {i+1} ---\n{chunk}" 
                                   for i, chunk in enumerate(context)])
        
        prompt = f"""Ты — эксперт по бенгальским кошкам. Отвечай на вопросы пользователя, используя только предоставленную информацию из базы знаний.

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context_text}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}

ИНСТРУКЦИИ:
- Отвечай только на основе предоставленного контекста
- Если в контексте нет информации для ответа, честно скажи об этом
- Отвечай подробно и информативно
- Используй естественный, дружелюбный тон
- Структурируй ответ при необходимости

ОТВЕТ:"""
        return prompt
    
    def ask(self, question: str, top_k: int = 5) -> str:
        """
        Получение ответа на вопрос с использованием RAG
        
        Args:
            question: Вопрос пользователя
            top_k: Количество релевантных чанков для контекста
            
        Returns:
            Ответ LLM
        """
        if not question.strip():
            return "[ERROR] Пожалуйста, введите вопрос."
        
        print(f"\n[SEARCH] Поиск релевантной информации для вопроса: «{question}»")
        
        # Поиск релевантных чанков
        relevant_chunks = self._search_relevant_chunks(question, top_k)
        
        print(f"   [OK] Найдено {len(relevant_chunks)} релевантных чанков")
        
        # Подготовка контекста
        context = [chunk[0] for chunk in relevant_chunks]
        
        # Формирование промпта
        prompt = self._build_prompt(question, context)
        
        # Запрос к LLM
        print("   [LLM] Генерация ответа...")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Оптимальная модель для RAG по цене/качеству
                messages=[
                    {
                        "role": "system",
                        "content": "Ты — полезный ассистент, эксперт по бенгальским кошкам."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
            answer = response.choices[0].message.content
            # Очистка от потенциально проблемных символов для Windows консоли
            return answer.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except Exception as e:
            return f"[ERROR] Ошибка при генерации ответа: {e}"

    def ask_with_relevance(
        self,
        question: str,
        threshold: float = 0.5,
        top_k: int = 5,
        show_scores: bool = True
    ) -> str:
        """
        Получение ответа на вопрос с использованием RAG и оценкой релевантности чанков.

        Args:
            question: Вопрос пользователя
            threshold: Порог релевантности для фильтрации чанков
            top_k: Максимальное количество чанков для контекста
            show_scores: Показывать оценки релевантности в выводе

        Returns:
            Ответ LLM
        """
        if not question.strip():
            return "[ERROR] Пожалуйста, введите вопрос."

        print(f"\n[SEARCH] Поиск релевантной информации для вопроса: «{question}»")

        # Поиск релевантных чанков с оценкой
        relevant_chunks = self._search_relevant_chunks_with_score(
            query=question,
            threshold=threshold,
            top_k=top_k
        )

        if not relevant_chunks:
            print(f"   [WARN] Не найдено чанков с score >= {threshold}")
            return "[INFO] Не удалось найти достаточно релевантную информацию в базе знаний. Попробуйте переформулировать вопрос."

        print(f"   [OK] Найдено {len(relevant_chunks)} релевантных чанков (threshold >= {threshold})")

        # Вывод оценок релевантности
        if show_scores:
            print("\n   [SCORES] Оценки релевантности:")
            for i, (text, metadata, score) in enumerate(relevant_chunks, 1):
                preview = text[:80] + "..." if len(text) > 80 else text
                print(f"      {i}. Score: {score:.4f} | {preview}")

        # Подготовка контекста
        context = [chunk[0] for chunk in relevant_chunks]

        # Формирование промпта
        prompt = self._build_prompt(question, context)

        # Запрос к LLM
        print("\n   [LLM] Генерация ответа...")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Ты — полезный ассистент, эксперт по бенгальским кошкам."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
            answer = response.choices[0].message.content
            return answer.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except Exception as e:
            return f"[ERROR] Ошибка при генерации ответа: {e}"


def main():
    """Главная функция для интерактивного режима"""
    print("=" * 60)
    print("[RAG-CHAT] Бенгальские кошки")
    print("=" * 60)
    
    # Инициализация системы
    try:
        rag_system = RAGChatSystem()
    except Exception as e:
        print(f"[ERROR] Ошибка при инициализации: {e}")
        return
    
    print("\n[INFO] Введите ваш вопрос о бенгальских кошках")
    print("   (введите 'exit' или 'quit' для выхода)\n")
    
    while True:
        try:
            # Получение вопроса от пользователя
            question = input("[QUESTION] Ваш вопрос: ").strip()
            
            # Проверка на выход
            if question.lower() in ['exit', 'quit', 'выход', 'q']:
                print("\n[BYE] До свидания!")
                break
            
            if not question:
                continue
            
            # Получение ответа
            answer = rag_system.ask(question, top_k=5)
            
            # Вывод ответа
            print("\n" + "=" * 60)
            print("[ANSWER]")
            print("=" * 60)
            print(answer)
            print("=" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n[BYE] До свидания!")
            break
        except Exception as e:
            print(f"\n[ERROR] Ошибка: {e}\n")


if __name__ == "__main__":
    main()
