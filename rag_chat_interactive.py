#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная RAG-система с интерактивным интерфейсом.
Фаза 1 улучшений: меню, цветной вывод, история, настройки.
"""

import os
import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict
from relevance_scorer_openai import RelevanceScorerOpenAI
from datetime import datetime
from colorama import Fore, Style, Back, init

# Инициализация colorama
init(autoreset=True)

# Загрузка переменных окружения
load_dotenv()


class AnswerHistory:
    """Класс для управления историей вопросов и ответов"""

    def __init__(self, reports_dir: str = "reports"):
        self.history: List[Dict] = []
        self.reports_dir = reports_dir
        self._ensure_reports_dir()

    def _ensure_reports_dir(self):
        """Создание папки для отчетов, если она не существует"""
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
            print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Создана папка для отчетов: {self.reports_dir}")

    def add(self, question: str, answer: str, mode: str, metadata: Dict = None):
        """Добавление ответа в историю"""
        self.history.append({
            "timestamp": datetime.now(),
            "question": question,
            "answer": answer,
            "mode": mode,
            "metadata": metadata or {}
        })

    def get_all(self) -> List[Dict]:
        """Получение всей истории"""
        return self.history

    def search(self, keyword: str) -> List[Dict]:
        """Поиск по истории"""
        keyword_lower = keyword.lower()
        return [item for item in self.history
                if keyword_lower in item['question'].lower()]

    def clear(self):
        """Очистка истории"""
        self.history.clear()

    def export_to_markdown(self, filename: str = None):
        """Экспорт истории в Markdown"""
        if not self.history:
            return None

        if filename is None:
            filename = f"rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        # Полный путь к файлу в папке reports
        filepath = os.path.join(self.reports_dir, filename)

        content = f"""# История вопросов RAG-системы

**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Всего вопросов:** {len(self.history)}

---

"""

        for i, item in enumerate(self.history, 1):
            timestamp = item['timestamp'].strftime('%H:%M:%S')
            mode = item['mode']
            metadata = item.get('metadata', {})

            content += f"""## Вопрос {i}

**Время:** {timestamp}
**Режим:** {mode}
"""

            if 'threshold' in metadata:
                content += f"**Порог:** {metadata['threshold']}\n"
            if 'chunks' in metadata:
                content += f"**Чанков:** {metadata['chunks']}\n"
            if 'avg_score' in metadata:
                content += f"**Средний score:** {metadata['avg_score']:.3f}\n"

            content += f"""
### Вопрос
{item['question']}

### Ответ
{item['answer']}
"""

            if 'sources' in metadata:
                content += "\n### Использованные источники\n"
                for source in metadata['sources']:
                    content += f"- Score: {source['score']:.4f} | \"{source['text'][:60]}...\"\n"

            content += "\n---\n\n"

        content += """
*Отчет сгенерирован автоматически RAG-чат v2.5.0*
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return filepath

    def export_to_json(self, filename: str = None):
        """Экспорт истории в JSON"""
        if not self.history:
            return None

        if filename is None:
            filename = f"rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Полный путь к файлу в папке reports
        filepath = os.path.join(self.reports_dir, filename)

        data = {
            "export_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_questions": len(self.history),
            "history": []
        }

        for item in self.history:
            data["history"].append({
                "timestamp": item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                "question": item['question'],
                "answer": item['answer'],
                "mode": item['mode'],
                "metadata": item.get('metadata', {})
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return filepath


class RAGChatSystem:
    """Улучшенная RAG-система с историей и настройками"""

    def __init__(self, rag_db_path: str = "бенгальские кошки RAG", use_relevance_scorer: bool = True):
        self.rag_db_path = os.path.abspath(rag_db_path)
        self.index_path = os.path.join(self.rag_db_path, "index.faiss")
        self.dataset_path = os.path.join(self.rag_db_path, "dataset.json")
        self.metadata_path = os.path.join(self.rag_db_path, "metadata.json")

        self.client = OpenAI(
            api_key=os.getenv("PROXI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )

        self.use_relevance_scorer = use_relevance_scorer
        self.relevance_scorer = None

        self.index = None
        self.dataset = None
        self.metadata = None

        self.history = AnswerHistory()

        self._load_rag_database()

    def _load_rag_database(self):
        """Загрузка FAISS индекса и датасета"""
        print(f"{Fore.CYAN}[LOAD] Загрузка базы данных RAG...{Style.RESET_ALL}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"   {Fore.GREEN}[OK]{Style.RESET_ALL} Модель эмбеддингов: {self.metadata['model_name']}")
        print(f"   {Fore.GREEN}[OK]{Style.RESET_ALL} Размерность: {self.metadata['embedding_dim']}")
        print(f"   {Fore.GREEN}[OK]{Style.RESET_ALL} Количество векторов: {self.metadata['total_vectors']}")

        try:
            self.index = faiss.read_index(self.index_path)
        except Exception as e:
            import shutil
            import tempfile
            print(f"   {Fore.YELLOW}[WARN]{Style.RESET_ALL} Используем обходной путь для загрузки индекса...")
            temp_dir = tempfile.mkdtemp()
            temp_index_path = os.path.join(temp_dir, "temp_index.faiss")
            try:
                shutil.copy2(self.index_path, temp_index_path)
                self.index = faiss.read_index(temp_index_path)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"   {Fore.GREEN}[OK]{Style.RESET_ALL} Индекс FAISS загружен: {self.index.ntotal} векторов")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        print(f"   {Fore.GREEN}[OK]{Style.RESET_ALL} Датасет загружен: {len(self.dataset)} чанков")

        if self.use_relevance_scorer:
            print(f"\n{Fore.CYAN}[RELEVANCE]{Style.RESET_ALL} Инициализация оценщика релевантности...")
            try:
                self.relevance_scorer = RelevanceScorerOpenAI()
                print(f"   {Fore.GREEN}[OK]{Style.RESET_ALL} Оценщик релевантности готов\n")
            except Exception as e:
                print(f"   {Fore.YELLOW}[WARN]{Style.RESET_ALL} Ошибка при инициализации оценщика: {e}")
                print(f"   {Fore.YELLOW}[INFO]{Style.RESET_ALL} Продолжаем без оценки релевантности\n")
                self.use_relevance_scorer = False
        else:
            print()

    def _create_embedding(self, text: str) -> np.ndarray:
        """Создание эмбеддинга для текста запроса"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array([response.data[0].embedding], dtype=np.float32)

    def _search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, dict]]:
        """Поиск релевантных чанков по запросу"""
        query_embedding = self._create_embedding(query)
        distances, indices = self.index.search(query_embedding, top_k)
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
        """Поиск релевантных чанков с оценкой релевантности"""
        if not self.relevance_scorer:
            print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Оценщик релевантности не доступен")
            standard_results = self._search_relevant_chunks(query, top_k)
            results = []
            for text, metadata, distance in standard_results:
                score = 1.0 / (1.0 + float(distance))
                results.append((text, metadata, score))
            return results

        chunk_texts = [chunk["text"] for chunk in self.dataset]
        scored_results = self.relevance_scorer.search_relevant_docs(
            query=query,
            docs=chunk_texts,
            threshold=threshold,
            k=top_k
        )

        results = []
        for doc_text, score in scored_results:
            for chunk in self.dataset:
                if chunk["text"] == doc_text:
                    results.append((chunk["text"], chunk["metadata"], score))
                    break

        return results

    def _build_prompt(self, question: str, context: List[str]) -> str:
        """Формирование промпта для LLM с контекстом"""
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
        """Получение ответа на вопрос (базовый режим)"""
        if not question.strip():
            return f"{Fore.RED}[ERROR]{Style.RESET_ALL} Пожалуйста, введите вопрос."

        print(f"\n{Fore.CYAN}[SEARCH]{Style.RESET_ALL} Поиск релевантной информации для вопроса: «{Fore.YELLOW}{question}{Style.RESET_ALL}»")

        relevant_chunks = self._search_relevant_chunks(question, top_k)
        print(f"   {Fore.GREEN}[OK]{Style.RESET_ALL} Найдено {len(relevant_chunks)} релевантных чанков")

        context = [chunk[0] for chunk in relevant_chunks]
        prompt = self._build_prompt(question, context)

        print(f"   {Fore.CYAN}[LLM]{Style.RESET_ALL} Генерация ответа...")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Ты — полезный ассистент, эксперт по бенгальским кошкам."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            answer = response.choices[0].message.content
            return answer.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except Exception as e:
            return f"{Fore.RED}[ERROR]{Style.RESET_ALL} Ошибка при генерации ответа: {e}"

    def ask_with_relevance(
        self,
        question: str,
        threshold: float = 0.5,
        top_k: int = 5,
        show_scores: bool = True
    ) -> Tuple[str, Dict]:
        """Получение ответа на вопрос с оценкой релевантности"""
        if not question.strip():
            return f"{Fore.RED}[ERROR]{Style.RESET_ALL} Пожалуйста, введите вопрос.", {}

        print(f"\n{Fore.CYAN}[SEARCH]{Style.RESET_ALL} Поиск релевантной информации для вопроса: «{Fore.YELLOW}{question}{Style.RESET_ALL}»")

        relevant_chunks = self._search_relevant_chunks_with_score(
            query=question,
            threshold=threshold,
            top_k=top_k
        )

        if not relevant_chunks:
            print(f"   {Fore.YELLOW}[WARN]{Style.RESET_ALL} Не найдено чанков с score >= {threshold}")
            return f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Не удалось найти достаточно релевантную информацию в базе знаний. Попробуйте переформулировать вопрос.", {}

        print(f"   {Fore.GREEN}[OK]{Style.RESET_ALL} Найдено {len(relevant_chunks)} релевантных чанков (threshold >= {threshold})")

        metadata = {
            "threshold": threshold,
            "chunks": len(relevant_chunks)
        }

        if show_scores:
            print(f"\n   {Fore.BLUE}[SCORES]{Style.RESET_ALL} Оценки релевантности:")
            scores = []
            for i, (text, meta, score) in enumerate(relevant_chunks, 1):
                preview = text[:70] + "..." if len(text) > 70 else text
                print(f"      {Fore.CYAN}{i}.{Style.RESET_ALL} {Fore.BLUE}Score:{Style.RESET_ALL} {score:.4f} | {preview}")
                scores.append(score)

            avg_score = sum(scores) / len(scores) if scores else 0
            metadata["avg_score"] = avg_score
            metadata["sources"] = [
                {"text": text, "score": score}
                for text, _, score in relevant_chunks
            ]

        context = [chunk[0] for chunk in relevant_chunks]
        prompt = self._build_prompt(question, context)

        print(f"\n   {Fore.CYAN}[LLM]{Style.RESET_ALL} Генерация ответа...")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Ты — полезный ассистент, эксперт по бенгальским кошкам."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            answer = response.choices[0].message.content
            return answer.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore'), metadata
        except Exception as e:
            return f"{Fore.RED}[ERROR]{Style.RESET_ALL} Ошибка при генерации ответа: {e}", {}


class RAGChatUI:
    """Интерфейс пользователя для RAG-системы"""

    def __init__(self):
        self.rag = RAGChatSystem(use_relevance_scorer=True)
        self.settings = {
            "threshold": 0.5,
            "top_k": 5,
            "show_scores": True,
            "show_chunks": False
        }

    def show_header(self):
        """Отображение заголовка"""
        print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{' ' * 15}RAG-чат: Бенгальские кошки v2.5.0{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{' ' * 18}Авторы: Line_GV, Koda, Алиса{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")

    def show_menu(self):
        """Отображение главного меню"""
        print(f"{Fore.GREEN}ГЛАВНОЕ МЕНЮ:{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}1.{Style.RESET_ALL} Задать вопрос (базовый поиск)")
        print(f"  {Fore.YELLOW}2.{Style.RESET_ALL} Задать вопрос (с оценкой релевантности)")
        print(f"  {Fore.YELLOW}3.{Style.RESET_ALL} История вопросов")
        print(f"  {Fore.YELLOW}4.{Style.RESET_ALL} Настройки")
        print(f"  {Fore.YELLOW}5.{Style.RESET_ALL} Экспорт истории")
        print(f"  {Fore.YELLOW}6.{Style.RESET_ALL} Информация о системе")
        print(f"  {Fore.YELLOW}c.{Style.RESET_ALL} Очистить экран")
        print(f"  {Fore.YELLOW}0.{Style.RESET_ALL} Выход")

    def ask_question_basic(self):
        """Режим: базовый поиск"""
        print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}РЕЖИМ: Базовый поиск{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")

        question = input(f"{Fore.YELLOW}❓ Ваш вопрос: {Style.RESET_ALL}").strip()

        if not question:
            print(f"\n{Fore.YELLOW}[INFO]{Style.RESET_ALL} Вопрос не введен")
            return

        if question.lower() in ['exit', 'quit', 'выход', 'q', '0']:
            return

        start_time = datetime.now()

        answer = self.rag.ask(question, top_k=self.settings["top_k"])

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print(f"\n{Fore.GREEN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}ОТВЕТ:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 70}{Style.RESET_ALL}\n")
        print(answer)
        print(f"\n{Fore.GREEN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}⏱️  Время:{Style.RESET_ALL} {elapsed:.2f} сек | {Fore.CYAN}📊 Чанков:{Style.RESET_ALL} {self.settings['top_k']} | {Fore.CYAN}🤖 Модель:{Style.RESET_ALL} GPT-4o-mini")
        print(f"{Fore.GREEN}{'=' * 70}{Style.RESET_ALL}\n")

        self.rag.history.add(
            question=question,
            answer=answer,
            mode="basic",
            metadata={"chunks": self.settings["top_k"], "time": elapsed}
        )

    def ask_question_with_relevance(self):
        """Режим: поиск с оценкой релевантности"""
        print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}РЕЖИМ: Поиск с оценкой релевантности{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Текущие настройки:{Style.RESET_ALL} threshold={self.settings['threshold']}, top_k={self.settings['top_k']}\n")

        question = input(f"{Fore.YELLOW}❓ Ваш вопрос: {Style.RESET_ALL}").strip()

        if not question:
            print(f"\n{Fore.YELLOW}[INFO]{Style.RESET_ALL} Вопрос не введен")
            return

        if question.lower() in ['exit', 'quit', 'выход', 'q', '0']:
            return

        start_time = datetime.now()

        answer, metadata = self.rag.ask_with_relevance(
            question=question,
            threshold=self.settings["threshold"],
            top_k=self.settings["top_k"],
            show_scores=self.settings["show_scores"]
        )

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        if "avg_score" in metadata:
            avg_score = metadata["avg_score"]
        else:
            avg_score = 0

        print(f"\n{Fore.GREEN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}ОТВЕТ:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 70}{Style.RESET_ALL}\n")
        print(answer)
        print(f"\n{Fore.GREEN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}⏱️  Время:{Style.RESET_ALL} {elapsed:.2f} сек | {Fore.CYAN}📊 Чанков:{Style.RESET_ALL} {metadata.get('chunks', 0)} | {Fore.CYAN}🎯 Средний score:{Style.RESET_ALL} {avg_score:.3f}")
        print(f"{Fore.GREEN}{'=' * 70}{Style.RESET_ALL}\n")

        self.rag.history.add(
            question=question,
            answer=answer,
            mode="relevance",
            metadata={**metadata, "time": elapsed}
        )

    def show_history(self):
        """Отображение истории вопросов"""
        history = self.rag.history.get_all()

        if not history:
            print(f"\n{Fore.YELLOW}[INFO]{Style.RESET_ALL} История пуста\n")
            return

        print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}ИСТОРИЯ ВОПРОСОВ ({len(history)} шт.){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")

        for i, item in enumerate(history, 1):
            timestamp = item['timestamp'].strftime('%H:%M:%S')
            mode = item['mode']
            question = item['question'][:50] + "..." if len(item['question']) > 50 else item['question']

            mode_color = Fore.GREEN if mode == "basic" else Fore.BLUE
            print(f"{Fore.CYAN}{i}.{Style.RESET_ALL} [{Fore.WHITE}{timestamp}{Style.RESET_ALL}] {mode_color}{mode.upper()}{Style.RESET_ALL}")
            print(f"   {Fore.YELLOW}Вопрос:{Style.RESET_ALL} {question}\n")

        print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")

        # Поиск по истории
        search = input(f"\n{Fore.YELLOW}🔍 Поиск по истории (Enter для пропуска): {Style.RESET_ALL}").strip()
        if search:
            results = self.rag.history.search(search)
            if results:
                print(f"\n{Fore.GREEN}Найдено {len(results)} результатов:{Style.RESET_ALL}\n")
                for i, item in enumerate(results, 1):
                    timestamp = item['timestamp'].strftime('%H:%M:%S')
                    print(f"{Fore.CYAN}{i}.{Style.RESET_ALL} [{Fore.WHITE}{timestamp}{Style.RESET_ALL}] {item['question']}")
            else:
                print(f"\n{Fore.YELLOW}Ничего не найдено{Style.RESET_ALL}")

    def show_settings(self):
        """Отображение и изменение настроек"""
        while True:
            print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}НАСТРОЙКИ{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")

            print(f"  {Fore.CYAN}1.{Style.RESET_ALL} threshold (порог релевантности): {Fore.YELLOW}{self.settings['threshold']}{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}2.{Style.RESET_ALL} top_k (макс. чанков): {Fore.YELLOW}{self.settings['top_k']}{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}3.{Style.RESET_ALL} show_scores (показывать оценки): {Fore.YELLOW}{self.settings['show_scores']}{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}0.{Style.RESET_ALL} Назад")

            choice = input(f"\n{Fore.YELLOW}Ваш выбор: {Style.RESET_ALL}").strip()

            if choice == '1':
                try:
                    value = float(input(f"{Fore.YELLOW}Введите threshold (0.0-1.0): {Style.RESET_ALL}"))
                    if 0.0 <= value <= 1.0:
                        self.settings['threshold'] = value
                        print(f"{Fore.GREEN}[OK]{Style.RESET_ALL} threshold изменен на {value}")
                    else:
                        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Значение должно быть от 0.0 до 1.0")
                except ValueError:
                    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Некорректное значение")

            elif choice == '2':
                try:
                    value = int(input(f"{Fore.YELLOW}Введите top_k (1-10): {Style.RESET_ALL}"))
                    if 1 <= value <= 10:
                        self.settings['top_k'] = value
                        print(f"{Fore.GREEN}[OK]{Style.RESET_ALL} top_k изменен на {value}")
                    else:
                        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Значение должно быть от 1 до 10")
                except ValueError:
                    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Некорректное значение")

            elif choice == '3':
                self.settings['show_scores'] = not self.settings['show_scores']
                status = "включено" if self.settings['show_scores'] else "выключено"
                print(f"{Fore.GREEN}[OK]{Style.RESET_ALL} show_scores {status}")

            elif choice == '0':
                break

            else:
                print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Некорректный выбор")

    def export_history(self):
        """Экспорт истории"""
        history = self.rag.history.get_all()

        if not history:
            print(f"\n{Fore.YELLOW}[INFO]{Style.RESET_ALL} История пуста, нечего экспортировать\n")
            return

        print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}ЭКСПОРТ ИСТОРИИ ({len(history)} записей){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")

        print(f"  {Fore.CYAN}1.{Style.RESET_ALL} Экспорт в Markdown (.md)")
        print(f"  {Fore.CYAN}2.{Style.RESET_ALL} Экспорт в JSON (.json)")
        print(f"  {Fore.CYAN}0.{Style.RESET_ALL} Назад")

        choice = input(f"\n{Fore.YELLOW}Ваш выбор: {Style.RESET_ALL}").strip()

        if choice == '1':
            try:
                print(f"\n{Fore.CYAN}[EXPORT]{Style.RESET_ALL} Экспорт в Markdown...")
                filename = self.rag.history.export_to_markdown()
                if filename:
                    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} История экспортирована в файл: {Fore.YELLOW}{filename}{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Ошибка при экспорте\n")
            except Exception as e:
                print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Ошибка при экспорте: {e}\n")

        elif choice == '2':
            try:
                print(f"\n{Fore.CYAN}[EXPORT]{Style.RESET_ALL} Экспорт в JSON...")
                filename = self.rag.history.export_to_json()
                if filename:
                    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} История экспортирована в файл: {Fore.YELLOW}{filename}{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Ошибка при экспорте\n")
            except Exception as e:
                print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Ошибка при экспорте: {e}\n")

        elif choice == '0':
            print()  # Просто возвращаемся в меню
        else:
            print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} Некорректный выбор\n")

    def show_system_info(self):
        """Отображение информации о системе"""
        print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}ИНФОРМАЦИЯ О СИСТЕМЕ{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")

        print(f"{Fore.CYAN}Версия:{Style.RESET_ALL} 2.5.0")
        print(f"{Fore.CYAN}Авторы:{Style.RESET_ALL} Line_GV, Koda (AI Assistant), Алиса (AI Consultant)")
        print(f"\n{Fore.CYAN}База данных RAG:{Style.RESET_ALL}")
        print(f"  Модель эмбеддингов: {self.rag.metadata['model_name']}")
        print(f"  Размерность: {self.rag.metadata['embedding_dim']}")
        print(f"  Векторов: {self.rag.metadata['total_vectors']}")
        print(f"  Чанков: {len(self.rag.dataset)}")
        print(f"\n{Fore.CYAN}LLM модель:{Style.RESET_ALL} GPT-4o-mini")
        print(f"{Fore.CYAN}API:{Style.RESET_ALL} OpenAI (через proxyapi.ru)")
        print(f"{Fore.CYAN}Оценка релевантности:{Style.RESET_ALL} {'Включена' if self.rag.use_relevance_scorer else 'Выключена'}")

        history = self.rag.history.get_all()
        print(f"\n{Fore.CYAN}История:{Style.RESET_ALL} {len(history)} вопросов")

        print(f"\n{Fore.CYAN}Текущие настройки:{Style.RESET_ALL}")
        print(f"  threshold: {self.settings['threshold']}")
        print(f"  top_k: {self.settings['top_k']}")
        print(f"  show_scores: {self.settings['show_scores']}")

        print(f"\n{Fore.CYAN}Папка для отчетов:{Style.RESET_ALL} {self.rag.history.reports_dir}")

        print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")

    def run(self):
        """Запуск главного цикла интерфейса"""
        self.show_header()

        while True:
            try:
                # Очистка буфера ввода перед показом меню
                import sys
                if sys.stdin in sys.__dict__:
                    try:
                        import msvcrt
                        while msvcrt.kbhit():
                            msvcrt.getch()
                    except:
                        pass

                self.show_menu()

                choice = input(f"\n{Fore.YELLOW}Ваш выбор: {Style.RESET_ALL}").strip()

                if choice == '1':
                    self.ask_question_basic()
                elif choice == '2':
                    self.ask_question_with_relevance()
                elif choice == '3':
                    self.show_history()
                elif choice == '4':
                    self.show_settings()
                elif choice == '5':
                    self.export_history()
                elif choice == '6':
                    self.show_system_info()
                elif choice == 'c' or choice == 'C' or choice == 'cls' or choice == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    self.show_header()
                elif choice == '0':
                    print(f"\n{Fore.GREEN}[BYE]{Style.RESET_ALL} До свидания! 👋\n")
                    break
                else:
                    print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} Некорректный выбор. Попробуйте снова.\n")
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}[INFO]{Style.RESET_ALL} Прервано пользователем\n")
                continue
            except Exception as e:
                print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} Произошла ошибка: {e}\n")


def main():
    """Главная функция"""
    try:
        ui = RAGChatUI()
        ui.run()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}[BYE]{Style.RESET_ALL} До свидания! 👋\n")
    except Exception as e:
        print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} Произошла ошибка: {e}\n")


if __name__ == "__main__":
    main()
