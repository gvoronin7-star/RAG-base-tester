# API Reference

**Версия:** 2.5.0 (Production Release)

> **Примечание:** Это детальная документация API. Основная документация: [README.md](README.md)

---

## Содержание

- [RAGChatSystem](#ragchatsystem)
- [RelevanceScorerOpenAI](#relevancescoreropenai)
- [AnswerHistory](#answerhistory)
- [Константы](#константы)

---

## RAGChatSystem

Основной класс для работы с RAG-системой.

### Инициализация

```python
from rag_chat_interactive import RAGChatSystem

rag = RAGChatSystem(
    rag_db_path="бенгальские кошки RAG",
    use_relevance_scorer=True
)
```

### Параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `rag_db_path` | str | `"бенгальские кошки RAG"` | Путь к базе данных FAISS |
| `use_relevance_scorer` | bool | `False` | Использовать ли модуль оценки релевантности |

### Методы

#### ask()

Базовый поиск без оценки релевантности.

```python
answer = rag.ask(question="Какой характер у бенгалов?", top_k=5)
```

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `question` | str | ✅ | Вопрос пользователя |
| `top_k` | int | ❌ (5) | Количество чанков для поиска |

**Возвращает:** `str` — сгенерированный ответ

**Пример:**

```python
answer = rag.ask("Какой характер у бенгалов?")
print(answer)
# Бенгальские кошки активные, любопытные и игривые...
```

---

#### ask_with_relevance()

Поиск с оценкой релевантности и фильтрацией.

```python
answer, metadata = rag.ask_with_relevance(
    question="Какой характер у бенгалов?",
    threshold=0.5,
    top_k=5,
    show_scores=True
)
```

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `question` | str | ✅ | Вопрос пользователя |
| `threshold` | float | ❌ (0.5) | Порог релевантности (0.0-1.0) |
| `top_k` | int | ❌ (5) | Максимальное количество чанков |
| `show_scores` | bool | ❌ (True) | Показывать ли оценки |

**Возвращает:** `Tuple[str, Dict]` — ответ и метаданные

**Метаданные:**

```python
{
    "chunks_used": 3,
    "threshold": 0.5,
    "top_k": 5,
    "scores": [
        {"text": "...", "score": 0.6947},
        {"text": "...", "score": 0.6921},
        ...
    ],
    "avg_score": 0.687
}
```

**Пример:**

```python
answer, metadata = rag.ask_with_relevance(
    "Какой характер у бенгалов?",
    threshold=0.5,
    show_scores=True
)

print(f"Ответ: {answer}")
print(f"Использовано чанков: {metadata['chunks_used']}")
print(f"Средний score: {metadata['avg_score']:.3f}")
```

---

#### get_system_info()

Получение информации о системе.

```python
info = rag.get_system_info()
```

**Возвращает:** `Dict` — информация о системе

```python
{
    "rag_db_path": "бенгальские кошки RAG",
    "index_loaded": True,
    "index_size": 210,
    "index_dimension": 1536,
    "use_relevance_scorer": True,
    "model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small"
}
```

---

## RelevanceScorerOpenAI

Модуль для независимого использования оценки релевантности.

### Инициализация

```python
from relevance_scorer_openai import RelevanceScorerOpenAI

scorer = RelevanceScorerOpenAI()
```

### Методы

#### search_relevant_docs()

Поиск релевантных документов с оценкой сходства.

```python
results = scorer.search_relevant_docs(
    query="Текст запроса",
    docs=["Документ 1", "Документ 2", ...],
    threshold=0.5,
    k=5
)
```

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `query` | str | ✅ | Текст запроса |
| `docs` | List[str] | ✅ | Список документов |
| `threshold` | float | ❌ (0.5) | Порог релевантности |
| `k` | int | ❌ (5) | Максимальное количество результатов |

**Возвращает:** `List[Tuple[str, float]]` — список (документ, score)

**Пример:**

```python
docs = [
    "Бенгальские кошки активные и игривые",
    "Они любят воду",
    "Характер бенгалов уникален"
]

results = scorer.search_relevant_docs(
    query="Какой характер у бенгалов?",
    docs=docs,
    threshold=0.5
)

for doc, score in results:
    print(f"Score: {score:.4f} | {doc}")
```

---

#### compute_cosine_similarity()

Вычисление косинусного сходства между двумя векторами.

```python
similarity = scorer.compute_cosine_similarity(vector1, vector2)
```

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `vector1` | np.ndarray | ✅ | Первый вектор |
| `vector2` | np.ndarray | ✅ | Второй вектор |

**Возвращает:** `float` — значение сходства (0.0-1.0)

**Пример:**

```python
import numpy as np

v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([1.0, 2.0, 3.0])

similarity = scorer.compute_cosine_similarity(v1, v2)
print(f"Similarity: {similarity:.4f}")  # 1.0
```

---

## AnswerHistory

Класс для управления историей вопросов и ответов.

### Инициализация

```python
from rag_chat_interactive import AnswerHistory

history = AnswerHistory()
```

### Методы

#### add()

Добавление записи в историю.

```python
history.add(
    question="Какой характер у бенгалов?",
    answer="Бенгальские кошки активные...",
    mode="relevance",
    metadata={"chunks_used": 3, "avg_score": 0.687}
)
```

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `question` | str | ✅ | Вопрос |
| `answer` | str | ✅ | Ответ |
| `mode` | str | ❌ ("basic") | Режим поиска |
| `metadata` | Dict | ❌ ({}) | Метаданные |

---

#### search()

Поиск в истории по ключевым словам.

```python
results = history.search("характер")
```

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `keyword` | str | ✅ | Ключевое слово |

**Возвращает:** `List[Dict]` — найденные записи

---

#### export_to_markdown()

Экспорт истории в Markdown файл.

```python
history.export_to_markdown("reports/history.md")
```

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `filepath` | str | ✅ | Путь к файлу |

---

#### export_to_json()

Экспорт истории в JSON файл.

```python
history.export_to_json("reports/history.json")
```

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `filepath` | str | ✅ | Путь к файлу |

---

#### clear()

Очистка истории.

```python
history.clear()
```

---

#### get_all()

Получение всех записей.

```python
all_entries = history.get_all()
```

**Возвращает:** `List[Dict]` — все записи

---

#### count()

Количество записей.

```python
count = history.count()
```

**Возвращает:** `int` — количество записей

---

## Константы

### Модели

```python
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
```

### Настройки по умолчанию

```python
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.7
```

### Пути

```python
DEFAULT_RAG_DB_PATH = "бенгальские кошки RAG"
REPORTS_DIR = "reports"
```

---

## Примеры использования

### Полный пример с историей

```python
from rag_chat_interactive import RAGChatSystem, AnswerHistory

# Инициализация
rag = RAGChatSystem(use_relevance_scorer=True)
history = AnswerHistory()

# Задать вопрос
question = "Какой характер у бенгалов?"
answer, metadata = rag.ask_with_relevance(question)

# Добавить в историю
history.add(question, answer, mode="relevance", metadata=metadata)

# Экспорт
history.export_to_markdown("reports/session.md")
history.export_to_json("reports/session.json")
```

### Независимое использование модуля релевантности

```python
from relevance_scorer_openai import RelevanceScorerOpenAI

scorer = RelevanceScorerOpenAI()

docs = load_documents()  # Ваша функция загрузки
results = scorer.search_relevant_docs(
    query="Ваш запрос",
    docs=docs,
    threshold=0.6
)

for doc, score in results:
    print(f"[{score:.3f}] {doc[:100]}...")
```

---

## Обработка ошибок

### Пример с обработкой исключений

```python
from rag_chat_interactive import RAGChatSystem
import os

try:
    rag = RAGChatSystem(rag_db_path="путь/к/базе")
    answer = rag.ask("Ваш вопрос")
    print(answer)

except FileNotFoundError as e:
    print(f"Ошибка: база данных не найдена: {e}")

except Exception as e:
    print(f"Произошла ошибка: {e}")
```

---

## Типы данных

```python
from typing import List, Dict, Tuple, Optional

# Тип для результатов поиска
SearchResult = Tuple[str, float]

# Тип для метаданных
Metadata = Dict[str, Any]

# Тип для записи истории
HistoryEntry = Dict[str, Any]
```

---

## Производительность

### Рекомендации

- Используйте `threshold` для фильтрации малорелевантных чанков
- Уменьшайте `top_k` для ускорения поиска
- Кэшируйте эмбеддинги для повторяющихся запросов

### Бенчмарки

| Операция | Время |
|----------|-------|
| Загрузка индекса | ~0.5 сек |
| Эмбеддинг запроса | ~0.3 сек |
| Поиск FAISS (k=5) | ~0.1 сек |
| Генерация ответа | ~1.5 сек |
| Общее время | ~2.4 сек |

---

## Поддержка

Для вопросов и предложений:
- **Line_GV** — [@Line_GV](https://t.me/Line_GV)

---

*RAG-чат: Бенгальские кошки v2.5.0 — Production Release*  
*API Reference*
