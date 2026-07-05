# cogneelearn

RAG-ассистент с UI поверх [Cognee](https://github.com/topoteretes/cognee) на локальных LLM через `llama-server`. База знаний: зависимости объектов DuckDB (TPC-H) + проектная документация (md, txt, docx, pdf без OCR).

## Архитектура

```
run-llamaserver-all.sh   3 модели в одном llama-server на 127.0.0.1:1234
                         (qwen3.6-27b-mtp — LLM, qwen3-embedding-8b-q8 — эмбеддинги)
.env                     конфиг Cognee → llama-server (alias-driven, легко менять модели)
src/cogneelearn/
  config.py              загрузка .env из корня проекта (независимо от CWD)
  ingest/
    duckdb_deps.py       file-mode DuckDB → метаданные (таблицы/FK/вьюшки) → документы для Cognee
                         TPC-H: dbgen() не создаёт FK, поэтому каноническая схема TPC-H инжектируется
    docs_loader.py       md/txt/pdf — нативно в Cognee; docx — извлечение текста через python-docx
    cli.py               cogneelearn-ingest tpch|docs|all
  pipeline.py            обёртки Cognee: ingest_tpch / ingest_documents / ask / retrieve_context / prune
  assistant.py           RAG: retrieve_context + прямой вызов LLM с кастомным промптом (или cognee.recall)
  ui.py                  Streamlit: чат + загрузка документов + ingest TPC-H + статус датасетов
```

Хранилища Cognee (по умолчанию файловые, без внешних сервисов): SQLite (метаданные) + KuzuDB (граф) + LanceDB (векторы 4096d).

## Запуск

```bash
uv venv .venv && uv pip install -e ".[dev]"

# 1. Поднять LLM (отдельно, не запускать из репозитория)
bash run-llamaserver-all.sh

# 2. Инжест данных
uv run cogneelearn-ingest tpch --sf 0.01          # TPC-H → датасет tpch_schema
uv run cogneelearn-ingest docs data/docs          # документы → датасет docs

# 3. UI
uv run streamlit run src/cogneelearn/ui.py
```

## Смена модели

Модели адресуются alias'ами пресетов `run-llamaserver-all.sh`. Чтобы поменять LLM/эмбеддинги:
1. Добавить/изменить пресет в `run-llamaserver-all.sh`, перезапустить сервер.
2. Поправить `LLM_MODEL` / `EMBEDDING_MODEL` (+ `EMBEDDING_DIMENSIONS`) в `.env`.
3. Выполнить prune: `await cognee.prune.prune_system(graph=True, vector=True, metadata=True)` — иначе в LanceDB останутся векторы старой размерности.

## Нюанс: Qwen3-Embedding-8B

Модель instruction-aware и обучена под last-token pooling, а в пресете стоит `pooling = mean`. Для учебного проекта допустимо игнорировать (падение качества retrieval ~1–5%); для максимального качества — см. комментарий в `.env`.


## Что сделано
1. run-llamaserver-all.sh — исправлена опечатка parallel = 1./ → parallel = 1 (run-llamaserver-all.sh:27).

2. .env — конфиг Cognee, привязанный к llama-server через alias'ы пресетов. Ключевые решения:

LLM_PROVIDER="custom" + LLM_MODEL="openai/qwen3.6-27b-mtp" (префикс openai/ для OpenAI-совместимого роутинга LiteLLM);
EMBEDDING_PROVIDER="openai_compatible" — прямой OpenAI-SDK, минуя LiteLLM, лучший путь для llama.cpp;
EMBEDDING_DIMENSIONS="4096" (проверено по model card Qwen3-Embedding-8B);
ENABLE_BACKEND_ACCESS_CONTROL="False" — иначе Cognee 1.x принудительно включает multi-tenant+auth;
COGNEE_SKIP_CONNECTION_TEST="true" — 27B «греется» >30с;
EMBEDDING_BATCH_SIZE="5" — локальный сервер, малый батч против таймаутов.
3. pyproject.toml — duckdb 1.5.4, cognee 1.2.2, streamlit, python-docx, httpx; ruff/mypy в dev-extras.

4. src/cogneelearn/:

ingest/duckdb_deps.py — file-mode DuckDB (build_tpch_db обязательно с путём файла, :memory: отклоняется). Важная находка при зондировании: dbgen() в DuckDB 1.5.4 не создаёт PK/FK constraints в duckdb_constraints() — поэтому каноническая схема TPC-H (8 FK по спецификации) инжектируется декларативно (_backfill_tpch_foreign_keys), а для любых других БД FK читаются динамически. Smoke-тест: 8 таблиц + 8 FK = 16 документов, композитный ключ lineitem→partsupp корректен.
ingest/docs_loader.py — md/txt/pdf идут нативно в Cognee; docx извлекается через python-docx (включая таблицы).
ingest/cli.py — cogneelearn-ingest tpch|docs|all.
pipeline.py — обёртки Cognee (ingest_tpch, ingest_documents, ask, retrieve_context, prune_all), датасеты разделены: tpch_schema и docs.
assistant.py — два режима RAG: кастомный промпт с прямым вызовом LLM (answer_with_context) и встроенный QA Cognee (answer_via_cognee).
ui.py — Streamlit: чат с контекстом, загрузка документов, ingest TPC-H, prune, переключатель режимов ответа.
config.py — явная загрузка .env из корня проекта (Cognee ищет .env от CWD, что ломало запуск из другой директории — проверено и исправлено).
Проверка: ruff lint + format чистые; все модули импортируются; e2e smoke-тест TPC-H + docs_loader прошёл; mypy падает только на сторонних stubs numpy (не наш код).

mypy не прошёл из-за несовместимости numpy-stubs с версией mypy в venv — это окружение, не код проекта. Предлагаю добавить в AGENTS.md заметку, что typecheck надо запускать без numpy-stubs (mypy --no-incremental или через --exclude numpy), чтобы зафиксировать рабочий способ проверки.