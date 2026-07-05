"""Cognee pipeline wrappers: ingest datasets, cognify, recall, prune.

This module is the single integration point between the app and the Cognee
library. Everything above it (assistant, UI, CLI) calls these helpers so the
Cognee API surface is contained in one place and easy to retarget when Cognee
ships breaking changes.

Datasets
--------
We keep DuckDB schema and project documents in *separate* Cognee datasets so
they can be cognified/pruned independently:

- ``tpch_schema`` — serialized DuckDB objects (tables, FKs, views).
- ``docs``        — project documents (md/txt/docx/pdf).

A third dataset, ``main_dataset`` (Cognee's default), is left untouched.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cognee
from cognee.modules.search.types import SearchType

# Load .env from the project root BEFORE importing cognee, so the local-LLM
# config is applied regardless of the current working directory.
from . import config as _config  # noqa: F401 — side effect: loads .env
from .ingest.docs_loader import LoadedDoc, load_directory, load_document
from .ingest.duckdb_deps import (
    build_tpch_db,
    serialize_for_cognee,
)

logger = logging.getLogger(__name__)

TPCH_DATASET = "tpch_schema"
DOCS_DATASET = "docs"


@dataclass
class IngestResult:
    """Outcome of an ingestion + cognify run."""

    dataset: str
    items_added: int
    ok: bool
    error: str | None = None


# ─── lifecycle ────────────────────────────────────────────────────────────────
async def prune_all(metadata: bool = False) -> None:
    """Wipe graph + vectors (+ optionally metadata) and rebuild from empty.

    Call after switching LLM/embedding model or ``EMBEDDING_DIMENSIONS`` —
    otherwise stale vectors of the old dimensionality break search.
    """
    logger.warning("Pruning Cognee (metadata=%s)", metadata)
    await cognee.prune.prune_system(graph=True, vector=True, metadata=metadata, cache=True)


async def list_datasets() -> list[str]:
    """Return the names of datasets Cognee currently knows about."""
    try:
        return await cognee.datasets.list_datasets()
    except Exception as exc:  # noqa: BLE001 — UI should not crash on listing
        logger.error("list_datasets failed: %s", exc)
        return []


# ─── DuckDB / TPC-H ───────────────────────────────────────────────────────────
async def ingest_tpch(
    db_path: str | Path,
    dataset: str = TPCH_DATASET,
    cognify: bool = True,
) -> IngestResult:
    """Serialize a DuckDB database and ingest it into Cognee.

    The file must already exist. To create a TPC-H database from scratch, call
    :func:`build_tpch_db` first (or :func:`ensure_tpch_db` below for the full
    build-or-load flow).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return IngestResult(
            dataset=dataset, items_added=0, ok=False, error=f"DB not found: {db_path}"
        )

    docs = serialize_for_cognee(db_path)
    if not docs:
        return IngestResult(
            dataset=dataset, items_added=0, ok=False, error="No objects extracted from database"
        )

    try:
        await cognee.add([d.text for d in docs], dataset_name=dataset)
        if cognify:
            await cognee.cognify(datasets=dataset)
    except Exception as exc:  # noqa: BLE001
        logger.exception("TPC-H ingestion failed")
        return IngestResult(dataset=dataset, items_added=len(docs), ok=False, error=str(exc))
    return IngestResult(dataset=dataset, items_added=len(docs), ok=True)


async def ensure_tpch_db(db_path: str | Path, sf: float = 0.01, overwrite: bool = False) -> Path:
    """Build a TPC-H database if missing, then return its path."""
    db_path = Path(db_path)
    if overwrite or not db_path.exists():
        build_tpch_db(db_path, sf=sf, overwrite=overwrite)
    return db_path


# ─── Documents ────────────────────────────────────────────────────────────────
async def ingest_documents(
    source: str | Path,
    dataset: str = DOCS_DATASET,
    cognify: bool = True,
) -> IngestResult:
    """Ingest a single document or a directory of documents into Cognee."""
    source_path = Path(source)
    if source_path.is_dir():
        loaded = load_directory(source_path)
    else:
        loaded = [load_document(source_path)]

    if not loaded:
        return IngestResult(
            dataset=dataset, items_added=0, ok=False, error="No supported documents found"
        )

    try:
        for doc in loaded:
            await _add_one(doc, dataset)
        if cognify:
            await cognee.cognify(datasets=dataset)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Document ingestion failed")
        return IngestResult(dataset=dataset, items_added=len(loaded), ok=False, error=str(exc))
    return IngestResult(dataset=dataset, items_added=len(loaded), ok=True)


async def _add_one(doc: LoadedDoc, dataset: str) -> None:
    """Feed a single loaded document to Cognee.

    Native formats (md/txt/pdf) go in as file paths so Cognee's own loaders
    handle them; docx goes in as already-extracted text.
    """
    if doc.kind == "native" and doc.path is not None:
        await cognee.add(doc.path, dataset_name=dataset)
    elif doc.text is not None:
        # Prefix with the filename so the graph can attribute chunks to a source.
        text = f"[Source file: {doc.name}]\n\n{doc.text}"
        await cognee.add(text, dataset_name=dataset)
    else:
        logger.warning("Skipping empty document: %s", doc.name)


# ─── Recall / search ──────────────────────────────────────────────────────────
@dataclass
class AnswerResult:
    """A recall answer with the retrieved context that backed it."""

    answer: str
    context: list[str]
    raw: list[Any]


async def ask(
    question: str,
    datasets: list[str] | None = None,
    top_k: int = 15,
) -> AnswerResult:
    """Ask a question against the knowledge graph and return answer + context.

    Uses ``cognee.recall`` with ``auto_route=True`` (Cognee picks the best
    search strategy: graph, vector, or hybrid). The returned entries are a mix
    of QA answers and graph-context fragments; we separate them for the UI.
    """
    if datasets is None:
        datasets = [TPCH_DATASET, DOCS_DATASET]

    raw = await cognee.recall(
        question,
        datasets=datasets,
        top_k=top_k,
        auto_route=True,
    )

    answer_parts: list[str] = []
    context_parts: list[str] = []
    for entry in raw or []:
        # RecallResponse entries are discriminated by `source`. We pull text
        # from the common attributes defensively rather than relying on a
        # specific entry subclass.
        text = _extract_entry_text(entry)
        if not text:
            continue
        source = getattr(entry, "source", None)
        if source == "QA":
            answer_parts.append(text)
        else:
            context_parts.append(text)

    answer = (
        "\n\n".join(answer_parts)
        if answer_parts
        else (context_parts[0] if context_parts else "No answer found.")
    )
    return AnswerResult(answer=answer, context=context_parts, raw=raw)


async def retrieve_context(
    query: str,
    datasets: list[str] | None = None,
    top_k: int = 15,
) -> list[str]:
    """Return only retrieved context fragments, no LLM answer.

    Useful when the caller wants to build its own prompt (e.g. a custom
    assistant persona) instead of using Cognee's default QA prompt.
    """
    if datasets is None:
        datasets = [TPCH_DATASET, DOCS_DATASET]
    raw = await cognee.recall(
        query,
        datasets=datasets,
        top_k=top_k,
        auto_route=True,
        only_context=True,
    )
    return [t for t in (_extract_entry_text(e) for e in raw or []) if t]


async def search_graph(
    query: str,
    datasets: list[str] | None = None,
    top_k: int = 15,
) -> list[Any]:
    """Run a raw graph-completion search (V1 API) and return SearchResults.

    Exposed for the UI's "inspect graph" view where we want the structured
    results, not a flattened answer.
    """
    if datasets is None:
        datasets = [TPCH_DATASET, DOCS_DATASET]
    return await cognee.search(
        query,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=datasets,
        top_k=top_k,
    )


# ─── helpers ──────────────────────────────────────────────────────────────────
def _extract_entry_text(entry: Any) -> str:
    """Pull a human-readable string out of a recall/search entry."""
    for attr in ("answer", "text", "content", "payload", "response"):
        val = getattr(entry, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    # Fall back to str() of the entry — better than dropping it silently.
    try:
        s = str(entry)
    except Exception:  # noqa: BLE001
        return ""
    return s if s and not s.startswith("<") else ""


def run(coro: Any) -> Any:
    """Run an async coroutine from sync code (UI/CLI entrypoints).

    Uses a fresh event loop so it is safe to call from Streamlit callbacks
    that may already be inside a loop-managed context.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an existing loop (Streamlit): nest under asyncio.
            import nest_asyncio  # type: ignore

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)
