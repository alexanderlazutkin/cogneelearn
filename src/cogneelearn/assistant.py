"""RAG assistant: build a grounded answer from Cognee context + local LLM.

Two modes:

1. ``answer_with_context`` — retrieve context via :func:`pipeline.retrieve_context`,
   then call the local LLM directly with a custom system prompt. This gives full
   control over the assistant persona and how sources are cited.

2. ``answer_via_cognee`` — delegate to ``cognee.recall``, which runs Cognee's
   own QA prompt end to end. Simpler, less control.

The UI offers both so the project can compare a hand-rolled RAG prompt against
Cognee's built-in pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .pipeline import AnswerResult, ask, retrieve_context, run

logger = logging.getLogger(__name__)

# Loaded from environment at first use; see _llm_settings.
_LLM_SETTINGS: dict[str, str] | None = None

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledge-base assistant for a data-engineering project. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know — do not invent. "
    "When the context mentions database objects (tables, columns, foreign keys, views), "
    "reference them explicitly. Cite sources by their label when available."
)


@dataclass
class LLMSettings:
    endpoint: str
    api_key: str
    model: str


def _load_llm_settings() -> LLMSettings:
    """Read LLM endpoint/model/key from the environment (single-user)."""
    import os

    return LLMSettings(
        endpoint=os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:1234/v1"),
        api_key=os.environ.get("LLM_API_KEY", "."),
        model=os.environ.get("LLM_MODEL", "openai/qwen3.6-27b-mtp"),
    )


def _strip_prefix(model: str) -> str:
    """Cognee uses LiteLLM prefixes (openai/...); the raw API wants the bare id."""
    return model.split("/", 1)[-1] if "/" in model else model


async def answer_with_context(
    question: str,
    datasets: list[str] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    top_k: int = 15,
) -> AnswerResult:
    """Retrieve context from Cognee, then ask the local LLM directly.

    This is the hand-rolled RAG path: we control the prompt and citation
    format. The local LLM is reached through its OpenAI-compatible endpoint
    (the same one Cognee uses) so results are directly comparable.
    """
    context = await retrieve_context(question, datasets=datasets, top_k=top_k)
    answer_text = await _call_llm(question, context, system_prompt)
    return AnswerResult(answer=answer_text, context=context, raw=[])


async def answer_via_cognee(
    question: str,
    datasets: list[str] | None = None,
    top_k: int = 15,
) -> AnswerResult:
    """Delegate the full QA loop to ``cognee.recall`` (built-in prompt)."""
    return await ask(question, datasets=datasets, top_k=top_k)


async def _call_llm(question: str, context: list[str], system_prompt: str) -> str:
    """Call the local llama-server chat endpoint with a grounded prompt."""
    settings = _load_llm_settings()
    context_block = _format_context(context)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_block}\n\n"
                f"Question:\n{question}\n\n"
                "Answer (grounded in the context above):"
            ),
        },
    ]

    payload = {
        "model": _strip_prefix(settings.model),
        "messages": messages,
        "temperature": 0.2,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {settings.api_key}"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.endpoint}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        logger.error("Malformed LLM response: %s", data)
        raise RuntimeError("LLM returned no answer content") from exc


def _format_context(context: list[str]) -> str:
    if not context:
        return "(no context retrieved)"
    blocks = []
    for i, chunk in enumerate(context, 1):
        blocks.append(f"[{i}] {chunk}")
    return "\n\n".join(blocks)


# ─── sync wrappers for UI/CLI ─────────────────────────────────────────────────
def answer(
    question: str,
    mode: str = "context",
    datasets: list[str] | None = None,
    top_k: int = 15,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> AnswerResult:
    """Sync entrypoint used by the UI/CLI.

    ``mode``: ``"context"`` → hand-rolled RAG prompt; ``"cognee"`` → Cognee QA.
    """
    if mode == "cognee":
        return run(answer_via_cognee(question, datasets=datasets, top_k=top_k))
    return run(
        answer_with_context(question, datasets=datasets, system_prompt=system_prompt, top_k=top_k)
    )
