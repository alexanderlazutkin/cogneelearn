"""Streamlit UI: chat with the RAG assistant, ingest data, inspect datasets.

Run with::

    streamlit run src/cogneelearn/ui.py
    # or, after `uv pip install -e .`:
    cogneelearn-ui
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

from . import assistant as A
from . import pipeline as P

logger = logging.getLogger("cogneelearn.ui")

SIDEBAR_CFG = {
    "tpch_db": "data/tpch/tpch.duckdb",
    "docs_dir": "data/docs",
    "default_sf": 0.01,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(
        page_title="Cognee RAG Assistant",
        page_icon="🧠",
        layout="wide",
    )
    st.title("🧠 Cognee RAG Assistant")
    st.caption(
        "Knowledge base over DuckDB/TPC-H dependencies and project documents. "
        "Backed by a local llama-server LLM."
    )

    _render_sidebar()
    _render_chat()
    _render_datasets_panel()


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Ingest data")

        sf = st.number_input(
            "TPC-H scale factor",
            min_value=0.001,
            max_value=10.0,
            value=SIDEBAR_CFG["default_sf"],
            step=0.01,
            format="%.3f",
            help="0.01 ≈ 10MB. Raise for larger datasets.",
        )
        db_path = st.text_input("DuckDB file path", value=SIDEBAR_CFG["tpch_db"])

        c1, c2 = st.columns(2)
        if c1.button("Build + ingest TPC-H", type="primary"):
            _do_ingest_tpch(db_path, sf)
        if c2.button("Prune all", help="Wipe graph + vectors. Use after switching models."):
            _do_prune()

        st.divider()
        st.subheader("Project documents")
        docs_dir = st.text_input("Documents directory", value=SIDEBAR_CFG["docs_dir"])
        if st.button("Ingest documents"):
            _do_ingest_docs(docs_dir)

        uploaded = st.file_uploader(
            "…or upload a file (md/txt/docx/pdf)",
            type=["md", "txt", "docx", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Ingest uploaded files"):
            _do_ingest_uploads(uploaded)

        st.divider()
        st.subheader("Settings")
        st.session_state.setdefault("mode", "context")
        st.radio(
            "Answer mode",
            options=["context", "cognee"],
            index=0,
            key="mode",
            help=(
                "context: hand-rolled RAG prompt with the local LLM. "
                "cognee: Cognee's built-in QA pipeline."
            ),
            format_func=lambda x: "Custom RAG prompt" if x == "context" else "Cognee QA",
        )
        st.number_input(
            "Top-K context chunks",
            min_value=1,
            max_value=50,
            value=15,
            step=1,
            key="top_k",
        )


def _render_chat() -> None:
    st.subheader("Chat")
    st.session_state.setdefault("messages", [])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("context"):
                with st.expander(f"Retrieved context ({len(msg['context'])} chunks)"):
                    for i, chunk in enumerate(msg["context"], 1):
                        st.markdown(f"**[{i}]**")
                        st.text(chunk)

    question = st.chat_input("Ask about the schema, dependencies, or documents…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                res: A.AnswerResult = A.answer(
                    question,
                    mode=st.session_state.mode,
                    top_k=st.session_state.top_k,
                )
            except Exception as exc:  # noqa: BLE001 — show error in chat, don't crash
                st.error(f"Answer failed: {exc}")
                return
        st.markdown(res.answer)
        if res.context:
            with st.expander(f"Retrieved context ({len(res.context)} chunks)"):
                for i, chunk in enumerate(res.context, 1):
                    st.markdown(f"**[{i}]**")
                    st.text(chunk)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": res.answer,
                "context": res.context,
            }
        )


def _render_datasets_panel() -> None:
    with st.expander("Datasets", expanded=False):
        try:
            datasets = P.run(P.list_datasets())
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not list datasets: {exc}")
            return
        if not datasets:
            st.info("No datasets yet. Ingest TPC-H or documents from the sidebar.")
            return
        for name in datasets:
            st.markdown(f"- `{name}`")


# ─── ingest actions ───────────────────────────────────────────────────────────
def _do_ingest_tpch(db_path: str, sf: float) -> None:
    with st.status("Building and ingesting TPC-H…"):
        try:
            path = P.run(P.ensure_tpch_db(db_path, sf=sf, overwrite=False))
            st.write(f"Database: `{path}`")
            res = P.run(P.ingest_tpch(path))
        except Exception as exc:  # noqa: BLE001
            st.error(f"TPC-H ingest failed: {exc}")
            return
    if res.ok:
        st.success(f"TPC-H ingested: {res.items_added} objects → `{res.dataset}`")
    else:
        st.error(f"TPC-H ingest failed: {res.error}")


def _do_ingest_docs(docs_dir: str) -> None:
    if not Path(docs_dir).is_dir():
        st.warning(f"Directory not found: {docs_dir}")
        return
    with st.status("Ingesting documents…"):
        res = P.run(P.ingest_documents(docs_dir))
    if res.ok:
        st.success(f"Documents ingested: {res.items_added} files → `{res.dataset}`")
    else:
        st.error(f"Documents ingest failed: {res.error}")


def _do_ingest_uploads(uploads) -> None:
    docs_dir = Path(SIDEBAR_CFG["docs_dir"])
    docs_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for up in uploads:
        dest = docs_dir / up.name
        dest.write_bytes(up.getvalue())
        saved.append(str(dest))
    with st.status(f"Ingesting {len(saved)} uploaded file(s)…"):
        results = [P.run(P.ingest_documents(p, cognify=False)) for p in saved]
        # Cognify once after all files are added.
        if any(r.ok for r in results):
            P.run(_cognify_docs())
    ok = sum(1 for r in results if r.ok)
    if ok == len(results):
        st.success(f"Uploaded {ok} file(s) ingested → `docs`")
    else:
        st.warning(f"{ok}/{len(results)} files ingested; check logs.")


async def _cognify_docs() -> None:
    import cognee

    # data_per_batch берётся из COGNEE_DATA_PER_BATCH (по умолчанию 2).
    # См. pipeline._data_per_batch() — синхронизировано с parallel в llama-server.
    await cognee.cognify(datasets=P.DOCS_DATASET, data_per_batch=P._data_per_batch())


def _do_prune() -> None:
    with st.status("Pruning Cognee…"):
        try:
            P.run(P.prune_all(metadata=True))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prune failed: {exc}")
            return
    st.success("Pruned. Re-ingest data to rebuild the graph.")
    st.session_state.messages.clear()


if __name__ == "__main__":
    main()
    sys.exit(0)
