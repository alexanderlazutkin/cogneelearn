"""CLI entrypoint for ingestion: build TPC-H DB and/or load documents.

Examples
--------
Build TPC-H and ingest it into Cognee::

    cogneelearn-ingest tpch --sf 0.01

Ingest a folder of project documents::

    cogneelearn-ingest docs data/docs

Rebuild TPC-H and ingest docs in one go::

    cogneelearn-ingest all --sf 0.01 --docs data/docs

Prune everything before re-ingesting (after switching models)::

    cogneelearn-ingest tpch --sf 0.01 --prune
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .. import pipeline as P

logger = logging.getLogger("cogneelearn.ingest.cli")

DEFAULT_DB_PATH = "data/tpch/tpch.duckdb"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cogneelearn-ingest",
        description="Ingest DuckDB/TPC-H and project documents into Cognee.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tpch = sub.add_parser("tpch", help="Build/ingest a TPC-H DuckDB database.")
    p_tpch.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB file path.")
    p_tpch.add_argument("--sf", type=float, default=0.01, help="TPC-H scale factor.")
    p_tpch.add_argument("--no-cognify", action="store_true", help="Only add, skip cognify.")
    p_tpch.add_argument("--prune", action="store_true", help="Prune Cognee before ingest.")
    p_tpch.add_argument(
        "--overwrite", action="store_true", help="Rebuild the .duckdb file even if it exists."
    )

    p_docs = sub.add_parser("docs", help="Ingest project documents.")
    p_docs.add_argument("source", help="File or directory of documents.")
    p_docs.add_argument("--no-cognify", action="store_true", help="Only add, skip cognify.")
    p_docs.add_argument("--prune", action="store_true", help="Prune Cognee before ingest.")

    p_all = sub.add_parser("all", help="Ingest TPC-H then documents.")
    p_all.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB file path.")
    p_all.add_argument("--sf", type=float, default=0.01, help="TPC-H scale factor.")
    p_all.add_argument("--docs", default="data/docs", help="Documents directory.")
    p_all.add_argument("--prune", action="store_true", help="Prune Cognee before ingest.")
    p_all.add_argument(
        "--overwrite", action="store_true", help="Rebuild the .duckdb file even if it exists."
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)

    if getattr(args, "prune", False):
        P.run(P.prune_all(metadata=True))

    rc = 0
    if args.command in ("tpch", "all"):
        db_path = P.run(P.ensure_tpch_db(args.db, sf=args.sf, overwrite=args.overwrite))
        no_cog = getattr(args, "no_cognify", False)
        res = P.run(P.ingest_tpch(db_path, cognify=not no_cog))
        _report("TPC-H", res)
        rc = rc or (0 if res.ok else 1)

    if args.command in ("docs", "all"):
        source = getattr(args, "docs", None) if args.command == "all" else args.source
        no_cog = getattr(args, "no_cognify", False)
        if not Path(source).exists():
            logger.error("Source not found: %s", source)
            return 1
        res = P.run(P.ingest_documents(source, cognify=not no_cog))
        _report("Docs", res)
        rc = rc or (0 if res.ok else 1)

    return rc


def _report(label: str, res: P.IngestResult) -> None:
    status = "OK" if res.ok else "FAIL"
    print(
        f"[{label}] {status}: {res.items_added} items -> dataset '{res.dataset}'"
        + (f" ({res.error})" if res.error else "")
    )


if __name__ == "__main__":
    sys.exit(main())
