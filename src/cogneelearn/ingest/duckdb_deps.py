"""Extract DuckDB schema and object dependencies for Cognee ingestion.

The module turns a file-backed DuckDB database into a list of human-readable
documents (one per table + one per view + one per FK relationship) that Cognee
ingests into its knowledge graph. Text is intentionally declarative so the LLM
can extract entities (tables, columns) and relations (references, depends_on).

Usage::

    from cogneelearn.ingest.duckdb_deps import build_tpch_db, serialize_for_cognee

    db_path = build_tpch_db("data/tpch/tpch.duckdb", sf=0.01)
    docs = serialize_for_cognee(db_path)
    # -> feed each doc.text into cognee.add()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

# ─── TPC-H schema (specification) ─────────────────────────────────────────────
# DuckDB's dbgen() creates the 8 base tables and their data but does NOT declare
# primary/foreign-key constraints (verified against duckdb 1.5.4 —
# duckdb_constraints() returns no PK/FK rows for a dbgen database). We therefore
# record the canonical TPC-H relationships here so the knowledge graph still
# reflects the real schema. For any other DuckDB database that *does* declare
# constraints, they are read dynamically from duckdb_constraints().
TPCH_TABLES = [
    "region",
    "nation",
    "part",
    "supplier",
    "partsupp",
    "customer",
    "orders",
    "lineitem",
]

# Each entry: (child_table, child_columns, parent_table, parent_columns)
# Mirrors the TPC-H benchmark specification v2.18.1 §1.4.
TPCH_FOREIGN_KEYS: list[tuple[str, list[str], str, list[str]]] = [
    ("nation", ["n_regionkey"], "region", ["r_regionkey"]),
    ("customer", ["c_nationkey"], "nation", ["n_nationkey"]),
    ("supplier", ["s_nationkey"], "nation", ["n_nationkey"]),
    ("orders", ["o_custkey"], "customer", ["c_custkey"]),
    ("partsupp", ["ps_partkey"], "part", ["p_partkey"]),
    ("partsupp", ["ps_suppkey"], "supplier", ["s_suppkey"]),
    ("lineitem", ["l_orderkey"], "orders", ["o_orderkey"]),
    (
        "lineitem",
        ["l_partkey", "l_suppkey"],
        "partsupp",
        ["ps_partkey", "ps_suppkey"],
    ),
]


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_nullable: bool
    default: str | None = None
    comment: str | None = None


@dataclass
class TableInfo:
    name: str
    schema: str
    columns: list[ColumnInfo] = field(default_factory=list)
    comment: str | None = None
    sql: str | None = None
    row_count: int | None = None


@dataclass
class ForeignKey:
    child_table: str
    child_columns: list[str]
    parent_table: str
    parent_columns: list[str]
    constraint_name: str | None = None


@dataclass
class ViewInfo:
    name: str
    schema: str
    sql: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class SchemaSnapshot:
    tables: list[TableInfo] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    views: list[ViewInfo] = field(default_factory=list)

    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]


# ─── DB creation ──────────────────────────────────────────────────────────────
def build_tpch_db(db_path: str | Path, sf: float = 0.01, overwrite: bool = True) -> Path:
    """Create a file-backed DuckDB database populated with the TPC-H dataset.

    Parameters
    ----------
    db_path:
        File path for the DuckDB database. A parent directory must exist.
        Must be a real file path — ``:memory:`` is rejected because the whole
        point of this project is a persistent, inspectable database.
    sf:
        Scale factor passed to ``dbgen``. ``0.01`` (~10MB) is enough to
        exercise the schema without bloating the repo; raise it for real loads.
    overwrite:
        If True and the file exists, it is deleted first so dbgen runs against
        an empty database.

    Returns
    -------
    Path
        The resolved database file path.
    """
    db_path = Path(db_path)
    if str(db_path) in (":memory:", "memory"):
        raise ValueError("build_tpch_db requires a file path, not :memory:")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and db_path.exists():
        db_path.unlink()
        # DuckDB also writes a <name>.wal next to the main file; clean it too.
        wal = db_path.with_suffix(db_path.suffix + ".wal")
        if wal.exists():
            wal.unlink()

    con = duckdb.connect(str(db_path))
    try:
        logger.info("Generating TPC-H (sf=%s) into %s", sf, db_path)
        con.execute("CALL dbgen(sf = ?)", [sf])
    finally:
        con.close()
    return db_path


# ─── Schema extraction ───────────────────────────────────────────────────────
def extract_schema(db_path: str | Path, schema: str = "main") -> SchemaSnapshot:
    """Read tables, columns, foreign keys and views from a DuckDB file."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = _extract_tables(con, schema)
        fks = _extract_foreign_keys(con, schema)
        views = _extract_views(con, schema)
    finally:
        con.close()

    snapshot = SchemaSnapshot(tables=tables, foreign_keys=fks, views=views)

    # TPC-H databases built by dbgen() carry no FK constraints in metadata, so
    # backfill the canonical relationships when the schema looks like TPC-H.
    _backfill_tpch_foreign_keys(snapshot)
    return snapshot


def _extract_tables(con: duckdb.DuckDBPyConnection, schema: str) -> list[TableInfo]:
    rows = con.execute(
        """
        SELECT table_name, comment, sql
        FROM duckdb_tables()
        WHERE schema_name = ? AND NOT internal
        ORDER BY table_name
        """,
        [schema],
    ).fetchall()

    tables: list[TableInfo] = []
    for table_name, comment, sql in rows:
        col_rows = con.execute(
            """
            SELECT column_name, data_type, is_nullable, column_default, comment
            FROM duckdb_columns()
            WHERE schema_name = ? AND table_name = ?
            ORDER BY column_index
            """,
            [schema, table_name],
        ).fetchall()
        columns = [
            ColumnInfo(
                name=col_name,
                data_type=data_type,
                is_nullable=bool(is_nullable),
                default=default,
                comment=comment,
            )
            for col_name, data_type, is_nullable, default, comment in col_rows
        ]

        row_count: int | None = None
        try:
            row_count = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{table_name}"').fetchone()[0]
        except duckdb.Error as exc:
            logger.debug("Could not count rows of %s: %s", table_name, exc)

        tables.append(
            TableInfo(
                name=table_name,
                schema=schema,
                columns=columns,
                comment=comment,
                sql=sql,
                row_count=row_count,
            )
        )
    return tables


def _extract_foreign_keys(con: duckdb.DuckDBPyConnection, schema: str) -> list[ForeignKey]:
    """Read FK constraints declared in the database.

    Works for any DuckDB database that declares real foreign keys. TPC-H
    databases built by ``dbgen()`` declare none — see ``_backfill_tpch_foreign_keys``.
    """
    rows = con.execute(
        """
        SELECT table_name, constraint_name,
               constraint_column_names, referenced_table, referenced_column_names
        FROM duckdb_constraints()
        WHERE schema_name = ? AND constraint_type = 'FK'
        ORDER BY table_name, constraint_index
        """,
        [schema],
    ).fetchall()
    fks: list[ForeignKey] = []
    for table_name, constraint_name, child_cols, parent_table, parent_cols in rows:
        if not parent_table:
            continue
        fks.append(
            ForeignKey(
                child_table=table_name,
                child_columns=list(child_cols or []),
                parent_table=parent_table,
                parent_columns=list(parent_cols or []),
                constraint_name=constraint_name,
            )
        )
    return fks


def _extract_views(con: duckdb.DuckDBPyConnection, schema: str) -> list[ViewInfo]:
    rows = con.execute(
        """
        SELECT view_name, sql
        FROM duckdb_views()
        WHERE schema_name = ? AND NOT internal
        ORDER BY view_name
        """,
        [schema],
    ).fetchall()
    # Skip DuckDB's built-in inspect views that live in `main` temporarily.
    builtin = {
        "duckdb_columns",
        "duckdb_constraints",
        "duckdb_databases",
        "duckdb_indexes",
        "duckdb_logs",
        "duckdb_schemas",
        "duckdb_tables",
        "duckdb_types",
        "duckdb_views",
        "pragma_database_list",
        "sqlite_master",
        "sqlite_schema",
        "sqlite_temp_master",
        "sqlite_temp_schema",
    }
    views: list[ViewInfo] = []
    for view_name, sql in rows:
        if view_name in builtin or not sql:
            continue
        depends_on = _parse_view_dependencies(sql)
        views.append(ViewInfo(name=view_name, schema=schema, sql=sql, depends_on=depends_on))
    return views


def _parse_view_dependencies(sql: str) -> list[str]:
    """Best-effort extraction of table names referenced in a view's SQL.

    Resolves table names with a permissive scan for ``FROM``/``JOIN`` clauses.
    Schema-qualified names (``main.orders``) are reduced to the bare table.
    This is intentionally conservative — false positives are preferable here
    because they only add extra edges to the graph.
    """
    deps: list[str] = []
    pattern = re.compile(
        r"(?:FROM|JOIN)\s+(?:\"(?P<q1>[^\"]+)\"|(?P<q2>[\w.]+))",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sql):
        name = match.group("q1") or match.group("q2")
        if not name:
            continue
        bare = name.split(".")[-1].strip('"')
        # Drop CTE aliases / subquery noise.
        if bare and bare not in deps and bare not in {"(", "select"}:
            deps.append(bare)
    return deps


def _backfill_tpch_foreign_keys(snapshot: SchemaSnapshot) -> None:
    """Add canonical TPC-H FKs when the schema matches the TPC-H table set.

    dbgen() does not declare constraints, so a freshly built TPC-H database has
    zero FKs in metadata. Detect the TPC-H table set by name and inject the
    spec-defined relationships so the knowledge graph is still correct.
    Existing (dynamically read) FKs are kept; deduplicated by signature.
    """
    present = {t.name for t in snapshot.tables}
    tpch_set = set(TPCH_TABLES)
    if not tpch_set.issubset(present):
        return  # Not a TPC-H schema; leave FKs as read from metadata.

    existing = {
        (fk.child_table, tuple(fk.child_columns), fk.parent_table, tuple(fk.parent_columns))
        for fk in snapshot.foreign_keys
    }
    for child, child_cols, parent, parent_cols in TPCH_FOREIGN_KEYS:
        sig = (child, tuple(child_cols), parent, tuple(parent_cols))
        if sig in existing:
            continue
        snapshot.foreign_keys.append(
            ForeignKey(
                child_table=child,
                child_columns=list(child_cols),
                parent_table=parent,
                parent_columns=list(parent_cols),
                constraint_name=f"tpch_fk_{child}_{parent}",
            )
        )
        existing.add(sig)


# ─── Serialization for Cognee ────────────────────────────────────────────────
@dataclass
class CogneeDoc:
    """A single document to ingest into Cognee."""

    id: str
    kind: str  # "table" | "foreign_key" | "view"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def serialize_for_cognee(db_path: str | Path, schema: str = "main") -> list[CogneeDoc]:
    """Build Cognee-ready documents describing every object in the database.

    Each document is a short, declarative paragraph that the LLM turns into
    graph entities and relationships during ``cognify()``.
    """
    snapshot = extract_schema(db_path, schema=schema)
    docs: list[CogneeDoc] = []

    for table in snapshot.tables:
        docs.append(_serialize_table(table, snapshot))
    for fk in snapshot.foreign_keys:
        docs.append(_serialize_foreign_key(fk))
    for view in snapshot.views:
        docs.append(_serialize_view(view))
    return docs


def _serialize_table(table: TableInfo, snapshot: SchemaSnapshot) -> CogneeDoc:
    col_lines: list[str] = []
    for col in table.columns:
        parts = [f"{col.name} ({col.data_type})"]
        if not col.is_nullable:
            parts.append("NOT NULL")
        if col.default is not None:
            parts.append(f"DEFAULT {col.default}")
        if col.comment:
            parts.append(f"-- {col.comment}")
        col_lines.append("  - " + ", ".join(parts))

    outgoing = [fk for fk in snapshot.foreign_keys if fk.child_table == table.name]
    incoming = [fk for fk in snapshot.foreign_keys if fk.parent_table == table.name]

    rel_lines: list[str] = []
    for fk in outgoing:
        rel_lines.append(
            f"  - references {fk.parent_table} via "
            f"{_join(fk.child_columns)} -> {_join(fk.parent_columns)}"
        )
    for fk in incoming:
        rel_lines.append(
            f"  - referenced by {fk.child_table} via "
            f"{_join(fk.child_columns)} -> {_join(fk.parent_columns)}"
        )

    text = f"Table {table.name} (schema: {table.schema})"
    if table.comment:
        text += f" — {table.comment}"
    if table.row_count is not None:
        text += f" [{table.row_count} rows]"
    text += "\nColumns:\n" + "\n".join(col_lines)
    if rel_lines:
        text += "\nRelationships:\n" + "\n".join(rel_lines)

    return CogneeDoc(
        id=f"table:{table.name}",
        kind="table",
        text=text,
        metadata={
            "table_name": table.name,
            "schema": table.schema,
            "column_count": len(table.columns),
            "row_count": table.row_count,
        },
    )


def _serialize_foreign_key(fk: ForeignKey) -> CogneeDoc:
    text = (
        f"Foreign key relationship: table {fk.child_table} references table "
        f"{fk.parent_table}. "
        f"Columns: {_join(fk.child_columns)} -> {_join(fk.parent_columns)}."
    )
    return CogneeDoc(
        id=f"fk:{fk.child_table}->{fk.parent_table}",
        kind="foreign_key",
        text=text,
        metadata={
            "child_table": fk.child_table,
            "parent_table": fk.parent_table,
            "child_columns": fk.child_columns,
            "parent_columns": fk.parent_columns,
        },
    )


def _serialize_view(view: ViewInfo) -> CogneeDoc:
    deps = ", ".join(view.depends_on) if view.depends_on else "none"
    text = f"View {view.name} (schema: {view.schema}) depends on tables: {deps}.\nSQL:\n{view.sql}"
    return CogneeDoc(
        id=f"view:{view.name}",
        kind="view",
        text=text,
        metadata={
            "view_name": view.name,
            "depends_on": view.depends_on,
        },
    )


def _join(items: list[str]) -> str:
    return ", ".join(items) if len(items) != 1 else items[0]
