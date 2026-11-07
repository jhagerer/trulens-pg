"""Shared database access for the benchmark repository and the metric store."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Self

from sqlalchemy import Engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from trulens_pg.bench.models import SCHEMA_PLACEHOLDER
from trulens_pg.config import DEFAULT_BENCH_SCHEMA, Settings, validate_schema_name
from trulens_pg.errors import ReadOnlyError

READ_ONLY_SQLSTATE = "TPG01"


def _sqlstate(exc: DBAPIError) -> str | None:
    orig = exc.orig
    return getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)


class BenchDB:
    """Base class: one engine, one schema, one transaction per public method."""

    def __init__(self, engine: Engine, bench_schema: str = DEFAULT_BENCH_SCHEMA) -> None:
        self.bench_schema = validate_schema_name(bench_schema)
        self.raw_engine = engine
        self.engine = engine.execution_options(
            schema_translate_map={SCHEMA_PLACEHOLDER: self.bench_schema}
        )
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @classmethod
    def from_context(cls) -> Self:
        """Use the engine and schema of the last `trulens_pg.connect()` call."""
        from trulens_pg.session import get_context

        ctx = get_context()
        return cls(ctx.engine, ctx.bench_schema)

    @classmethod
    def from_url(cls, database_url: str | None = None, bench_schema: str | None = None) -> Self:
        """Create a store without a TruSession (CLI, Streamlit app)."""
        from trulens_pg.errors import NotConnectedError
        from trulens_pg.session import create_bench_engine

        settings = Settings.from_env()
        url = database_url or settings.database_url
        if url is None:
            raise NotConnectedError("No database URL: pass one or set TRULENS_PG_DATABASE_URL.")
        return cls(create_bench_engine(url), bench_schema or settings.bench_schema)

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        """One database transaction; trigger errors become `ReadOnlyError`."""
        try:
            with self._sessions.begin() as session:
                yield session
        except DBAPIError as exc:
            if _sqlstate(exc) == READ_ONLY_SQLSTATE:
                message = str(exc.orig).splitlines()[0] if exc.orig else str(exc)
                raise ReadOnlyError(message) from exc
            raise

    def quote(self, name: str) -> str:
        """Quote an identifier with the dialect's rules."""
        return self.raw_engine.dialect.identifier_preparer.quote_identifier(name)
