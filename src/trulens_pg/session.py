"""`connect()`: TruSession on Postgres, driver check and bench schema check."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url

from trulens_pg import migrations
from trulens_pg.config import DEFAULT_BENCH_SCHEMA, Settings, validate_schema_name
from trulens_pg.errors import NotConnectedError, UnsupportedDatabaseError

logger = logging.getLogger(__name__)

SUPPORTED_DRIVERS = ("psycopg", "psycopg2")


def normalize_database_url(database_url: str | URL) -> URL:
    """Validate a Postgres URL and return it with an explicit synchronous driver.

    ``postgresql://`` becomes ``postgresql+psycopg://`` (psycopg 3 is a core
    dependency). Asynchronous drivers such as asyncpg are rejected, because
    TruLens 2.16 uses synchronous SQLAlchemy sessions.
    """
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise UnsupportedDatabaseError(
            f"trulens-pg supports PostgreSQL only, got dialect {url.get_backend_name()!r}."
        )
    driver = url.get_driver_name() if "+" in url.drivername else None
    if driver is None:
        return url.set(drivername="postgresql+psycopg")
    if driver not in SUPPORTED_DRIVERS:
        raise UnsupportedDatabaseError(
            f"Driver {driver!r} is not supported: TruLens needs a synchronous driver. "
            f"Use postgresql+psycopg:// (psycopg 3) or postgresql+psycopg2://."
        )
    return url


@dataclass
class PgContext:
    """State shared by `wrap()`, `PgRunDao`, the repository and the metric store."""

    session: Any
    engine: Engine
    bench_schema: str = DEFAULT_BENCH_SCHEMA

    @property
    def bench_engine(self) -> Engine:
        """Engine whose ORM statements map the placeholder schema to `bench_schema`."""
        return self.engine.execution_options(schema_translate_map={"bench": self.bench_schema})


_CONTEXT: PgContext | None = None


def get_context() -> PgContext:
    """Return the context created by the last `connect()` call."""
    if _CONTEXT is None:
        raise NotConnectedError("Call trulens_pg.connect(database_url) first.")
    return _CONTEXT


def set_context(context: PgContext | None) -> None:
    """Replace the current context (used by tests and the Streamlit app)."""
    global _CONTEXT  # noqa: PLW0603 - one context per process, like TruSession
    _CONTEXT = context


def create_bench_engine(database_url: str | URL, **engine_kwargs: Any) -> Engine:
    """Create a SQLAlchemy engine for the bench schema without a TruSession."""
    engine_kwargs.setdefault("pool_pre_ping", True)
    return create_engine(normalize_database_url(database_url), **engine_kwargs)


def connect(
    database_url: str | URL | None = None,
    bench_schema: str | None = None,
    *,
    migrate: bool = False,
    check_schema: bool = True,
    **session_kwargs: Any,
) -> Any:
    """Create a TruLens ``TruSession`` on Postgres and check the bench schema.

    Args:
        database_url: SQLAlchemy URL; defaults to ``TRULENS_PG_DATABASE_URL``.
        bench_schema: Schema of the benchmark and metric store; defaults to
            ``TRULENS_PG_BENCH_SCHEMA`` or ``bench``.
        migrate: Create or upgrade the bench schema before checking it.
        check_schema: Check that the bench schema is at the latest revision.
            Pass ``False`` to use only the run backend (G1, G2 for plain tables).
        **session_kwargs: Passed to ``TruSession`` (for example
            ``database_redact_keys=True``).

    Returns:
        The ``TruSession``.
    """
    from trulens.core import TruSession

    settings = Settings.from_env()
    raw_url = database_url or settings.database_url
    if raw_url is None:
        raise NotConnectedError("No database URL: pass one or set TRULENS_PG_DATABASE_URL.")
    url = normalize_database_url(raw_url)
    schema = validate_schema_name(bench_schema or settings.bench_schema)

    session = TruSession(database_url=url.render_as_string(hide_password=False), **session_kwargs)
    engine: Engine = session.connector.db.engine

    if migrate:
        migrations.upgrade(engine, schema)
    if check_schema:
        migrations.check(engine, schema)

    set_context(PgContext(session=session, engine=engine, bench_schema=schema))
    logger.info("trulens-pg connected to %s (bench schema %s)", url.render_as_string(), schema)
    return session
