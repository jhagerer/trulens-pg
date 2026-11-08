"""`connect()`: a TruLens ``TruSession`` on Postgres with a synchronous driver."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.engine import URL, make_url

from trulens_pg.errors import NotConnectedError, UnsupportedDatabaseError

logger = logging.getLogger(__name__)

SUPPORTED_DRIVERS = ("psycopg", "psycopg2")
DATABASE_URL_ENV = "TRULENS_PG_DATABASE_URL"


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
    """State shared by `TruApp`, `attach()` and `PgRunDao`."""

    session: Any
    engine: Engine


_CONTEXT: PgContext | None = None


def get_context() -> PgContext:
    """Return the context created by the last `connect()` call."""
    if _CONTEXT is None:
        raise NotConnectedError("Call trulens_pg.connect(database_url) first.")
    return _CONTEXT


def set_context(context: PgContext | None) -> None:
    """Replace the current context (used by tests)."""
    global _CONTEXT  # noqa: PLW0603 - one context per process, like TruSession
    _CONTEXT = context


def connect(database_url: str | URL | None = None, **session_kwargs: Any) -> Any:
    """Create a TruLens ``TruSession`` on Postgres.

    Args:
        database_url: SQLAlchemy URL; defaults to ``TRULENS_PG_DATABASE_URL``.
        **session_kwargs: Passed to ``TruSession`` (for example
            ``database_redact_keys=True``).

    Returns:
        The ``TruSession``.
    """
    from trulens.core import TruSession

    raw_url = database_url or os.environ.get(DATABASE_URL_ENV)
    if not raw_url:
        raise NotConnectedError(f"No database URL: pass one or set {DATABASE_URL_ENV}.")
    url = normalize_database_url(raw_url)
    session = TruSession(database_url=url.render_as_string(hide_password=False), **session_kwargs)
    set_context(PgContext(session=session, engine=session.connector.db.engine))
    logger.info("trulens-pg connected to %s", url.render_as_string())
    return session
