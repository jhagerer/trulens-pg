"""Database access of the metric store: one engine, one schema, one transaction per call."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self

from sqlalchemy import Engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from trulens_pg.config import DEFAULT_SCHEMA, Settings, validate_schema_name
from trulens_pg.errors import ReadOnlyError
from trulens_pg.metrics.models import SCHEMA_PLACEHOLDER

if TYPE_CHECKING:
    from trulens_pg.metrics.prompts import PromptRegistry

READ_ONLY_SQLSTATE = "TPG01"


def _sqlstate(exc: DBAPIError) -> str | None:
    orig = exc.orig
    return getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)


class StoreDB:
    """Base class: one engine, one schema, one transaction per public method."""

    def __init__(
        self,
        engine: Engine,
        schema: str = DEFAULT_SCHEMA,
        *,
        trulens_db: Any | None = None,
    ) -> None:
        self.schema = validate_schema_name(schema)
        self._trulens_db = trulens_db
        self._prompts: PromptRegistry | None = None
        self.raw_engine = engine
        self.engine = engine.execution_options(
            schema_translate_map={SCHEMA_PLACEHOLDER: self.schema}
        )
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @classmethod
    def from_context(cls) -> Self:
        """Use the engine and schema of the last `trulens_pg.connect()` call."""
        from trulens_pg.session import get_context

        ctx = get_context()
        return cls(ctx.engine, ctx.schema, trulens_db=ctx.session.connector.db)

    @classmethod
    def from_url(cls, database_url: str | None = None, schema: str | None = None) -> Self:
        """Create a store without a TruSession (CLI, scripts)."""
        from trulens_pg.errors import NotConnectedError
        from trulens_pg.session import create_store_engine

        settings = Settings.from_env()
        url = database_url or settings.database_url
        if url is None:
            raise NotConnectedError("No database URL: pass one or set TRULENS_PG_DATABASE_URL.")
        return cls(create_store_engine(url), schema or settings.schema)

    @property
    def prompts(self) -> PromptRegistry:
        """TruLens prompt registry on the same database (opened lazily)."""
        from trulens_pg.metrics.prompts import PromptRegistry

        if self._prompts is None:
            self._prompts = (
                PromptRegistry(self._trulens_db)
                if self._trulens_db is not None
                else PromptRegistry.from_engine(self.raw_engine)
            )
        return self._prompts

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
