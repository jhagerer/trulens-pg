"""Alembic migrations for the trulens_pg schema.

Only these migrations touch the trulens_pg schema; TruLens migrations keep managing
the TruLens tables. The Alembic version table lives inside the trulens_pg schema.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect

from trulens_pg.config import validate_schema_name
from trulens_pg.errors import SchemaNotReadyError

VERSION_TABLE = "trulens_pg_alembic_version"
_SCRIPT_LOCATION = str(Path(__file__).parent)


def _config(engine: Engine, schema: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.attributes["engine"] = engine
    cfg.attributes["schema"] = validate_schema_name(schema)
    return cfg


def head_revision() -> str:
    """Return the latest revision shipped with this package."""
    script = ScriptDirectory(_SCRIPT_LOCATION)
    head = script.get_current_head()
    if head is None:  # pragma: no cover - the package always ships migrations
        raise SchemaNotReadyError("No migrations found.")
    return head


def current_revision(engine: Engine, schema: str) -> str | None:
    """Return the revision of the trulens_pg schema, or ``None`` if it is not set up."""
    validate_schema_name(schema)
    with engine.connect() as conn:
        if not inspect(conn).has_table(VERSION_TABLE, schema=schema):
            return None
        ctx = MigrationContext.configure(
            conn, opts={"version_table": VERSION_TABLE, "version_table_schema": schema}
        )
        return ctx.get_current_revision()


def upgrade(engine: Engine, schema: str, revision: str = "head") -> None:
    """Create or upgrade the trulens_pg schema to ``revision``."""
    command.upgrade(_config(engine, schema), revision)


def downgrade(engine: Engine, schema: str, revision: str) -> None:
    """Downgrade the trulens_pg schema to ``revision`` (``base`` removes all tables)."""
    command.downgrade(_config(engine, schema), revision)


def check(engine: Engine, schema: str) -> None:
    """Raise `SchemaNotReadyError` unless the trulens_pg schema is at the head revision."""
    current = current_revision(engine, schema)
    head = head_revision()
    if current != head:
        state = "is missing" if current is None else f"is at revision {current}"
        raise SchemaNotReadyError(
            f"The trulens_pg schema {schema!r} {state}; expected revision {head}. "
            f"Run `trulens-pg db upgrade` or call connect(..., migrate=True)."
        )
