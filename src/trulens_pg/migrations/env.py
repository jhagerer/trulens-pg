"""Alembic environment for the trulens_pg schema (used programmatically only)."""

from __future__ import annotations

from alembic import context
from sqlalchemy import Engine, text

from trulens_pg.migrations import VERSION_TABLE

config = context.config
engine: Engine = config.attributes["engine"]
schema: str = config.attributes["schema"]


def run_migrations_online() -> None:
    with engine.connect() as connection:
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        connection.commit()
        context.configure(
            connection=connection,
            version_table=VERSION_TABLE,
            version_table_schema=schema,
            transaction_per_migration=True,
        )
        with context.begin_transaction():
            context.run_migrations(schema=schema)


if context.is_offline_mode():  # pragma: no cover
    raise RuntimeError("Offline migrations are not supported.")
run_migrations_online()
