"""Shared fixtures.

Integration tests (marker ``pg``) need ``TRULENS_PG_TEST_DATABASE_URL``, for
example ``postgresql+psycopg://postgres:postgres@localhost:5432/trulens_pg_test``.
They are skipped when the variable is not set. The TruLens tables are reset and
the schema ``tpg_eval`` is recreated.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("TRULENS_OTEL_TRACING", "1")

TEST_URL = os.environ.get("TRULENS_PG_TEST_DATABASE_URL")
EVAL_SCHEMA = "tpg_eval"

SAMPLE_ROWS = [
    {
        "item_id": f"q{i:03d}",
        "question": f"Question {i} about motor insurance?",
        "answer": f"Answer {i}, see clause KFZ-10{i:02d}.",
        "split": "test" if i % 2 else "dev",
        "tags": ["motor"] if i % 3 else ["home"],
    }
    for i in range(1, 7)
]


@pytest.fixture(scope="session")
def pg_url() -> str:
    if not TEST_URL:
        pytest.skip("TRULENS_PG_TEST_DATABASE_URL is not set")
    return TEST_URL


@pytest.fixture(scope="session")
def pg_session(pg_url: str) -> Any:
    """One TruSession per test session (TruSession is a singleton)."""
    import trulens_pg as tpg

    session = tpg.connect(pg_url)
    session.reset_database()
    return session


@pytest.fixture
def ctx(pg_session: Any) -> Any:
    """Recreate the schema tpg_eval before each test; returns the PgContext."""
    from sqlalchemy import text

    from trulens_pg.session import get_context

    context = get_context()
    with context.engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{EVAL_SCHEMA}" CASCADE'))
        conn.execute(text(f'CREATE SCHEMA "{EVAL_SCHEMA}"'))
    return context


@pytest.fixture
def claims_table(ctx: Any) -> str:
    """The table ``tpg_eval.claims_faq`` with 6 questions; returns its name."""
    from sqlalchemy import text

    with ctx.engine.begin() as conn:
        conn.execute(
            text(
                f"CREATE TABLE {EVAL_SCHEMA}.claims_faq (item_id text, question text, "
                "answer text, split text, tags text[])"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO {EVAL_SCHEMA}.claims_faq VALUES "
                "(:item_id, :question, :answer, :split, :tags)"
            ),
            SAMPLE_ROWS,
        )
    return f"{EVAL_SCHEMA}.claims_faq"
