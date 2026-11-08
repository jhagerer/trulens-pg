"""Shared fixtures.

Integration tests (marker ``pg``) need ``TRULENS_PG_TEST_DATABASE_URL``, for
example ``postgresql+psycopg://postgres:postgres@localhost:5432/trulens_pg_test``.
They are skipped when the variable is not set. The database is reset, and the
schemas ``tpg_test`` and ``tpg_eval`` are recreated.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("TRULENS_OTEL_TRACING", "1")

TEST_URL = os.environ.get("TRULENS_PG_TEST_DATABASE_URL")
TEST_SCHEMA = "tpg_test"


class FakeJudge:
    """Offline stand-in for a TruLens LLM provider.

    Scores 3 (of 0..3) when the user prompt mentions a clause number such as
    ``KFZ-1001``, otherwise 1. Records every call.
    """

    model_engine = "fake-judge-1"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_score_and_reasons(
        self,
        system_prompt: str,
        user_prompt: str | None = None,
        min_score_val: int = 0,
        max_score_val: int = 10,
        temperature: float = 0.0,
    ) -> tuple[float, dict[str, str]]:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        raw = max_score_val if user_prompt and "-10" in user_prompt else min_score_val + 1
        return (raw - min_score_val) / (max_score_val - min_score_val), {"reason": f"raw={raw}"}

    def correctness_with_cot_reasons(
        self,
        text: str,
        criteria: str | None = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> tuple[float, dict[str, str]]:
        self.calls.append({"text": text, "criteria": criteria})
        return (1.0 if "KFZ" in (text or "") else 0.0), {"reason": "fake"}


@pytest.fixture
def fake_judge() -> FakeJudge:
    return FakeJudge()


@pytest.fixture(scope="session")
def pg_url() -> str:
    if not TEST_URL:
        pytest.skip("TRULENS_PG_TEST_DATABASE_URL is not set")
    return TEST_URL


@pytest.fixture(scope="session")
def pg_session(pg_url: str) -> Any:
    """One TruSession per test session (TruSession is a singleton)."""
    from sqlalchemy import text

    import trulens_pg as tpg
    from trulens_pg.session import create_store_engine

    engine = create_store_engine(pg_url)
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{TEST_SCHEMA}" CASCADE'))
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{EVAL_SCHEMA}" CASCADE'))
    engine.dispose()

    session = tpg.connect(pg_url, schema=TEST_SCHEMA, migrate=True)
    session.reset_database()
    return session


@pytest.fixture
def ctx(pg_session: Any) -> Any:
    """Clean the metric tables and the evaluation table before each test."""
    from sqlalchemy import text

    from trulens_pg.session import get_context

    context = get_context()
    with context.engine.begin() as conn:
        conn.execute(
            text(
                f'TRUNCATE "{TEST_SCHEMA}".run_metric, "{TEST_SCHEMA}".metric_set, '
                f'"{TEST_SCHEMA}".metric_calibration, "{TEST_SCHEMA}".metric_label, '
                f'"{TEST_SCHEMA}".metric_version, "{TEST_SCHEMA}".metric CASCADE'
            )
        )
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{EVAL_SCHEMA}" CASCADE'))
        conn.execute(text(f'CREATE SCHEMA "{EVAL_SCHEMA}"'))
    return context


@pytest.fixture
def store(ctx: Any) -> Any:
    from trulens_pg.metrics import MetricStore

    return MetricStore(ctx.engine, ctx.schema, trulens_db=ctx.session.connector.db)


EVAL_SCHEMA = "tpg_eval"

SAMPLE_ROWS = [
    {
        "item_id": f"q{i:03d}",
        "question": f"Question {i} about motor insurance?",
        "answer": f"Answer {i}, see clause KFZ-10{i:02d}.",
        "context_ids": [f"c{i % 3}"],
        "split": "test" if i % 2 else "dev",
        "tags": ["motor"] if i % 3 else ["home"],
    }
    for i in range(1, 7)
]

CHUNK_TEXTS = {f"c{i}": f"Clause text {i}" for i in range(3)}

GOLDEN_SET = [
    {
        "query": row["question"],
        "expected_response": row["answer"],
        "expected_chunks": [{"text": CHUNK_TEXTS[c]} for c in row["context_ids"]],
    }
    for row in SAMPLE_ROWS
]


@pytest.fixture
def claims_table(ctx: Any) -> str:
    """The table ``tpg_eval.claims_faq`` with 6 questions; returns its name."""
    from sqlalchemy import text

    with ctx.engine.begin() as conn:
        conn.execute(
            text(
                f"CREATE TABLE {EVAL_SCHEMA}.claims_faq (item_id text, question text, "
                "answer text, context_ids text[], split text, tags text[])"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO {EVAL_SCHEMA}.claims_faq VALUES "
                "(:item_id, :question, :answer, :context_ids, :split, :tags)"
            ),
            SAMPLE_ROWS,
        )
    return f"{EVAL_SCHEMA}.claims_faq"
