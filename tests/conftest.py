"""Shared fixtures.

Integration tests (marker ``pg``) need ``TRULENS_PG_TEST_DATABASE_URL``, for
example ``postgresql+psycopg://postgres:postgres@localhost:5432/trulens_pg_test``.
They are skipped when the variable is not set. The database is reset.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("TRULENS_OTEL_TRACING", "1")

TEST_URL = os.environ.get("TRULENS_PG_TEST_DATABASE_URL")
TEST_SCHEMA = "bench_test"


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
    import trulens_pg as tpg
    from trulens_pg import migrations
    from trulens_pg.session import create_bench_engine

    engine = create_bench_engine(pg_url)
    if migrations.current_revision(engine, TEST_SCHEMA) is not None:
        migrations.downgrade(engine, TEST_SCHEMA, "base")
    engine.dispose()

    session = tpg.connect(pg_url, bench_schema=TEST_SCHEMA, migrate=True)
    session.reset_database()
    return session


@pytest.fixture
def ctx(pg_session: Any) -> Any:
    """Clean bench tables before each test; yields the PgContext."""
    from sqlalchemy import text

    from trulens_pg.session import get_context

    context = get_context()
    with context.engine.begin() as conn:
        views = conn.execute(
            text("SELECT viewname FROM pg_views WHERE schemaname = :s"), {"s": TEST_SCHEMA}
        ).scalars()
        for view in list(views):
            conn.execute(text(f'DROP VIEW "{TEST_SCHEMA}"."{view}"'))
        conn.execute(
            text(
                f'TRUNCATE "{TEST_SCHEMA}".run_metric, "{TEST_SCHEMA}".metric_set, '
                f'"{TEST_SCHEMA}".metric_calibration, "{TEST_SCHEMA}".metric_label, '
                f'"{TEST_SCHEMA}".metric_version, "{TEST_SCHEMA}".metric, '
                f'"{TEST_SCHEMA}".run_link, "{TEST_SCHEMA}".review, "{TEST_SCHEMA}".item, '
                f'"{TEST_SCHEMA}".dataset_version, "{TEST_SCHEMA}".document, '
                f'"{TEST_SCHEMA}".corpus, "{TEST_SCHEMA}".dataset CASCADE'
            )
        )
    return context


@pytest.fixture
def repo(ctx: Any) -> Any:
    from trulens_pg.bench import Repository

    return Repository(ctx.engine, ctx.bench_schema)


@pytest.fixture
def store(ctx: Any) -> Any:
    from trulens_pg.metrics import MetricStore

    return MetricStore(ctx.engine, ctx.bench_schema)


SAMPLE_ITEMS = [
    {
        "external_id": f"q{i:03d}",
        "query": f"Question {i} about motor insurance?",
        "ground_truth_answer": f"Answer {i}, see clause KFZ-10{i:02d}.",
        "ground_truth_context_ids": [f"c{i % 3}"],
        "split": "test" if i % 2 else "dev",
        "tags": ["motor"] if i % 3 else ["home"],
        "difficulty": "easy",
        "language": "en",
        "contains_pii": i == 4,
    }
    for i in range(1, 7)
]

SAMPLE_DOCS = [
    {"chunk_id": f"c{i}", "text": f"Clause text {i}", "title": f"Clause {i}"} for i in range(3)
]


@pytest.fixture
def published(repo: Any) -> Any:
    """A published version ``claims_faq@1.0.0`` with 6 items and a corpus."""
    repo.create_dataset("claims_faq", domain="insurance", owner="tester")
    repo.create_corpus("policies")
    repo.add_documents("policies", SAMPLE_DOCS)
    draft = repo.create_draft("claims_faq", copy_from=None, created_by="tester")
    repo.set_corpus(draft.version_id, "policies")
    repo.add_items(draft.version_id, SAMPLE_ITEMS)
    return repo.publish_version(draft.version_id, published_by="tester", require_review=False)
