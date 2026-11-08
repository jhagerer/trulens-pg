"""End-to-end tests of the TruLens Run API on Postgres (G1, G2)."""

from __future__ import annotations

import time
import uuid
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import text
from trulens.apps.app import TruApp
from trulens.core import Metric, Selector
from trulens.core.otel.instrument import instrument
from trulens.core.run import RunConfig
from trulens.otel.semconv.trace import SpanAttributes

import trulens_pg as tpg
from trulens_pg.errors import SourceError

pytestmark = pytest.mark.pg


class ClaimsRag:
    """Deterministic RAG app: answers cite clause KFZ-10xx for motor questions."""

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> list[str]:
        return ["Clause text 1", "Clause text 0"]

    @instrument()
    def answer(self, query: str) -> str:
        contexts = self.retrieve(query)
        if "motor" in query:
            return f"Covered under clause KFZ-1001 ({len(contexts)} sources)."
        return "Not covered."


def wait_for_invocation(run: Any, timeout: float = 60.0) -> str:
    """Poll until the invocation is complete (the documented TruLens pattern)."""
    deadline = time.monotonic() + timeout
    while True:
        status = run.get_status().name
        if status.startswith("INVOCATION_COMPLETED") or status == "COMPLETED":
            return str(status)
        if time.monotonic() > deadline:
            raise TimeoutError(status)
        time.sleep(0.5)


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def test_g1_plain_truapp_fails_and_tpg_truapp_fixes_it(ctx: Any) -> None:
    rag = ClaimsRag()
    plain = TruApp(rag, app_name=_unique("plain"), app_version="v1", main_method=rag.answer)
    with pytest.raises(ValueError, match="Run DAO not initialized"):
        plain.add_run(RunConfig(run_name="r", dataset_name="df", dataset_spec={"input": "input"}))

    attached = tpg.attach(plain)
    assert isinstance(attached.run_dao, tpg.PgRunDao)

    tru_app = tpg.TruApp(rag, app_name=_unique("claims"), app_version="v1", main_method=rag.answer)
    assert isinstance(tru_app, TruApp)  # a native TruLens TruApp
    assert isinstance(tru_app.run_dao, tpg.PgRunDao)
    run = tru_app.add_run(
        RunConfig(run_name=_unique("run"), dataset_name="inline", dataset_spec={"input": "input"})
    )
    run.start(input_df=pd.DataFrame({"input": ["motor glass damage?", "home flood?"]}))
    wait_for_invocation(run)

    def length(answer: str) -> float:
        return min(len(answer) / 100, 1.0)

    from trulens.core import Metric, Selector

    metric = Metric(
        implementation=length, name="length", selectors={"answer": Selector.select_record_output()}
    )
    run.compute_metrics([metric])
    records, metrics = ctx.session.get_records_and_feedback(
        app_name=tru_app.app_name, run_name=run.run_name
    )
    assert "length" in metrics
    assert len(records) == 2


def test_g2_table_source_and_column_check(ctx: Any) -> None:
    table = _unique("questions")
    with ctx.engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS tpg_eval"))
        conn.execute(
            text(
                f"CREATE TABLE tpg_eval.{table} (item_id text, input text, "
                "ground_truth_output text, contexts jsonb, tags text[])"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO tpg_eval.{table} VALUES "
                "('1', 'motor glass?', 'yes', '[\"c1\", \"c2\"]', '{motor}'), "
                "('2', 'home flood?', 'no', '[]', '{home}')"
            )
        )
    rag = ClaimsRag()
    tru_app = tpg.TruApp(rag, app_name=_unique("claims"), app_version="v1", main_method=rag.answer)

    with pytest.raises(SourceError, match="lacks column"):
        tru_app.add_run(
            RunConfig(
                run_name=_unique("bad"),
                dataset_name=f"tpg_eval.{table}",
                source_type="TABLE",
                dataset_spec={"input": "question"},
            )
        )
    with pytest.raises(SourceError, match="does not exist"):
        tru_app.add_run(
            RunConfig(
                run_name=_unique("bad"),
                dataset_name="tpg_eval.missing_table",
                source_type="TABLE",
                dataset_spec={"input": "input"},
            )
        )

    df = tru_app.run_dao.fetch_source_data(f"tpg_eval.{table}?tags=motor")
    assert list(df["input"]) == ["motor glass?"]
    assert df.loc[0, "contexts"] == ["c1", "c2"]  # JSONB arrives as a Python list

    run = tru_app.add_run(
        RunConfig(
            run_name=_unique("run"),
            dataset_name=f"tpg_eval.{table}",
            source_type="TABLE",
            dataset_spec={"input": "input", "ground_truth_output": "ground_truth_output"},
        )
    )
    run.start()  # no input_df: read through PgRunDao.fetch_source_data
    status = wait_for_invocation(run)
    assert status == "INVOCATION_COMPLETED"


def test_log_ingestion_from_table(ctx: Any) -> None:
    """LOG_INGESTION mode scores stored outputs and contexts without calling the app."""
    from trulens.core import Metric, Selector
    from trulens.core.enums import Mode

    table = _unique("logged")
    with ctx.engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS tpg_eval"))
        conn.execute(
            text(f"CREATE TABLE tpg_eval.{table} (input text, output text, contexts text[])")
        )
        conn.execute(
            text(
                f"INSERT INTO tpg_eval.{table} VALUES "
                "('q1', 'a1 KFZ-1001', '{ctx a,ctx b}'), ('q2', 'a2', '{ctx c}')"
            )
        )
    tru_app = tpg.TruApp(None, app_name=_unique("logged_app"), app_version="v1")
    run = tru_app.add_run(
        RunConfig(
            run_name=_unique("run"),
            dataset_name=f"tpg_eval.{table}",
            source_type="TABLE",
            mode=Mode.LOG_INGESTION,
            dataset_spec={
                "input": "input",
                "record_root.output": "output",
                "retrieval.retrieved_contexts": "contexts",
            },
        )
    )
    run.start()

    def n_contexts(contexts: list[str]) -> float:
        return len(contexts) / 2

    metric = Metric(
        implementation=n_contexts,
        name="n_contexts",
        selectors={"contexts": Selector.select_context(collect_list=True)},
    )
    run.compute_metrics([metric])
    records, metrics = ctx.session.get_records_and_feedback(
        app_name=tru_app.app_name, run_name=run.run_name
    )
    assert "n_contexts" in metrics
    assert sorted(records["n_contexts"]) == [0.5, 1.0]


def test_metric_reads_ground_truth_from_the_table_run(ctx: Any, claims_table: str) -> None:
    """Ground truth comes from the run's table via dataset_spec; no second dataset is passed."""

    def cites_gold_clause(answer: str, gold: str) -> float:
        clause = gold.rsplit("clause ", maxsplit=1)[-1].rstrip(".")
        return 1.0 if clause in answer else 0.0

    gold_metric = Metric(
        implementation=cites_gold_clause,
        name="cites_gold_clause",
        selectors={
            "answer": Selector.select_record_output(),
            "gold": Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
            ),
        },
    )
    rag = ClaimsRag()
    tru_app = tpg.TruApp(rag, app_name=_unique("claims"), app_version="v1", main_method=rag.answer)
    run_name = _unique("gold")
    run = tru_app.add_run(
        RunConfig(
            run_name=run_name,
            dataset_name=f"{claims_table}?split=test",
            source_type="TABLE",
            dataset_spec={"input": "question", "ground_truth_output": "answer"},
        )
    )
    run.start()
    assert wait_for_invocation(run) == "INVOCATION_COMPLETED"
    run.compute_metrics([gold_metric])

    records, metrics = ctx.session.get_records_and_feedback(
        app_name=tru_app.app_name, run_name=run_name
    )
    assert metrics == ["cites_gold_clause"]
    assert len(records) == 3  # split=test: items 1, 3, 5
    # ClaimsRag always cites KFZ-1001, the gold clause of item 1 only.
    assert sorted(records["cites_gold_clause"]) == [0.0, 0.0, 1.0]
