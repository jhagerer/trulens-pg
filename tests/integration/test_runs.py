"""End-to-end tests of the Run API on Postgres (G1, G2, stored metrics)."""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import text
from trulens.apps.app import TruApp
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


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def test_g1_plain_truapp_fails_and_wrap_fixes_it(ctx: Any) -> None:
    rag = ClaimsRag()
    plain = TruApp(rag, app_name=_unique("plain"), app_version="v1", main_method=rag.answer)
    with pytest.raises(ValueError, match="Run DAO not initialized"):
        plain.add_run(RunConfig(run_name="r", dataset_name="df", dataset_spec={"input": "input"}))

    attached = tpg.attach(plain)
    assert isinstance(attached.run_dao, tpg.PgRunDao)

    tru_app = tpg.wrap(rag, app_name=_unique("claims"), app_version="v1", main_method=rag.answer)
    run = tru_app.add_run(
        RunConfig(run_name=_unique("run"), dataset_name="inline", dataset_spec={"input": "input"})
    )
    run.start(input_df=pd.DataFrame({"input": ["motor glass damage?", "home flood?"]}))
    tpg.metrics.resolve.wait_for_invocation(run, timeout=60, poll_interval=0.5)

    def length(answer: str) -> float:
        return min(len(answer) / 100, 1.0)

    from trulens.core import Metric, Selector

    metric = Metric(
        implementation=length, name="length", selectors={"answer": Selector.select_record_output()}
    )
    tpg.compute_metrics(run, [metric], timeout=60, poll_interval=0.5)
    records, metrics = ctx.session.get_records_and_feedback(
        app_name=tru_app.app_name, run_name=run.run_name
    )
    assert "length" in metrics
    assert len(records) == 2


def test_g2_table_source_and_column_check(ctx: Any) -> None:
    table = _unique("questions")
    with ctx.engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS bench_eval"))
        conn.execute(
            text(
                f"CREATE TABLE bench_eval.{table} (item_id text, input text, "
                "ground_truth_output text, contexts jsonb, tags text[])"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO bench_eval.{table} VALUES "
                "('1', 'motor glass?', 'yes', '[\"c1\", \"c2\"]', '{motor}'), "
                "('2', 'home flood?', 'no', '[]', '{home}')"
            )
        )
    rag = ClaimsRag()
    tru_app = tpg.wrap(rag, app_name=_unique("claims"), app_version="v1", main_method=rag.answer)

    with pytest.raises(SourceError, match="lacks column"):
        tru_app.add_run(
            RunConfig(
                run_name=_unique("bad"),
                dataset_name=f"bench_eval.{table}",
                source_type="TABLE",
                dataset_spec={"input": "question"},
            )
        )
    with pytest.raises(SourceError, match="does not exist"):
        tru_app.add_run(
            RunConfig(
                run_name=_unique("bad"),
                dataset_name="bench_eval.missing_table",
                source_type="TABLE",
                dataset_spec={"input": "input"},
            )
        )

    df = tru_app.run_dao.fetch_source_data(f"bench_eval.{table}?tags=motor")
    assert list(df["input"]) == ["motor glass?"]
    assert df.loc[0, "contexts"] == ["c1", "c2"]  # JSONB arrives as a Python list

    run = tru_app.add_run(
        RunConfig(
            run_name=_unique("run"),
            dataset_name=f"bench_eval.{table}",
            source_type="TABLE",
            dataset_spec={"input": "input", "ground_truth_output": "ground_truth_output"},
        )
    )
    run.start()  # no input_df: read through PgRunDao.fetch_source_data
    status = tpg.metrics.resolve.wait_for_invocation(run, timeout=60, poll_interval=0.5)
    assert status == "INVOCATION_COMPLETED"


def test_log_ingestion_from_table(ctx: Any) -> None:
    """LOG_INGESTION mode scores stored outputs and contexts without calling the app."""
    from trulens.core import Metric, Selector
    from trulens.core.enums import Mode

    table = _unique("logged")
    with ctx.engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS bench_eval"))
        conn.execute(
            text(f"CREATE TABLE bench_eval.{table} (input text, output text, contexts text[])")
        )
        conn.execute(
            text(
                f"INSERT INTO bench_eval.{table} VALUES "
                "('q1', 'a1 KFZ-1001', '{ctx a,ctx b}'), ('q2', 'a2', '{ctx c}')"
            )
        )
    tru_app = tpg.wrap(None, app_name=_unique("logged_app"), app_version="v1")
    run = tru_app.add_run(
        RunConfig(
            run_name=_unique("run"),
            dataset_name=f"bench_eval.{table}",
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
    tpg.compute_metrics(run, [metric], timeout=60, poll_interval=0.5)
    records, metrics = ctx.session.get_records_and_feedback(
        app_name=tru_app.app_name, run_name=run.run_name
    )
    assert "n_contexts" in metrics
    assert sorted(records["n_contexts"]) == [0.5, 1.0]


def test_benchmark_source_with_stored_metrics(
    ctx: Any, repo: Any, store: Any, published: Any, fake_judge: Any
) -> None:
    # Stored metrics: a deterministic python_ref and a calibrated llm_judge.
    store.create_metric("has_policy_number", "python_ref", description="Cites a policy number")
    ref = store.create_draft("has_policy_number", python_ref="contains_policy_number")
    store.publish(ref.metric_version_id, published_by="tester")

    store.create_metric("policy_citation", "llm_judge")
    draft = store.create_draft(
        "policy_citation",
        prompt_template="Question: {question}\nAnswer: {answer}",
        rubric="3 if the answer cites a clause number, 0 otherwise.",
        judge={"provider": "openai", "model": "gpt-4o-mini"},
    )
    store.add_labels(
        "policy_citation",
        [
            {"question": "q1", "answer": "See KFZ-1001", "human_score": 3},
            {"question": "q2", "answer": "No idea", "human_score": 0},
            {"question": "q3", "answer": "KFZ-1002 applies", "human_score": 3},
        ],
    )
    cal = store.calibrate(draft.metric_version_id, fake_judge)
    assert cal.n_labels == 3
    assert cal.spearman == pytest.approx(1.0)
    store.publish(draft.metric_version_id, published_by="tester", min_labels=3)
    store.create_metric("context_recall_at_2", "ground_truth")
    gt = store.create_draft("context_recall_at_2", params={"method": "recall_at_k", "k": 2})
    store.publish(gt.metric_version_id, published_by="tester")
    store.create_metric_set(
        "claims-rag-v1",
        ["has_policy_number@1", "policy_citation@1", "context_recall_at_2@1"],
        dataset="claims_faq",
    )

    rag = ClaimsRag()
    tru_app = tpg.wrap(rag, app_name=_unique("claims"), app_version="v3", main_method=rag.answer)
    run_name = _unique("v3_test")
    run = tru_app.add_run(
        RunConfig(
            run_name=run_name,
            dataset_name=tpg.bench_source("claims_faq", "1.0.0", split="test"),
            source_type="TABLE",
            dataset_spec={"input": "input", "ground_truth_output": "ground_truth_output"},
            llm_judge_name="fake-judge-1",
        )
    )
    run.start()
    tpg.compute_metrics(run, ["claims-rag-v1"], judge=fake_judge, timeout=60, poll_interval=0.5)

    records, metrics = ctx.session.get_records_and_feedback(
        app_name=tru_app.app_name, run_name=run_name
    )
    assert set(metrics) == {"has_policy_number@1", "policy_citation@1", "context_recall_at_2@1"}
    assert len(records) == 3  # split=test: items 1, 3, 5
    assert records["has_policy_number@1"].mean() == pytest.approx(1.0)
    # Golden chunks of items 1, 3, 5 are c1, c0, c2; ClaimsRag retrieves c1 and c0.
    assert records["context_recall_at_2@1"].mean() == pytest.approx(2 / 3)

    version = repo.get_version("claims_faq", "1.0.0")
    links = repo.runs_for_version(version.version_id)
    assert list(links["run_name"]) == [run_name]
    assert links.loc[0, "judge_model"] == "fake-judge-1"
    used = store.run_metrics(run_name, tru_app.app_name)
    assert sorted(used["name"]) == ["context_recall_at_2", "has_policy_number", "policy_citation"]
    assert set(used["judge_model"].dropna()) == {"fakejudge:fake-judge-1"}

    summary = repo.run_metric_summary(version.version_id, ctx.session)
    assert set(summary["metric"]) == set(metrics)
    assert (summary["n"] == 3).all()


def test_unpublished_benchmark_version_is_not_a_source(ctx: Any, repo: Any) -> None:
    repo.create_dataset("drafty")
    repo.create_draft("drafty", copy_from=None)
    rag = ClaimsRag()
    tru_app = tpg.wrap(rag, app_name=_unique("claims"), app_version="v1", main_method=rag.answer)
    with pytest.raises(SourceError, match="only published versions"):
        tru_app.add_run(
            RunConfig(
                run_name=_unique("r"),
                dataset_name="bench:drafty@1.0.0",
                source_type="TABLE",
                dataset_spec={"input": "input"},
            )
        )
