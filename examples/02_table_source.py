"""G2: RunConfig(source_type="TABLE") reads a Postgres table.

Loads the example questions into the table eval.claims_faq, runs the app on the
test split without passing input_df, and scores each answer against the ground
truth answer of its row. The metric gets the ground truth from the run
(dataset_spec "ground_truth_output"), not from a second dataset.

    export TRULENS_PG_DATABASE_URL=postgresql+psycopg://user:password@localhost/evals
    python examples/02_table_source.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rag_app import ClaimsRag
from sqlalchemy import Text, text
from sqlalchemy.dialects.postgresql import ARRAY
from trulens.core import Metric, Selector
from trulens.core.run import RunConfig
from trulens.otel.semconv.trace import SpanAttributes

import trulens_pg as tpg

session = tpg.connect()
engine = tpg.get_context().engine

items = pd.read_json(Path(__file__).parent / "claims_faq" / "items.jsonl", lines=True)
with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS eval"))
items.to_sql(
    "claims_faq",
    engine,
    schema="eval",
    if_exists="replace",
    index=False,
    dtype={"ground_truth_context_ids": ARRAY(Text), "tags": ARRAY(Text)},
)

rag = ClaimsRag()
tru_app = tpg.wrap(rag, app_name="claims_rag", app_version="v1", main_method=rag.answer)
run = tru_app.add_run(
    RunConfig(
        run_name="v1_test_split",
        dataset_name="eval.claims_faq?split=test",
        source_type="TABLE",
        dataset_spec={"input": "query", "ground_truth_output": "ground_truth_answer"},
    )
)
run.start()  # no input_df: rows come from eval.claims_faq
print(run.get_status())


def cites_gold_clause(answer: str, gold: str) -> float:
    """1.0 if the answer names the clause that the ground truth answer names."""
    clauses = [w.strip("().,") for w in gold.split() if "-" in w and w[:1].isupper()]
    return 1.0 if any(c in answer for c in clauses) else 0.0


gold_metric = Metric(
    implementation=cites_gold_clause,
    name="cites_gold_clause",
    selectors={
        "answer": Selector.select_record_output(),
        "gold": Selector(  # the ground_truth_output column of the run's table
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
        ),
    },
)
print(run.compute_metrics([gold_metric]))
print(session.get_leaderboard())
