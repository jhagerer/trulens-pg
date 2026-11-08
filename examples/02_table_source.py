"""G2: RunConfig(source_type="TABLE") reads a Postgres table.

Loads the example questions into the table eval.claims_faq, then runs the app on
the test split without passing input_df.

    export TRULENS_PG_DATABASE_URL=postgresql+psycopg://user:password@localhost/evals
    python examples/02_table_source.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rag_app import ClaimsRag
from sqlalchemy import Text, text
from sqlalchemy.dialects.postgresql import ARRAY
from trulens.core.run import RunConfig

import trulens_pg as tpg

session = tpg.connect(migrate=True)
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
