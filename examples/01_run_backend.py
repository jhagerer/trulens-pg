"""G1: the TruLens Run API on Postgres without passing connector=.

export TRULENS_PG_DATABASE_URL=postgresql+psycopg://user:password@localhost/evals
python examples/01_run_backend.py
"""

from __future__ import annotations

import pandas as pd
from rag_app import ClaimsRag
from trulens.core import Metric, Selector
from trulens.core.run import RunConfig

import trulens_pg as tpg

session = tpg.connect()  # URL from TRULENS_PG_DATABASE_URL
rag = ClaimsRag()
tru_app = tpg.TruApp(rag, app_name="claims_rag", app_version="v1", main_method=rag.answer)

run = tru_app.add_run(
    RunConfig(
        run_name="v1_inline", dataset_name="inline_questions", dataset_spec={"input": "input"}
    )
)
run.start(
    input_df=pd.DataFrame({"input": ["Is a broken windscreen covered?", "Is my bike insured?"]})
)


def cites_clause(answer: str) -> float:
    return 1.0 if "clause" in answer else 0.0


metric = Metric(
    implementation=cites_clause,
    name="cites_clause",
    selectors={"answer": Selector.select_record_output()},
)
print(run.compute_metrics([metric]))  # the TruLens Run API
print(session.get_leaderboard())
