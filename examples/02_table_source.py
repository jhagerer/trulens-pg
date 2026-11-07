"""G2: RunConfig(source_type="TABLE") reads a published benchmark version from Postgres.

Load the example benchmark first:

    trulens-pg db upgrade
    trulens-pg bench import-corpus examples/claims_faq/corpus.jsonl --corpus policies --create
    trulens-pg bench import examples/claims_faq/items.jsonl --dataset claims_faq --create \
        --corpus policies --version 1.0.0 --publish --changelog "Example benchmark"
"""

from __future__ import annotations

from rag_app import ClaimsRag
from trulens.core.run import RunConfig

import trulens_pg as tpg

session = tpg.connect()
rag = ClaimsRag()
tru_app = tpg.wrap(rag, app_name="claims_rag", app_version="v1", main_method=rag.answer)

run = tru_app.add_run(
    RunConfig(
        run_name="v1_test_split",
        dataset_name=tpg.bench_source("claims_faq", "1.0.0", split="test"),
        source_type="TABLE",
        dataset_spec={"input": "input", "ground_truth_output": "ground_truth_output"},
    )
)
run.start()  # no input_df: rows come from the view bench.v_claims_faq__1_0_0
print(run.get_status())
