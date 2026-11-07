"""Custom metrics as data: create, calibrate, publish, bundle in a set, and run by name.

Run 02_table_source.py first. Uses the offline judge; pass a TruLens provider for real scores.
"""

from __future__ import annotations

import json
from pathlib import Path

from offline_judge import OfflineJudge
from rag_app import ClaimsRag
from trulens.core.run import RunConfig

import trulens_pg as tpg
from trulens_pg.errors import ValidationError

session = tpg.connect()
store = tpg.MetricStore.from_context()
judge = OfflineJudge()

try:
    store.create_metric(
        "policy_citation", "llm_judge", description="The answer cites the policy clause it uses"
    )
    store.create_metric("has_policy_number", "python_ref")
    store.create_metric("context_recall_at_2", "ground_truth")
except ValidationError:
    print("Metrics exist already; creating new draft versions.")

citation = store.create_draft(
    "policy_citation",
    prompt_template="Question: {question}\n\nAnswer: {answer}",
    rubric="3: the answer names the clause number it relies on.\n0: no clause number.",
    score_min=0,
    score_max=3,
)
labels = [
    json.loads(x)
    for x in (Path(__file__).parent / "claims_faq" / "labels.jsonl").read_text().splitlines()
    if x
]
store.add_labels("policy_citation", labels)
calibration = store.calibrate(citation.metric_version_id, judge)
print(
    f"Calibration: n={calibration.n_labels} spearman={calibration.spearman} "
    f"mae={calibration.mean_abs_error}"
)
citation = store.publish(
    citation.metric_version_id, published_by="example", changelog="first version"
)

number = store.create_draft("has_policy_number", python_ref="contains_policy_number")
number = store.publish(number.metric_version_id, published_by="example")
recall = store.create_draft("context_recall_at_2", params={"method": "recall_at_k", "k": 2})
recall = store.publish(recall.metric_version_id, published_by="example")

set_name = f"claims-rag-c{citation.version}"
store.create_metric_set(set_name, [citation.ref, number.ref, recall.ref], dataset="claims_faq")

rag = ClaimsRag()
tru_app = tpg.wrap(rag, app_name="claims_rag", app_version="v2", main_method=rag.answer)
run = tru_app.add_run(
    RunConfig(
        run_name=f"v2_{set_name}",
        dataset_name="bench:claims_faq@1.0.0",
        source_type="TABLE",
        dataset_spec={"input": "input", "ground_truth_output": "ground_truth_output"},
        llm_judge_name=judge.model_engine,
    )
)
run.start()
print(tpg.compute_metrics(run, [set_name], judge=judge))
print(session.get_leaderboard())
