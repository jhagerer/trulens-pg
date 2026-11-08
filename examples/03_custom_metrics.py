"""Custom metrics as data: create, calibrate, publish, bundle in a set, and run by name.

Run 02_table_source.py first (it creates the table eval.claims_faq). Uses the offline
judge; pass a TruLens provider for real scores.
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
    # Stored as a version of the TruLens prompt "trulens-pg.policy_citation" (Prompts tab);
    # the metric version pins that exact prompt version.
    prompt_text="Question: {{question}}\n\nAnswer: {{answer}}",
    rubric="3: the answer names the clause number it relies on.\n0: no clause number.",
    score_min=0,
    score_max=3,
)
data = Path(__file__).parent / "claims_faq"
items = [json.loads(x) for x in (data / "items.jsonl").read_text().splitlines() if x]
chunks = {
    doc["chunk_id"]: doc["text"]
    for doc in map(json.loads, (data / "corpus.jsonl").read_text().splitlines())
}

# Human scores: reference answers that cite a clause get 3, answers without a clause get 0.
labels = []
for item in items[:24]:
    answer = item["ground_truth_answer"]
    short = answer.split(" (clause")[0].split(" (Klausel")[0]
    labels.append(
        {"question": item["query"], "answer": answer, "human_score": 3 * (short != answer)}
    )
    labels.append({"question": item["query"], "answer": short, "human_score": 0})
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
store.create_metric_set(set_name, [citation.ref, number.ref, recall.ref])

# Ground truth for context_recall_at_2: the expected chunk texts of each question.
golden_set = [
    {
        "query": item["query"],
        "expected_response": item["ground_truth_answer"],
        "expected_chunks": [{"text": chunks[c]} for c in item["ground_truth_context_ids"]],
    }
    for item in items
]

rag = ClaimsRag()
tru_app = tpg.wrap(rag, app_name="claims_rag", app_version="v2", main_method=rag.answer)
run = tru_app.add_run(
    RunConfig(
        run_name=f"v2_{set_name}",
        dataset_name="eval.claims_faq",
        source_type="TABLE",
        dataset_spec={"input": "query", "ground_truth_output": "ground_truth_answer"},
        llm_judge_name=judge.model_engine,
    )
)
run.start()
print(tpg.compute_metrics(run, [set_name], judge=judge, golden_set=golden_set))
print(session.get_leaderboard())
print(f"Judge prompt version: {citation.prompt_version_id}")
