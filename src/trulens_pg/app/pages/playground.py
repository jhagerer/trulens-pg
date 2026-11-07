"""Playground: run a draft on 5 to 50 benchmark items or on pasted text."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import EDITOR
from trulens_pg.metrics.build import JUDGE_KINDS, build_implementation
from trulens_pg.metrics.store import call_on_label
from trulens_pg.sources import SourceOptions, apply_sampling

st.title("Playground")
store = common.store()
repo = common.repo()
user = common.require(EDITOR)

name = common.pick_metric()
if name is None:
    st.stop()
row = common.pick_metric_version(name)
if row is None:
    st.stop()
defn = store.definition(row.metric_version_id)
inputs = defn.resolved_inputs()

source = st.radio("Examples", ["Benchmark items", "Pasted text"], horizontal=True)
examples: list[dict[str, Any]] = []
golden: list[dict[str, Any]] | None = None
if source == "Benchmark items":
    dataset = common.pick_dataset()
    version = common.pick_version(dataset, statuses=common.FROZEN) if dataset else None
    if version is None:
        st.stop()
    cols = st.columns(2)
    n = int(cols[0].slider("Items", 5, 50, 10))
    seed = cols[1].text_input("Sample seed", "playground")
    frame = repo.version_frame(version.version_id)
    frame = apply_sampling(
        frame, SourceOptions(sample=min(1.0, n / max(len(frame), 1)) or 1.0, seed=seed)
    )
    frame = frame.head(n)
    golden = repo.golden_set(version.version_id)
    texts = {g["query"]: [c["text"] for c in g["expected_chunks"]] for g in golden}
    st.caption(
        "The answer is the reference answer (ground_truth_output); contexts are the "
        "ground-truth chunks from the corpus."
    )
    for item in frame.to_dict(orient="records"):
        examples.append(
            {
                "question": item["input"],
                "answer": item["ground_truth_output"],
                "contexts": texts.get(item["input"], []),
                "ground_truth": item["ground_truth_output"],
            }
        )
else:
    question = st.text_area("Question")
    answer = st.text_area("Answer")
    contexts = st.text_area("Contexts (one per line)")
    ground_truth = st.text_area("Ground truth answer")
    if question or answer:
        examples.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [c for c in contexts.splitlines() if c.strip()],
                "ground_truth": ground_truth,
                "query": question,
            }
        )

published = store.list_versions(name)
published = published[published["status"] == "published"]
baseline = None
if not published.empty and int(published.iloc[0].version) != defn.version:
    if st.checkbox(f"Compare with published v{int(published.iloc[0].version)}"):
        baseline = store.definition(published.iloc[0].metric_version_id)

per_example = (
    max(1, max((len(e.get("contexts") or []) for e in examples), default=1))
    if "context_each" in inputs.values()
    else 1
)
calls = len(examples) * per_example * (2 if baseline else 1)
if defn.kind in JUDGE_KINDS:
    st.info(f"This run makes about {calls} judge call(s), billed by the judge provider.")

if st.button("Run", type="primary", disabled=not examples):
    rows = []
    with common.handled():
        impl, _, info = build_implementation(defn, golden_set=golden)
        base_impl = build_implementation(baseline, golden_set=golden)[0] if baseline else None
        progress = st.progress(0.0)
        for i, example in enumerate(examples):
            label = {**example, "query": example["question"]}
            score, reason = call_on_label(impl, inputs, label)
            result = {"question": example["question"], "score": score, "reason": reason}
            if base_impl is not None and baseline is not None:
                result["baseline_score"] = call_on_label(
                    base_impl, baseline.resolved_inputs(), label
                )[0]
            rows.append(result)
            progress.progress((i + 1) / len(examples))
        st.caption(
            f"Judge: {info.get('provider')}:{info.get('model')}" if info else "No judge used."
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
