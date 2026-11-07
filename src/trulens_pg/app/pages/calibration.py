"""Calibration: enter human scores and compute agreement with the judge."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import EDITOR
from trulens_pg.bench import io as bench_io

st.title("Calibration")
store = common.store()
repo = common.repo()
user = common.require(EDITOR)

name = common.pick_metric()
if name is None:
    st.stop()
row = common.pick_metric_version(name, statuses=common.EDITABLE)
if row is None:
    st.stop()
defn = store.definition(row.metric_version_id)
labels = store.list_labels(name)
min_labels = common.settings().min_calibration_labels
st.metric("Labels", len(labels), help=f"Publishing judge metrics needs at least {min_labels}.")
st.dataframe(labels.drop(columns=["label_id"]), hide_index=True, use_container_width=True)

st.subheader("Add a label")
mode = st.radio("Example from", ["Benchmark item", "Free text"], horizontal=True)
prefill = {"question": "", "answer": "", "contexts": "", "ground_truth": "", "item_id": None}
if mode == "Benchmark item":
    dataset = common.pick_dataset()
    version = common.pick_version(dataset, statuses=common.FROZEN) if dataset else None
    if version is not None:
        frame = repo.version_frame(version.version_id)
        if not frame.empty:
            pick = st.selectbox(
                "Item",
                range(len(frame)),
                format_func=lambda i: f"{frame.iloc[i].external_id}: {frame.iloc[i].input[:80]}",
            )
            item = frame.iloc[int(pick)]
            prefill.update(
                question=item.input,
                answer=item.ground_truth_output or "",
                ground_truth=item.ground_truth_output or "",
                item_id=item.item_id,
            )
with st.form("label", clear_on_submit=True):
    question = st.text_area("Question", prefill["question"])
    answer = st.text_area("Answer to score", prefill["answer"])
    contexts = st.text_area("Contexts (one per line)", prefill["contexts"])
    ground_truth = st.text_area("Ground truth", prefill["ground_truth"])
    score = st.slider("Human score", defn.score_min, defn.score_max, defn.score_min)
    if st.form_submit_button("Add label"):
        common.run_write(
            lambda: store.add_labels(
                name,
                [
                    {
                        "item_id": prefill["item_id"],
                        "question": question,
                        "answer": answer,
                        "contexts": [c for c in contexts.splitlines() if c.strip()],
                        "ground_truth": ground_truth or None,
                        "human_score": score,
                        "labeler": user.name,
                    }
                ],
            ),
            lambda n: f"Added {n} label.",
        )

with st.expander(
    "Upload labels (CSV/JSONL: question, answer, contexts, ground_truth, human_score)"
):
    upload = st.file_uploader("Labels file", type=["csv", "jsonl", "json"])
    if upload is not None and st.button("Add labels from file"):
        with common.handled():
            frame = bench_io.read_table(upload, bench_io.detect_format(upload.name))
            cols = [
                c
                for c in ("question", "answer", "contexts", "ground_truth", "human_score")
                if c in frame
            ]
            records = [{str(k): v for k, v in r.items()} for r in frame[cols].to_dict("records")]
            for r in records:
                r["labeler"] = user.name
                if isinstance(r.get("contexts"), str):
                    r["contexts"] = common.text_to_list(r["contexts"])
            common.run_write(
                lambda: store.add_labels(name, records), lambda n: f"Added {n} label(s)."
            )

st.subheader("Run calibration")
st.info(f"Calibration scores {len(labels)} label(s): about {len(labels)} judge call(s).")
if st.button("Calibrate", type="primary", disabled=labels.empty):
    with common.handled(), st.spinner("Scoring labels ..."):
        result = store.calibrate(row.metric_version_id, golden_set=None)
        cols = st.columns(3)
        cols[0].metric("Labels", result.n_labels)
        cols[1].metric("Spearman", "-" if result.spearman is None else f"{result.spearman:.2f}")
        cols[2].metric(
            "Mean abs. error",
            "-" if result.mean_abs_error is None else f"{result.mean_abs_error:.2f}",
        )
        chart = (
            alt.Chart(result.scores)
            .mark_circle(size=80)
            .encode(
                x=alt.X(
                    "human_normalized:Q", title="human score (0..1)", scale=alt.Scale(domain=[0, 1])
                ),
                y=alt.Y(
                    "judge_score:Q", title="judge score (0..1)", scale=alt.Scale(domain=[0, 1])
                ),
                tooltip=["question", "human_score", "judge_score", "reason"],
            )
        )
        diagonal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(strokeDash=[4, 4])
            .encode(x="x", y="y")
        )
        st.altair_chart(chart + diagonal, use_container_width=True)
        st.dataframe(result.scores, hide_index=True, use_container_width=True)
