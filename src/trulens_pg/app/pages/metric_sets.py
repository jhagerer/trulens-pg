"""Metric sets: bundle fixed metric versions for a benchmark, e.g. claims-rag-v1."""

from __future__ import annotations

import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import PUBLISHER

st.title("Metric sets")
store = common.store()
repo = common.repo()
user = common.current_user()

st.dataframe(store.list_metric_sets(), hide_index=True, use_container_width=True)
st.caption('Runs reference a set by name: compute_metrics(run, ["claims-rag-v1"], judge).')

refs = []
for metric in store.list_metrics().itertuples():
    versions = store.list_versions(metric.name)
    refs += [f"{metric.name}@{v.version}" for v in versions.itertuples() if v.status == "published"]

with st.form("metric_set", clear_on_submit=True):
    st.subheader("Create metric set")
    name = st.text_input("Name, e.g. claims-rag-v1")
    datasets = repo.list_datasets()
    dataset = st.selectbox(
        "Benchmark (optional)", ["", *([] if datasets.empty else datasets["name"])]
    )
    chosen = st.multiselect("Published metric versions", refs)
    if st.form_submit_button("Create", disabled=not user.is_publisher):
        common.require(PUBLISHER)
        common.run_write(
            lambda: store.create_metric_set(
                name.strip(), chosen, dataset=dataset or None, created_by=user.name
            ),
            lambda s: f"Created metric set {s.name} with {len(chosen)} metric version(s).",
        )
