"""Metrics: browse custom metrics with kind, latest version, status and calibration."""

from __future__ import annotations

import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import EDITOR
from trulens_pg.metrics.build import KINDS

st.title("Metrics")
store = common.store()

metrics = store.list_metrics()
if metrics.empty:
    st.info("No custom metrics yet.")
else:
    cols = st.columns(2)
    kinds = cols[0].multiselect("Kind", KINDS)
    statuses = cols[1].multiselect(
        "Latest status", ["draft", "in_review", "published", "deprecated"]
    )
    view = metrics
    if kinds:
        view = view[view["kind"].isin(kinds)]
    if statuses:
        view = view[view["latest_status"].isin(statuses)]
    st.dataframe(view, hide_index=True, use_container_width=True)
    name = st.selectbox("Show versions of", list(view["name"]))
    if name:
        st.dataframe(store.list_versions(str(name)), hide_index=True, use_container_width=True)

with st.form("create_metric", clear_on_submit=True):
    st.subheader("Create metric")
    new_name = st.text_input("Name (lower-case, e.g. policy_citation)")
    kind = st.selectbox(
        "Kind",
        KINDS,
        help=(
            "llm_judge: prompt template scored by an LLM judge · criteria: allow-listed TruLens "
            "judge method with your criteria · ground_truth: IR metric against the benchmark · "
            "python_ref: reviewed function from an installed package"
        ),
    )
    description = st.text_area("Description")
    higher = st.checkbox("Higher score is better", value=True)
    if st.form_submit_button("Create", disabled=not common.can(EDITOR)):
        user = common.require(EDITOR)
        common.run_write(
            lambda: store.create_metric(
                new_name.strip(),
                str(kind),
                description=description or None,
                owner=user.name,
                higher_is_better=higher,
            ),
            lambda m: f"Created metric {m.name} (1 row). Define it in the Metric editor.",
        )
