"""Metric publish: compare the draft with the last version and publish after calibration."""

from __future__ import annotations

import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import PUBLISHER

st.title("Metric publish")
store = common.store()
user = common.current_user()

name = common.pick_metric()
if name is None:
    st.stop()
row = common.pick_metric_version(name, statuses=common.EDITABLE)
if row is not None:
    diff = store.diff(row.metric_version_id)
    st.subheader("Changes against the last published version")
    if diff.empty:
        st.info("No differences.")
    for change in diff.itertuples():
        with st.expander(change.field, expanded=True):
            cols = st.columns(2)
            cols[0].caption("published")
            cols[0].code(str(change.base))
            cols[1].caption("draft")
            cols[1].code(str(change.draft))

    calibration = store.latest_calibration(row.metric_version_id)
    if calibration is not None:
        st.caption(
            f"Calibration: {calibration.n_labels} labels, Spearman {calibration.spearman}, "
            f"mean abs. error {calibration.mean_abs_error}, judge {calibration.judge_model}"
        )
    blockers = store.publish_blockers(row.metric_version_id)
    for blocker in blockers:
        st.warning(blocker)
    if not user.is_publisher:
        st.info("Publishing needs the publisher role.")
    changelog = st.text_area("Changelog", row.changelog or "")
    if st.button(
        "Publish", type="primary", disabled=bool(blockers) or not user.is_publisher or not changelog
    ):
        common.require(PUBLISHER)
        common.run_write(
            lambda: store.publish(
                row.metric_version_id, published_by=user.name, changelog=changelog
            ),
            lambda d: f'Published {d.ref} (1 row). Use it as "{d.ref}" in compute_metrics().',
        )

st.subheader("Deprecate a published version")
published = common.pick_metric_version(name, statuses=("published",), key="deprecate")
if published is not None and st.button("Deprecate", disabled=not user.is_publisher):
    common.require(PUBLISHER)
    common.run_write(
        lambda: store.deprecate(published.metric_version_id),
        lambda _: f"{name}@{published.version} is deprecated (1 row).",
    )
