"""Runs: runs of one dataset version, metric means and spread per app version."""

from __future__ import annotations

import altair as alt
import streamlit as st

from trulens_pg.app import common

st.title("Runs")
repo = common.repo()
common.current_user()

dataset = common.pick_dataset()
if dataset is None:
    st.stop()
version = common.pick_version(dataset, statuses=common.FROZEN)
if version is None:
    st.stop()

st.code(f'dataset_name="bench:{dataset}@{version.version}"', language="python")
runs = repo.runs_for_version(version.version_id)
if runs.empty:
    st.info("No runs use this version yet.")
    st.stop()
st.dataframe(runs, hide_index=True, use_container_width=True)

dashboard = common.settings().dashboard_url
if dashboard:
    st.link_button("Open traces in the TruLens dashboard", dashboard)

with common.handled(), st.spinner("Reading results from the TruLens tables ..."):
    summary = repo.run_metric_summary(version.version_id, common.trulens_session(), runs=runs)
if summary.empty:
    st.info("No metric results yet.")
    st.stop()

platforms = sorted(summary["platform"].unique())
judges = sorted(str(j) for j in summary["judge_model"].fillna("-").unique())
cols = st.columns(2)
platform = cols[0].multiselect("Platform", platforms, default=platforms)
judge = cols[1].multiselect("Judge", judges, default=judges)
summary = summary[
    summary["platform"].isin(platform) & summary["judge_model"].fillna("-").isin(judge)
]
st.dataframe(summary, hide_index=True, use_container_width=True)

for metric, group in summary.groupby("metric"):
    frame = group.assign(
        lower=group["mean"] - group["std"].fillna(0), upper=group["mean"] + group["std"].fillna(0)
    )
    base = alt.Chart(frame).encode(x=alt.X("app_version:N", title="app version"))
    bars = base.mark_bar().encode(
        y=alt.Y("mean:Q", title=str(metric), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("run_name:N", legend=None),
        tooltip=["run_name", "app_version", "judge_model", "mean", "std", "n"],
    )
    rules = base.mark_rule().encode(y="lower:Q", y2="upper:Q")
    st.altair_chart(
        (bars + rules).properties(title=str(metric), height=220), use_container_width=True
    )
