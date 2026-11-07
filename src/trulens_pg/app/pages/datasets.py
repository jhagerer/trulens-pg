"""Datasets: list, search and create benchmarks; see versions, status and item counts."""

from __future__ import annotations

import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import EDITOR

st.title("Datasets")
repo = common.repo()

search = st.text_input("Search", placeholder="name, domain or description")
datasets = repo.list_datasets(search or None)
if datasets.empty:
    st.info("No datasets found.")
else:
    domains = sorted(d for d in datasets["domain"].dropna().unique())
    chosen = st.multiselect("Domain", domains)
    if chosen:
        datasets = datasets[datasets["domain"].isin(chosen)]
    st.dataframe(datasets, hide_index=True, use_container_width=True)

    name = st.selectbox("Show versions of", list(datasets["name"]))
    if name:
        st.dataframe(repo.list_versions(name), hide_index=True, use_container_width=True)

left, right = st.columns(2)
with left, st.form("create_dataset", clear_on_submit=True):
    st.subheader("Create dataset")
    new_name = st.text_input("Name (lower-case slug, e.g. claims_faq)")
    domain = st.text_input("Domain")
    description = st.text_area("Description")
    if st.form_submit_button("Create", disabled=not common.can(EDITOR)):
        user = common.require(EDITOR)
        common.run_write(
            lambda: repo.create_dataset(
                new_name.strip(), domain=domain or None, owner=user.name, description=description
            ),
            lambda ds: f"Created dataset {ds.name} (1 row).",
        )

with right, st.form("create_draft"):
    st.subheader("Create draft version")
    names = [] if datasets.empty else list(repo.list_datasets()["name"])
    target = st.selectbox("Dataset", names, key="draft_dataset")
    copy = st.radio("Start from", ["latest version", "empty"], horizontal=True)
    bump = st.radio("Increase", ["minor", "patch", "major"], horizontal=True)
    version = st.text_input("Version (optional, e.g. 2.0.0)")
    if st.form_submit_button("Create draft", disabled=not common.can(EDITOR) or not names):
        user = common.require(EDITOR)
        common.run_write(
            lambda: repo.create_draft(
                str(target),
                version.strip() or None,
                copy_from="latest" if copy == "latest version" else None,
                bump=bump,
                created_by=user.name,
            ),
            lambda v: f"Created draft {target}@{v.version} with {v.item_count} item(s).",
        )
