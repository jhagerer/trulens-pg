"""Items: edit items of a draft; bulk-tag; set split; filter by tag, difficulty, language."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import EDITOR
from trulens_pg.bench.repository import PAGE_SIZE

st.title("Items")
repo = common.repo()
user = common.current_user()

dataset = common.pick_dataset()
if dataset is None:
    st.stop()
version = common.pick_version(dataset)
if version is None:
    st.stop()
editable = version.status in common.EDITABLE and user.is_editor
if version.status not in common.EDITABLE:
    st.info(f"Version {version.version} is {version.status} and read-only. Create a new draft.")

cols = st.columns(5)
split = cols[0].selectbox("Split", ["", "dev", "test"])
tags = common.text_to_list(cols[1].text_input("Tags (any of, comma-separated)"))
difficulty = cols[2].text_input("Difficulty")
language = cols[3].text_input("Language")
search = cols[4].text_input("Search text")
filters = {
    "split": split or None,
    "tags": tags,
    "difficulty": difficulty or None,
    "language": language or None,
    "search": search or None,
}
total = repo.count_items(version.version_id, **filters)
pages = max(1, math.ceil(total / PAGE_SIZE))
page = st.number_input(f"Page (of {pages}, {total:,} items)", 1, pages, 1) - 1
items = repo.list_items(version.version_id, page=int(page), **filters)
view = common.lists_to_text(items, ("ground_truth_context_ids", "tags"))
view.insert(0, "select", False)

edited = st.data_editor(
    view,
    hide_index=True,
    use_container_width=True,
    disabled=not editable,
    key=f"editor_{version.version_id}_{page}",
    column_config={
        "select": st.column_config.CheckboxColumn("✓", width="small"),
        "item_id": None,
        "external_id": st.column_config.TextColumn("external_id", disabled=True),
        "query": st.column_config.TextColumn("query", width="large", required=True),
        "ground_truth_answer": st.column_config.TextColumn("ground_truth_answer", width="large"),
        "ground_truth_context_ids": st.column_config.TextColumn("context ids (comma-separated)"),
        "split": st.column_config.SelectboxColumn("split", options=["dev", "test"]),
        "tags": st.column_config.TextColumn("tags (comma-separated)"),
        "contains_pii": st.column_config.CheckboxColumn("PII"),
        "metadata": None,
    },
)

if not editable:
    st.stop()

selected = list(edited.loc[edited["select"], "item_id"])
original = view.set_index("item_id")
changes = []
for row in edited.drop(columns=["select"]).to_dict(orient="records"):
    before = original.loc[row["item_id"]].drop(labels=["select"]).to_dict()
    diff = {
        k: v
        for k, v in row.items()
        if k not in ("item_id", "metadata")
        and not (pd.isna(v) and pd.isna(before.get(k)))
        and v != before.get(k)
    }
    if diff:
        for key in ("tags", "ground_truth_context_ids"):
            if key in diff:
                diff[key] = common.text_to_list(diff[key])
        changes.append({"item_id": row["item_id"], **diff})

c1, c2 = st.columns(2)
if c1.button(f"Save {len(changes)} changed row(s)", disabled=not changes, type="primary"):
    common.require(EDITOR)
    if (
        common.run_write(lambda: repo.update_items(changes), lambda n: f"Updated {n} item(s).")
        is not None
    ):
        st.rerun()

with c2.expander(f"Bulk actions on {len(selected)} selected item(s)"):
    add = common.text_to_list(st.text_input("Add tags"))
    remove = common.text_to_list(st.text_input("Remove tags"))
    if st.button("Apply tags", disabled=not selected or not (add or remove)):
        common.run_write(
            lambda: repo.bulk_tag(selected, add=add, remove=remove),
            lambda n: f"Tagged {n} item(s).",
        )
    new_split = st.selectbox("Set split", ["dev", "test", "(none)"])
    if st.button("Set split", disabled=not selected):
        common.run_write(
            lambda: repo.set_split(selected, None if new_split == "(none)" else new_split),
            lambda n: f"Set split of {n} item(s).",
        )
    if st.button("Delete selected", disabled=not selected):
        common.run_write(lambda: repo.delete_items(selected), lambda n: f"Deleted {n} item(s).")

with st.expander("Add item"), st.form("add_item", clear_on_submit=True):
    query = st.text_area("Query")
    answer = st.text_area("Ground truth answer")
    context_ids = st.text_input("Ground truth context ids (comma-separated)")
    item_tags = st.text_input("Tags (comma-separated)")
    item_split = st.selectbox("Split", ["test", "dev"])
    pii = st.checkbox("Contains personal data")
    if st.form_submit_button("Add"):
        common.run_write(
            lambda: repo.add_items(
                version.version_id,
                [
                    {
                        "query": query,
                        "ground_truth_answer": answer or None,
                        "ground_truth_context_ids": common.text_to_list(context_ids),
                        "tags": common.text_to_list(item_tags),
                        "split": item_split,
                        "contains_pii": pii,
                    }
                ],
            ),
            lambda n: f"Added {n} item.",
        )
