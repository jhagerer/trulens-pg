"""Review: work through a review queue (accept, edit, reject) and see who reviewed what."""

from __future__ import annotations

import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import EDITOR

st.title("Review")
repo = common.repo()
user = common.require(EDITOR)

dataset = common.pick_dataset()
if dataset is None:
    st.stop()
version = common.pick_version(dataset, statuses=common.EDITABLE)
if version is None:
    st.stop()

if version.status == "draft":
    st.info("This version is a draft. Submit it to start the review.")
    if st.button("Submit for review", type="primary", disabled=not user.is_editor):
        if (
            common.run_write(
                lambda: repo.submit_for_review(version.version_id),
                lambda _: f"{dataset}@{version.version} is in review (1 row).",
            )
            is not None
        ):
            st.rerun()
    st.stop()

only_open = st.toggle("Open items only", value=True)
queue = repo.review_queue(version.version_id, open_only=only_open)
open_count = repo.open_review_count(version.version_id)
st.progress(
    1 - open_count / max(int(version.item_count), 1),
    text=f"{open_count} open of {int(version.item_count)} item(s)",
)
if queue.empty:
    st.success("No open items. Continue on the Publish page.")
else:
    key = f"review_pos_{version.version_id}"
    pos = min(st.session_state.get(key, 0), len(queue) - 1)
    nav = st.columns([1, 1, 6])
    if nav[0].button("◀ Previous", disabled=pos == 0):
        pos -= 1
    if nav[1].button("Next ▶", disabled=pos >= len(queue) - 1):
        pos += 1
    st.session_state[key] = pos
    row = queue.iloc[pos]
    item = repo.list_items(version.version_id, search=row.external_id, page_size=5)
    item = item[item["external_id"] == row.external_id].iloc[0]

    st.subheader(f"{pos + 1}/{len(queue)} · {item.external_id}")
    if row.decision:
        st.caption(
            f"Last decision: {row.decision} by {row.reviewer} ({row.comment or 'no comment'})"
        )
    with st.form(f"review_{item.item_id}"):
        query = st.text_area("Query", item.query)
        answer = st.text_area("Ground truth answer", item.ground_truth_answer or "")
        contexts = st.text_input(
            "Ground truth context ids", ", ".join(item.ground_truth_context_ids)
        )
        comment = st.text_input("Comment")
        buttons = st.columns(3)
        accept = buttons[0].form_submit_button("Accept", type="primary")
        edit = buttons[1].form_submit_button("Save edit")
        reject = buttons[2].form_submit_button("Reject")
    decision = "accept" if accept else "edit" if edit else "reject" if reject else None
    if decision:
        edits = None
        if decision == "edit":
            edits = {
                "query": query,
                "ground_truth_answer": answer or None,
                "ground_truth_context_ids": common.text_to_list(contexts),
            }
        if (
            common.run_write(
                lambda: repo.add_review(
                    item.item_id,
                    reviewer=user.name,
                    decision=decision,
                    comment=comment,
                    edits=edits,
                ),
                lambda _: f"Recorded '{decision}' for {item.external_id} (1 row).",
            )
            is not None
        ):
            st.rerun()

with st.expander("Review history"):
    st.dataframe(repo.review_history(version.version_id), hide_index=True, use_container_width=True)
    if st.button("Return version to draft"):
        common.run_write(
            lambda: repo.return_to_draft(version.version_id),
            lambda _: f"{dataset}@{version.version} is a draft again (1 row).",
        )
