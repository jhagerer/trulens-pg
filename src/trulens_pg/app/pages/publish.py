"""Publish: compare a draft with the last published version and publish with a changelog."""

from __future__ import annotations

import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import PUBLISHER

st.title("Publish")
repo = common.repo()
user = common.current_user()

dataset = common.pick_dataset()
if dataset is None:
    st.stop()
version = common.pick_version(dataset, statuses=common.EDITABLE)

if version is not None:
    diff = repo.diff_versions(version.version_id)
    counts = diff["change"].value_counts() if not diff.empty else {}
    cols = st.columns(4)
    cols[0].metric("Items", int(version.item_count))
    cols[1].metric("Added", int(counts.get("added", 0)))
    cols[2].metric("Changed", int(counts.get("changed", 0)))
    cols[3].metric("Removed", int(counts.get("removed", 0)))
    st.dataframe(diff, hide_index=True, use_container_width=True)

    open_reviews = repo.open_review_count(version.version_id)
    blocked = version.status != "in_review" or open_reviews > 0
    if version.status != "in_review":
        st.warning("Submit the version for review first (Review page).")
    elif open_reviews:
        st.warning(f"{open_reviews} item(s) have open reviews; publishing is blocked.")
    if not user.is_publisher:
        st.info("Publishing needs the publisher role.")

    changelog = st.text_area("Changelog", version.changelog or "")
    if st.button(
        "Publish", type="primary", disabled=blocked or not user.is_publisher or not changelog
    ):
        common.require(PUBLISHER)
        result = common.run_write(
            lambda: repo.publish_version(
                version.version_id, published_by=user.name, changelog=changelog
            ),
            lambda r: (
                f"Published {r.dataset}@{r.version}: {r.item_count} item(s), "
                f"{r.removed_rejected} rejected item(s) removed, view {r.view}. "
                f'Use dataset_name="bench:{r.dataset}@{r.version}".'
            ),
        )
        if result is not None and result.duplicate_of:
            st.warning(f"Same content hash as version(s) {', '.join(result.duplicate_of)}.")

st.subheader("Deprecate a published version")
published = common.pick_version(dataset, statuses=("published",), key="deprecate_version")
if published is not None and st.button("Deprecate", disabled=not user.is_publisher):
    common.require(PUBLISHER)
    common.run_write(
        lambda: repo.deprecate_version(published.version_id),
        lambda _: f"{dataset}@{published.version} is deprecated (1 row).",
    )
