"""Entry point of the Streamlit app: ``trulens-pg app`` or ``streamlit run main.py``."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from trulens_pg.app import common

PAGES = Path(__file__).parent / "pages"

st.set_page_config(page_title="trulens-pg", layout="wide")

user = common.current_user()

benchmark_pages = [
    st.Page(PAGES / "datasets.py", title="Datasets", default=True),
    st.Page(PAGES / "import_items.py", title="Import"),
    st.Page(PAGES / "items.py", title="Items"),
    st.Page(PAGES / "corpus.py", title="Corpus"),
    st.Page(PAGES / "review.py", title="Review"),
    st.Page(PAGES / "publish.py", title="Publish"),
    st.Page(PAGES / "runs.py", title="Runs"),
    st.Page(PAGES / "export.py", title="Export"),
]
metric_pages = [
    st.Page(PAGES / "metrics.py", title="Metrics"),
    st.Page(PAGES / "metric_editor.py", title="Metric editor"),
    st.Page(PAGES / "playground.py", title="Playground"),
    st.Page(PAGES / "calibration.py", title="Calibration"),
    st.Page(PAGES / "metric_publish.py", title="Metric publish"),
    st.Page(PAGES / "metric_sets.py", title="Metric sets"),
]

with st.sidebar:
    st.caption(f"Signed in as **{user.name}** · roles: {', '.join(sorted(user.roles)) or 'viewer'}")
    if user.auth == "none":
        st.warning("Authentication is off (TRULENS_PG_AUTH=none). Use for local work only.")
    if not common.settings().roles:
        st.caption("TRULENS_PG_ROLES is not set: every user is editor and publisher.")
    if user.auth == "oidc":
        st.button("Log out", on_click=st.logout)

st.navigation({"Benchmarks": benchmark_pages, "Metrics": metric_pages}).run()
