"""Corpus: browse documents and chunks; check that every ground-truth context ID exists."""

from __future__ import annotations

import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import EDITOR
from trulens_pg.bench import io as bench_io

st.title("Corpus")
repo = common.repo()
user = common.current_user()

corpora = repo.list_corpora()
st.dataframe(corpora, hide_index=True, use_container_width=True)

if not corpora.empty:
    corpus = st.selectbox("Corpus", list(corpora["name"]))
    search = st.text_input("Search documents")
    docs = repo.search_documents(str(corpus), search or None, limit=100)
    st.caption(f"{len(docs)} document(s) shown (max. 100)")
    for doc in docs.itertuples():
        with st.expander(f"{doc.chunk_id} · {doc.title or ''}"):
            st.write(doc.text)
            if doc.source_uri:
                st.caption(doc.source_uri)

st.subheader("Check a dataset version")
dataset = common.pick_dataset()
if dataset is not None:
    version = common.pick_version(dataset, key="check_version")
    if version is not None:
        if version.status in common.EDITABLE and user.is_editor and not corpora.empty:
            target = st.selectbox("Corpus of this version", list(corpora["name"]), key="set_corpus")
            if st.button("Set corpus"):
                common.run_write(
                    lambda: repo.set_corpus(version.version_id, str(target)),
                    lambda _: f"Version {version.version} uses corpus {target} (1 row).",
                )
        with common.handled():
            missing = repo.missing_context_ids(version.version_id)
            if missing.empty:
                st.success("Every ground_truth_context_id exists in the corpus.")
            else:
                st.error(f"{len(missing)} context id(s) are missing from the corpus.")
                st.dataframe(missing, hide_index=True, use_container_width=True)

if user.is_editor:
    st.subheader("Add documents")
    with st.form("new_corpus", clear_on_submit=True):
        name = st.text_input("New corpus name")
        uri = st.text_input("Source URI")
        if st.form_submit_button("Create corpus"):
            common.require(EDITOR)
            common.run_write(
                lambda: repo.create_corpus(name.strip(), source_uri=uri or None),
                lambda c: f"Created corpus {c.name} (1 row).",
            )
    if not corpora.empty:
        target = st.selectbox("Upload into corpus", list(corpora["name"]), key="upload_corpus")
        upload = st.file_uploader(
            "CSV, JSONL or Parquet with columns chunk_id, text, title (optional), source_uri",
            type=["csv", "jsonl", "json", "parquet"],
        )
        if upload is not None and st.button("Upload documents"):
            with common.handled():
                frame = bench_io.read_table(upload, bench_io.detect_format(upload.name))
                keep = [
                    c for c in ("chunk_id", "text", "title", "source_uri") if c in frame.columns
                ]
                records = [
                    {str(k): v for k, v in r.items()} for r in frame[keep].to_dict("records")
                ]
                common.run_write(
                    lambda: repo.add_documents(str(target), records),
                    lambda n: f"Wrote {n} document(s) to {target}.",
                )
