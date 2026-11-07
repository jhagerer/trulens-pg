"""Import: upload CSV, JSONL, Parquet or load a Hugging Face dataset into a new draft."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import EDITOR
from trulens_pg.bench import io as bench_io
from trulens_pg.bench.schemas import ITEM_FIELDS

st.title("Import")
repo = common.repo()
user = common.require(EDITOR)

dataset = common.pick_dataset()
if dataset is None:
    st.stop()

source = st.radio("Source", ["File upload", "Hugging Face"], horizontal=True)
df: pd.DataFrame | None = None
with common.handled():
    if source == "File upload":
        upload = st.file_uploader(
            "CSV, JSONL or Parquet", type=["csv", "tsv", "jsonl", "json", "parquet"]
        )
        if upload is not None:
            df = bench_io.read_table(upload, bench_io.detect_format(upload.name))
    else:
        path = st.text_input("Dataset path, e.g. org/name")
        cols = st.columns(3)
        split = cols[0].text_input("Split", value="test")
        config = cols[1].text_input("Config name (optional)")
        limit = cols[2].number_input("Max rows", 1, bench_io.BROWSER_IMPORT_LIMIT, 1000)
        if path and st.button("Load"):
            st.session_state["hf_frame"] = bench_io.load_huggingface(
                path, split=split, name=config or None, limit=int(limit)
            )
        df = st.session_state.get("hf_frame")

if df is None:
    st.stop()
if len(df) > bench_io.BROWSER_IMPORT_LIMIT:
    st.error(
        f"{len(df):,} rows: imports above {bench_io.BROWSER_IMPORT_LIMIT:,} rows run in the CLI: "
        f"`trulens-pg bench import FILE --dataset {dataset}`."
    )
    st.stop()

st.caption(f"{len(df):,} rows, columns: {', '.join(map(str, df.columns))}")
guess = bench_io.guess_mapping([str(c) for c in df.columns])
options = ["—", *map(str, df.columns)]
st.subheader("Column mapping")
mapping: dict[str, str] = {}
grid = st.columns(3)
for i, field in enumerate(ITEM_FIELDS):
    default = guess.get(field)
    choice = grid[i % 3].selectbox(
        field,
        options,
        index=options.index(default) if default in options else 0,
        key=f"map_{field}",
    )
    if choice != "—":
        mapping[field] = choice
st.caption("Unmapped columns are stored in the item metadata.")

with common.handled():
    report = bench_io.validate_frame(df, mapping)
    valid, invalid = len(report.items), len(report.errors)
    st.metric(
        "Valid rows",
        f"{valid:,}",
        delta=f"{invalid} invalid" if invalid else None,
        delta_color="inverse",
    )
    if report.errors:
        st.dataframe(pd.DataFrame(report.errors), hide_index=True, use_container_width=True)
    if report.duplicate_ids:
        st.error(f"Duplicate external_id values: {report.duplicate_ids[:20]}")
    st.dataframe(
        pd.DataFrame([i.model_dump() for i in report.items[:50]]),
        hide_index=True,
        use_container_width=True,
    )

    mode = st.radio(
        "Write to",
        ["new draft from the latest version (update matching external_id)", "new empty draft"],
    )
    version = st.text_input("New version (optional)")
    if st.button("Write items", type="primary", disabled=not report.ok or not user.is_editor):
        append = mode.startswith("new draft from")

        def write() -> tuple[str, int]:
            draft = repo.create_draft(
                dataset,
                version.strip() or None,
                copy_from="latest" if append else None,
                created_by=user.name,
            )
            n = repo.add_items(draft.version_id, report.items, upsert=append)
            return draft.version, n

        common.run_write(write, lambda r: f"Wrote {r[1]:,} item(s) to draft {dataset}@{r[0]}.")
