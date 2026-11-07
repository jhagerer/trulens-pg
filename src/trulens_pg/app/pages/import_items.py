"""Import: upload CSV, JSONL, Parquet or load a Hugging Face dataset into a new draft."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trulens_pg import trulens_store
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

source = st.radio("Source", ["File upload", "Hugging Face", "Production traces"], horizontal=True)
df: pd.DataFrame | None = None
with common.handled():
    if source == "File upload":
        upload = st.file_uploader(
            "CSV, JSONL or Parquet", type=["csv", "tsv", "jsonl", "json", "parquet"]
        )
        if upload is not None:
            df = bench_io.read_table(upload, bench_io.detect_format(upload.name))
    elif source == "Hugging Face":
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
    elif source == "Production traces":
        session = common.trulens_session()
        st.caption(
            "Recorded traces are converted with TruLens curate_records_to_dataset "
            "(column mapping, provenance metadata, error report)."
        )
        cols = st.columns(2)
        app_name = cols[0].text_input("App name")
        app_version = cols[1].text_input("App version (optional)")
        if app_name:
            records, metric_names = session.get_records_and_feedback(
                app_name=app_name, app_versions=[app_version] if app_version else None
            )
            st.caption(f"{len(records):,} recorded trace(s)")
            cols = st.columns(2)
            since = cols[0].date_input("From", value=None)
            until = cols[1].date_input("Until", value=None)
            cols = st.columns(3)
            metric = cols[0].selectbox("Filter by metric", ["", *metric_names])
            min_score = cols[1].number_input("Minimum score", value=0.0)
            max_score = cols[2].number_input("Maximum score", value=1.0)
            selected = trulens_store.filter_traces(
                records,
                since=pd.Timestamp(since, tz="UTC") if since else None,
                until=pd.Timestamp(until, tz="UTC") + pd.Timedelta(days=1) if until else None,
                metric=metric or None,
                min_score=min_score if metric else None,
                max_score=max_score if metric else None,
            )
            st.caption(f"{len(selected):,} trace(s) selected")
            use_output = st.checkbox(
                "Use the recorded output as draft ground truth answer (check it in Review)"
            )
            if st.button("Convert traces", disabled=selected.empty):
                result, items = trulens_store.traces_to_items(
                    common.store().trulens,
                    selected,
                    dataset=dataset,
                    expected_response="output" if use_output else None,
                )
                st.session_state["trace_items"] = (result, items)
            if "trace_items" in st.session_state:
                result, items = st.session_state["trace_items"]
                st.info(
                    f"Curation: {result.accepted} accepted, {result.duplicates} duplicate(s), "
                    f"{result.rejected} rejected (staged as TruLens dataset {result.dataset_name})."
                )
                if result.rejected:
                    st.dataframe(result.errors_df(), hide_index=True, use_container_width=True)
                df = pd.DataFrame(items)

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
