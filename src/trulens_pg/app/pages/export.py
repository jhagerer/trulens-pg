"""Export: a published version to JSONL, CSV, Parquet, or a Snowflake table."""

from __future__ import annotations

import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import PUBLISHER
from trulens_pg.bench import io as bench_io

st.title("Export")
repo = common.repo()
user = common.require(PUBLISHER)

dataset = common.pick_dataset()
if dataset is None:
    st.stop()
version = common.pick_version(dataset, statuses=common.FROZEN)
if version is None:
    st.stop()

include_pii = st.checkbox("Include items that contain personal data (contains_pii)")
frame = repo.version_frame(version.version_id, include_pii=include_pii)
st.caption(f"{len(frame):,} item(s) in the benchmark contract columns")
st.dataframe(frame.head(20), hide_index=True, use_container_width=True)

target = st.selectbox("Target", ["jsonl", "csv", "parquet", "snowflake"])
if target == "snowflake":
    table = st.text_input("Snowflake table (DATABASE.SCHEMA.TABLE)")
    connection = st.text_input("Connection name (~/.snowflake/connections.toml)")
    if st.button(
        "Export to Snowflake", type="primary", disabled=not table or not user.is_publisher
    ):

        def export() -> int:
            params = {"connection_name": connection} if connection else {}
            return bench_io.export_snowflake(frame, table, connection_params=params)

        common.run_write(export, lambda n: f"Exported {n:,} item(s) to {table}.")
else:
    with common.handled():
        st.download_button(
            f"Download {target.upper()}",
            data=bench_io.to_bytes(frame, target),  # type: ignore[arg-type]
            file_name=f"{dataset}-{version.version}.{target}",
            disabled=not user.is_publisher,
        )
