# Changelog

## Unreleased

First implementation of the specification "trulens-pg: Postgres-compatible TruLens evaluation
and benchmark management" (v0.1, 2026-11-06).

- `connect()`, `wrap()`, `attach()`: TruLens Run API on Postgres without `connector=` (G1).
- `PgRunDao.fetch_source_data`: `source_type="TABLE"` for tables, views and
  `bench:<dataset>@<version>` with source options `split`, `tags`, `limit`, `sample`, `seed` (G2).
- Bench schema with Alembic migration `0001`: benchmark store, metric store, read-only
  triggers for published versions, versioned views.
- Benchmark repository: import (CSV, JSONL, Parquet, Hugging Face), edit, review, publish with
  content hash, deprecate, corpus checks, export (JSONL, CSV, Parquet, Snowflake).
- Custom metric store: kinds `llm_judge`, `criteria`, `ground_truth`, `python_ref`;
  calibration (Spearman correlation, mean absolute error); publish gate; metric sets;
  `resolve_metrics()` and `compute_metrics()`.
- Streamlit dashboard with 8 benchmark pages and 6 metric pages, OIDC / proxy login, roles (G3).
- CLI `trulens-pg db | bench | app`.
- Example benchmark (50 items, 15-chunk corpus) and runnable example scripts.
