# Changelog

## Unreleased

Changes for the updated specification (2026-11-07):

- Judge prompts of `llm_judge` metrics live in the TruLens prompt registry. Migration `0002`
  replaces `metric_version.prompt_template` with `prompt_version_id` (an exact
  `PromptVersion`, never a label); placeholders use the TruLens syntax `{{name}}` and are
  rendered locally. `MetricStore.create_draft(prompt_text=...)` creates the prompt version.
- Publishing a benchmark version also writes the TruLens dataset `<dataset>@<version>`
  (`Dataset` plus `GroundTruth` rows); new `Repository.sync_trulens_dataset()` and CLI
  `trulens-pg bench sync-trulens`.
- Import page: new source "Production traces", converted with TruLens
  `curate_records_to_dataset` and copied with provenance into a draft.
- Metric editor: write a new prompt version or pick one from the registry; the publish diff
  shows the prompt text.
- Contract tests cover the TruLens prompt, dataset and curation APIs.

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
