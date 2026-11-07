# trulens-pg

Postgres-compatible TruLens evaluation and benchmark management.

trulens-pg makes the [TruLens](https://github.com/truera/trulens) Run API work on PostgreSQL as
it works on Snowflake. It adds three parts in front of one Postgres database. TruLens code
stays unchanged.

| Gap on Postgres with TruLens 2.16 | Fix in trulens-pg |
|---|---|
| **G1** The run backend is wired only with `connector=`; otherwise `ValueError: Run DAO not initialized` | `tpg.wrap()` / `tpg.attach()` install `PgRunDao` |
| **G2** `source_type="TABLE"` fails with `NotImplementedError` | `PgRunDao.fetch_source_data` reads tables, views and published benchmark versions |
| **G3** No dashboard to manage benchmark datasets and metrics | Streamlit dashboard on a versioned `bench` schema |

It also provides a **custom metric store**: LLM-judge metrics are defined, calibrated against
human labels and versioned as Postgres rows. They run by name on Postgres and, as client-side
TruLens `Metric` objects, on Snowflake.

Status: alpha (spec v0.1, milestones M1 to M4 implemented, M5 partly; see [Status](#status)).

## Install

```bash
pip install "trulens-pg @ git+https://github.com/jhagerer/trulens-pg.git"
# with extras: app (Streamlit), openai, litellm (judges), snowflake (export), hf, parquet
pip install "trulens-pg[app,openai] @ git+https://github.com/jhagerer/trulens-pg.git"
```

Requirements: Python 3.12 or 3.13, PostgreSQL 14 to 17, TruLens 2.16.x, a synchronous driver
(psycopg 3, installed by default, or psycopg2). `postgresql+asyncpg://` URLs are rejected.

## Quickstart

```python
import trulens_pg as tpg
from trulens.core.run import RunConfig

session = tpg.connect("postgresql+psycopg://user@host/evals", migrate=True)
tru_app = tpg.wrap(rag, app_name="claims_rag", app_version="v3", main_method=rag.answer)  # G1
run = tru_app.add_run(
    RunConfig(
        run_name="v3_test",
        dataset_name="bench:claims_faq@1.0.0",  # published benchmark version
        source_type="TABLE",  # G2
        dataset_spec={"input": "input", "ground_truth_output": "ground_truth_output"},
    )
)
run.start()
tpg.compute_metrics(run, ["claims-rag-v1"], judge=judge)  # stored metric set
```

`examples/` contains a runnable walk-through on a 50-item example benchmark with a
15-chunk corpus, a keyword-retrieval RAG app and an offline judge (no API key needed):

```bash
export TRULENS_PG_DATABASE_URL=postgresql+psycopg://user:password@localhost/evals
trulens-pg db upgrade
trulens-pg bench import-corpus examples/claims_faq/corpus.jsonl --corpus policies --create
trulens-pg bench import examples/claims_faq/items.jsonl --dataset claims_faq --create \
    --corpus policies --version 1.0.0 --publish --changelog "Example benchmark"
cd examples
python 01_run_backend.py     # G1: Run API without connector=
python 02_table_source.py    # G2: source_type="TABLE" from a benchmark version
python 03_custom_metrics.py  # metric store: create, calibrate, publish, run by name
```

## Python API

| Function | Behaviour |
|---|---|
| `connect(database_url=None, bench_schema="bench", *, migrate=False, check_schema=True)` | Rejects async drivers, creates the `TruSession`, checks that the bench schema is at the latest Alembic revision (`migrate=True` upgrades it). |
| `wrap(app, *, app_name, app_version, main_method, feedbacks=None, **kw)` | `TruApp(..., connector=session.connector)` plus `PgRunDao`. |
| `attach(tru_app)` | Installs `PgRunDao` on an existing `TruApp`, `TruChain`, `TruGraph` or `TruLlama`. |
| `resolve_metrics(metrics, judge=None, *, dataset=None)` | `"name"` (latest published), `"name@2"`, a metric set (`"claims-rag-v1"` or `"set:claims-rag-v1"`) or `Metric` objects → list of TruLens `Metric`. |
| `compute_metrics(run, metrics, judge=None)` | Waits for `INVOCATION_COMPLETED`, resolves, calls `run.compute_metrics`, records metric versions and judge in `bench.run_metric`. |
| `bench_source(dataset, version, *, split, tags, limit, sample, seed)` | Builds a source name with options. |
| `Repository`, `MetricStore` | Benchmark store and metric store (`.from_context()` or `.from_url()`). |

### Run sources (G2)

`RunConfig.dataset_name` accepts `table`, `schema.table`, a view, or
`bench:<dataset>@<version>`. Reads use SQLAlchemy reflection and `select()`; identifiers are
validated, never formatted into SQL. A column named in `dataset_spec` that is missing fails
`add_run()` before any invocation. JSONB and array columns arrive as Python lists, so
`RETRIEVAL.RETRIEVED_CONTEXTS` works in `LOG_INGESTION` mode.

Source options are written as a query string in the source name, so they are stored with the
run and can be repeated exactly:

```text
bench:claims_faq@1.2.0?split=test&tags=motor,home&limit=50
bench:claims_faq@1.2.0?sample=0.2&seed=2026      # deterministic by hash of the item ID
```

### Custom metric kinds

| Kind | Stored definition | Built as |
|---|---|---|
| `llm_judge` | Prompt template (`{question}`, `{answer}`, `{contexts}`, `{ground_truth}`), rubric, score range, few-shot examples | Fills the template and calls the judge's `generate_score_and_reasons()` (score normalised to 0..1, plus reasons) |
| `criteria` | Criteria text for one allow-listed provider method (`correctness`, `coherence`, `relevance`, `context_relevance`, ...) | `<method>_with_cot_reasons` of the judge with the criteria |
| `ground_truth` | One `GroundTruthAgreement` method (`recall_at_k`, `precision_at_k`, `ndcg_at_k`, `mrr`, `ir_hit_rate`, `agreement_measure`) and `k` | Golden set from the benchmark version and its corpus |
| `python_ref` | Name of an entry point in the group `trulens_pg.metrics` | The function of an installed, reviewed package |

The database holds no executable code: templates allow plain placeholder names only (no
attribute access, indexing or format specs), and `python_ref` resolves only entry points of
installed packages. Each built metric is named `<name>@<version>` in the TruLens dashboard;
judge provider, model and temperature are stored in the result metadata and in
`bench.run_metric`.

Register your own `python_ref` metric in your package's `pyproject.toml`:

```toml
[project.entry-points."trulens_pg.metrics"]
policy_regex = "my_package.metrics:policy_regex"
```

## Benchmark store

The schema `bench` (configurable) holds `dataset`, `dataset_version`, `item`, `corpus`,
`document`, `review`, `run_link` and the metric tables `metric`, `metric_version`,
`metric_label`, `metric_calibration`, `metric_set`, `run_metric`. Only trulens-pg migrations
touch it; TruLens keeps managing its own tables.

- Published versions are read-only, enforced by database triggers (not only in the app). The
  only allowed change is `published` → `deprecated`. Changes create a new draft.
- `content_hash` is SHA-256 over the sorted, normalised items; publishing reports versions
  with the same hash.
- Each published version gets a view `bench.v_<dataset>__<major>_<minor>_<patch>` with the
  TruLens `dataset_spec` names `input`, `ground_truth_output`, `ground_truth_context_ids`,
  `item_id`, `split`, `tags`.
- Items with `contains_pii = true` are left out of exports unless `include_pii=True`.

## Streamlit dashboard (G3)

```bash
pip install "trulens-pg[app] @ git+https://github.com/jhagerer/trulens-pg.git"
trulens-pg app --port 8502
# or: docker build -t trulens-pg-app . && docker run -p 8502:8502 -e TRULENS_PG_DATABASE_URL=... trulens-pg-app
```

Pages — Benchmarks: Datasets, Import, Items, Corpus, Review, Publish, Runs, Export.
Metrics: Metrics, Metric editor, Playground, Calibration, Metric publish, Metric sets.
Pages use only the repository and metric store; every write is one transaction and is
confirmed with the affected row count. Playground and calibration show the number of judge
calls before they start.

| Variable | Meaning |
|---|---|
| `TRULENS_PG_DATABASE_URL` | Postgres URL |
| `TRULENS_PG_BENCH_SCHEMA` | Bench schema (default `bench`) |
| `TRULENS_PG_DASHBOARD_URL` | TruLens dashboard, linked from the Runs page |
| `TRULENS_PG_AUTH` | `oidc` (default, `st.login` with `[auth]` in `.streamlit/secrets.toml`), `proxy` (user from `X-Forwarded-Email` or `TRULENS_PG_AUTH_HEADER`), `none` (local only) |
| `TRULENS_PG_ROLES` | e.g. `alice@x.com=publisher; *=editor`. Unset: every user is editor and publisher |
| `TRULENS_PG_MIN_CALIBRATION_LABELS` | Labels needed before a judge metric is published (default 20) |
| `TRULENS_PG_MIN_SPEARMAN` | Optional minimum Spearman correlation for publishing |
| `TRULENS_PG_JUDGE_PROVIDER`, `_MODEL`, `_TEMPERATURE` | Default judge (`openai`, `azure_openai`, `litellm`); API keys come from the provider's own variables |

Roles: an *editor* imports, edits, reviews, labels and runs the playground; a *publisher*
also publishes, deprecates, exports and creates metric sets.

## CLI

```text
trulens-pg db upgrade | current | downgrade REV
trulens-pg bench list [DATASET]
trulens-pg bench import FILE --dataset NAME [--create] [--version X.Y.Z] [--map query=question]
                         [--append] [--corpus NAME] [--publish] [--dry-run] [--hf PATH]
trulens-pg bench import-corpus FILE --corpus NAME [--create]
trulens-pg bench export DATASET@VERSION --target jsonl|csv|parquet|snowflake [--include-pii]
                         [--snowflake-table DB.SCHEMA.TABLE --snowflake-connection NAME]
trulens-pg app [--port 8502]
```

Imports above 100,000 rows belong in the CLI; the app refuses them.

## Cross-platform use with Snowflake

`trulens-pg bench export claims_faq@1.0.0 --target snowflake --snowflake-table DB.EVAL.CLAIMS_FAQ_1_0_0`
writes the contract columns, so the same `dataset_spec` works there. Stored metrics run on
Snowflake through `resolve_metrics()`, which returns client-side `Metric` objects for
`run.compute_metrics()`. Scores are comparable only with the same judge model; the judge is
recorded per result. `Repository.link_run(..., platform="snowflake")` records Snowflake runs
in `bench.run_link`.

## Design decisions on the open questions of the spec

| Question (spec section 8) | Decision in this implementation |
|---|---|
| Where do source options live? | In the source name as a query string (see [Run sources](#run-sources-g2)). They are stored with the run in TruLens' `source_info.name`, need no side table, and `PgRunDao.fetch_source_data(source_name)` receives them without knowing the run. |
| Minimum labels and agreement before publish | Configurable: `TRULENS_PG_MIN_CALIBRATION_LABELS` (default 20) and `TRULENS_PG_MIN_SPEARMAN` (default: no minimum). |
| Calibration for which kinds? | Required for judge-based kinds (`llm_judge`, `criteria`). `ground_truth` IR metrics and `python_ref` functions are deterministic and need no calibration. |
| Allow-list for `criteria` | `correctness`, `coherence`, `conciseness`, `helpfulness`, `harmfulness`, `maliciousness`, `controversiality`, `insensitivity`, `relevance`, `context_relevance`. |
| Dataset names | Lower-case slugs (`claims_faq`), so view names are valid identifiers; long names go into `description`. |
| Item identity across versions | `external_id` (stable; derived from the query text if the source has none). Diffs and sampling use it. |
| Review gate | Publishing needs status `in_review` and an `accept` or `reject` decision for every item; `edit` keeps an item open; rejected items are removed on publish. The CLI can publish without review (`--publish`) for automated imports. |
| Python version | 3.12 and 3.13 (the spec listed 3.10 to 3.13; this repository targets 3.12). |

## Known limits

- `PgRunDao` and `attach()` depend on TruLens internals (`DefaultRunDao`, `App.run_dao`,
  private `_object_name`). The version pin is `trulens-core>=2.16,<2.17`;
  `tests/unit/test_trulens_contract.py` runs weekly against the newest TruLens release.
- TruLens 2.16 writes a NaN metric score into a JSON column, which Postgres rejects; the whole
  batch of evaluation spans is lost. trulens-pg turns NaN scores of stored metrics into
  `SkipEval`. Your own `Metric` objects should not return NaN on Postgres.
- The Runs page reads TruLens results for Postgres runs; Snowflake runs are listed with
  their metadata only.
- The read target "100,000 items in under 10 s" (spec section 8) is not measured yet.

## Development

```bash
uv sync                                  # Python 3.12, all dev tools
uv run pre-commit install                # ruff, ruff-format, mypy, gitleaks, file checks
uv run pytest tests/unit                 # no network, no database
export TRULENS_PG_TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost/trulens_pg_test
uv run pytest                            # plus integration tests (database is reset)
```

Tooling: [uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/) (line length
100), [mypy](https://mypy.readthedocs.io/) (strict), [gitleaks](https://github.com/gitleaks/gitleaks),
[pre-commit](https://pre-commit.com/). CI runs the same hooks, unit tests, and integration
tests against `postgres:14/16/17-alpine` on Python 3.12 and 3.13. Version numbers come from
git tags (`vX.Y.Z`); the release workflow attaches the wheel to each GitHub release.

## Status

| Milestone | State |
|---|---|
| M1 package, run backend wiring (G1) | Implemented and tested |
| M2 table sources (G2) | Implemented and tested; 100k-row performance target not measured |
| M3 custom metric store | Implemented and tested with an offline judge; not yet run against a live judge provider |
| M4 benchmark store and dashboard (G3) | Implemented; pages covered by Streamlit AppTest smoke tests; usability test pending |
| M5 cross-platform | Snowflake export implemented but not tested against a Snowflake account; upstream TruLens pull requests not opened |

## Licence

MIT, see [LICENSE](LICENSE). TruLens is MIT-licensed as well.
