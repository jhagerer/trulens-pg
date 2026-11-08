# trulens-pg

Postgres-compatible TruLens evaluation and benchmark management.

trulens-pg makes the [TruLens](https://github.com/truera/trulens) Run API work on PostgreSQL as
it works on Snowflake, and adds a versioned benchmark store and a custom metric store in the
same database. TruLens code stays unchanged.

## What trulens-pg does

Each example assumes `import trulens_pg as tpg` and the objects of the examples before it.
`rag` stands for your RAG app, `pd` is pandas, and `RunConfig` comes from `trulens.core.run`.

**1. Connect TruLens to Postgres.** Creates the `TruSession` and the `bench` schema.

```python
session = tpg.connect("postgresql+psycopg://user:password@host/evals", migrate=True)
```

**2. Use the TruLens Run API without `connector=`.** Plain TruLens raises
`Run DAO not initialized` on Postgres.

```python
tru_app = tpg.wrap(rag, app_name="claims_rag", app_version="v1", main_method=rag.answer)
# or for an existing TruApp, TruChain, TruGraph or TruLlama: tpg.attach(tru_chain)
run = tru_app.add_run(
    RunConfig(run_name="smoke", dataset_name="inline", dataset_spec={"input": "input"})
)
run.start(input_df=pd.DataFrame({"input": ["Is glass damage covered?"]}))
```

**3. Read run inputs from a Postgres table.** Plain TruLens raises `NotImplementedError` for
`source_type="TABLE"` on Postgres.

```python
run = tru_app.add_run(
    RunConfig(
        run_name="from_table",
        source_type="TABLE",
        dataset_name="eval.questions?split=test&limit=100",  # table or view, with optional filters
        dataset_spec={"input": "question", "ground_truth_output": "answer"},
    )
)
run.start()
```

**4. Keep benchmarks as versioned datasets.** A published version is read-only, gets the view
`bench.v_claims_faq__1_0_0`, and is also written as the TruLens dataset `claims_faq@1.0.0`.

```python
from trulens_pg.bench import Repository, io

repo = Repository.from_context()
repo.create_dataset("claims_faq")
draft = repo.create_draft("claims_faq", "1.0.0", copy_from=None)
repo.add_items(draft.version_id, io.validate_frame(io.read_table("claims_faq.csv")).items)
repo.submit_for_review(draft.version_id)
for item_id in repo.review_queue(draft.version_id)["item_id"]:
    repo.add_review(item_id, reviewer="alice", decision="accept")
repo.publish_version(draft.version_id, published_by="bob", changelog="First version")
```

**5. Run your app on a benchmark version.**

```python
run = tru_app.add_run(
    RunConfig(
        run_name="v1_on_1.0.0",
        source_type="TABLE",
        dataset_name="bench:claims_faq@1.0.0",
        dataset_spec={"input": "input", "ground_truth_output": "ground_truth_output"},
    )
)
run.start()
```

**6. Store the documents that answers should come from.** A draft version gets a corpus;
`missing_context_ids` lists `ground_truth_context_ids` entries that are not in it.

```python
repo.create_corpus("policies")
repo.add_documents("policies", [{"chunk_id": "KFZ-1001", "title": "Glass damage", "text": "..."}])
draft = repo.create_draft("claims_faq")  # version 1.1.0, copied from 1.0.0
repo.set_corpus(draft.version_id, "policies")
print(repo.missing_context_ids(draft.version_id))  # empty if every ID exists
```

**7. Turn production traces into benchmark items.** Uses TruLens `curate_records_to_dataset`
and keeps the source record in the item metadata.

```python
from trulens_pg import trulens_store

records, _ = session.get_records_and_feedback(app_name="claims_rag")
failures = trulens_store.filter_traces(records, metric="policy_citation@1", max_score=0.5)
_, items = trulens_store.traces_to_items(repo.trulens, failures, dataset="claims_faq")
repo.add_items(repo.create_draft("claims_faq").version_id, items)
```

**8. Export a benchmark version.** Items with personal data are left out by default.

```python
frame = repo.version_frame(repo.get_version("claims_faq", "1.0.0").version_id)
io.write_file(frame, "claims_faq-1.0.0.parquet")
io.export_snowflake(
    frame, "EVALS.BENCH.CLAIMS_FAQ_1_0_0", connection_params={"connection_name": "prod"}
)
```

**9. Define an LLM-judge metric as data.** The prompt is stored as a version in the TruLens
prompt registry; the metric pins that exact version. Publishing requires a calibration
against human scores.

```python
store = tpg.MetricStore.from_context()
store.create_metric("policy_citation", "llm_judge")
draft = store.create_draft(
    "policy_citation",
    prompt_text="Question: {{question}}\nAnswer: {{answer}}",
    rubric="3: the answer names the policy clause it uses. 0: it names none.",
    judge={"provider": "openai", "model": "gpt-4o-mini"},
)
store.add_labels("policy_citation", [{"question": "...", "answer": "...", "human_score": 3}])
print(store.calibrate(draft.metric_version_id).spearman)  # agreement with the human scores
store.publish(draft.metric_version_id, published_by="bob")  # needs 20 labels by default
```

**10. Use the other metric kinds and bundle metrics in a set.**

```python
store.create_metric("context_recall_at_5", "ground_truth")  # compares with the benchmark
store.create_draft("context_recall_at_5", params={"method": "recall_at_k", "k": 5})
store.create_metric("guideline_correctness", "criteria")  # allow-listed TruLens judge method
store.create_draft(
    "guideline_correctness",
    params={"method": "correctness"},
    criteria="Follows the claims guideline.",
)
store.create_metric("has_policy_number", "python_ref")  # function of an installed package
store.create_draft("has_policy_number", python_ref="contains_policy_number")
# after publishing each version:
store.create_metric_set(
    "claims-rag-v1", ["policy_citation@1", "context_recall_at_5@1", "has_policy_number@1"]
)
```

**11. Compute stored metrics by name.** Accepts metric names, `name@version`, metric sets and
TruLens `Metric` objects.

```python
tpg.compute_metrics(run, ["claims-rag-v1"])
metrics = tpg.resolve_metrics(
    ["policy_citation@1"]
)  # TruLens Metric objects, also for Snowflake runs
```

**12. Compare runs per benchmark version.**

```python
version = repo.get_version("claims_faq", "1.0.0")
print(repo.runs_for_version(version.version_id))  # runs that used this version
print(repo.run_metric_summary(version.version_id, session))  # mean, spread, count per metric
```

**13. Use the command line.**

```bash
trulens-pg db upgrade
trulens-pg bench import claims_faq.csv --dataset claims_faq --create --version 1.0.0 --publish
trulens-pg bench export claims_faq@1.0.0 --target jsonl --out claims_faq.jsonl
```

## Example: EnterpriseRAG-Bench

[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) has 500 questions
about 500,000 internal documents of a fictional company. Download `all_documents.zip`
from its latest release and `questions.jsonl` from the repository. `agent` stands for your RAG
agent; `agent.answer(question)` returns the answer text.

```python
import zipfile
from pathlib import Path

import pandas as pd
from trulens.core.run import RunConfig

import trulens_pg as tpg
from trulens_pg.bench import Repository, io

session = tpg.connect("postgresql+psycopg://user:password@host/evals", migrate=True)
repo = Repository.from_context()

# Documents: file names start with the document ID, "dsid_<id>__<name>.txt"
repo.create_corpus("erb_documents")
with zipfile.ZipFile("all_documents.zip") as archive:
    repo.add_documents(
        "erb_documents",
        (
            {"chunk_id": Path(name).name.split("__")[0], "text": archive.read(name).decode()}
            for name in archive.namelist()
            if name.endswith(".txt")
        ),
    )

# Questions as the published benchmark version enterprise_rag_bench@1.0.0
repo.create_dataset("enterprise_rag_bench")
draft = repo.create_draft("enterprise_rag_bench", "1.0.0", copy_from=None)
repo.set_corpus(draft.version_id, "erb_documents")
questions = pd.read_json("questions.jsonl", lines=True)
repo.add_items(
    draft.version_id,
    io.validate_frame(
        questions,
        {
            "external_id": "question_id",
            "query": "question",
            "ground_truth_answer": "gold_answer",
            "ground_truth_context_ids": "expected_doc_ids",
            "tags": "question_type",
        },
    ).items,
)
repo.publish_version(draft.version_id, published_by="me", require_review=False)

# Run the agent on all 500 questions
tru_app = tpg.wrap(agent, app_name="my_agent", app_version="v1", main_method=agent.answer)
run = tru_app.add_run(
    RunConfig(
        run_name="erb_v1",
        source_type="TABLE",
        dataset_name="bench:enterprise_rag_bench@1.0.0",
        dataset_spec={"input": "input", "ground_truth_output": "ground_truth_output"},
    )
)
run.start()
```

The columns `answer_facts` and `source_types` are kept in the item metadata. To score the
answers, compute metrics as in examples 9 to 11, for example a `ground_truth` metric with
`params={"method": "agreement_measure"}`, which compares each answer with the gold answer.

## Install

```bash
pip install "trulens-pg @ git+https://github.com/jhagerer/trulens-pg.git"
# with extras: openai, litellm (judges), snowflake (export), hf, parquet
pip install "trulens-pg[openai] @ git+https://github.com/jhagerer/trulens-pg.git"
```

Requirements: Python 3.12 or 3.13, PostgreSQL 14 to 17, TruLens 2.16.x, a synchronous driver
(psycopg 3, installed by default, or psycopg2). `postgresql+asyncpg://` URLs are rejected.

## Runnable examples

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
| `llm_judge` | Pinned version of a judge prompt in the TruLens prompt registry, with placeholders (`{{question}}`, `{{answer}}`, `{{contexts}}`, `{{ground_truth}}`), rubric, score range, few-shot examples | Renders the pinned prompt version locally and calls the judge's `generate_score_and_reasons()` (score normalised to 0..1, plus reasons) |
| `criteria` | Criteria text for one allow-listed provider method (`correctness`, `coherence`, `relevance`, `context_relevance`, ...) | `<method>_with_cot_reasons` of the judge with the criteria |
| `ground_truth` | One `GroundTruthAgreement` method (`recall_at_k`, `precision_at_k`, `ndcg_at_k`, `mrr`, `ir_hit_rate`, `agreement_measure`) and `k` | Golden set from the benchmark version and its corpus |
| `python_ref` | Name of an entry point in the group `trulens_pg.metrics` | The function of an installed, reviewed package |

The database holds no executable code: judge prompts use TruLens `{{variable}}`
placeholders only (no statements, expressions or filters) and are rendered locally without a
model call, and `python_ref` resolves only entry points of installed packages. Each built metric is named `<name>@<version>` in the TruLens dashboard;
judge provider, model and temperature are stored in the result metadata and in
`bench.run_metric`.

#### Judge prompts in the TruLens prompt registry

Judge prompts live in the TruLens prompt registry (tables `Prompt`, `PromptVersion`,
`PromptLabel`, TruLens 2.14 or later). They get immutable versions, labels such as
`production`, and a diff view in the TruLens Prompts tab. `bench.metric_version.prompt_version_id`
pins an exact `PromptVersion` id, never a label, so moving a label cannot change a published
metric.

```python
store = tpg.MetricStore.from_context()
store.create_metric("policy_citation", "llm_judge")
# Creates a version of the prompt "trulens-pg.policy_citation" and pins it:
draft = store.create_draft(
    "policy_citation",
    prompt_text="Question: {{question}}\n\nAnswer: {{answer}}",
    rubric="3: the answer names the clause it relies on. 0: no clause.",
)
# Or pin a version created elsewhere (TruSession.create_prompt_version):
store.create_draft("policy_citation", prompt_version_id="prompt_version_hash_...")
```

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
- On publish, the version is also written as a TruLens dataset (`Dataset` plus `GroundTruth`
  rows, in the TruLens tables) named `<dataset>@<version>`. Mapping: `query` → `query`,
  `external_id` → `query_id`, `ground_truth_answer` → `expected_response`, the texts of the
  `ground_truth_context_ids` chunks → `expected_chunks`; split, tags, difficulty, language,
  PII flag and metadata → `meta`. `TruSession.get_ground_truth("claims_faq@1.0.0")` and
  `GroundTruthAgreement` then work on published benchmarks. Versions, review, corpus and
  metric tables stay in `bench`, because TruLens datasets have none of these. Writing is
  idempotent; `trulens-pg bench sync-trulens` repeats it.

## CLI

```text
trulens-pg db upgrade | current | downgrade REV
trulens-pg bench list [DATASET]
trulens-pg bench import FILE --dataset NAME [--create] [--version X.Y.Z] [--map query=question]
                         [--append] [--corpus NAME] [--publish] [--dry-run] [--hf PATH]
trulens-pg bench import-corpus FILE --corpus NAME [--create]
trulens-pg bench sync-trulens [DATASET@VERSION]   # write published versions as TruLens datasets
trulens-pg bench export DATASET@VERSION --target jsonl|csv|parquet|snowflake [--include-pii]
                         [--snowflake-table DB.SCHEMA.TABLE --snowflake-connection NAME]
```

Imports above 100,000 rows belong in the CLI; the app refuses them.

## Cross-platform use with Snowflake

`trulens-pg bench export claims_faq@1.0.0 --target snowflake --snowflake-table DB.EVAL.CLAIMS_FAQ_1_0_0`
writes the contract columns, so the same `dataset_spec` works there. Stored metrics run on
Snowflake through `resolve_metrics()`, which returns client-side `Metric` objects for
`run.compute_metrics()`. Scores are comparable only with the same judge model; the judge is
recorded per result. `Repository.link_run(..., platform="snowflake")` records Snowflake runs
in `bench.run_link`.

Open question (spec section 8): whether the Snowflake connector supports the TruLens prompt
and dataset tables is not verified. `resolve_metrics()` reads the pinned prompt version from
the Postgres registry and builds the `Metric` on the client, so a Snowflake run works when the
client can reach the Postgres database. Otherwise the judge prompt text and the dataset must
be exported separately.

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
| Old metric versions with `prompt_template` | Migration `0002` replaces the column with `prompt_version_id`. It stops with an error if a stored version still holds a template; move such templates into the registry first. |
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
- Publishing writes the bench schema and the TruLens dataset in two transactions (TruLens
  uses its own sessions). If the second write fails, the version stays published and the
  error says so; `trulens-pg bench sync-trulens` repeats the idempotent write.
- Each trace import leaves its staging dataset in the TruLens tables as a provenance record.

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
| M4 benchmark store (G3) | Benchmark store, review and publish implemented (Python API and CLI); the Streamlit dashboard was removed on request |
| M5 cross-platform | Snowflake export implemented but not tested against a Snowflake account; upstream TruLens pull requests not opened |

## Licence

MIT, see [LICENSE](LICENSE). TruLens is MIT-licensed as well.
