# trulens-pg

Postgres-compatible TruLens evaluation with stored custom metrics.

trulens-pg makes the [TruLens](https://github.com/truera/trulens) Run API work on PostgreSQL as
it works on Snowflake: runs read their inputs from Postgres tables, and custom metrics are
stored, calibrated and versioned in the same database. TruLens code stays unchanged.

## What trulens-pg does

Each example assumes `import trulens_pg as tpg` and the objects of the examples before it.
`rag` stands for your RAG app, `pd` is pandas, and `RunConfig` comes from `trulens.core.run`.

**1. Connect TruLens to Postgres.** Creates the `TruSession` and the `trulens_pg` schema of
the metric store.

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

**3. Read run inputs from a Postgres table or view.** Plain TruLens raises
`NotImplementedError` for `source_type="TABLE"` on Postgres.

```python
run = tru_app.add_run(
    RunConfig(
        run_name="from_table",
        source_type="TABLE",
        dataset_name="eval.questions?split=test&limit=100",  # options: split, tags, limit, sample, seed
        dataset_spec={"input": "question", "ground_truth_output": "answer"},
    )
)
run.start()
```

**4. Define an LLM-judge metric as data.** The prompt is stored as a version in the TruLens
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

**5. Use the other metric kinds and bundle metrics in a set.**

```python
store.create_metric("context_recall_at_5", "ground_truth")  # compares with a golden set
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

**6. Compute stored metrics by name.** Accepts metric names, `name@version`, metric sets and
TruLens `Metric` objects. `ground_truth` metrics need a golden set, for example read from a
Postgres table.

```python
golden = pd.read_sql(
    "SELECT question AS query, answer AS expected_response, chunks AS expected_chunks FROM eval.golden",
    tpg.get_context().engine,
)
tpg.compute_metrics(run, ["claims-rag-v1"], golden_set=golden)
metrics = tpg.resolve_metrics(
    ["policy_citation@1"]
)  # TruLens Metric objects, also for Snowflake runs
print(store.run_metrics(run.run_name, "claims_rag"))  # metric versions and judge used by the run
```

**7. Manage the schema from the command line.**

```bash
trulens-pg db upgrade   # create or upgrade the trulens_pg schema
trulens-pg db current   # show the schema revision
```

## Example: EnterpriseRAG-Bench

[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) has 500 questions
about 500,000 internal documents of a fictional company. Your agent indexes the
documents itself; trulens-pg only needs the questions. Download `questions.jsonl` from the
repository. `agent` stands for your RAG agent; `agent.answer(question)` returns the answer text.

```python
import pandas as pd
from trulens.core.run import RunConfig

import trulens_pg as tpg

session = tpg.connect("postgresql+psycopg://user:password@host/evals", migrate=True)

# Questions as a Postgres table
questions = pd.read_json("questions.jsonl", lines=True)
questions[["question_id", "question_type", "question", "gold_answer"]].to_sql(
    "enterprise_rag_bench", tpg.get_context().engine, if_exists="replace", index=False
)

# Run the agent on all 500 questions
tru_app = tpg.wrap(agent, app_name="my_agent", app_version="v1", main_method=agent.answer)
run = tru_app.add_run(
    RunConfig(
        run_name="erb_v1",
        source_type="TABLE",
        dataset_name="enterprise_rag_bench",
        dataset_spec={"input": "question", "ground_truth_output": "gold_answer"},
    )
)
run.start()
```

To score the answers against the gold answers, publish a `ground_truth` metric with
`params={"method": "agreement_measure"}` (example 5) and pass the golden set:
`tpg.compute_metrics(run, ["gold_agreement"], judge=judge, golden_set=questions.rename(columns={"question": "query", "gold_answer": "expected_response"}))`.

## Install

```bash
pip install "trulens-pg @ git+https://github.com/jhagerer/trulens-pg.git"
# with extras: openai, litellm (judges)
pip install "trulens-pg[openai] @ git+https://github.com/jhagerer/trulens-pg.git"
```

Requirements: Python 3.12 or 3.13, PostgreSQL 14 to 17, TruLens 2.16.x, a synchronous driver
(psycopg 3, installed by default, or psycopg2). `postgresql+asyncpg://` URLs are rejected.

## Runnable examples

`examples/` runs without an API key: a keyword-retrieval RAG app over 15 example policy
clauses, 50 example questions, and an offline judge.

```bash
export TRULENS_PG_DATABASE_URL=postgresql+psycopg://user:password@localhost/evals
cd examples
python 01_run_backend.py     # Run API without connector=
python 02_table_source.py    # loads the questions into eval.claims_faq; source_type="TABLE"
python 03_custom_metrics.py  # metric store: create, calibrate, publish, run by name
```

## Python API

| Function | Behaviour |
|---|---|
| `connect(database_url=None, schema="trulens_pg", *, migrate=False, check_schema=True)` | Rejects async drivers, creates the `TruSession`, checks that the metric store schema is at the latest Alembic revision (`migrate=True` upgrades it). |
| `wrap(app, *, app_name, app_version, main_method, feedbacks=None, **kw)` | `TruApp(..., connector=session.connector)` plus `PgRunDao`. |
| `attach(tru_app)` | Installs `PgRunDao` on an existing `TruApp`, `TruChain`, `TruGraph` or `TruLlama`. |
| `resolve_metrics(metrics, judge=None, *, golden_set=None)` | `"name"` (latest published), `"name@2"`, a metric set (`"claims-rag-v1"` or `"set:claims-rag-v1"`) or `Metric` objects → list of TruLens `Metric`. |
| `compute_metrics(run, metrics, judge=None, *, golden_set=None)` | Waits for `INVOCATION_COMPLETED`, resolves, calls `run.compute_metrics`, records metric versions and judge in the `run_metric` table. |
| `MetricStore` | Metric definitions, labels, calibration, publish, metric sets (`.from_context()` or `.from_url()`). |

### Run sources

`RunConfig.dataset_name` accepts `table`, `schema.table` or a view. Reads use SQLAlchemy
reflection and `select()`; identifiers are validated, never formatted into SQL. A column named
in `dataset_spec` that is missing fails `add_run()` before any invocation. JSONB and array
columns arrive as Python lists, so `RETRIEVAL.RETRIEVED_CONTEXTS` works in `LOG_INGESTION`
mode.

Source options are written as a query string in the source name, so they are stored with the
run and can be repeated exactly:

```text
eval.questions?split=test&tags=motor,home&limit=50   # needs the columns split and tags
eval.questions?sample=0.2&seed=2026                   # deterministic by hash of the row ID
```

The row ID for `sample` is the first existing column of `external_id`, `item_id`, `input_id`,
`question_id`, `id`.

### Custom metric kinds

| Kind | Stored definition | Built as |
|---|---|---|
| `llm_judge` | Pinned version of a judge prompt in the TruLens prompt registry, with placeholders (`{{question}}`, `{{answer}}`, `{{contexts}}`, `{{ground_truth}}`), rubric, score range, few-shot examples | Renders the pinned prompt version locally and calls the judge's `generate_score_and_reasons()` (score normalised to 0..1, plus reasons) |
| `criteria` | Criteria text for one allow-listed provider method (`correctness`, `coherence`, `relevance`, `context_relevance`, ...) | `<method>_with_cot_reasons` of the judge with the criteria |
| `ground_truth` | One `GroundTruthAgreement` method (`recall_at_k`, `precision_at_k`, `ndcg_at_k`, `mrr`, `ir_hit_rate`, `agreement_measure`) and `k` | `GroundTruthAgreement` on the `golden_set` passed to `compute_metrics()` |
| `python_ref` | Name of an entry point in the group `trulens_pg.metrics` | The function of an installed, reviewed package |

The database holds no executable code: judge prompts use TruLens `{{variable}}`
placeholders only (no statements, expressions or filters) and are rendered locally without a
model call, and `python_ref` resolves only entry points of installed packages. Each built metric is named `<name>@<version>` in the TruLens dashboard;
judge provider, model and temperature are stored in the result metadata and in
the `run_metric` table.

#### Judge prompts in the TruLens prompt registry

Judge prompts live in the TruLens prompt registry (tables `Prompt`, `PromptVersion`,
`PromptLabel`, TruLens 2.14 or later). They get immutable versions, labels such as
`production`, and a diff view in the TruLens Prompts tab. `metric_version.prompt_version_id`
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

## Cross-platform use with Snowflake

Stored metrics run on Snowflake through `resolve_metrics()`, which returns client-side
`Metric` objects for `run.compute_metrics()`. Scores are comparable only with the same judge
model; the judge is recorded per result. `resolve_metrics()` reads the pinned prompt version
from the Postgres prompt registry, so the client must reach the Postgres database. Whether the
Snowflake connector supports the TruLens prompt tables is not verified (open question in spec
section 8).

## Design decisions

| Question | Decision |
|---|---|
| Where do source options live? | In the source name as a query string (see [Run sources](#run-sources)). They are stored with the run in TruLens' `source_info.name` and need no side table. |
| Minimum labels and agreement before publish | Configurable: `TRULENS_PG_MIN_CALIBRATION_LABELS` (default 20) and `TRULENS_PG_MIN_SPEARMAN` (default: no minimum). |
| Calibration for which kinds? | Required for judge-based kinds (`llm_judge`, `criteria`). `ground_truth` metrics and `python_ref` functions need no calibration. |
| Allow-list for `criteria` | `correctness`, `coherence`, `conciseness`, `helpfulness`, `harmfulness`, `maliciousness`, `controversiality`, `insensitivity`, `relevance`, `context_relevance`. |
| Benchmark datasets | Plain Postgres tables or views, read through `source_type="TABLE"`. trulens-pg has no own dataset store. |
| Python version | 3.12 and 3.13 (the spec listed 3.10 to 3.13; this repository targets 3.12). |

## Known limits

- `PgRunDao` and `attach()` depend on TruLens internals (`DefaultRunDao`, `App.run_dao`,
  private `_object_name`). The version pin is `trulens-core>=2.16,<2.17`;
  `tests/unit/test_trulens_contract.py` runs weekly against the newest TruLens release.
- TruLens 2.16 writes a NaN metric score into a JSON column, which Postgres rejects; the whole
  batch of evaluation spans is lost. trulens-pg turns NaN scores of stored metrics into
  `SkipEval`. Your own `Metric` objects should not return NaN on Postgres.
- A `python_ref` metric version stores the entry point name, not the function code. Changing
  the installed package changes the behaviour of an already published version.
- The read target "100,000 rows in under 10 s" (spec section 8) is not measured yet.

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
| M4 benchmark store and dashboard (G3) | Removed on request: benchmarks are plain Postgres tables |
| M5 cross-platform | Stored metrics usable as client-side `Metric` objects; not tested against Snowflake |

## Licence

MIT, see [LICENSE](LICENSE). TruLens is MIT-licensed as well.
