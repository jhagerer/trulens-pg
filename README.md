# trulens-pg

The [TruLens](https://github.com/truera/trulens) Run API on PostgreSQL, with Postgres tables
as run sources.

TruLens 2.16 already stores traces and results in Postgres, but two Run API features that
work on Snowflake fail there. trulens-pg adds exactly these two; everything else is plain
TruLens code.

| On Postgres with TruLens 2.16 | With trulens-pg |
|---|---|
| `add_run()` raises `ValueError: Run DAO not initialized` unless the app gets `connector=` | `tpg.TruApp()` / `tpg.attach()` install the run backend |
| `source_type="TABLE"` fails: `NotImplementedError: Source table fetch not supported by this DAO` | Runs read tables and views from Postgres |

## What trulens-pg does

Each example assumes `import trulens_pg as tpg` and the objects of the examples before it.
`rag` stands for your RAG app, `pd` is pandas, and `RunConfig` comes from `trulens.core.run`.

**1. Connect TruLens to Postgres.** Creates the TruLens `TruSession` and rejects asynchronous
drivers, which TruLens cannot use.

```python
session = tpg.connect("postgresql+psycopg://user:password@host/evals")
```

**2. Use the Run API without `connector=`.**

```python
tru_app = tpg.TruApp(rag, app_name="claims_rag", app_version="v1", main_method=rag.answer)
# same arguments as trulens.apps.app.TruApp; for an existing TruChain, TruGraph or TruLlama: tpg.attach(tru_chain)
run = tru_app.add_run(
    RunConfig(run_name="smoke", dataset_name="inline", dataset_spec={"input": "input"})
)
run.start(input_df=pd.DataFrame({"input": ["Is glass damage covered?"]}))
```

**3. Read the run inputs from a Postgres table or view.** Optional filters follow the table
name. A column named in `dataset_spec` that does not exist fails `add_run()` at once.

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

**4. Score the run against the ground truth of its table.** The `ground_truth_output` column
from `dataset_spec` is stored on each record, so a TruLens `Metric` selects it from the run.
No dataset is passed again.

```python
from trulens.core import Metric, Selector
from trulens.otel.semconv.trace import SpanAttributes

ground_truth = Selector(
    span_type=SpanAttributes.SpanType.RECORD_ROOT,
    span_attribute=SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
)
exact_match = Metric(
    implementation=lambda answer, gold: float(answer.strip() == gold.strip()),
    name="exact_match",
    selectors={"answer": Selector.select_record_output(), "gold": ground_truth},
)
run.compute_metrics([exact_match])
```

**5. Score stored outputs from a table without calling the app.** TruLens' `LOG_INGESTION`
mode; array and JSONB columns arrive as Python lists.

```python
from trulens.core.enums import Mode

logged = tpg.TruApp(None, app_name="prod_logs", app_version="2026-11")
run = logged.add_run(
    RunConfig(
        run_name="nov_logs",
        source_type="TABLE",
        dataset_name="logs.answers",
        mode=Mode.LOG_INGESTION,
        dataset_spec={
            "input": "question",
            "record_root.output": "answer",
            "retrieval.retrieved_contexts": "contexts",
        },
    )
)
run.start()
```

## Example: EnterpriseRAG-Bench

[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) has 500 questions
about 500,000 internal documents of a fictional company. Your agent indexes the
documents itself; trulens-pg only needs the questions. Download `questions.jsonl` from the
repository. `agent` stands for your RAG agent; `agent.answer(question)` returns the answer text.

```python
import pandas as pd
from trulens.core import Metric, Selector
from trulens.core.run import RunConfig
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.openai import OpenAI

import trulens_pg as tpg

session = tpg.connect("postgresql+psycopg://user:password@host/evals")

# Questions as a Postgres table
questions = pd.read_json("questions.jsonl", lines=True)
questions[["question_id", "question_type", "question", "gold_answer"]].to_sql(
    "enterprise_rag_bench", tpg.get_context().engine, if_exists="replace", index=False
)

# Run the agent on all 500 questions
tru_app = tpg.TruApp(agent, app_name="my_agent", app_version="v1", main_method=agent.answer)
run = tru_app.add_run(
    RunConfig(
        run_name="erb_v1",
        source_type="TABLE",
        dataset_name="enterprise_rag_bench",
        dataset_spec={"input": "question", "ground_truth_output": "gold_answer"},
    )
)
run.start()

# Score each answer against its gold answer with an LLM judge
judge = OpenAI(model_engine="gpt-4o-mini")
gold_agreement = Metric(
    implementation=lambda answer, gold: judge.generate_score(
        system_prompt=f"Rate from 0 to 10 how well the answer agrees with this reference answer: {gold}",
        user_prompt=answer,
    ),
    name="gold_agreement",
    selectors={
        "answer": Selector.select_record_output(),
        "gold": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
        ),
    },
)
run.compute_metrics([gold_agreement])
print(session.get_leaderboard())
```

**Evaluate retrieval and other intermediate steps.** `instrument_method` records methods of
your agent as trace spans without changing its code; call it before `tpg.TruApp(...)`. Metrics
then select these spans. The example assumes `agent.retrieve(query)` returns a list of text
chunks and `agent.rewrite_query(question)` returns a string.

```python
from trulens.core.otel.instrument import instrument_method

# Before tpg.TruApp(...): record the steps of every agent call
instrument_method(
    type(agent),
    "retrieve",
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes={
        SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
    },
)
instrument_method(type(agent), "rewrite_query")  # any other step

# After run.start(): one metric per step
context_relevance = Metric(  # is each retrieved chunk relevant to the question?
    implementation=judge.context_relevance,
    name="context_relevance",
    selectors={
        "question": Selector.select_record_input(),
        "context": Selector.select_context(collect_list=False),
    },
)
rewrite_quality = Metric(  # does the rewritten query keep the meaning of the question?
    implementation=lambda question, rewritten: judge.generate_score(
        system_prompt=f"Rate from 0 to 10 how well this search query keeps the meaning of: {question}",
        user_prompt=rewritten,
    ),
    name="rewrite_quality",
    selectors={
        "question": Selector.select_record_input(),
        "rewritten": Selector(function_name="rewrite_query", function_attribute="return"),
    },
)
run.compute_metrics([context_relevance, rewrite_quality])
```

## Install

```bash
pip install "trulens-pg @ git+https://github.com/jhagerer/trulens-pg.git"
# with a TruLens judge provider: extras openai or litellm
pip install "trulens-pg[openai] @ git+https://github.com/jhagerer/trulens-pg.git"
```

Requirements: Python 3.12 or 3.13, PostgreSQL 14 to 17, TruLens 2.16.x, a synchronous driver
(psycopg 3, installed by default, or psycopg2). `postgresql+asyncpg://` URLs are rejected.

## Runnable examples

`examples/` runs without an API key: a keyword-retrieval RAG app over 15 example policy
clauses and 50 example questions.

```bash
export TRULENS_PG_DATABASE_URL=postgresql+psycopg://user:password@localhost/evals
cd examples
python 01_run_backend.py   # Run API without connector=
python 02_table_source.py  # loads the questions into eval.claims_faq, runs and scores the app
```

## Python API

| Function | Behaviour |
|---|---|
| `connect(database_url=None, **session_kwargs)` | Creates the `TruSession` (URL default: `TRULENS_PG_DATABASE_URL`); `postgresql://` becomes `postgresql+psycopg://`; asynchronous drivers are rejected. |
| `TruApp(app, main_method=None, **kwargs)` | The TruLens `TruApp` (same arguments, subclass of `trulens.apps.app.TruApp`); `connector` defaults to the `connect()` session, and `PgRunDao` is installed. |
| `attach(tru_app)` | Installs `PgRunDao` on an existing TruLens app (`TruApp`, `TruChain`, `TruGraph`, `TruLlama`). |
| `PgRunDao` | The TruLens `DefaultRunDao` plus `fetch_source_data()` for Postgres tables and views. |

### Run sources

`RunConfig.dataset_name` accepts `table`, `schema.table` or a view. Reads use SQLAlchemy
reflection and `select()`; identifiers are validated, never formatted into SQL.

Source options are written as a query string in the source name, so they are stored with the
run and can be repeated exactly:

```text
eval.questions?split=test&tags=motor,home&limit=50   # needs the columns split and tags
eval.questions?sample=0.2&seed=2026                   # deterministic by hash of the row ID
```

The row ID for `sample` is the first existing column of `external_id`, `item_id`, `input_id`,
`question_id`, `id`.

### What a run passes from its table to metrics

In normal app runs TruLens 2.16 uses two `dataset_spec` entries: `input` (passed to the app)
and `ground_truth_output` (stored on each record's root span, selectable by metrics as in
example 4). Other entries are passed to the app as extra positional arguments. In
`LOG_INGESTION` mode, every `dataset_spec` entry becomes a span attribute.

## Known limits

- `PgRunDao` and `attach()` depend on TruLens internals (`DefaultRunDao`, `App.run_dao`,
  private `_object_name`). The version pin is `trulens-core>=2.16,<2.17`;
  `tests/unit/test_trulens_contract.py` runs weekly against the newest TruLens release.
- TruLens 2.16 writes a NaN metric score into a JSON column, which Postgres rejects; then the
  whole batch of evaluation spans is lost. A metric without a score for a record should
  raise `trulens.core.metric.SkipEval` instead of returning NaN.
- Metrics given by name (`run.compute_metrics(["groundedness"])`) are Snowflake server-side
  metrics; on Postgres TruLens skips them with a warning. Use `Metric` objects.
- The read target "100,000 rows in under 10 s" is not measured yet.

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

## Licence

MIT, see [LICENSE](LICENSE). TruLens is MIT-licensed as well.
