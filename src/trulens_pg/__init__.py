"""trulens-pg: the TruLens Run API on PostgreSQL, plus a custom metric store.

Quickstart::

    import trulens_pg as tpg
    from trulens.core.run import RunConfig

    session = tpg.connect("postgresql+psycopg://user@host/evals", migrate=True)
    tru_app = tpg.wrap(rag, app_name="claims_rag", app_version="v3", main_method=rag.answer)
    run = tru_app.add_run(
        RunConfig(
            run_name="v3_test",
            dataset_name="eval.claims_faq",
            source_type="TABLE",
            dataset_spec={"input": "question", "ground_truth_output": "answer"},
        )
    )
    run.start()
    tpg.compute_metrics(run, ["policy_citation@1"], judge=judge)
"""

from trulens_pg.dao import PgRunDao
from trulens_pg.errors import (
    MetricResolutionError,
    NotConnectedError,
    NotFoundError,
    ReadOnlyError,
    SchemaNotReadyError,
    SourceError,
    TrulensPgError,
    UnsupportedDatabaseError,
    ValidationError,
    WorkflowError,
)
from trulens_pg.metrics import JudgeConfig, MetricStore, compute_metrics, resolve_metrics
from trulens_pg.session import connect, get_context
from trulens_pg.wiring import attach, wrap

try:
    from trulens_pg._version import __version__
except ImportError:  # pragma: no cover - source tree without build
    __version__ = "0.0.0"

__all__ = [
    "JudgeConfig",
    "MetricResolutionError",
    "MetricStore",
    "NotConnectedError",
    "NotFoundError",
    "PgRunDao",
    "ReadOnlyError",
    "SchemaNotReadyError",
    "SourceError",
    "TrulensPgError",
    "UnsupportedDatabaseError",
    "ValidationError",
    "WorkflowError",
    "__version__",
    "attach",
    "compute_metrics",
    "connect",
    "get_context",
    "resolve_metrics",
    "wrap",
]
