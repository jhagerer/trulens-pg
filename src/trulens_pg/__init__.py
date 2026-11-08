"""trulens-pg: the TruLens Run API on PostgreSQL, with Postgres tables as run sources.

Quickstart::

    import trulens_pg as tpg
    from trulens.core.run import RunConfig

    session = tpg.connect("postgresql+psycopg://user@host/evals")
    tru_app = tpg.TruApp(rag, app_name="claims_rag", app_version="v3", main_method=rag.answer)
    run = tru_app.add_run(
        RunConfig(
            run_name="v3_test",
            dataset_name="eval.claims_faq",
            source_type="TABLE",
            dataset_spec={"input": "question", "ground_truth_output": "answer"},
        )
    )
    run.start()
    run.compute_metrics([my_metric])  # TruLens Metric objects
"""

from trulens_pg.dao import PgRunDao
from trulens_pg.errors import (
    NotConnectedError,
    SourceError,
    TrulensPgError,
    UnsupportedDatabaseError,
)
from trulens_pg.session import connect, get_context
from trulens_pg.wiring import TruApp, attach

try:
    from trulens_pg._version import __version__
except ImportError:  # pragma: no cover - source tree without build
    __version__ = "0.0.0"

__all__ = [
    "NotConnectedError",
    "PgRunDao",
    "SourceError",
    "TruApp",
    "TrulensPgError",
    "UnsupportedDatabaseError",
    "__version__",
    "attach",
    "connect",
    "get_context",
]
