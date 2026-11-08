"""`TruApp` and `attach()`: install the Postgres run backend on a TruLens app (closes G1).

TruLens 2.16 sets ``App.run_dao`` only when ``connector=`` is passed to the app
constructor; otherwise ``add_run()`` raises ``ValueError: Run DAO not
initialized``. These functions set the run backend and the private run
identifiers (``_object_name``, ``_object_type``, ``_object_version``) that
``App.add_run()`` reads.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trulens.apps.app import TruApp as _NativeTruApp

from trulens_pg.dao import PgRunDao
from trulens_pg.session import PgContext, get_context

OBJECT_TYPE = "APP"


def make_run_dao(context: PgContext | None = None) -> PgRunDao:
    """Create a `PgRunDao` for the current (or given) context."""
    ctx = context or get_context()
    return PgRunDao(db=ctx.session.connector.db, engine=ctx.engine)


def attach(tru_app: Any, *, context: PgContext | None = None) -> Any:
    """Install `PgRunDao` on an existing ``TruApp``, ``TruChain``, ``TruGraph`` or ``TruLlama``.

    Returns the same app object, so calls can be chained.
    """
    if not hasattr(tru_app, "run_dao") or not hasattr(tru_app, "add_run"):
        raise TypeError(f"Expected a TruLens app (TruApp, TruChain, ...), got {type(tru_app)!r}.")
    tru_app.run_dao = make_run_dao(context)
    tru_app._object_name = tru_app.app_name or "default_app"
    tru_app._object_type = OBJECT_TYPE
    tru_app._object_version = tru_app.app_version or "base"
    return tru_app


class TruApp(_NativeTruApp):  # type: ignore[misc]
    """The TruLens ``TruApp`` with the Postgres session and run backend.

    Takes the same arguments as ``trulens.apps.app.TruApp``. ``connector`` defaults
    to the connector of the `connect()` session, and `PgRunDao` is installed, so
    ``add_run()`` works on Postgres.

    Example:
        ``tpg.TruApp(rag, app_name="claims_rag", app_version="v1", main_method=rag.answer)``
    """

    def __init__(self, app: Any, main_method: Callable[..., Any] | None = None, **kwargs: Any):
        ctx = get_context()
        kwargs.setdefault("connector", ctx.session.connector)
        super().__init__(app, main_method=main_method, **kwargs)
        attach(self, context=ctx)
