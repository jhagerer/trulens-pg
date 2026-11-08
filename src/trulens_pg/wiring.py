"""`wrap()` and `attach()`: install the Postgres run backend on a TruLens app (closes G1).

TruLens 2.16 sets ``App.run_dao`` only when ``connector=`` is passed to the app
constructor; otherwise ``add_run()`` raises ``ValueError: Run DAO not
initialized``. These functions set the run backend and the private run
identifiers (``_object_name``, ``_object_type``, ``_object_version``) that
``App.add_run()`` reads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

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


def wrap(
    app: Any,
    *,
    app_name: str,
    app_version: str,
    main_method: Callable[..., Any] | None = None,
    feedbacks: Iterable[Any] | None = None,
    context: PgContext | None = None,
    **kwargs: Any,
) -> Any:
    """Create a ``TruApp`` that uses the Postgres session and run backend.

    Equivalent to ``TruApp(app, ..., connector=session.connector)`` followed by
    `attach()`.
    """
    from trulens.apps.app import TruApp

    ctx = context or get_context()
    if main_method is not None:
        kwargs["main_method"] = main_method
    tru_app = TruApp(
        app,
        app_name=app_name,
        app_version=app_version,
        feedbacks=list(feedbacks) if feedbacks is not None else None,
        connector=ctx.session.connector,
        **kwargs,
    )
    return attach(tru_app, context=ctx)
