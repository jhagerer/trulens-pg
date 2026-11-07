"""Helpers shared by the app pages. Pages access Postgres only through the repository."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd
import streamlit as st

from trulens_pg.app import auth
from trulens_pg.bench import Repository
from trulens_pg.bench.models import EDITABLE_STATUSES, FROZEN_STATUSES
from trulens_pg.config import Settings
from trulens_pg.errors import TrulensPgError
from trulens_pg.metrics import MetricStore


def settings() -> Settings:
    return Settings.from_env()


@st.cache_resource(show_spinner=False)
def _repo(url: str, schema: str) -> Repository:
    return Repository.from_url(url, schema)


@st.cache_resource(show_spinner=False)
def _store(url: str, schema: str) -> MetricStore:
    return MetricStore.from_url(url, schema)


def _url() -> tuple[str, str]:
    cfg = settings()
    if not cfg.database_url:
        st.error("Set TRULENS_PG_DATABASE_URL to the Postgres database of the bench schema.")
        st.stop()
    return str(cfg.database_url), cfg.bench_schema


def repo() -> Repository:
    return _repo(*_url())


def store() -> MetricStore:
    return _store(*_url())


@st.cache_resource(show_spinner="Connecting to TruLens ...")
def _trulens_session(url: str, schema: str) -> Any:
    from trulens_pg import connect

    return connect(url, schema)


def trulens_session() -> Any:
    """TruSession on the same database (Runs page and playground)."""
    return _trulens_session(*_url())


def current_user() -> auth.User:
    """Log the user in (OIDC, proxy header or none) and return name and roles."""
    cached = st.session_state.get("_tpg_user")
    if isinstance(cached, auth.User):
        return cached
    cfg = settings()
    mode = cfg.auth
    if mode == "none":
        name = auth.local_user()
    elif mode == "proxy":
        header = os.environ.get("TRULENS_PG_AUTH_HEADER")
        found = auth.user_from_headers(dict(st.context.headers), header)
        if not found:
            st.error("No user header from the authenticating proxy; access denied.")
            st.stop()
        name = str(found)
    else:
        try:
            logged_in = bool(st.user.is_logged_in)
        except Exception:  # auth not configured in secrets.toml
            st.error(
                "OIDC login is not configured. Add an [auth] section to "
                ".streamlit/secrets.toml, or set TRULENS_PG_AUTH=proxy (reverse proxy) "
                "or TRULENS_PG_AUTH=none (local development)."
            )
            st.stop()
        if not logged_in:
            st.title("trulens-pg")
            st.button("Log in", on_click=st.login, type="primary")
            st.stop()
        name = str(st.user.get("email") or st.user.get("name") or "unknown")
    try:
        roles = auth.roles_for(name, cfg.roles)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    user = auth.User(name=name, roles=roles, auth=mode)
    st.session_state["_tpg_user"] = user
    return user


def require(role: str) -> auth.User:
    user = current_user()
    allowed = user.is_publisher if role == auth.PUBLISHER else user.is_editor
    if not allowed:
        st.warning(f"This action needs the role '{role}'. Your roles: {sorted(user.roles) or '-'}.")
    return user


def can(role: str) -> bool:
    user = current_user()
    return user.is_publisher if role == auth.PUBLISHER else user.is_editor


@contextmanager
def handled() -> Iterator[None]:
    """Show trulens-pg errors as messages instead of tracebacks."""
    try:
        yield
    except TrulensPgError as exc:
        st.error(str(exc))
    except ValueError as exc:  # pydantic validation of user input
        st.error(str(exc))


def run_write[T](action: Callable[[], T], success: Callable[[T], str]) -> T | None:
    """Run one write (one transaction) and confirm it with the affected row count."""
    try:
        result = action()
    except (TrulensPgError, ValueError) as exc:
        st.error(str(exc))
        return None
    st.success(success(result))
    return result


def pick_dataset(key: str = "dataset", label: str = "Dataset") -> str | None:
    datasets = repo().list_datasets()
    if datasets.empty:
        st.info("No datasets yet. Create one on the Datasets page.")
        return None
    names = list(datasets["name"])
    current = st.session_state.get("tpg_dataset")
    index = names.index(current) if current in names else 0
    name = st.selectbox(label, names, index=index, key=key)
    st.session_state["tpg_dataset"] = name
    return str(name)


def pick_version(
    dataset: str,
    *,
    statuses: tuple[str, ...] | None = None,
    key: str = "version",
    label: str = "Version",
) -> pd.Series | None:
    """Select a version of ``dataset`` with one of ``statuses``; returns its row."""
    versions = repo().list_versions(dataset)
    if statuses is not None and not versions.empty:
        versions = versions[versions["status"].isin(statuses)]
    if versions.empty:
        wanted = ", ".join(statuses) if statuses else "any"
        st.info(f"Dataset {dataset} has no version with status: {wanted}.")
        return None
    labels = [f"{r.version} ({r.status}, {r.item_count} items)" for r in versions.itertuples()]
    choice = st.selectbox(label, range(len(labels)), format_func=lambda i: labels[i], key=key)
    return versions.iloc[int(choice)]


EDITABLE = tuple(EDITABLE_STATUSES)
FROZEN = tuple(FROZEN_STATUSES)


def pick_metric(key: str = "metric") -> str | None:
    metrics = store().list_metrics()
    if metrics.empty:
        st.info("No custom metrics yet. Create one on the Metrics page.")
        return None
    names = list(metrics["name"])
    current = st.session_state.get("tpg_metric")
    index = names.index(current) if current in names else 0
    name = st.selectbox("Metric", names, index=index, key=key)
    st.session_state["tpg_metric"] = name
    return str(name)


def pick_metric_version(
    metric: str, *, statuses: tuple[str, ...] | None = None, key: str = "metric_version"
) -> pd.Series | None:
    versions = store().list_versions(metric)
    if statuses is not None and not versions.empty:
        versions = versions[versions["status"].isin(statuses)]
    if versions.empty:
        st.info(f"Metric {metric} has no version with status: {', '.join(statuses or ())}.")
        return None
    labels = [f"v{r.version} ({r.status})" for r in versions.itertuples()]
    choice = st.selectbox("Version", range(len(labels)), format_func=lambda i: labels[i], key=key)
    return versions.iloc[int(choice)]


def lists_to_text(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out:
            out[col] = out[col].map(lambda v: ", ".join(v) if isinstance(v, list) else (v or ""))
    return out


def text_to_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [p.strip() for p in str(value).split(",") if p.strip()]
