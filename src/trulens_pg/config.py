"""Configuration from environment variables.

| Variable                              | Meaning                                         |
|---------------------------------------|-------------------------------------------------|
| ``TRULENS_PG_DATABASE_URL``           | SQLAlchemy URL of the Postgres database         |
| ``TRULENS_PG_BENCH_SCHEMA``           | Schema of the benchmark and metric store        |
| ``TRULENS_PG_DASHBOARD_URL``          | URL of the TruLens dashboard (linked from app)  |
| ``TRULENS_PG_ROLES``                  | Role list for the Streamlit app                 |
| ``TRULENS_PG_AUTH``                   | ``oidc``, ``proxy`` or ``none`` (app login)     |
| ``TRULENS_PG_MIN_CALIBRATION_LABELS`` | Labels needed before a judge metric is published |
| ``TRULENS_PG_MIN_SPEARMAN``           | Optional minimum Spearman correlation           |
| ``TRULENS_PG_JUDGE_PROVIDER``         | ``openai``, ``azure_openai`` or ``litellm``     |
| ``TRULENS_PG_JUDGE_MODEL``            | Judge model or Azure deployment name            |
| ``TRULENS_PG_JUDGE_TEMPERATURE``      | Judge temperature (default 0.0)                 |
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from trulens_pg.errors import ValidationError

DEFAULT_BENCH_SCHEMA = "bench"
DEFAULT_MIN_CALIBRATION_LABELS = 20

_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def validate_schema_name(schema: str) -> str:
    """Return ``schema`` if it is a plain lower-case Postgres identifier."""
    if not _IDENT.match(schema):
        raise ValidationError(
            f"Invalid schema name {schema!r}: use lower-case letters, digits and '_'."
        )
    return schema


@dataclass(frozen=True)
class Settings:
    """Settings resolved from the environment."""

    database_url: str | None = None
    bench_schema: str = DEFAULT_BENCH_SCHEMA
    dashboard_url: str | None = None
    roles: str = ""
    auth: str = "oidc"
    min_calibration_labels: int = DEFAULT_MIN_CALIBRATION_LABELS
    min_spearman: float | None = None
    judge: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ if environ is None else environ)
        min_spearman = env.get("TRULENS_PG_MIN_SPEARMAN")
        judge = {
            key.removeprefix("TRULENS_PG_JUDGE_").lower(): value
            for key, value in env.items()
            if key.startswith("TRULENS_PG_JUDGE_") and value
        }
        return cls(
            database_url=env.get("TRULENS_PG_DATABASE_URL") or None,
            bench_schema=validate_schema_name(
                env.get("TRULENS_PG_BENCH_SCHEMA") or DEFAULT_BENCH_SCHEMA
            ),
            dashboard_url=env.get("TRULENS_PG_DASHBOARD_URL") or None,
            roles=env.get("TRULENS_PG_ROLES", ""),
            auth=(env.get("TRULENS_PG_AUTH") or "oidc").lower(),
            min_calibration_labels=int(
                env.get("TRULENS_PG_MIN_CALIBRATION_LABELS") or DEFAULT_MIN_CALIBRATION_LABELS
            ),
            min_spearman=float(min_spearman) if min_spearman else None,
            judge=judge,
        )
