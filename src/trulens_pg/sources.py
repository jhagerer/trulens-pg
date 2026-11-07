"""Source names for ``RunConfig(source_type="TABLE")`` and safe table reads.

Accepted source names (``RunConfig.dataset_name``):

* ``table`` or ``view`` in the database's search path,
* ``schema.table`` or ``schema.view``,
* ``bench:<dataset>@<version>`` for a published benchmark version.

Source options (open question in spec section 8) are part of the source name,
written as a URL query string. They are stored with the run in TruLens'
``source_info.name``, so a run can be repeated exactly::

    bench:claims_faq@1.2.0?split=test&tags=motor,home&limit=50
    bench:claims_faq@1.2.0?sample=0.2&seed=2026

| Option   | Meaning                                                           |
|----------|-------------------------------------------------------------------|
| ``split``  | keep rows whose ``split`` column equals the value                |
| ``tags``   | keep rows whose ``tags`` array overlaps the comma-separated list |
| ``limit``  | keep at most this many rows (after sampling)                     |
| ``sample`` | keep a deterministic fraction (0 < f <= 1) by hash of the row ID |
| ``seed``   | seed for ``sample`` (default ``trulens-pg``)                     |

Reads use SQLAlchemy table reflection and ``select()``. Identifiers are checked
against a strict pattern and never formatted into SQL strings.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlencode

import pandas as pd
from sqlalchemy import Engine, MetaData, Table, select
from sqlalchemy.exc import NoSuchTableError

from trulens_pg.errors import SourceError

BENCH_PREFIX = "bench:"
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]{0,62}$")
_DATASET = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_OPTIONS = frozenset({"split", "tags", "limit", "sample", "seed"})
_ID_COLUMNS = ("external_id", "item_id", "input_id")


def check_dataset_name(name: str) -> str:
    """Return ``name`` if it is a valid benchmark dataset name (lower-case slug)."""
    if not _DATASET.match(name):
        raise SourceError(
            f"Invalid dataset name {name!r}: use 1 to 40 lower-case letters, digits and '_', "
            "starting with a letter."
        )
    return name


def check_semver(version: str) -> str:
    """Return ``version`` if it has the form MAJOR.MINOR.PATCH."""
    if not _SEMVER.match(version):
        raise SourceError(f"Invalid version {version!r}: expected MAJOR.MINOR.PATCH, e.g. 1.0.0.")
    return version


def parse_semver(version: str) -> tuple[int, int, int]:
    major, minor, patch = check_semver(version).split(".")
    return int(major), int(minor), int(patch)


def view_name(dataset: str, version: str) -> str:
    """Name of the view of a published version: ``v_<dataset>__<major>_<minor>_<patch>``."""
    return f"v_{check_dataset_name(dataset)}__{check_semver(version).replace('.', '_')}"


@dataclass(frozen=True)
class SourceOptions:
    """Row filters and sampling applied when a source is read."""

    split: str | None = None
    tags: tuple[str, ...] = ()
    limit: int | None = None
    sample: float | None = None
    seed: str = "trulens-pg"

    @classmethod
    def parse(cls, query: str) -> SourceOptions:
        raw = parse_qs(query, keep_blank_values=False, strict_parsing=False)
        unknown = set(raw) - _OPTIONS
        if unknown:
            raise SourceError(
                f"Unknown source option(s): {sorted(unknown)}. Allowed: {sorted(_OPTIONS)}."
            )
        values = {key: vals[-1] for key, vals in raw.items()}
        try:
            limit = int(values["limit"]) if "limit" in values else None
            sample = float(values["sample"]) if "sample" in values else None
        except ValueError as exc:
            raise SourceError(f"Invalid source option value: {exc}") from exc
        if limit is not None and limit < 1:
            raise SourceError("Source option 'limit' must be at least 1.")
        if sample is not None and not 0.0 < sample <= 1.0:
            raise SourceError("Source option 'sample' must be in (0, 1].")
        tags = tuple(t.strip() for t in values.get("tags", "").split(",") if t.strip())
        return cls(
            split=values.get("split") or None,
            tags=tags,
            limit=limit,
            sample=sample,
            seed=values.get("seed") or "trulens-pg",
        )

    def to_query(self) -> str:
        params: dict[str, str] = {}
        if self.split:
            params["split"] = self.split
        if self.tags:
            params["tags"] = ",".join(self.tags)
        if self.sample is not None:
            params["sample"] = repr(self.sample)
            if self.seed != "trulens-pg":
                params["seed"] = self.seed
        if self.limit is not None:
            params["limit"] = str(self.limit)
        return urlencode(params, safe=",")

    @property
    def is_empty(self) -> bool:
        return self == SourceOptions()


@dataclass(frozen=True)
class SourceRef:
    """A parsed source name."""

    raw: str
    table: str
    schema: str | None = None
    dataset: str | None = None
    version: str | None = None
    options: SourceOptions = field(default_factory=SourceOptions)

    @property
    def is_benchmark(self) -> bool:
        return self.dataset is not None

    @property
    def qualified(self) -> str:
        return f"{self.schema}.{self.table}" if self.schema else self.table


def parse_source_name(source_name: str, bench_schema: str = "bench") -> SourceRef:
    """Parse a source name into a `SourceRef` (no database access)."""
    if not source_name or not source_name.strip():
        raise SourceError("Empty source name.")
    name, _, query = source_name.strip().partition("?")
    options = SourceOptions.parse(query) if query else SourceOptions()

    if name.startswith(BENCH_PREFIX):
        spec = name.removeprefix(BENCH_PREFIX)
        dataset, sep, version = spec.partition("@")
        if not sep:
            raise SourceError(
                f"Benchmark source {source_name!r} needs an exact version: "
                f"bench:{dataset}@<MAJOR.MINOR.PATCH>."
            )
        return SourceRef(
            raw=source_name,
            table=view_name(dataset, version),
            schema=bench_schema,
            dataset=dataset,
            version=version,
            options=options,
        )

    parts = name.split(".")
    if len(parts) > 2 or not all(_IDENT.match(p) for p in parts):
        raise SourceError(
            f"Invalid source name {source_name!r}: expected table, schema.table or "
            "bench:<dataset>@<version>."
        )
    if len(parts) == 2:
        return SourceRef(raw=source_name, schema=parts[0], table=parts[1], options=options)
    return SourceRef(raw=source_name, table=parts[0], options=options)


def bench_source(
    dataset: str,
    version: str,
    *,
    split: str | None = None,
    tags: tuple[str, ...] | list[str] = (),
    limit: int | None = None,
    sample: float | None = None,
    seed: str = "trulens-pg",
) -> str:
    """Build a ``bench:<dataset>@<version>`` source name with options.

    Example:
        >>> bench_source("claims_faq", "1.2.0", split="test", limit=50)
        'bench:claims_faq@1.2.0?split=test&limit=50'
    """
    check_dataset_name(dataset)
    check_semver(version)
    opts = SourceOptions(split=split, tags=tuple(tags), limit=limit, sample=sample, seed=seed)
    query = opts.to_query()
    return f"{BENCH_PREFIX}{dataset}@{version}" + (f"?{query}" if query else "")


def reflect_table(engine: Engine, ref: SourceRef) -> Table:
    """Reflect the table or view of ``ref``; raise `SourceError` if it does not exist."""
    try:
        return Table(ref.table, MetaData(), schema=ref.schema, autoload_with=engine)
    except NoSuchTableError as exc:
        if ref.is_benchmark:
            raise SourceError(
                f"Benchmark version {ref.dataset}@{ref.version} has no view {ref.qualified}; "
                "only published versions can be used as run sources."
            ) from exc
        raise SourceError(f"Table or view {ref.qualified!r} does not exist.") from exc


def check_columns(table: Table, dataset_spec: Mapping[str, str], source_name: str) -> None:
    """Fail early if a column named in ``dataset_spec`` is missing from the source."""
    columns = set(table.columns.keys())
    missing = sorted({col for col in dataset_spec.values() if col not in columns})
    if missing:
        raise SourceError(
            f"Source {source_name!r} lacks column(s) {missing} named in dataset_spec. "
            f"Available columns: {sorted(columns)}."
        )


def _sample_key(seed: str, row_id: Any) -> float:
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def apply_sampling(df: pd.DataFrame, options: SourceOptions) -> pd.DataFrame:
    """Keep a deterministic fraction of rows, chosen by hash of the row ID column."""
    if options.sample is None or options.sample >= 1.0:
        return df
    id_col = next((c for c in _ID_COLUMNS if c in df.columns), None)
    if id_col is None:
        raise SourceError(
            f"Source option 'sample' needs one of the columns {list(_ID_COLUMNS)} as row ID."
        )
    keys = df[id_col].map(lambda v: _sample_key(options.seed, v))
    return df.loc[keys < options.sample].reset_index(drop=True)


def read_source(engine: Engine, ref: SourceRef) -> pd.DataFrame:
    """Read a table, view or benchmark version into a DataFrame.

    JSONB and array columns arrive as Python lists and dicts (psycopg converts
    them), so ``RETRIEVAL.RETRIEVED_CONTEXTS`` works in ``LOG_INGESTION`` mode.
    """
    table = reflect_table(engine, ref)
    opts = ref.options
    stmt = select(table)
    if opts.split is not None:
        if "split" not in table.c:
            raise SourceError(f"Source option 'split' needs a 'split' column in {ref.qualified}.")
        stmt = stmt.where(table.c.split == opts.split)
    if opts.tags:
        if "tags" not in table.c:
            raise SourceError(f"Source option 'tags' needs a 'tags' column in {ref.qualified}.")
        stmt = stmt.where(table.c.tags.overlap(list(opts.tags)))
    order_col = next((table.c[c] for c in _ID_COLUMNS if c in table.c), None)
    if order_col is not None:
        stmt = stmt.order_by(order_col)
    if opts.limit is not None and opts.sample is None:
        stmt = stmt.limit(opts.limit)

    with engine.connect() as conn:
        result = conn.execute(stmt)
        df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))

    df = apply_sampling(df, opts)
    if opts.limit is not None:
        df = df.head(opts.limit).reset_index(drop=True)
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, uuid.UUID)).any():
            df[col] = df[col].map(lambda v: str(v) if isinstance(v, uuid.UUID) else v)
    return df
