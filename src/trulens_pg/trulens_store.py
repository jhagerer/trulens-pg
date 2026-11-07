"""Access to TruLens' own tables: prompt registry and datasets (TruLens 2.14 or later).

* Judge prompts of ``llm_judge`` metrics live in the TruLens prompt registry
  (``Prompt``, ``PromptVersion``, ``PromptLabel``). They get immutable versions,
  labels and a diff view in the TruLens Prompts tab. A metric version pins an
  exact ``PromptVersion`` id, never a label.
* Each published benchmark version is also written as a TruLens dataset
  ``<dataset>@<version>`` (``Dataset`` plus ``GroundTruth`` rows), so
  ``TruSession.get_ground_truth`` and ``GroundTruthAgreement`` work on it.
* Production traces are converted with TruLens ``curate_records_to_dataset``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import Engine

from trulens_pg.errors import NotFoundError, ValidationError

PROMPT_TAGS = ("trulens-pg", "judge")
PROMPT_SLUG_PREFIX = "trulens-pg."
_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")


def default_prompt_slug(metric_name: str) -> str:
    """Registry slug of the judge prompt of a metric: ``trulens-pg.<metric>``."""
    return f"{PROMPT_SLUG_PREFIX}{metric_name}"


def prompt_variables(text: str) -> list[str]:
    """Variables of a judge prompt; raise for anything but ``{{name}}`` placeholders.

    Uses the TruLens template rules: no statements (``{% %}``), no expressions,
    only plain names such as ``{{question}}``.
    """
    from trulens.core.schema.prompt import (
        VARIABLE_PATTERN,
        InvalidTemplateError,
        validate_template,
    )

    try:
        validate_template([text])
    except InvalidTemplateError as exc:
        raise ValidationError(f"Invalid judge prompt: {exc}") from exc
    names: list[str] = []
    for name in VARIABLE_PATTERN.findall(text):
        if name not in names:
            names.append(name)
    return names


def make_resolved_prompt(
    text: str, slug: str = "inline", variables: list[str] | None = None
) -> Any:
    """Build a ``ResolvedPrompt`` in memory (tests, playground on unsaved text)."""
    from trulens.core.schema.prompt import Prompt, PromptVersion, ResolvedPrompt

    prompt = Prompt(slug=slug)
    version = PromptVersion(
        prompt_id=prompt.prompt_id,
        prompt_type="text",
        text=text,
        variables=variables if variables is not None else prompt_variables(text),
    )
    return ResolvedPrompt(prompt=prompt, version=version)


class TrulensStore:
    """Thin wrapper around a TruLens ``SQLAlchemyDB`` on the same database."""

    def __init__(self, db: Any) -> None:
        self.db = db

    @classmethod
    def from_engine(cls, engine: Engine, *, migrate: bool = True) -> TrulensStore:
        """Open the TruLens tables through ``engine`` (creates or upgrades them if needed)."""
        from trulens.core.database.sqlalchemy import SQLAlchemyDB

        db = SQLAlchemyDB.from_tru_args(database_engine=engine)
        if migrate:
            db.migrate_database()
        return cls(db)

    @classmethod
    def from_context(cls) -> TrulensStore:
        from trulens_pg.session import get_context

        return cls(get_context().session.connector.db)

    # ---------------------------------------------------------------- prompts

    def create_judge_prompt_version(
        self,
        slug: str,
        text: str,
        *,
        description: str | None = None,
        change_note: str | None = None,
        created_by: str | None = None,
    ) -> Any:
        """Create (or reuse, if the text is unchanged) a version of a text prompt."""
        from trulens.core import prompt as core_prompt

        if not _SLUG.match(slug):
            raise ValidationError(f"Invalid prompt slug {slug!r}.")
        variables = prompt_variables(text)
        if not variables:
            raise ValidationError("The judge prompt has no {{variable}} placeholders.")
        if self.db.get_prompt(slug=slug) is None:
            core_prompt.create_prompt(
                self.db, slug=slug, description=description, tags=list(PROMPT_TAGS)
            )
        return core_prompt.create_prompt_version(
            self.db,
            prompt=slug,
            text=text,
            variables=variables,
            change_note=change_note,
            created_by=created_by,
        )

    def resolve_prompt_version(self, version_id: str) -> Any:
        """Load one exact prompt version as a ``ResolvedPrompt`` (renders locally)."""
        from trulens.core import prompt as core_prompt

        version = self.db.get_prompt_version(version_id)
        if version is None:
            raise NotFoundError(f"TruLens prompt version {version_id!r} does not exist.")
        return core_prompt.resolve_prompt(self.db, prompt=version.prompt_id, version_id=version_id)

    def list_prompts(self) -> pd.DataFrame:
        frame: pd.DataFrame = self.db.get_prompts()
        return frame

    def list_prompt_versions(self, slug: str) -> pd.DataFrame:
        from trulens.core import prompt as core_prompt

        try:
            prompt = core_prompt.as_prompt(self.db, slug)
        except ValueError as exc:
            raise NotFoundError(str(exc)) from exc
        frame: pd.DataFrame = self.db.get_prompt_versions(prompt.prompt_id)
        return frame

    # --------------------------------------------------------------- datasets

    def write_dataset(
        self,
        name: str,
        ground_truths: Iterable[Mapping[str, Any]],
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        """Write a TruLens dataset with ground-truth rows (idempotent: ids are content hashes)."""
        from trulens.core.schema.dataset import Dataset
        from trulens.core.schema.groundtruth import GroundTruth

        if meta is not None and "content_hash" in meta:
            existing = self.db.get_datasets()
            for other in existing[existing["name"] == name].itertuples():
                other_hash = (other.meta or {}).get("content_hash")
                if other_hash != meta["content_hash"]:
                    raise ValidationError(
                        f"TruLens already has a dataset {name!r} with other content "
                        f"(content_hash {other_hash}); TruLens would merge both under one name."
                    )
        dataset_id: str = self.db.insert_dataset(dataset=Dataset(name=name, meta=dict(meta or {})))
        batch: list[Any] = []
        for row in ground_truths:
            batch.append(GroundTruth(dataset_id=dataset_id, **dict(row)))
            if len(batch) >= 500:
                self.db.batch_insert_ground_truth(batch)
                batch = []
        if batch:
            self.db.batch_insert_ground_truth(batch)
        return dataset_id

    def ground_truth(self, name: str) -> pd.DataFrame:
        frame: pd.DataFrame | None = self.db.get_ground_truths_by_dataset(name)
        if frame is None or frame.empty:
            raise NotFoundError(f"TruLens dataset {name!r} does not exist or is empty.")
        return frame

    def curate_traces(
        self,
        records: pd.DataFrame,
        *,
        staging_name: str,
        mapping: Mapping[str, Any] | None = None,
    ) -> tuple[Any, pd.DataFrame]:
        """Convert recorded traces with TruLens ``curate_records_to_dataset``.

        The rows are written to the TruLens dataset ``staging_name`` (provenance
        of the source records is kept in each row's ``meta``) and returned.

        Returns:
            ``(CurationResult, ground truth rows)``.
        """
        from trulens.core import dataset as core_dataset

        result = core_dataset.curate_records_to_dataset(
            dataset_name=staging_name,
            records=records,
            db=self.db,
            mapping=core_dataset.TraceDatasetMapping(**dict(mapping or {})),
            dataset_metadata={"source": "trulens-pg import", "created_at": _now_iso()},
            on_error=core_dataset.ON_ERROR_COLLECT,
        )
        rows = (
            self.db.get_ground_truths_by_dataset(staging_name)
            if result.accepted
            else pd.DataFrame()
        )
        if rows is None:
            rows = pd.DataFrame()
        if not rows.empty and "ground_truth_id" in rows:
            rows = rows[rows["ground_truth_id"].isin(result.ground_truth_ids)]
        return result, rows.reset_index(drop=True)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def staging_dataset_name(dataset: str) -> str:
    """Name of the TruLens dataset that holds one trace import before it enters a draft."""
    return f"trulens-pg-import/{dataset}/{datetime.now(UTC):%Y%m%dT%H%M%SZ}"


def ground_truths_to_items(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Map curated TruLens ground truths to benchmark item dictionaries.

    ``query_id`` → ``external_id``, ``expected_response`` → ``ground_truth_answer``;
    expected chunks (texts, not corpus ids) and the provenance ``meta`` go into
    the item metadata.
    """
    items: list[dict[str, Any]] = []
    for row in rows.to_dict(orient="records"):
        meta = dict(row.get("meta") or {})
        chunks = row.get("expected_chunks")
        if isinstance(chunks, list) and chunks:
            meta["expected_chunks"] = chunks
        query_id = row.get("query_id")
        items.append(
            {
                "external_id": f"trace-{query_id}" if query_id else None,
                "query": row.get("query"),
                "ground_truth_answer": row.get("expected_response"),
                "metadata": meta,
            }
        )
    return items


def filter_traces(
    records: pd.DataFrame,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    metric: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
) -> pd.DataFrame:
    """Filter a ``get_records_and_feedback`` frame by time range and metric score."""
    out = records
    if "ts" in out and (since is not None or until is not None):
        ts = pd.to_datetime(out["ts"], utc=True)
        if since is not None:
            out = out[ts >= pd.Timestamp(since).tz_convert("UTC")]
            ts = ts[out.index]
        if until is not None:
            out = out[ts <= pd.Timestamp(until).tz_convert("UTC")]
    if metric is not None and metric in out:
        scores = pd.to_numeric(out[metric], errors="coerce")
        if min_score is not None:
            out = out[scores >= min_score]
            scores = scores[out.index]
        if max_score is not None:
            out = out[scores <= max_score]
    return out.reset_index(drop=True)


def traces_to_items(
    store: TrulensStore,
    records: pd.DataFrame,
    *,
    dataset: str,
    expected_response: str | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Convert production traces into benchmark item dictionaries.

    TruLens ``curate_records_to_dataset`` validates the rows, normalises inputs
    and outputs, keeps the provenance (source record id, app name and version)
    and reports rejected rows. Its result is staged in the TruLens dataset
    `staging_dataset_name` and mapped to items with `ground_truths_to_items`.

    Args:
        store: TruLens tables of the database.
        records: Rows from ``TruSession.get_records_and_feedback``.
        dataset: Benchmark dataset the items are meant for (names the staging set).
        expected_response: Column to use as draft ground truth answer (for
            example ``output``; reviewers must check it), or ``None``.

    Returns:
        ``(CurationResult, items)``.
    """
    mapping: dict[str, Any] = {"query": "input", "query_id": "record_id"}
    if expected_response is not None:
        mapping["expected_response"] = expected_response
    result, rows = store.curate_traces(
        records, staging_name=staging_dataset_name(dataset), mapping=mapping
    )
    return result, ground_truths_to_items(rows)
