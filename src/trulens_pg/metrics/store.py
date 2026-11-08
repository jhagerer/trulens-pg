"""Custom metric store: create, version, calibrate, publish (spec sections 4.5 and 5)."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from trulens_pg import hashing
from trulens_pg.config import Settings
from trulens_pg.errors import NotFoundError, ValidationError, WorkflowError
from trulens_pg.metrics import models as m
from trulens_pg.metrics.build import (
    JUDGE_KINDS,
    KINDS,
    GoldenSet,
    MetricDefinition,
    build_implementation,
    validate_definition,
)
from trulens_pg.metrics.db import StoreDB
from trulens_pg.metrics.judges import judge_model_label

DEFINITION_FIELDS = (
    "prompt_version_id",
    "rubric",
    "criteria",
    "score_min",
    "score_max",
    "inputs",
    "judge",
    "params",
    "python_ref",
    "examples",
)
_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")
_SET_CHARS = _NAME_CHARS | {"-", "."}

LABEL_FIELD_BY_SELECTOR = {
    "record_input": "question",
    "record_output": "answer",
    "context": "contexts",
    "context_each": "contexts",
    "ground_truth": "ground_truth",
}


def check_metric_name(name: str) -> str:
    if not name or not name[0].isalpha() or not set(name) <= _NAME_CHARS or len(name) > 63:
        raise ValidationError(
            f"Invalid metric name {name!r}: use lower-case letters, digits and '_', "
            "starting with a letter."
        )
    return name


def check_set_name(name: str) -> str:
    if not name or not name[0].isalpha() or not set(name.lower()) <= _SET_CHARS or len(name) > 63:
        raise ValidationError(f"Invalid metric set name {name!r}.")
    return name


def parse_metric_ref(ref: str) -> tuple[str, int | None]:
    """Split ``name@version``; the version is ``None`` for ``name``."""
    name, sep, version = ref.strip().partition("@")
    if not sep:
        return name, None
    if not version.isdigit():
        raise ValidationError(f"Metric version must be an integer: {ref!r}.")
    return name, int(version)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation (average ranks for ties); ``None`` if undefined."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = pd.Series(list(x), dtype=float).rank().to_numpy()
    ry = pd.Series(list(y), dtype=float).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def mean_abs_error(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or not x:
        return None
    return float(np.mean(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))))


def _prompt_text(prompt: Any) -> str | None:
    if prompt is None:
        return None
    version = prompt.version
    if version.text is not None:
        return str(version.text)
    return "\n\n".join(f"[{m_.role.value}] {m_.content}" for m_ in version.messages or [])


@dataclass
class CalibrationResult:
    metric: str
    version: int
    n_labels: int
    spearman: float | None
    mean_abs_error: float | None
    judge_model: str | None
    scores: pd.DataFrame


def _score_of(result: Any) -> float:
    value = result[0] if isinstance(result, tuple) else result
    return float(value)


def _reason_of(result: Any) -> str | None:
    if isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], Mapping):
        reason = result[1].get("reason")
        return str(reason) if reason is not None else None
    return None


def call_on_label(
    implementation: Callable[..., Any], inputs: Mapping[str, str], label: Mapping[str, Any]
) -> tuple[float, str | None]:
    """Score one labelled example with a metric implementation (normalised to 0..1)."""
    kwargs: dict[str, Any] = {}
    per_context: str | None = None
    for arg, selector in inputs.items():
        value = label.get(LABEL_FIELD_BY_SELECTOR[selector])
        if selector == "context" and value is None:
            value = []
        if selector == "context_each":
            per_context = arg
        kwargs[arg] = value
    if per_context is not None:
        contexts = list(kwargs[per_context] or [])
        if not contexts:
            raise ValidationError("The label has no contexts for a per-context metric.")
        results = [implementation(**{**kwargs, per_context: c}) for c in contexts]
        return float(np.mean([_score_of(r) for r in results])), _reason_of(results[0])
    result = implementation(**kwargs)
    return _score_of(result), _reason_of(result)


class MetricStore(StoreDB):
    """Data access for custom metric definitions in the trulens_pg schema."""

    # ------------------------------------------------------------------ metrics

    def create_metric(
        self,
        name: str,
        kind: str,
        *,
        description: str | None = None,
        owner: str | None = None,
        higher_is_better: bool = True,
    ) -> m.Metric:
        check_metric_name(name)
        if kind not in KINDS:
            raise ValidationError(f"Kind must be one of {KINDS}.")
        with self.transaction() as s:
            if s.scalar(select(m.Metric).where(m.Metric.name == name)) is not None:
                raise ValidationError(f"Metric {name!r} already exists.")
            metric = m.Metric(
                name=name,
                kind=kind,
                description=description,
                owner=owner,
                higher_is_better=higher_is_better,
            )
            s.add(metric)
        return metric

    def _metric(self, s: Session, name: str) -> m.Metric:
        metric = s.scalar(select(m.Metric).where(m.Metric.name == name))
        if metric is None:
            raise NotFoundError(f"Metric {name!r} does not exist.")
        return metric

    def get_metric(self, name: str) -> m.Metric:
        with self.transaction() as s:
            return self._metric(s, name)

    def list_metrics(self) -> pd.DataFrame:
        """Metrics with latest version, status, calibration and the sets that use them."""
        with self.transaction() as s:
            sets = list(s.scalars(select(m.MetricSet)))
            rows = []
            for metric in s.scalars(select(m.Metric).order_by(m.Metric.name)):
                versions = metric.versions
                latest = versions[-1] if versions else None
                published = [v for v in versions if v.status == "published"]
                calibration = (
                    s.scalar(
                        select(m.MetricCalibration)
                        .where(m.MetricCalibration.metric_version_id == latest.metric_version_id)
                        .order_by(m.MetricCalibration.created_at.desc())
                        .limit(1)
                    )
                    if latest
                    else None
                )
                ids = {v.metric_version_id for v in versions}
                rows.append(
                    {
                        "name": metric.name,
                        "kind": metric.kind,
                        "description": metric.description,
                        "owner": metric.owner,
                        "latest_version": latest.version if latest else None,
                        "latest_status": latest.status if latest else None,
                        "latest_published": published[-1].version if published else None,
                        "n_labels": calibration.n_labels if calibration else None,
                        "spearman": calibration.spearman if calibration else None,
                        "mean_abs_error": calibration.mean_abs_error if calibration else None,
                        "metric_sets": ", ".join(
                            sorted(ms.name for ms in sets if ids & set(ms.metric_version_ids))
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def list_versions(self, name: str) -> pd.DataFrame:
        with self.transaction() as s:
            metric = self._metric(s, name)
            return pd.DataFrame(
                [
                    {
                        "metric_version_id": str(v.metric_version_id),
                        "version": v.version,
                        "status": v.status,
                        "judge": judge_model_label(v.judge),
                        "content_hash": v.content_hash,
                        "published_by": v.published_by,
                        "published_at": v.published_at,
                        "changelog": v.changelog,
                    }
                    for v in reversed(metric.versions)
                ]
            )

    # ----------------------------------------------------------------- versions

    def _prompt(self, version_id: str | None) -> Any:
        """Load a pinned prompt version (versions are immutable, so they are cached)."""
        if not version_id:
            return None
        cache: dict[str, Any] = self.__dict__.setdefault("_prompt_cache", {})
        if version_id not in cache:
            cache[version_id] = self.prompts.resolve_prompt_version(version_id)
        return cache[version_id]

    def _prompt_fields(
        self,
        name: str,
        fields: dict[str, Any],
        prompt_text: str | None,
        prompt_slug: str | None,
        created_by: str | None,
    ) -> dict[str, Any]:
        """Turn ``prompt_text`` into a new registry version; check a given version id."""
        from trulens_pg.metrics.prompts import default_prompt_slug

        if prompt_text is not None:
            version = self.prompts.create_judge_prompt_version(
                prompt_slug or default_prompt_slug(name),
                prompt_text,
                description=f"Judge prompt of the trulens-pg metric {name}",
                change_note=fields.get("changelog"),
                created_by=created_by,
            )
            fields["prompt_version_id"] = version.version_id
        elif fields.get("prompt_version_id"):
            self._prompt(fields["prompt_version_id"])  # raises NotFoundError
        return fields

    def _definition(self, metric: m.Metric, version: m.MetricVersion) -> MetricDefinition:
        return MetricDefinition(
            name=metric.name,
            kind=metric.kind,
            version=version.version,
            description=metric.description,
            higher_is_better=metric.higher_is_better,
            prompt_version_id=version.prompt_version_id,
            prompt=self._prompt(version.prompt_version_id),
            rubric=version.rubric,
            criteria=version.criteria,
            score_min=version.score_min,
            score_max=version.score_max,
            inputs=dict(version.inputs or {}),
            judge=dict(version.judge or {}),
            params=dict(version.params or {}),
            python_ref=version.python_ref,
            examples=list(version.examples or []),
            metric_version_id=str(version.metric_version_id),
            status=version.status,
        )

    def _mv(self, s: Session, metric_version_id: uuid.UUID | str) -> m.MetricVersion:
        mv = s.get(m.MetricVersion, uuid.UUID(str(metric_version_id)))
        if mv is None:
            raise NotFoundError(f"Metric version {metric_version_id} does not exist.")
        return mv

    def create_draft(
        self,
        name: str,
        *,
        copy_from: str | int | None = "latest",
        created_by: str | None = None,
        prompt_text: str | None = None,
        prompt_slug: str | None = None,
        **fields: Any,
    ) -> MetricDefinition:
        """Create the next version as a draft, copied from the latest version by default.

        Args:
            name: Metric name.
            copy_from: ``"latest"``, a version number, or ``None`` for an empty draft.
            created_by: User name.
            prompt_text: Judge prompt with ``{{variable}}`` placeholders; stored as a
                new version in the TruLens prompt registry (slug ``prompt_slug``,
                default ``trulens-pg.<name>``) and pinned by its version id.
            prompt_slug: Registry slug for ``prompt_text``.
            **fields: Definition fields, for example ``prompt_version_id`` to pin a
                version picked from the registry.
        """
        unknown = set(fields) - set(DEFINITION_FIELDS) - {"changelog"}
        if unknown:
            raise ValidationError(f"Unknown metric field(s): {sorted(unknown)}")
        fields = self._prompt_fields(name, fields, prompt_text, prompt_slug, created_by)
        with self.transaction() as s:
            metric = self._metric(s, name)
            versions = metric.versions
            base: m.MetricVersion | None = None
            if copy_from == "latest" and versions:
                base = versions[-1]
            elif isinstance(copy_from, int):
                base = next((v for v in versions if v.version == copy_from), None)
                if base is None:
                    raise NotFoundError(f"Metric version {name}@{copy_from} does not exist.")
            values: dict[str, Any] = (
                {f: getattr(base, f) for f in DEFINITION_FIELDS} if base is not None else {}
            )
            values.update(fields)
            mv = m.MetricVersion(
                metric_id=metric.metric_id,
                version=(versions[-1].version + 1) if versions else 1,
                status="draft",
                created_by=created_by,
                **{k: v for k, v in values.items() if v is not None},
            )
            s.add(mv)
            s.flush()
            s.refresh(mv)
            return self._definition(metric, mv)

    def update_draft(
        self,
        metric_version_id: uuid.UUID | str,
        *,
        prompt_text: str | None = None,
        prompt_slug: str | None = None,
        updated_by: str | None = None,
        **fields: Any,
    ) -> MetricDefinition:
        """Change a draft; ``prompt_text`` creates a new registry version (see `create_draft`)."""
        unknown = set(fields) - set(DEFINITION_FIELDS) - {"changelog"}
        if unknown:
            raise ValidationError(f"Unknown metric field(s): {sorted(unknown)}")
        with self.transaction() as s:
            mv = self._mv(s, metric_version_id)
            if mv.status not in m.EDITABLE_STATUSES:
                raise WorkflowError(f"Version {mv.version} is {mv.status}; create a new draft.")
            name = mv.metric.name
        fields = self._prompt_fields(name, fields, prompt_text, prompt_slug, updated_by)
        with self.transaction() as s:
            mv = self._mv(s, metric_version_id)
            if mv.status not in m.EDITABLE_STATUSES:
                raise WorkflowError(f"Version {mv.version} is {mv.status}; create a new draft.")
            for key, value in fields.items():
                setattr(mv, key, value)
            s.flush()
            return self._definition(mv.metric, mv)

    def definition(self, metric_version_id: uuid.UUID | str) -> MetricDefinition:
        with self.transaction() as s:
            mv = self._mv(s, metric_version_id)
            return self._definition(mv.metric, mv)

    def get_definition(self, ref: str, *, allow_unpublished: bool = False) -> MetricDefinition:
        """Resolve ``name`` (latest published) or ``name@version``."""
        name, version = parse_metric_ref(ref)
        with self.transaction() as s:
            metric = self._metric(s, name)
            if version is None:
                published = [v for v in metric.versions if v.status == "published"]
                if not published:
                    raise NotFoundError(f"Metric {name!r} has no published version.")
                return self._definition(metric, published[-1])
            mv = next((v for v in metric.versions if v.version == version), None)
            if mv is None:
                raise NotFoundError(f"Metric version {name}@{version} does not exist.")
            if mv.status == "deprecated":
                raise WorkflowError(f"Metric version {name}@{version} is deprecated.")
            if mv.status != "published" and not allow_unpublished:
                raise WorkflowError(
                    f"Metric version {name}@{version} is {mv.status}; only published versions "
                    "run by name (use the Playground for drafts)."
                )
            return self._definition(metric, mv)

    def diff(
        self, metric_version_id: uuid.UUID | str, base_version: int | None = None
    ) -> pd.DataFrame:
        """Fields that differ from ``base_version`` (default: last published)."""
        with self.transaction() as s:
            mv = self._mv(s, metric_version_id)
            others = [v for v in mv.metric.versions if v.version != mv.version]
            if base_version is None:
                published = [v for v in others if v.status in m.FROZEN_STATUSES]
                base = published[-1] if published else None
            else:
                base = next((v for v in others if v.version == base_version), None)
            rows = [
                {
                    "field": f,
                    "base": None if base is None else getattr(base, f),
                    "draft": getattr(mv, f),
                }
                for f in DEFINITION_FIELDS
                if base is None or getattr(base, f) != getattr(mv, f)
            ]
            base_prompt = None if base is None else base.prompt_version_id
            draft_prompt = mv.prompt_version_id
        if base_prompt != draft_prompt:
            rows.append(
                {
                    "field": "prompt text",
                    "base": _prompt_text(self._prompt(base_prompt)),
                    "draft": _prompt_text(self._prompt(draft_prompt)),
                }
            )
        return pd.DataFrame(rows, columns=["field", "base", "draft"])

    # ------------------------------------------------------------------- labels

    def add_labels(self, name: str, labels: Iterable[Mapping[str, Any]]) -> int:
        """Add human-scored examples: question, answer, contexts, ground_truth, human_score."""
        allowed = {
            "question",
            "answer",
            "contexts",
            "ground_truth",
            "human_score",
            "labeler",
        }
        with self.transaction() as s:
            metric = self._metric(s, name)
            score_ranges = [(v.score_min, v.score_max) for v in metric.versions]
            lo = min((r[0] for r in score_ranges), default=None)
            hi = max((r[1] for r in score_ranges), default=None)
            count = 0
            for label in labels:
                unknown = set(label) - allowed
                if unknown:
                    raise ValidationError(f"Unknown label field(s): {sorted(unknown)}")
                if label.get("human_score") is None:
                    raise ValidationError("Each label needs a human_score.")
                score = float(label["human_score"])
                if lo is not None and hi is not None and not lo <= score <= hi:
                    raise ValidationError(f"human_score {score} is outside [{lo}, {hi}].")
                contexts = label.get("contexts") or []
                if isinstance(contexts, str):
                    contexts = [contexts]
                s.add(
                    m.MetricLabel(
                        metric_id=metric.metric_id,
                        question=label.get("question"),
                        answer=label.get("answer"),
                        contexts=list(contexts),
                        ground_truth=label.get("ground_truth"),
                        human_score=score,
                        labeler=label.get("labeler"),
                    )
                )
                count += 1
            return count

    def list_labels(self, name: str) -> pd.DataFrame:
        with self.transaction() as s:
            metric = self._metric(s, name)
            labels = s.scalars(
                select(m.MetricLabel)
                .where(m.MetricLabel.metric_id == metric.metric_id)
                .order_by(m.MetricLabel.created_at)
            )
            rows = [
                {
                    "label_id": str(lb.label_id),
                    "question": lb.question,
                    "answer": lb.answer,
                    "contexts": list(lb.contexts or []),
                    "ground_truth": lb.ground_truth,
                    "human_score": lb.human_score,
                    "labeler": lb.labeler,
                    "created_at": lb.created_at,
                }
                for lb in labels
            ]
        return pd.DataFrame(
            rows,
            columns=[
                "label_id",
                "question",
                "answer",
                "contexts",
                "ground_truth",
                "human_score",
                "labeler",
                "created_at",
            ],
        )

    def delete_label(self, label_id: uuid.UUID | str) -> None:
        with self.transaction() as s:
            label = s.get(m.MetricLabel, uuid.UUID(str(label_id)))
            if label is not None:
                s.delete(label)

    # -------------------------------------------------------------- calibration

    def calibrate(
        self,
        metric_version_id: uuid.UUID | str,
        judge: Any = None,
        *,
        golden_set: GoldenSet | None = None,
        limit: int | None = None,
    ) -> CalibrationResult:
        """Score the labelled examples with this version and compare with the human scores.

        Judge scores are normalised to 0..1 by TruLens; human scores are normalised
        with the version's score range before the mean absolute error is computed.
        Each label costs one judge call (per context for per-context metrics).
        """
        defn = self.definition(metric_version_id)
        labels = self.list_labels(defn.name)
        if limit is not None:
            labels = labels.head(limit)
        if labels.empty:
            raise WorkflowError(f"Metric {defn.name!r} has no labels; add human scores first.")
        implementation, _extra, info = build_implementation(defn, judge, golden_set=golden_set)
        inputs = defn.resolved_inputs()
        span = defn.score_max - defn.score_min
        rows = []
        for record in labels.to_dict(orient="records"):
            label = {str(k): v for k, v in record.items()}
            score, reason = call_on_label(implementation, inputs, label)
            rows.append(
                {
                    "label_id": label["label_id"],
                    "question": label["question"],
                    "human_score": label["human_score"],
                    "human_normalized": (label["human_score"] - defn.score_min) / span,
                    "judge_score": score,
                    "reason": reason,
                }
            )
        scores = pd.DataFrame(rows)
        valid = scores[scores["judge_score"].map(lambda v: v is not None and not math.isnan(v))]
        rho = spearman(list(valid["judge_score"]), list(valid["human_normalized"]))
        mae = mean_abs_error(list(valid["judge_score"]), list(valid["human_normalized"]))
        judge_model = judge_model_label(info)
        with self.transaction() as s:
            s.add(
                m.MetricCalibration(
                    metric_version_id=uuid.UUID(str(metric_version_id)),
                    n_labels=len(valid),
                    spearman=rho,
                    mean_abs_error=mae,
                    judge_model=judge_model,
                )
            )
        return CalibrationResult(
            metric=defn.name,
            version=int(defn.version or 0),
            n_labels=len(valid),
            spearman=rho,
            mean_abs_error=mae,
            judge_model=judge_model,
            scores=scores,
        )

    def latest_calibration(self, metric_version_id: uuid.UUID | str) -> m.MetricCalibration | None:
        with self.transaction() as s:
            return s.scalar(
                select(m.MetricCalibration)
                .where(m.MetricCalibration.metric_version_id == uuid.UUID(str(metric_version_id)))
                .order_by(m.MetricCalibration.created_at.desc())
                .limit(1)
            )

    def publish_blockers(
        self,
        metric_version_id: uuid.UUID | str,
        *,
        min_labels: int | None = None,
        min_spearman: float | None = None,
    ) -> list[str]:
        """Reasons why a version cannot be published yet (empty list: publish allowed)."""
        settings = Settings.from_env()
        min_labels = settings.min_calibration_labels if min_labels is None else min_labels
        min_spearman = settings.min_spearman if min_spearman is None else min_spearman
        defn = self.definition(metric_version_id)
        blockers = validate_definition(defn)
        if defn.status not in m.EDITABLE_STATUSES:
            blockers.append(f"Version {defn.version} is already {defn.status}.")
        if defn.kind in JUDGE_KINDS:
            cal = self.latest_calibration(metric_version_id)
            if cal is None:
                blockers.append("No calibration yet: label examples and run the calibration.")
            else:
                if cal.n_labels < min_labels:
                    blockers.append(
                        f"Calibration used {cal.n_labels} labels; at least {min_labels} needed."
                    )
                if min_spearman is not None and (
                    cal.spearman is None or cal.spearman < min_spearman
                ):
                    blockers.append(
                        f"Spearman correlation {cal.spearman} is below the minimum {min_spearman}."
                    )
        return blockers

    def publish(
        self,
        metric_version_id: uuid.UUID | str,
        *,
        published_by: str,
        changelog: str | None = None,
        min_labels: int | None = None,
        min_spearman: float | None = None,
    ) -> MetricDefinition:
        """Publish a version after the calibration gate; it becomes read-only.

        Judge-based kinds (``llm_judge``, ``criteria``) need a calibration with
        at least ``min_labels`` labels (default ``TRULENS_PG_MIN_CALIBRATION_LABELS``).
        Deterministic kinds (``ground_truth``, ``python_ref``) need no calibration.
        """
        blockers = self.publish_blockers(
            metric_version_id, min_labels=min_labels, min_spearman=min_spearman
        )
        if blockers:
            raise WorkflowError("Cannot publish: " + " ".join(blockers))
        defn = self.definition(metric_version_id)
        with self.transaction() as s:
            mv = self._mv(s, metric_version_id)
            mv.inputs = defn.resolved_inputs()
            mv.content_hash = hashing.hash_object(defn.content())
            mv.status = "published"
            mv.published_by = published_by
            mv.published_at = datetime.now(UTC)
            if changelog is not None:
                mv.changelog = changelog
            s.flush()
            return self._definition(mv.metric, mv)

    def submit_for_review(self, metric_version_id: uuid.UUID | str) -> None:
        with self.transaction() as s:
            mv = self._mv(s, metric_version_id)
            if mv.status != "draft":
                raise WorkflowError(f"Only drafts can be submitted, version is {mv.status}.")
            mv.status = "in_review"

    def deprecate(self, metric_version_id: uuid.UUID | str) -> None:
        with self.transaction() as s:
            mv = self._mv(s, metric_version_id)
            if mv.status != "published":
                raise WorkflowError(f"Only published versions can be deprecated ({mv.status}).")
            mv.status = "deprecated"

    # -------------------------------------------------------------- metric sets

    def create_metric_set(
        self,
        name: str,
        refs: Sequence[str],
        *,
        created_by: str | None = None,
    ) -> m.MetricSet:
        """Bundle fixed, published metric versions (``name@version``), e.g. ``claims-rag-v1``."""
        check_set_name(name)
        if not refs:
            raise ValidationError("A metric set needs at least one metric version.")
        ids: list[uuid.UUID] = []
        for ref in refs:
            metric_name, version = parse_metric_ref(ref)
            if version is None:
                raise ValidationError(f"Metric sets need exact versions: {ref!r} -> name@version.")
            ids.append(uuid.UUID(str(self.get_definition(ref).metric_version_id)))
            _ = metric_name
        with self.transaction() as s:
            if s.scalar(select(m.MetricSet).where(m.MetricSet.name == name)) is not None:
                raise ValidationError(f"Metric set {name!r} already exists.")
            metric_set = m.MetricSet(name=name, metric_version_ids=ids, created_by=created_by)
            s.add(metric_set)
        return metric_set

    def has_metric_set(self, name: str) -> bool:
        with self.transaction() as s:
            return s.scalar(select(m.MetricSet.set_id).where(m.MetricSet.name == name)) is not None

    def has_metric(self, name: str) -> bool:
        with self.transaction() as s:
            return s.scalar(select(m.Metric.metric_id).where(m.Metric.name == name)) is not None

    def get_metric_set(self, name: str) -> list[MetricDefinition]:
        with self.transaction() as s:
            metric_set = s.scalar(select(m.MetricSet).where(m.MetricSet.name == name))
            if metric_set is None:
                raise NotFoundError(f"Metric set {name!r} does not exist.")
            result = []
            for mv_id in metric_set.metric_version_ids:
                mv = self._mv(s, mv_id)
                if mv.status == "deprecated":
                    raise WorkflowError(
                        f"Metric set {name!r} contains deprecated version "
                        f"{mv.metric.name}@{mv.version}."
                    )
                result.append(self._definition(mv.metric, mv))
            return result

    def list_metric_sets(self) -> pd.DataFrame:
        with self.transaction() as s:
            rows = []
            for ms in s.scalars(select(m.MetricSet).order_by(m.MetricSet.name)):
                refs = []
                for mv_id in ms.metric_version_ids:
                    mv = s.get(m.MetricVersion, mv_id)
                    refs.append(f"{mv.metric.name}@{mv.version}" if mv else str(mv_id))
                rows.append(
                    {
                        "name": ms.name,
                        "metrics": ", ".join(refs),
                        "created_by": ms.created_by,
                        "created_at": ms.created_at,
                    }
                )
        return pd.DataFrame(rows, columns=["name", "metrics", "created_by", "created_at"])

    # --------------------------------------------------------------------- runs

    def record_run_metrics(
        self,
        *,
        run_name: str,
        app_name: str,
        metric_version_ids: Iterable[uuid.UUID | str],
        judge_model: str | None = None,
    ) -> None:
        """Fill ``run_metric``: which metric versions (and judge) a run used."""
        with self.transaction() as s:
            for mv_id in metric_version_ids:
                stmt = insert(m.RunMetric).values(
                    run_name=run_name,
                    app_name=app_name,
                    metric_version_id=uuid.UUID(str(mv_id)),
                    judge_model=judge_model,
                )
                s.execute(
                    stmt.on_conflict_do_update(
                        constraint="uq_run_metric", set_={"judge_model": stmt.excluded.judge_model}
                    )
                )

    def run_metrics(self, run_name: str, app_name: str) -> pd.DataFrame:
        with self.transaction() as s:
            rows = s.execute(
                select(m.Metric.name, m.MetricVersion.version, m.RunMetric.judge_model)
                .join(
                    m.MetricVersion,
                    m.MetricVersion.metric_version_id == m.RunMetric.metric_version_id,
                )
                .join(m.Metric, m.Metric.metric_id == m.MetricVersion.metric_id)
                .where(m.RunMetric.run_name == run_name, m.RunMetric.app_name == app_name)
            ).all()
        return pd.DataFrame([r._asdict() for r in rows], columns=["name", "version", "judge_model"])
