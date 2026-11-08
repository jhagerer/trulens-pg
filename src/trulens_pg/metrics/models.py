"""SQLAlchemy models of the metric store (schema ``trulens_pg`` by default).

The models are declared with the placeholder schema ``trulens_pg``. The metric
store maps it to the configured schema with SQLAlchemy's ``schema_translate_map``,
so one set of models serves any schema name.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

SCHEMA_PLACEHOLDER = "trulens_pg"

VERSION_STATUSES = ("draft", "in_review", "published", "deprecated")
EDITABLE_STATUSES = ("draft", "in_review")
FROZEN_STATUSES = ("published", "deprecated")
METRIC_KINDS = ("llm_judge", "criteria", "ground_truth", "python_ref")


def _status_check(column: str = "status") -> CheckConstraint:
    values = ", ".join(f"'{s}'" for s in VERSION_STATUSES)
    return CheckConstraint(f"{column} IN ({values})", name=f"ck_{column}")


class Base(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA_PLACEHOLDER)
    type_annotation_map = {  # noqa: RUF012 - SQLAlchemy API
        dict[str, Any]: JSONB,
        list[str]: ARRAY(Text),
        uuid.UUID: UUID(as_uuid=True),
    }


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Metric(Base):
    __tablename__ = "metric"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('llm_judge', 'criteria', 'ground_truth', 'python_ref')", name="ck_metric_kind"
        ),
    )

    metric_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(63), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    owner: Mapped[str | None] = mapped_column(Text)
    higher_is_better: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = _created_at()

    versions: Mapped[list[MetricVersion]] = relationship(
        back_populates="metric", order_by="MetricVersion.version"
    )


class MetricVersion(Base):
    __tablename__ = "metric_version"
    __table_args__ = (
        UniqueConstraint("metric_id", "version", name="uq_metric_version"),
        _status_check(),
        CheckConstraint("score_max > score_min", name="ck_score_range"),
    )

    metric_version_id: Mapped[uuid.UUID] = _uuid_pk()
    metric_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("metric.metric_id", ondelete="RESTRICT"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    prompt_version_id: Mapped[str | None] = mapped_column(Text)
    """Exact TruLens ``PromptVersion`` id of the judge prompt (``llm_judge`` only)."""
    rubric: Mapped[str | None] = mapped_column(Text)
    criteria: Mapped[str | None] = mapped_column(Text)
    score_min: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    score_max: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    judge: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    python_ref: Mapped[str | None] = mapped_column(Text)
    examples: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    published_by: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    changelog: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    metric: Mapped[Metric] = relationship(back_populates="versions")


class MetricLabel(Base):
    __tablename__ = "metric_label"

    label_id: Mapped[uuid.UUID] = _uuid_pk()
    metric_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("metric.metric_id", ondelete="CASCADE"), nullable=False, index=True
    )
    question: Mapped[str | None] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text)
    contexts: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    ground_truth: Mapped[str | None] = mapped_column(Text)
    human_score: Mapped[float] = mapped_column(Float, nullable=False)
    labeler: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class MetricCalibration(Base):
    __tablename__ = "metric_calibration"

    calibration_id: Mapped[uuid.UUID] = _uuid_pk()
    metric_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("metric_version.metric_version_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    n_labels: Mapped[int] = mapped_column(Integer, nullable=False)
    spearman: Mapped[float | None] = mapped_column(Float)
    mean_abs_error: Mapped[float | None] = mapped_column(Float)
    judge_model: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class MetricSet(Base):
    __tablename__ = "metric_set"

    set_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(63), unique=True, nullable=False)
    metric_version_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False
    )
    created_by: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class RunMetric(Base):
    __tablename__ = "run_metric"
    __table_args__ = (
        UniqueConstraint("run_name", "app_name", "metric_version_id", name="uq_run_metric"),
    )

    run_metric_id: Mapped[uuid.UUID] = _uuid_pk()
    run_name: Mapped[str] = mapped_column(Text, nullable=False)
    app_name: Mapped[str] = mapped_column(Text, nullable=False)
    metric_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("metric_version.metric_version_id"), nullable=False, index=True
    )
    judge_model: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()
