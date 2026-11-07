"""SQLAlchemy models of the bench schema (spec section 5).

The models are declared with the placeholder schema ``bench``. The repository
maps it to the configured schema with SQLAlchemy's ``schema_translate_map``, so
one set of models serves any schema name.
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

SCHEMA_PLACEHOLDER = "bench"

VERSION_STATUSES = ("draft", "in_review", "published", "deprecated")
EDITABLE_STATUSES = ("draft", "in_review")
FROZEN_STATUSES = ("published", "deprecated")
SPLITS = ("dev", "test")
REVIEW_DECISIONS = ("accept", "edit", "reject")
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


class Dataset(Base):
    __tablename__ = "dataset"

    dataset_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    domain: Mapped[str | None] = mapped_column(Text)
    owner: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    versions: Mapped[list[DatasetVersion]] = relationship(
        back_populates="dataset", order_by="DatasetVersion.created_at"
    )


class Corpus(Base):
    __tablename__ = "corpus"

    corpus_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    snapshot_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _created_at()


class Document(Base):
    __tablename__ = "document"
    __table_args__ = (UniqueConstraint("corpus_id", "chunk_id", name="uq_document_chunk"),)

    doc_id: Mapped[uuid.UUID] = _uuid_pk()
    corpus_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("corpus.corpus_id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_id: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_version"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),
        _status_check(),
    )

    version_id: Mapped[uuid.UUID] = _uuid_pk()
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dataset.dataset_id", ondelete="RESTRICT"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    corpus_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("corpus.corpus_id"))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    item_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    published_by: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    changelog: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    dataset: Mapped[Dataset] = relationship(back_populates="versions")


class Item(Base):
    __tablename__ = "item"
    __table_args__ = (
        UniqueConstraint("version_id", "external_id", name="uq_item_external_id"),
        CheckConstraint("split IS NULL OR split IN ('dev', 'test')", name="ck_item_split"),
    )

    item_id: Mapped[uuid.UUID] = _uuid_pk()
    version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dataset_version.version_id", ondelete="CASCADE"), nullable=False, index=True
    )
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth_answer: Mapped[str | None] = mapped_column(Text)
    ground_truth_context_ids: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    split: Mapped[str | None] = mapped_column(String(8))
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    difficulty: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(Text)
    contains_pii: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class Review(Base):
    __tablename__ = "review"
    __table_args__ = (
        CheckConstraint("decision IN ('accept', 'edit', 'reject')", name="ck_review_decision"),
    )

    review_id: Mapped[uuid.UUID] = _uuid_pk()
    item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("item.item_id", ondelete="CASCADE"), nullable=False, index=True
    )
    reviewer: Mapped[str] = mapped_column(Text, nullable=False)
    decision: Mapped[str] = mapped_column(String(8), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class RunLink(Base):
    __tablename__ = "run_link"
    __table_args__ = (UniqueConstraint("run_name", "app_name", name="uq_run_link"),)

    run_link_id: Mapped[uuid.UUID] = _uuid_pk()
    run_name: Mapped[str] = mapped_column(Text, nullable=False)
    app_name: Mapped[str] = mapped_column(Text, nullable=False)
    app_version: Mapped[str | None] = mapped_column(Text)
    version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dataset_version.version_id"), nullable=False, index=True
    )
    platform: Mapped[str] = mapped_column(String(16), nullable=False, default="postgres")
    judge_model: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


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
    item_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("item.item_id", ondelete="SET NULL")
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
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("dataset.dataset_id"))
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
