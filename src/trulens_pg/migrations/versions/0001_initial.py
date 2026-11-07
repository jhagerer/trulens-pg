"""Initial bench schema: benchmark store, metric store, guards.

Revision ID: 0001
Revises:
Create Date: 2026-11-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TABLES = [
    "run_metric",
    "metric_set",
    "metric_calibration",
    "metric_label",
    "metric_version",
    "metric",
    "run_link",
    "review",
    "item",
    "dataset_version",
    "document",
    "corpus",
    "dataset",
]

# Published and deprecated versions are read-only. The only allowed change is
# published -> deprecated with every other column unchanged. The SQLSTATE TPG01
# lets the repository map the error to ReadOnlyError.
GUARD_VERSION = """
CREATE OR REPLACE FUNCTION "{s}".tpg_guard_version() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    IF OLD.status IN ('published', 'deprecated') THEN
      RAISE EXCEPTION 'trulens_pg: %.% version % is read-only (status %)',
        TG_TABLE_SCHEMA, TG_TABLE_NAME, OLD.version, OLD.status
        USING ERRCODE = 'TPG01';
    END IF;
    RETURN OLD;
  END IF;
  IF OLD.status IN ('published', 'deprecated') THEN
    IF NOT (OLD.status = 'published' AND NEW.status = 'deprecated'
            AND (to_jsonb(NEW) - 'status') = (to_jsonb(OLD) - 'status')) THEN
      RAISE EXCEPTION 'trulens_pg: %.% version % is read-only (status %)',
        TG_TABLE_SCHEMA, TG_TABLE_NAME, OLD.version, OLD.status
        USING ERRCODE = 'TPG01';
    END IF;
  END IF;
  RETURN NEW;
END;
$$;
"""

GUARD_ITEM = """
CREATE OR REPLACE FUNCTION "{s}".tpg_guard_item() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
  frozen boolean;
BEGIN
  SELECT bool_or(v.status IN ('published', 'deprecated')) INTO frozen
  FROM "{s}".dataset_version v
  WHERE v.version_id IN (
    CASE WHEN TG_OP = 'INSERT' THEN NULL ELSE OLD.version_id END,
    CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE NEW.version_id END
  );
  IF coalesce(frozen, false) THEN
    RAISE EXCEPTION 'trulens_pg: items of a published dataset version are read-only'
      USING ERRCODE = 'TPG01';
  END IF;
  IF TG_OP = 'DELETE' THEN
    RETURN OLD;
  END IF;
  RETURN NEW;
END;
$$;
"""

TRIGGERS = """
CREATE TRIGGER tpg_guard_dataset_version BEFORE UPDATE OR DELETE ON "{s}".dataset_version
  FOR EACH ROW EXECUTE FUNCTION "{s}".tpg_guard_version();
CREATE TRIGGER tpg_guard_metric_version BEFORE UPDATE OR DELETE ON "{s}".metric_version
  FOR EACH ROW EXECUTE FUNCTION "{s}".tpg_guard_version();
CREATE TRIGGER tpg_guard_item BEFORE INSERT OR UPDATE OR DELETE ON "{s}".item
  FOR EACH ROW EXECUTE FUNCTION "{s}".tpg_guard_item();
"""


def upgrade(bench_schema: str) -> None:
    s = bench_schema
    op.create_table(
        "corpus",
        sa.Column("corpus_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("snapshot_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("corpus_id"),
        sa.UniqueConstraint("name"),
        schema=s,
    )
    op.create_table(
        "dataset",
        sa.Column("dataset_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=40), nullable=False),
        sa.Column("domain", sa.Text(), nullable=True),
        sa.Column("owner", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("dataset_id"),
        sa.UniqueConstraint("name"),
        schema=s,
    )
    op.create_table(
        "metric",
        sa.Column("metric_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=63), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner", sa.Text(), nullable=True),
        sa.Column("higher_is_better", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('llm_judge', 'criteria', 'ground_truth', 'python_ref')", name="ck_metric_kind"
        ),
        sa.PrimaryKeyConstraint("metric_id"),
        sa.UniqueConstraint("name"),
        schema=s,
    )
    op.create_table(
        "dataset_version",
        sa.Column("version_id", sa.UUID(), nullable=False),
        sa.Column("dataset_id", sa.UUID(), nullable=False),
        sa.Column("version", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("corpus_id", sa.UUID(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("item_count", sa.Integer(), nullable=False),
        sa.Column("published_by", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("changelog", sa.Text(), nullable=True),
        sa.Column("created_by", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'in_review', 'published', 'deprecated')", name="ck_status"
        ),
        sa.ForeignKeyConstraint(
            ["corpus_id"],
            [f"{s}.corpus.corpus_id"],
        ),
        sa.ForeignKeyConstraint(["dataset_id"], [f"{s}.dataset.dataset_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("version_id"),
        sa.UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),
        schema=s,
    )
    op.create_index(
        op.f("ix_dataset_version_dataset_id"),
        "dataset_version",
        ["dataset_id"],
        unique=False,
        schema=s,
    )
    op.create_table(
        "document",
        sa.Column("doc_id", sa.UUID(), nullable=False),
        sa.Column("corpus_id", sa.UUID(), nullable=False),
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(["corpus_id"], [f"{s}.corpus.corpus_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("doc_id"),
        sa.UniqueConstraint("corpus_id", "chunk_id", name="uq_document_chunk"),
        schema=s,
    )
    op.create_index(
        op.f("ix_document_corpus_id"), "document", ["corpus_id"], unique=False, schema=s
    )
    op.create_table(
        "metric_set",
        sa.Column("set_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=63), nullable=False),
        sa.Column("dataset_id", sa.UUID(), nullable=True),
        sa.Column("metric_version_ids", postgresql.ARRAY(sa.UUID()), nullable=False),
        sa.Column("created_by", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            [f"{s}.dataset.dataset_id"],
        ),
        sa.PrimaryKeyConstraint("set_id"),
        sa.UniqueConstraint("name"),
        schema=s,
    )
    op.create_table(
        "metric_version",
        sa.Column("metric_version_id", sa.UUID(), nullable=False),
        sa.Column("metric_id", sa.UUID(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("prompt_template", sa.Text(), nullable=True),
        sa.Column("rubric", sa.Text(), nullable=True),
        sa.Column("criteria", sa.Text(), nullable=True),
        sa.Column("score_min", sa.Integer(), nullable=False),
        sa.Column("score_max", sa.Integer(), nullable=False),
        sa.Column("inputs", postgresql.JSONB(), nullable=False),
        sa.Column("judge", postgresql.JSONB(), nullable=False),
        sa.Column("params", postgresql.JSONB(), nullable=False),
        sa.Column("python_ref", sa.Text(), nullable=True),
        sa.Column("examples", postgresql.JSONB(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("published_by", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("changelog", sa.Text(), nullable=True),
        sa.Column("created_by", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'in_review', 'published', 'deprecated')", name="ck_status"
        ),
        sa.CheckConstraint("score_max > score_min", name="ck_score_range"),
        sa.ForeignKeyConstraint(["metric_id"], [f"{s}.metric.metric_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("metric_version_id"),
        sa.UniqueConstraint("metric_id", "version", name="uq_metric_version"),
        schema=s,
    )
    op.create_index(
        op.f("ix_metric_version_metric_id"), "metric_version", ["metric_id"], unique=False, schema=s
    )
    op.create_table(
        "item",
        sa.Column("item_id", sa.UUID(), nullable=False),
        sa.Column("version_id", sa.UUID(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("ground_truth_answer", sa.Text(), nullable=True),
        sa.Column("ground_truth_context_ids", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("split", sa.String(length=8), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("difficulty", sa.Text(), nullable=True),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("contains_pii", sa.Boolean(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint("split IS NULL OR split IN ('dev', 'test')", name="ck_item_split"),
        sa.ForeignKeyConstraint(
            ["version_id"], [f"{s}.dataset_version.version_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("item_id"),
        sa.UniqueConstraint("version_id", "external_id", name="uq_item_external_id"),
        schema=s,
    )
    op.create_index(op.f("ix_item_version_id"), "item", ["version_id"], unique=False, schema=s)
    op.create_table(
        "metric_calibration",
        sa.Column("calibration_id", sa.UUID(), nullable=False),
        sa.Column("metric_version_id", sa.UUID(), nullable=False),
        sa.Column("n_labels", sa.Integer(), nullable=False),
        sa.Column("spearman", sa.Float(), nullable=True),
        sa.Column("mean_abs_error", sa.Float(), nullable=True),
        sa.Column("judge_model", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["metric_version_id"], [f"{s}.metric_version.metric_version_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("calibration_id"),
        schema=s,
    )
    op.create_index(
        op.f("ix_metric_calibration_metric_version_id"),
        "metric_calibration",
        ["metric_version_id"],
        unique=False,
        schema=s,
    )
    op.create_table(
        "run_link",
        sa.Column("run_link_id", sa.UUID(), nullable=False),
        sa.Column("run_name", sa.Text(), nullable=False),
        sa.Column("app_name", sa.Text(), nullable=False),
        sa.Column("app_version", sa.Text(), nullable=True),
        sa.Column("version_id", sa.UUID(), nullable=False),
        sa.Column("platform", sa.String(length=16), nullable=False),
        sa.Column("judge_model", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            [f"{s}.dataset_version.version_id"],
        ),
        sa.PrimaryKeyConstraint("run_link_id"),
        sa.UniqueConstraint("run_name", "app_name", name="uq_run_link"),
        schema=s,
    )
    op.create_index(
        op.f("ix_run_link_version_id"), "run_link", ["version_id"], unique=False, schema=s
    )
    op.create_table(
        "run_metric",
        sa.Column("run_metric_id", sa.UUID(), nullable=False),
        sa.Column("run_name", sa.Text(), nullable=False),
        sa.Column("app_name", sa.Text(), nullable=False),
        sa.Column("metric_version_id", sa.UUID(), nullable=False),
        sa.Column("judge_model", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["metric_version_id"],
            [f"{s}.metric_version.metric_version_id"],
        ),
        sa.PrimaryKeyConstraint("run_metric_id"),
        sa.UniqueConstraint("run_name", "app_name", "metric_version_id", name="uq_run_metric"),
        schema=s,
    )
    op.create_index(
        op.f("ix_run_metric_metric_version_id"),
        "run_metric",
        ["metric_version_id"],
        unique=False,
        schema=s,
    )
    op.create_table(
        "metric_label",
        sa.Column("label_id", sa.UUID(), nullable=False),
        sa.Column("metric_id", sa.UUID(), nullable=False),
        sa.Column("item_id", sa.UUID(), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("contexts", postgresql.JSONB(), nullable=False),
        sa.Column("ground_truth", sa.Text(), nullable=True),
        sa.Column("human_score", sa.Float(), nullable=False),
        sa.Column("labeler", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["item_id"], [f"{s}.item.item_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["metric_id"], [f"{s}.metric.metric_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("label_id"),
        schema=s,
    )
    op.create_index(
        op.f("ix_metric_label_metric_id"), "metric_label", ["metric_id"], unique=False, schema=s
    )
    op.create_table(
        "review",
        sa.Column("review_id", sa.UUID(), nullable=False),
        sa.Column("item_id", sa.UUID(), nullable=False),
        sa.Column("reviewer", sa.Text(), nullable=False),
        sa.Column("decision", sa.String(length=8), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("decision IN ('accept', 'edit', 'reject')", name="ck_review_decision"),
        sa.ForeignKeyConstraint(["item_id"], [f"{s}.item.item_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("review_id"),
        schema=s,
    )
    op.create_index(op.f("ix_review_item_id"), "review", ["item_id"], unique=False, schema=s)

    op.execute(GUARD_VERSION.format(s=s))
    op.execute(GUARD_ITEM.format(s=s))
    for statement in TRIGGERS.format(s=s).strip().split(";\n"):
        op.execute(statement.rstrip(";") + ";")


def downgrade(bench_schema: str) -> None:
    s = bench_schema
    # Versioned benchmark views depend on bench.item; drop them first.
    op.execute(
        f"""
        DO $$
        DECLARE r record;
        BEGIN
          FOR r IN SELECT viewname FROM pg_views WHERE schemaname = '{s}' LOOP
            EXECUTE format('DROP VIEW IF EXISTS %I.%I', '{s}', r.viewname);
          END LOOP;
        END $$;
        """
    )
    for table in TABLES:
        op.drop_table(table, schema=s)
    op.execute(f'DROP FUNCTION IF EXISTS "{s}".tpg_guard_item()')
    op.execute(f'DROP FUNCTION IF EXISTS "{s}".tpg_guard_version()')
