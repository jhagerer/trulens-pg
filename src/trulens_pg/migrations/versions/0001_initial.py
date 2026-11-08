"""Metric store: custom metric definitions, labels, calibrations, metric sets.

Revision ID: 0001
Revises:
Create Date: 2026-11-08
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
]

# Published and deprecated metric versions are read-only. The only allowed change is
# published -> deprecated with every other column unchanged. The SQLSTATE TPG01 lets
# the metric store map the error to ReadOnlyError.
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

TRIGGER = """
CREATE TRIGGER tpg_guard_metric_version BEFORE UPDATE OR DELETE ON "{s}".metric_version
  FOR EACH ROW EXECUTE FUNCTION "{s}".tpg_guard_version()
"""


def upgrade(schema: str) -> None:
    s = schema
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
        "metric_set",
        sa.Column("set_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=63), nullable=False),
        sa.Column("metric_version_ids", postgresql.ARRAY(sa.UUID()), nullable=False),
        sa.Column("created_by", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("set_id"),
        sa.UniqueConstraint("name"),
        schema=s,
    )
    op.create_table(
        "metric_label",
        sa.Column("label_id", sa.UUID(), nullable=False),
        sa.Column("metric_id", sa.UUID(), nullable=False),
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
        sa.ForeignKeyConstraint(["metric_id"], [f"{s}.metric.metric_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("label_id"),
        schema=s,
    )
    op.create_index(
        op.f("ix_metric_label_metric_id"), "metric_label", ["metric_id"], unique=False, schema=s
    )
    op.create_table(
        "metric_version",
        sa.Column("metric_version_id", sa.UUID(), nullable=False),
        sa.Column("metric_id", sa.UUID(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("prompt_version_id", sa.Text(), nullable=True),
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

    op.execute(GUARD_VERSION.format(s=s))
    op.execute(TRIGGER.format(s=s))


def downgrade(schema: str) -> None:
    s = schema
    for table in TABLES:
        op.drop_table(table, schema=s)
    op.execute(f'DROP FUNCTION IF EXISTS "{s}".tpg_guard_version()')
