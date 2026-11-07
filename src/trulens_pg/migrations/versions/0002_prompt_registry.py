"""Judge prompts in the TruLens prompt registry: metric_version.prompt_version_id.

Replaces ``metric_version.prompt_template`` with ``prompt_version_id``, the id
of an exact TruLens ``PromptVersion`` (spec section 5, updated 2026-11-07).

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade(bench_schema: str) -> None:
    s = bench_schema
    bind = op.get_bind()
    stored = bind.execute(
        sa.text(f'SELECT count(*) FROM "{s}".metric_version WHERE prompt_template IS NOT NULL')
    ).scalar_one()
    if stored:
        raise RuntimeError(
            f"{stored} metric version(s) still hold a prompt_template. Copy each template into "
            "the TruLens prompt registry (TruSession.create_prompt_version, placeholders "
            "{{name}}) and create new metric versions before upgrading."
        )
    op.add_column("metric_version", sa.Column("prompt_version_id", sa.Text()), schema=s)
    op.drop_column("metric_version", "prompt_template", schema=s)


def downgrade(bench_schema: str) -> None:
    s = bench_schema
    op.add_column("metric_version", sa.Column("prompt_template", sa.Text()), schema=s)
    op.drop_column("metric_version", "prompt_version_id", schema=s)
