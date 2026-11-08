"""Judge prompts in the TruLens prompt registry (TruLens 2.14 or later).

Judge prompts of ``llm_judge`` metrics live in the TruLens prompt registry
(``Prompt``, ``PromptVersion``, ``PromptLabel``). They get immutable versions,
labels and a diff view in the TruLens Prompts tab. A metric version pins an
exact ``PromptVersion`` id, never a label.
"""

from __future__ import annotations

import re
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


class PromptRegistry:
    """Thin wrapper around the prompt registry of a TruLens ``SQLAlchemyDB``."""

    def __init__(self, db: Any) -> None:
        self.db = db

    @classmethod
    def from_engine(cls, engine: Engine, *, migrate: bool = True) -> PromptRegistry:
        """Open the TruLens tables through ``engine`` (creates or upgrades them if needed)."""
        from trulens.core.database.sqlalchemy import SQLAlchemyDB

        db = SQLAlchemyDB.from_tru_args(database_engine=engine)
        if migrate:
            db.migrate_database()
        return cls(db)

    @classmethod
    def from_context(cls) -> PromptRegistry:
        from trulens_pg.session import get_context

        return cls(get_context().session.connector.db)

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
