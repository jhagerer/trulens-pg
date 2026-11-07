"""Input models and result types of the benchmark repository."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ITEM_FIELDS = (
    "external_id",
    "query",
    "ground_truth_answer",
    "ground_truth_context_ids",
    "split",
    "tags",
    "difficulty",
    "language",
    "contains_pii",
    "metadata",
)
"""Content fields of a benchmark item (everything except IDs)."""


def _as_list(value: Any) -> list[str]:
    """Accept a list, a JSON list string, or a ``;``/``,``/``|`` separated string."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):  # NaN from pandas
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if hasattr(value, "tolist"):  # numpy arrays from Parquet
        return _as_list(value.tolist())
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return _as_list(parsed)
    for sep in (";", "|", ","):
        if sep in text:
            return [p.strip() for p in text.split(sep) if p.strip()]
    return [text]


def default_external_id(query: str) -> str:
    """Stable ID derived from the query text, used when the source has no ID column."""
    return "q-" + hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:12]


def _none_if_blank(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


class ItemIn(BaseModel):
    """One benchmark item as written by import, the app or the Python API."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    external_id: str | None = None
    query: str = Field(min_length=1)
    ground_truth_answer: str | None = None
    ground_truth_context_ids: list[str] = Field(default_factory=list)
    split: Literal["dev", "test"] | None = None
    tags: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    language: str | None = None
    contains_pii: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = {k: _none_if_blank(v) for k, v in data.items()}
            for key in ("ground_truth_context_ids", "tags"):
                if key in data:
                    data[key] = _as_list(data[key])
            if data.get("metadata") is None:
                data.pop("metadata", None)
            elif isinstance(data["metadata"], str):
                data["metadata"] = json.loads(data["metadata"])
            if data.get("contains_pii") is None:
                data.pop("contains_pii", None)
            if isinstance(data.get("split"), str):
                data["split"] = data["split"].strip().lower()
            if data.get("external_id") is not None:
                data["external_id"] = str(data["external_id"])
        return data

    @field_validator("contains_pii", mode="before")
    @classmethod
    def _bool(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "t"}
        return value

    @model_validator(mode="after")
    def _default_id(self) -> ItemIn:
        if not self.external_id:
            self.external_id = default_external_id(self.query)
        return self

    def to_row(self) -> dict[str, Any]:
        row = self.model_dump()
        row["metadata_"] = row.pop("metadata")
        return row


class DocumentIn(BaseModel):
    """One document or chunk of a corpus."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    title: str | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = {k: _none_if_blank(v) for k, v in data.items()}
            if data.get("metadata") is None:
                data.pop("metadata", None)
            elif isinstance(data["metadata"], str):
                data["metadata"] = json.loads(data["metadata"])
            if data.get("chunk_id") is not None:
                data["chunk_id"] = str(data["chunk_id"])
        return data


@dataclass
class ValidationReport:
    """Result of validating rows before they are written."""

    items: list[ItemIn] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.duplicate_ids


@dataclass
class PublishResult:
    dataset: str
    version: str
    content_hash: str
    item_count: int
    view: str
    removed_rejected: int = 0
    duplicate_of: list[str] = field(default_factory=list)
