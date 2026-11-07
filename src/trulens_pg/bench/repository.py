"""Benchmark repository: datasets, versions, items, review, publish, corpus, runs.

The Streamlit app and the CLI write to the bench schema only through this
module. Every public method runs in one database transaction.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import Select, Table, delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from trulens_pg import hashing
from trulens_pg.bench import models as m
from trulens_pg.bench.db import BenchDB
from trulens_pg.bench.schemas import (
    ITEM_FIELDS,
    DocumentIn,
    ItemIn,
    PublishResult,
    ValidationReport,
)
from trulens_pg.errors import NotFoundError, SourceError, ValidationError, WorkflowError
from trulens_pg.sources import check_dataset_name, check_semver, parse_semver, view_name

PAGE_SIZE = 500
CONTRACT_COLUMNS = (
    "item_id",
    "external_id",
    "input",
    "ground_truth_output",
    "ground_truth_context_ids",
    "split",
    "tags",
    "difficulty",
    "language",
    "contains_pii",
    "metadata",
)
"""Columns of a versioned view and of exports (TruLens ``dataset_spec`` names)."""

_EDITABLE_ITEM_FIELDS = frozenset(ITEM_FIELDS)


def _now() -> datetime:
    return datetime.now(UTC)


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def item_content(item: m.Item) -> dict[str, Any]:
    """Content fields of an item, used for hashing and diffs."""
    return {
        "external_id": item.external_id,
        "query": item.query,
        "ground_truth_answer": item.ground_truth_answer,
        "ground_truth_context_ids": sorted(item.ground_truth_context_ids or []),
        "split": item.split,
        "tags": sorted(item.tags or []),
        "difficulty": item.difficulty,
        "language": item.language,
        "contains_pii": bool(item.contains_pii),
        "metadata": item.metadata_ or {},
    }


def next_version(current: str | None, bump: str = "minor") -> str:
    """Next semantic version after ``current`` (``1.0.0`` if there is none)."""
    if current is None:
        return "1.0.0"
    major, minor, patch = parse_semver(current)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return f"{major}.{minor + 1}.0"


def validate_items(rows: Iterable[Mapping[str, Any]]) -> ValidationReport:
    """Validate rows as `ItemIn`; collect errors per row instead of raising."""
    report = ValidationReport()
    seen: set[str] = set()
    for index, row in enumerate(rows):
        try:
            item = ItemIn.model_validate(dict(row))
        except Exception as exc:  # pydantic.ValidationError and JSON errors
            report.errors.append({"row": index, "error": str(exc).replace("\n", " ")})
            continue
        assert item.external_id is not None
        if item.external_id in seen:
            report.duplicate_ids.append(item.external_id)
        seen.add(item.external_id)
        report.items.append(item)
    return report


class Repository(BenchDB):
    """Data access for benchmark datasets in the bench schema."""

    # ------------------------------------------------------------------ datasets

    def create_dataset(
        self,
        name: str,
        *,
        domain: str | None = None,
        owner: str | None = None,
        description: str | None = None,
    ) -> m.Dataset:
        check_dataset_name(name)
        with self.transaction() as s:
            if s.scalar(select(m.Dataset).where(m.Dataset.name == name)) is not None:
                raise ValidationError(f"Dataset {name!r} already exists.")
            dataset = m.Dataset(name=name, domain=domain, owner=owner, description=description)
            s.add(dataset)
        return dataset

    def _dataset(self, s: Session, name: str) -> m.Dataset:
        dataset = s.scalar(select(m.Dataset).where(m.Dataset.name == name))
        if dataset is None:
            raise NotFoundError(f"Dataset {name!r} does not exist.")
        return dataset

    def get_dataset(self, name: str) -> m.Dataset:
        with self.transaction() as s:
            return self._dataset(s, name)

    def list_datasets(self, search: str | None = None) -> pd.DataFrame:
        """One row per dataset with its latest version, status and item count."""
        with self.transaction() as s:
            stmt = select(m.Dataset).order_by(m.Dataset.name)
            if search:
                pattern = f"%{search}%"
                stmt = stmt.where(
                    or_(
                        m.Dataset.name.ilike(pattern),
                        m.Dataset.description.ilike(pattern),
                        m.Dataset.domain.ilike(pattern),
                    )
                )
            rows = []
            for ds in s.scalars(stmt):
                versions = sorted(ds.versions, key=lambda v: parse_semver(v.version))
                published = [v for v in versions if v.status == "published"]
                latest = versions[-1] if versions else None
                rows.append(
                    {
                        "name": ds.name,
                        "domain": ds.domain,
                        "owner": ds.owner,
                        "description": ds.description,
                        "versions": len(versions),
                        "latest_version": latest.version if latest else None,
                        "latest_status": latest.status if latest else None,
                        "latest_published": published[-1].version if published else None,
                        "item_count": latest.item_count if latest else 0,
                        "created_at": ds.created_at,
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ versions

    def list_versions(self, dataset: str) -> pd.DataFrame:
        with self.transaction() as s:
            ds = self._dataset(s, dataset)
            versions = sorted(ds.versions, key=lambda v: parse_semver(v.version), reverse=True)
            return pd.DataFrame(
                [
                    {
                        "version_id": str(v.version_id),
                        "version": v.version,
                        "status": v.status,
                        "item_count": v.item_count,
                        "content_hash": v.content_hash,
                        "published_by": v.published_by,
                        "published_at": v.published_at,
                        "changelog": v.changelog,
                        "created_at": v.created_at,
                    }
                    for v in versions
                ]
            )

    def _version(self, s: Session, version_id: uuid.UUID | str) -> m.DatasetVersion:
        version = s.get(m.DatasetVersion, _as_uuid(version_id))
        if version is None:
            raise NotFoundError(f"Dataset version {version_id} does not exist.")
        return version

    def get_version(self, dataset: str, version: str | None = None) -> m.DatasetVersion:
        """Return a version; without ``version`` the latest published one."""
        with self.transaction() as s:
            ds = self._dataset(s, dataset)
            if version is None:
                published = [v for v in ds.versions if v.status == "published"]
                if not published:
                    raise NotFoundError(f"Dataset {dataset!r} has no published version.")
                return max(published, key=lambda v: parse_semver(v.version))
            check_semver(version)
            found = next((v for v in ds.versions if v.version == version), None)
            if found is None:
                raise NotFoundError(f"Dataset version {dataset}@{version} does not exist.")
            return found

    def get_version_by_id(self, version_id: uuid.UUID | str) -> m.DatasetVersion:
        with self.transaction() as s:
            version = self._version(s, version_id)
            _ = version.dataset  # load for detached use
            return version

    def resolve_source(self, source: str) -> m.DatasetVersion:
        """Resolve ``bench:<dataset>@<version>`` (or ``<dataset>@<version>``)."""
        spec = source.removeprefix("bench:").partition("?")[0]
        dataset, sep, version = spec.partition("@")
        if not sep:
            raise SourceError(f"Expected <dataset>@<version>, got {source!r}.")
        return self.get_version(dataset, version)

    def create_draft(
        self,
        dataset: str,
        version: str | None = None,
        *,
        copy_from: str | None = "latest",
        bump: str = "minor",
        created_by: str | None = None,
        changelog: str | None = None,
    ) -> m.DatasetVersion:
        """Create a draft version, by default copied from the latest version.

        Args:
            dataset: Dataset name.
            version: New version number; default: `next_version` of the latest.
            copy_from: ``"latest"``, an exact version, or ``None`` for an empty draft.
            bump: Part to increase when ``version`` is not given.
            created_by: User name.
            changelog: Initial changelog text.
        """
        with self.transaction() as s:
            ds = self._dataset(s, dataset)
            existing = sorted(ds.versions, key=lambda v: parse_semver(v.version))
            latest = existing[-1] if existing else None
            new_version = (
                check_semver(version)
                if version
                else next_version(latest.version if latest else None, bump)
            )
            if any(v.version == new_version for v in existing):
                raise ValidationError(f"Version {dataset}@{new_version} already exists.")
            if latest is not None and parse_semver(new_version) <= parse_semver(latest.version):
                raise ValidationError(
                    f"Version {new_version} must be greater than {latest.version}."
                )
            source: m.DatasetVersion | None = None
            if copy_from == "latest":
                source = latest
            elif copy_from is not None:
                source = next((v for v in existing if v.version == copy_from), None)
                if source is None:
                    raise NotFoundError(f"Dataset version {dataset}@{copy_from} does not exist.")
            draft = m.DatasetVersion(
                dataset_id=ds.dataset_id,
                version=new_version,
                status="draft",
                corpus_id=source.corpus_id if source else None,
                created_by=created_by,
                changelog=changelog,
            )
            s.add(draft)
            s.flush()
            if source is not None:
                for item in s.scalars(select(m.Item).where(m.Item.version_id == source.version_id)):
                    s.add(
                        m.Item(
                            version_id=draft.version_id,
                            **{
                                ("metadata_" if k == "metadata" else k): v
                                for k, v in item_content(item).items()
                            },
                        )
                    )
                s.flush()
            draft.item_count = self._count(s, draft.version_id)
            return draft

    def _count(self, s: Session, version_id: uuid.UUID) -> int:
        return int(
            s.scalar(
                select(func.count()).select_from(m.Item).where(m.Item.version_id == version_id)
            )
            or 0
        )

    def _editable(self, s: Session, version_id: uuid.UUID | str) -> m.DatasetVersion:
        version = self._version(s, version_id)
        if version.status not in m.EDITABLE_STATUSES:
            raise WorkflowError(
                f"Version {version.version} is {version.status}; create a new draft to change it."
            )
        return version

    def set_corpus(self, version_id: uuid.UUID | str, corpus: str | None) -> None:
        with self.transaction() as s:
            version = self._editable(s, version_id)
            version.corpus_id = self._corpus(s, corpus).corpus_id if corpus else None

    def delete_draft(self, version_id: uuid.UUID | str) -> None:
        with self.transaction() as s:
            version = self._editable(s, version_id)
            s.delete(version)

    # --------------------------------------------------------------------- items

    def add_items(
        self,
        version_id: uuid.UUID | str,
        items: Iterable[ItemIn | Mapping[str, Any]],
        *,
        upsert: bool = False,
    ) -> int:
        """Insert items into a draft; with ``upsert`` replace items with the same external_id."""
        parsed = [i if isinstance(i, ItemIn) else ItemIn.model_validate(dict(i)) for i in items]
        ids = [i.external_id for i in parsed]
        duplicates = sorted({x for x in ids if ids.count(x) > 1 and x is not None})
        if duplicates:
            raise ValidationError(f"Duplicate external_id values: {duplicates[:10]}")
        with self.transaction() as s:
            version = self._editable(s, version_id)
            rows = [{"version_id": version.version_id, **i.to_row()} for i in parsed]
            if rows:
                table = m.Item.__table__
                assert isinstance(table, Table)
                for start in range(0, len(rows), 5000):
                    chunk = [
                        {("metadata" if k == "metadata_" else k): v for k, v in r.items()}
                        for r in rows[start : start + 5000]
                    ]
                    stmt = insert(table).values(chunk)
                    if upsert:
                        stmt = stmt.on_conflict_do_update(
                            constraint="uq_item_external_id",
                            set_={c: stmt.excluded[c] for c in ITEM_FIELDS if c != "external_id"},
                        )
                    s.execute(stmt)
            version.item_count = self._count(s, version.version_id)
            return len(rows)

    def update_items(self, updates: Sequence[Mapping[str, Any]]) -> int:
        """Update items by ``item_id``; each mapping holds ``item_id`` and changed fields."""
        changed = 0
        with self.transaction() as s:
            checked: set[uuid.UUID] = set()
            for upd in updates:
                fields = {k: v for k, v in upd.items() if k != "item_id"}
                unknown = set(fields) - _EDITABLE_ITEM_FIELDS
                if unknown:
                    raise ValidationError(f"Unknown item field(s): {sorted(unknown)}")
                item = s.get(m.Item, _as_uuid(upd["item_id"]))
                if item is None:
                    raise NotFoundError(f"Item {upd['item_id']} does not exist.")
                if item.version_id not in checked:
                    self._editable(s, item.version_id)
                    checked.add(item.version_id)
                self._apply_edits(item, fields)
                changed += 1
        return changed

    @staticmethod
    def _apply_edits(item: m.Item, fields: Mapping[str, Any]) -> None:
        unknown = set(fields) - _EDITABLE_ITEM_FIELDS
        if unknown:
            raise ValidationError(f"Unknown item field(s): {sorted(unknown)}")
        merged = {**item_content(item), **fields}
        valid = ItemIn.model_validate(merged).to_row()
        for key, value in valid.items():
            setattr(item, key, value)

    def delete_items(self, item_ids: Iterable[uuid.UUID | str]) -> int:
        ids = [_as_uuid(i) for i in item_ids]
        if not ids:
            return 0
        with self.transaction() as s:
            version_ids = set(s.scalars(select(m.Item.version_id).where(m.Item.item_id.in_(ids))))
            for vid in version_ids:
                self._editable(s, vid)
            result = s.execute(delete(m.Item).where(m.Item.item_id.in_(ids)))
            for vid in version_ids:
                self._version(s, vid).item_count = self._count(s, vid)
            return int(getattr(result, "rowcount", 0) or 0)

    def bulk_tag(
        self,
        item_ids: Iterable[uuid.UUID | str],
        *,
        add: Sequence[str] = (),
        remove: Sequence[str] = (),
    ) -> int:
        ids = [_as_uuid(i) for i in item_ids]
        with self.transaction() as s:
            items = list(s.scalars(select(m.Item).where(m.Item.item_id.in_(ids))))
            for vid in {i.version_id for i in items}:
                self._editable(s, vid)
            for item in items:
                tags = [t for t in (item.tags or []) if t not in set(remove)]
                tags += [t for t in add if t not in tags]
                item.tags = tags
            return len(items)

    def set_split(self, item_ids: Iterable[uuid.UUID | str], split: str | None) -> int:
        if split not in (None, *m.SPLITS):
            raise ValidationError(f"Split must be one of {m.SPLITS} or None.")
        ids = [_as_uuid(i) for i in item_ids]
        with self.transaction() as s:
            vids = set(s.scalars(select(m.Item.version_id).where(m.Item.item_id.in_(ids))))
            for vid in vids:
                self._editable(s, vid)
            result = s.execute(update(m.Item).where(m.Item.item_id.in_(ids)).values(split=split))
            return int(getattr(result, "rowcount", 0) or 0)

    def _items_stmt(
        self,
        version_id: uuid.UUID | str,
        *,
        split: str | None = None,
        tags: Sequence[str] = (),
        difficulty: str | None = None,
        language: str | None = None,
        search: str | None = None,
    ) -> Select[m.Item]:
        stmt = select(m.Item).where(m.Item.version_id == _as_uuid(version_id))
        if split:
            stmt = stmt.where(m.Item.split == split)
        if tags:
            stmt = stmt.where(m.Item.tags.overlap(list(tags)))
        if difficulty:
            stmt = stmt.where(m.Item.difficulty == difficulty)
        if language:
            stmt = stmt.where(m.Item.language == language)
        if search:
            pattern = f"%{search}%"
            stmt = stmt.where(
                or_(
                    m.Item.query.ilike(pattern),
                    m.Item.ground_truth_answer.ilike(pattern),
                    m.Item.external_id.ilike(pattern),
                )
            )
        return stmt

    def count_items(self, version_id: uuid.UUID | str, **filters: Any) -> int:
        stmt = self._items_stmt(version_id, **filters)
        with self.transaction() as s:
            return int(s.scalar(select(func.count()).select_from(stmt.subquery())) or 0)

    def list_items(
        self,
        version_id: uuid.UUID | str,
        *,
        page: int = 0,
        page_size: int = PAGE_SIZE,
        **filters: Any,
    ) -> pd.DataFrame:
        """One page of items (default 500 rows) as a DataFrame with ``item_id``."""
        stmt = (
            self._items_stmt(version_id, **filters)
            .order_by(m.Item.external_id)
            .offset(page * page_size)
            .limit(page_size)
        )
        with self.transaction() as s:
            rows = [{"item_id": str(i.item_id), **item_content(i)} for i in s.scalars(stmt)]
        columns = ["item_id", *ITEM_FIELDS]
        return pd.DataFrame(rows, columns=columns)

    def _all_items(self, s: Session, version_id: uuid.UUID) -> list[m.Item]:
        return list(
            s.scalars(
                select(m.Item).where(m.Item.version_id == version_id).order_by(m.Item.external_id)
            )
        )

    # -------------------------------------------------------------------- review

    def submit_for_review(self, version_id: uuid.UUID | str) -> None:
        with self.transaction() as s:
            version = self._version(s, version_id)
            if version.status != "draft":
                raise WorkflowError(f"Only drafts can be submitted, version is {version.status}.")
            version.status = "in_review"

    def return_to_draft(self, version_id: uuid.UUID | str) -> None:
        with self.transaction() as s:
            version = self._version(s, version_id)
            if version.status != "in_review":
                raise WorkflowError(f"Version is {version.status}, not in_review.")
            version.status = "draft"

    def add_review(
        self,
        item_id: uuid.UUID | str,
        *,
        reviewer: str,
        decision: str,
        comment: str | None = None,
        edits: Mapping[str, Any] | None = None,
    ) -> m.Review:
        """Record a review decision; ``edit`` applies ``edits`` to the item."""
        if decision not in m.REVIEW_DECISIONS:
            raise ValidationError(f"Decision must be one of {m.REVIEW_DECISIONS}.")
        if decision == "edit" and not edits:
            raise ValidationError("Decision 'edit' needs the edited fields.")
        with self.transaction() as s:
            item = s.get(m.Item, _as_uuid(item_id))
            if item is None:
                raise NotFoundError(f"Item {item_id} does not exist.")
            self._editable(s, item.version_id)
            if decision == "edit" and edits:
                self._apply_edits(item, edits)
            review = m.Review(
                item_id=item.item_id, reviewer=reviewer, decision=decision, comment=comment
            )
            s.add(review)
        return review

    def _latest_decisions(self, s: Session, version_id: uuid.UUID) -> dict[uuid.UUID, m.Review]:
        reviews = s.scalars(
            select(m.Review)
            .join(m.Item, m.Item.item_id == m.Review.item_id)
            .where(m.Item.version_id == version_id)
            .order_by(m.Review.created_at)
        )
        latest: dict[uuid.UUID, m.Review] = {}
        for review in reviews:
            latest[review.item_id] = review
        return latest

    def review_queue(self, version_id: uuid.UUID | str, *, open_only: bool = False) -> pd.DataFrame:
        """Items with their latest review decision (``None`` = open)."""
        vid = _as_uuid(version_id)
        with self.transaction() as s:
            latest = self._latest_decisions(s, vid)
            rows = []
            for item in self._all_items(s, vid):
                review = latest.get(item.item_id)
                if open_only and review is not None and review.decision != "edit":
                    continue
                rows.append(
                    {
                        "item_id": str(item.item_id),
                        "external_id": item.external_id,
                        "query": item.query,
                        "ground_truth_answer": item.ground_truth_answer,
                        "decision": review.decision if review else None,
                        "reviewer": review.reviewer if review else None,
                        "comment": review.comment if review else None,
                        "reviewed_at": review.created_at if review else None,
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "item_id",
                "external_id",
                "query",
                "ground_truth_answer",
                "decision",
                "reviewer",
                "comment",
                "reviewed_at",
            ],
        )

    def review_history(self, version_id: uuid.UUID | str) -> pd.DataFrame:
        with self.transaction() as s:
            rows = s.execute(
                select(
                    m.Item.external_id,
                    m.Review.reviewer,
                    m.Review.decision,
                    m.Review.comment,
                    m.Review.created_at,
                )
                .join(m.Item, m.Item.item_id == m.Review.item_id)
                .where(m.Item.version_id == _as_uuid(version_id))
                .order_by(m.Review.created_at.desc())
            ).all()
        return pd.DataFrame(
            [r._asdict() for r in rows],
            columns=["external_id", "reviewer", "decision", "comment", "created_at"],
        )

    def open_review_count(self, version_id: uuid.UUID | str) -> int:
        """Items without an ``accept`` or ``reject`` decision (``edit`` stays open)."""
        vid = _as_uuid(version_id)
        with self.transaction() as s:
            latest = self._latest_decisions(s, vid)
            return sum(
                1
                for item in self._all_items(s, vid)
                if item.item_id not in latest or latest[item.item_id].decision == "edit"
            )

    # ------------------------------------------------------------------- publish

    def diff_versions(
        self, version_id: uuid.UUID | str, base_version_id: uuid.UUID | str | None = None
    ) -> pd.DataFrame:
        """Added, changed and removed items compared with ``base`` (default: last published)."""
        vid = _as_uuid(version_id)
        with self.transaction() as s:
            version = self._version(s, vid)
            if base_version_id is None:
                published = [
                    v
                    for v in version.dataset.versions
                    if v.status in m.FROZEN_STATUSES and v.version_id != vid
                ]
                base = max(published, key=lambda v: parse_semver(v.version), default=None)
            else:
                base = self._version(s, base_version_id)
            new = {i.external_id: item_content(i) for i in self._all_items(s, vid)}
            old = (
                {i.external_id: item_content(i) for i in self._all_items(s, base.version_id)}
                if base
                else {}
            )
        rows: list[dict[str, Any]] = []
        for ext_id in sorted(new.keys() | old.keys()):
            before, after = old.get(ext_id), new.get(ext_id)
            if before is None:
                change, fields = "added", list[str]()
            elif after is None:
                change, fields = "removed", []
            elif before != after:
                change = "changed"
                fields = [k for k in after if after[k] != before.get(k)]
            else:
                continue
            rows.append(
                {
                    "external_id": ext_id,
                    "change": change,
                    "fields": ", ".join(fields),
                    "query": (after or before or {}).get("query"),
                }
            )
        return pd.DataFrame(rows, columns=["external_id", "change", "fields", "query"])

    def publish_version(
        self,
        version_id: uuid.UUID | str,
        *,
        published_by: str,
        changelog: str | None = None,
        require_review: bool = True,
    ) -> PublishResult:
        """Publish a version: hash, freeze, and create its view ``bench.v_<ds>__<ver>``.

        With ``require_review`` the version must be ``in_review`` and every item
        needs an ``accept`` or ``reject`` decision. Rejected items are removed
        before the hash is computed.
        """
        vid = _as_uuid(version_id)
        with self.transaction() as s:
            version = self._version(s, vid)
            if version.status not in m.EDITABLE_STATUSES:
                raise WorkflowError(f"Version {version.version} is already {version.status}.")
            removed = 0
            if require_review:
                if version.status != "in_review":
                    raise WorkflowError("Submit the version for review before publishing.")
                latest = self._latest_decisions(s, vid)
                items = self._all_items(s, vid)
                open_items = [
                    i
                    for i in items
                    if i.item_id not in latest or latest[i.item_id].decision == "edit"
                ]
                if open_items:
                    raise WorkflowError(
                        f"{len(open_items)} item(s) have open reviews; accept or reject them first."
                    )
                rejected = [i.item_id for i in items if latest[i.item_id].decision == "reject"]
                if rejected:
                    s.execute(delete(m.Item).where(m.Item.item_id.in_(rejected)))
                    removed = len(rejected)
            items = self._all_items(s, vid)
            if not items:
                raise WorkflowError("A version without items cannot be published.")
            content_hash = hashing.hash_records(
                (item_content(i) for i in items), sort_key="external_id"
            )
            duplicates = [
                v.version
                for v in version.dataset.versions
                if v.version_id != vid and v.content_hash == content_hash
            ]
            version.item_count = len(items)
            version.content_hash = content_hash
            version.status = "published"
            version.published_by = published_by
            version.published_at = _now()
            if changelog is not None:
                version.changelog = changelog
            s.flush()
            view = self._create_view(s, version)
            return PublishResult(
                dataset=version.dataset.name,
                version=version.version,
                content_hash=content_hash,
                item_count=len(items),
                view=f"{self.bench_schema}.{view}",
                removed_rejected=removed,
                duplicate_of=duplicates,
            )

    def deprecate_version(self, version_id: uuid.UUID | str) -> None:
        with self.transaction() as s:
            version = self._version(s, version_id)
            if version.status != "published":
                raise WorkflowError(
                    f"Only published versions can be deprecated ({version.status})."
                )
            version.status = "deprecated"

    def _create_view(self, s: Session, version: m.DatasetVersion) -> str:
        name = view_name(version.dataset.name, version.version)
        vid = str(_as_uuid(version.version_id))  # validated UUID literal
        q = self.quote
        schema = q(self.bench_schema)
        s.execute(
            text(
                f"CREATE OR REPLACE VIEW {schema}.{q(name)} AS "
                "SELECT i.item_id::text AS item_id, i.external_id, i.query AS input, "
                "i.ground_truth_answer AS ground_truth_output, i.ground_truth_context_ids, "
                "i.split, i.tags, i.difficulty, i.language, i.contains_pii, i.metadata "
                f"FROM {schema}.item AS i WHERE i.version_id = '{vid}'::uuid"
            )
        )
        return name

    def refresh_views(self) -> list[str]:
        """Create the views of all published and deprecated versions (idempotent)."""
        with self.transaction() as s:
            versions = s.scalars(
                select(m.DatasetVersion).where(m.DatasetVersion.status.in_(m.FROZEN_STATUSES))
            )
            return [self._create_view(s, v) for v in versions]

    # -------------------------------------------------------------------- export

    def version_frame(
        self, version_id: uuid.UUID | str, *, include_pii: bool = False
    ) -> pd.DataFrame:
        """Items in the benchmark contract columns; PII items only with ``include_pii``."""
        vid = _as_uuid(version_id)
        with self.transaction() as s:
            rows = [
                {
                    "item_id": str(i.item_id),
                    "external_id": i.external_id,
                    "input": i.query,
                    "ground_truth_output": i.ground_truth_answer,
                    "ground_truth_context_ids": list(i.ground_truth_context_ids or []),
                    "split": i.split,
                    "tags": list(i.tags or []),
                    "difficulty": i.difficulty,
                    "language": i.language,
                    "contains_pii": bool(i.contains_pii),
                    "metadata": i.metadata_ or {},
                }
                for i in self._all_items(s, vid)
                if include_pii or not i.contains_pii
            ]
        return pd.DataFrame(rows, columns=list(CONTRACT_COLUMNS))

    # -------------------------------------------------------------------- corpus

    def create_corpus(
        self,
        name: str,
        *,
        source_uri: str | None = None,
        snapshot_date: datetime | None = None,
    ) -> m.Corpus:
        with self.transaction() as s:
            if s.scalar(select(m.Corpus).where(m.Corpus.name == name)) is not None:
                raise ValidationError(f"Corpus {name!r} already exists.")
            corpus = m.Corpus(name=name, source_uri=source_uri, snapshot_date=snapshot_date)
            s.add(corpus)
        return corpus

    def _corpus(self, s: Session, name: str) -> m.Corpus:
        corpus = s.scalar(select(m.Corpus).where(m.Corpus.name == name))
        if corpus is None:
            raise NotFoundError(f"Corpus {name!r} does not exist.")
        return corpus

    def list_corpora(self) -> pd.DataFrame:
        with self.transaction() as s:
            rows = s.execute(
                select(
                    m.Corpus.name,
                    m.Corpus.source_uri,
                    m.Corpus.snapshot_date,
                    m.Corpus.content_hash,
                    func.count(m.Document.doc_id).label("documents"),
                )
                .outerjoin(m.Document, m.Document.corpus_id == m.Corpus.corpus_id)
                .group_by(m.Corpus.corpus_id)
                .order_by(m.Corpus.name)
            ).all()
        return pd.DataFrame(
            [r._asdict() for r in rows],
            columns=["name", "source_uri", "snapshot_date", "content_hash", "documents"],
        )

    def add_documents(
        self, corpus: str, documents: Iterable[DocumentIn | Mapping[str, Any]]
    ) -> int:
        """Insert or replace documents by ``chunk_id``; updates the corpus content hash."""
        docs = [
            d if isinstance(d, DocumentIn) else DocumentIn.model_validate(dict(d))
            for d in documents
        ]
        with self.transaction() as s:
            c = self._corpus(s, corpus)
            table = m.Document.__table__
            assert isinstance(table, Table)
            for start in range(0, len(docs), 5000):
                chunk = [
                    {"corpus_id": c.corpus_id, **d.model_dump()} for d in docs[start : start + 5000]
                ]
                if not chunk:
                    continue
                stmt = insert(table).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_document_chunk",
                    set_={k: stmt.excluded[k] for k in ("title", "text", "source_uri", "metadata")},
                )
                s.execute(stmt)
            all_docs = s.scalars(select(m.Document).where(m.Document.corpus_id == c.corpus_id))
            c.content_hash = hashing.hash_records(
                ({"chunk_id": d.chunk_id, "text": d.text, "title": d.title} for d in all_docs),
                sort_key="chunk_id",
            )
            return len(docs)

    def search_documents(
        self, corpus: str, search: str | None = None, *, limit: int = 50, offset: int = 0
    ) -> pd.DataFrame:
        with self.transaction() as s:
            c = self._corpus(s, corpus)
            stmt = select(m.Document).where(m.Document.corpus_id == c.corpus_id)
            if search:
                pattern = f"%{search}%"
                stmt = stmt.where(
                    or_(
                        m.Document.text.ilike(pattern),
                        m.Document.title.ilike(pattern),
                        m.Document.chunk_id.ilike(pattern),
                    )
                )
            docs = s.scalars(stmt.order_by(m.Document.chunk_id).offset(offset).limit(limit))
            rows = [
                {
                    "chunk_id": d.chunk_id,
                    "title": d.title,
                    "text": d.text,
                    "source_uri": d.source_uri,
                }
                for d in docs
            ]
        return pd.DataFrame(rows, columns=["chunk_id", "title", "text", "source_uri"])

    def documents_by_chunk_id(
        self, corpus_id: uuid.UUID, chunk_ids: Iterable[str]
    ) -> dict[str, str]:
        ids = list(set(chunk_ids))
        if not ids:
            return {}
        with self.transaction() as s:
            rows = s.execute(
                select(m.Document.chunk_id, m.Document.text).where(
                    m.Document.corpus_id == corpus_id, m.Document.chunk_id.in_(ids)
                )
            ).all()
        return {r.chunk_id: r.text for r in rows}

    def missing_context_ids(self, version_id: uuid.UUID | str) -> pd.DataFrame:
        """``ground_truth_context_ids`` of a version that are not in its corpus."""
        vid = _as_uuid(version_id)
        with self.transaction() as s:
            version = self._version(s, vid)
            if version.corpus_id is None:
                raise WorkflowError("The version has no corpus; set one first.")
            known = set(
                s.scalars(
                    select(m.Document.chunk_id).where(m.Document.corpus_id == version.corpus_id)
                )
            )
            rows = [
                {"external_id": i.external_id, "missing_context_id": cid}
                for i in self._all_items(s, vid)
                for cid in i.ground_truth_context_ids or []
                if cid not in known
            ]
        return pd.DataFrame(rows, columns=["external_id", "missing_context_id"])

    def golden_set(self, version_id: uuid.UUID | str) -> list[dict[str, Any]]:
        """Version items in the format of TruLens ``GroundTruthAgreement``."""
        version = self.get_version_by_id(version_id)
        frame = self.version_frame(version.version_id, include_pii=True)
        texts: dict[str, str] = {}
        if version.corpus_id is not None:
            all_ids = [cid for ids in frame["ground_truth_context_ids"] for cid in ids]
            texts = self.documents_by_chunk_id(version.corpus_id, all_ids)
        return [
            {
                "query": row["input"],
                "expected_response": row["ground_truth_output"],
                "expected_chunks": [
                    {"text": texts.get(cid, cid), "expect_score": 1}
                    for cid in row["ground_truth_context_ids"]
                ],
            }
            for row in frame.to_dict(orient="records")
        ]

    # ---------------------------------------------------------------------- runs

    def link_run(
        self,
        *,
        run_name: str,
        app_name: str,
        version_id: uuid.UUID | str,
        app_version: str | None = None,
        judge_model: str | None = None,
        platform: str = "postgres",
    ) -> None:
        """Record which dataset version a run used (also for Snowflake runs)."""
        with self.transaction() as s:
            stmt = insert(m.RunLink).values(
                run_name=run_name,
                app_name=app_name,
                app_version=app_version,
                version_id=_as_uuid(version_id),
                judge_model=judge_model,
                platform=platform,
            )
            s.execute(
                stmt.on_conflict_do_update(
                    constraint="uq_run_link",
                    set_={
                        "version_id": stmt.excluded.version_id,
                        "app_version": stmt.excluded.app_version,
                        "judge_model": stmt.excluded.judge_model,
                        "platform": stmt.excluded.platform,
                    },
                )
            )

    def runs_for_version(self, version_id: uuid.UUID | str) -> pd.DataFrame:
        with self.transaction() as s:
            rows = s.execute(
                select(
                    m.RunLink.run_name,
                    m.RunLink.app_name,
                    m.RunLink.app_version,
                    m.RunLink.platform,
                    m.RunLink.judge_model,
                    m.RunLink.created_at,
                )
                .where(m.RunLink.version_id == _as_uuid(version_id))
                .order_by(m.RunLink.created_at.desc())
            ).all()
        return pd.DataFrame(
            [r._asdict() for r in rows],
            columns=[
                "run_name",
                "app_name",
                "app_version",
                "platform",
                "judge_model",
                "created_at",
            ],
        )

    def run_metric_summary(
        self, version_id: uuid.UUID | str, session: Any, *, runs: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Metric mean, standard deviation and count per run, read from the TruLens tables.

        Args:
            version_id: Dataset version.
            session: A ``TruSession`` on the same database.
            runs: Optional run list (default: `runs_for_version`); rows from other
                platforms are reported with their metadata only.
        """
        runs = self.runs_for_version(version_id) if runs is None else runs
        rows: list[dict[str, Any]] = []
        for run in runs.itertuples():
            if run.platform != "postgres":
                continue
            records, metric_cols = session.get_records_and_feedback(
                app_name=run.app_name, run_name=run.run_name
            )
            for metric in metric_cols:
                values = pd.to_numeric(records[metric], errors="coerce").dropna()
                rows.append(
                    {
                        "run_name": run.run_name,
                        "app_name": run.app_name,
                        "app_version": run.app_version,
                        "platform": run.platform,
                        "judge_model": run.judge_model,
                        "metric": metric,
                        "mean": float(values.mean()) if len(values) else None,
                        "std": float(values.std()) if len(values) > 1 else None,
                        "n": len(values),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "run_name",
                "app_name",
                "app_version",
                "platform",
                "judge_model",
                "metric",
                "mean",
                "std",
                "n",
            ],
        )
