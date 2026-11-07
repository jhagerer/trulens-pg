from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import text

from tests.conftest import SAMPLE_ITEMS, TEST_SCHEMA
from trulens_pg.errors import ReadOnlyError, ValidationError, WorkflowError

pytestmark = pytest.mark.pg


def test_publish_creates_hash_and_view(repo: Any, published: Any, ctx: Any) -> None:
    assert published.item_count == 6
    assert len(published.content_hash) == 64
    assert published.view == f"{TEST_SCHEMA}.v_claims_faq__1_0_0"
    with ctx.engine.connect() as conn:
        cols = (
            conn.execute(text(f"SELECT * FROM {published.view} ORDER BY external_id LIMIT 1"))
            .mappings()
            .one()
        )
    assert set(cols) >= {
        "input",
        "ground_truth_output",
        "ground_truth_context_ids",
        "item_id",
        "split",
        "tags",
    }
    assert isinstance(cols["tags"], list)
    datasets = repo.list_datasets()
    assert datasets.loc[0, "latest_published"] == "1.0.0"


def test_published_version_is_read_only_in_the_database(
    repo: Any, published: Any, ctx: Any
) -> None:
    version = repo.get_version("claims_faq", "1.0.0")
    with pytest.raises(ReadOnlyError), repo.transaction() as s:
        s.execute(
            text(f"UPDATE {TEST_SCHEMA}.item SET query = 'changed' WHERE version_id = :v"),
            {"v": version.version_id},
        )
    with pytest.raises(ReadOnlyError), repo.transaction() as s:
        s.execute(
            text(f"UPDATE {TEST_SCHEMA}.dataset_version SET changelog = 'x' WHERE version_id = :v"),
            {"v": version.version_id},
        )
    with pytest.raises(ReadOnlyError), repo.transaction() as s:
        s.execute(
            text(f"DELETE FROM {TEST_SCHEMA}.dataset_version WHERE version_id = :v"),
            {"v": version.version_id},
        )
    # The repository refuses as well, before the trigger.
    with pytest.raises(WorkflowError):
        repo.add_items(version.version_id, [{"query": "new"}])
    # Deprecation is the one allowed change.
    repo.deprecate_version(version.version_id)
    assert repo.get_version("claims_faq", "1.0.0").status == "deprecated"


def test_draft_copy_diff_and_duplicate_flag(repo: Any, published: Any) -> None:
    draft = repo.create_draft("claims_faq", created_by="tester")
    assert draft.version == "1.1.0"
    assert draft.item_count == 6
    assert repo.diff_versions(draft.version_id).empty

    items = repo.list_items(draft.version_id)
    first = items.iloc[0]
    repo.update_items([{"item_id": first.item_id, "query": "Changed question?"}])
    repo.delete_items([items.iloc[1].item_id])
    repo.add_items(draft.version_id, [{"external_id": "q100", "query": "New?"}])
    diff = repo.diff_versions(draft.version_id).set_index("external_id")
    assert diff.loc[first.external_id, "change"] == "changed"
    assert diff.loc[first.external_id, "fields"] == "query"
    assert diff.loc[items.iloc[1].external_id, "change"] == "removed"
    assert diff.loc["q100", "change"] == "added"

    same = repo.create_draft("claims_faq", "2.0.0", copy_from="1.0.0")
    result = repo.publish_version(same.version_id, published_by="tester", require_review=False)
    assert result.duplicate_of == ["1.0.0"]


def test_review_gate(repo: Any) -> None:
    repo.create_dataset("faq")
    draft = repo.create_draft("faq", copy_from=None)
    repo.add_items(draft.version_id, SAMPLE_ITEMS[:3])
    with pytest.raises(WorkflowError, match="Submit"):
        repo.publish_version(draft.version_id, published_by="p")
    repo.submit_for_review(draft.version_id)
    queue = repo.review_queue(draft.version_id)
    assert queue["decision"].isna().all()
    ids = list(queue["item_id"])
    repo.add_review(ids[0], reviewer="r1", decision="accept")
    repo.add_review(
        ids[1], reviewer="r1", decision="edit", edits={"ground_truth_answer": "Fixed answer."}
    )
    repo.add_review(ids[2], reviewer="r2", decision="reject", comment="duplicate")
    assert repo.open_review_count(draft.version_id) == 1  # edit stays open
    with pytest.raises(WorkflowError, match="open reviews"):
        repo.publish_version(draft.version_id, published_by="p")
    repo.add_review(ids[1], reviewer="r2", decision="accept")
    result = repo.publish_version(draft.version_id, published_by="p", changelog="first")
    assert result.item_count == 2
    assert result.removed_rejected == 1
    assert len(repo.review_history(draft.version_id)) == 3  # rejected item is gone


def test_items_paging_filters_and_bulk_actions(repo: Any) -> None:
    repo.create_dataset("faq")
    draft = repo.create_draft("faq", copy_from=None)
    repo.add_items(
        draft.version_id, [{"external_id": f"x{i:04d}", "query": f"q{i}"} for i in range(1200)]
    )
    assert repo.count_items(draft.version_id) == 1200
    page = repo.list_items(draft.version_id, page=2)
    assert len(page) == 200
    ids = list(repo.list_items(draft.version_id, page_size=10)["item_id"])
    assert repo.bulk_tag(ids, add=["motor", "glass"]) == 10
    assert repo.bulk_tag(ids[:5], remove=["glass"]) == 5
    assert repo.count_items(draft.version_id, tags=["glass"]) == 5
    assert repo.set_split(ids, "test") == 10
    assert repo.count_items(draft.version_id, split="test") == 10
    with pytest.raises(ValidationError):
        repo.set_split(ids, "train")
    assert repo.count_items(draft.version_id, search="q11") == 111  # q11, q110-q119, q1100-q1199


def test_upsert_and_duplicates(repo: Any) -> None:
    repo.create_dataset("faq")
    draft = repo.create_draft("faq", copy_from=None)
    repo.add_items(draft.version_id, [{"external_id": "a", "query": "q"}])
    with pytest.raises(ValidationError, match="Duplicate"):
        repo.add_items(
            draft.version_id,
            [{"external_id": "b", "query": "1"}, {"external_id": "b", "query": "2"}],
        )
    repo.add_items(draft.version_id, [{"external_id": "a", "query": "q2"}], upsert=True)
    assert list(repo.list_items(draft.version_id)["query"]) == ["q2"]


def test_corpus_checks_and_golden_set(repo: Any, published: Any) -> None:
    version = repo.get_version("claims_faq", "1.0.0")
    assert repo.missing_context_ids(version.version_id).empty
    draft = repo.create_draft("claims_faq")
    repo.add_items(draft.version_id, [{"query": "x", "ground_truth_context_ids": ["nope"]}])
    missing = repo.missing_context_ids(draft.version_id)
    assert list(missing["missing_context_id"]) == ["nope"]
    golden = repo.golden_set(version.version_id)
    assert golden[0]["expected_chunks"][0]["text"].startswith("Clause text")
    docs = repo.search_documents("policies", "text 2")
    assert list(docs["chunk_id"]) == ["c2"]
    assert repo.list_corpora().loc[0, "documents"] == 3


def test_export_respects_pii(repo: Any, published: Any) -> None:
    version = repo.get_version("claims_faq", "1.0.0")
    assert len(repo.version_frame(version.version_id)) == 5
    assert len(repo.version_frame(version.version_id, include_pii=True)) == 6


def test_versions_must_increase(repo: Any, published: Any) -> None:
    with pytest.raises(ValidationError):
        repo.create_draft("claims_faq", "0.9.0")
    with pytest.raises(ValidationError):
        repo.create_draft("claims_faq", "1.0.0")
