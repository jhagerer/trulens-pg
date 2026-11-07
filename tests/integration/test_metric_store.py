from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import text

from tests.conftest import TEST_SCHEMA
from trulens_pg.errors import (
    MetricResolutionError,
    NotFoundError,
    ReadOnlyError,
    ValidationError,
    WorkflowError,
)
from trulens_pg.metrics import resolve_metrics

pytestmark = pytest.mark.pg

LABELS = [
    {"question": "q1", "answer": "See KFZ-1001", "human_score": 3},
    {"question": "q2", "answer": "No idea", "human_score": 1},
]


def _judge_metric(store: Any) -> Any:
    store.create_metric("policy_citation", "llm_judge", owner="ds")
    return store.create_draft(
        "policy_citation",
        prompt_text="Q: {{question}}\nA: {{answer}}",
        rubric="Cites clause.",
        created_by="ds",
    )


def test_calibration_gate_and_versions(store: Any, fake_judge: Any) -> None:
    draft = _judge_metric(store)
    assert draft.version == 1
    blockers = store.publish_blockers(draft.metric_version_id, min_labels=2)
    assert any("No calibration" in b for b in blockers)
    store.add_labels("policy_citation", LABELS)
    with pytest.raises(ValidationError, match="outside"):
        store.add_labels("policy_citation", [{"question": "q", "answer": "a", "human_score": 9}])
    store.calibrate(draft.metric_version_id, fake_judge)
    with pytest.raises(WorkflowError, match="at least 20"):
        store.publish(draft.metric_version_id, published_by="p")
    published = store.publish(draft.metric_version_id, published_by="p", min_labels=2)
    assert published.status == "published"
    assert published.inputs == {"question": "record_input", "answer": "record_output"}

    with pytest.raises(WorkflowError):
        store.update_draft(draft.metric_version_id, rubric="changed")
    with pytest.raises(ReadOnlyError), store.transaction() as s:
        s.execute(
            text(
                f"UPDATE {TEST_SCHEMA}.metric_version SET rubric = 'x' WHERE metric_version_id = :i"
            ),
            {"i": draft.metric_version_id},
        )

    v2 = store.create_draft("policy_citation", rubric="Cites clause and page.")
    assert v2.version == 2
    assert v2.prompt_version_id == draft.prompt_version_id
    assert v2.prompt.version.text == "Q: {{question}}\nA: {{answer}}"
    diff = store.diff(v2.metric_version_id)
    assert list(diff["field"]) == ["rubric"]
    assert store.get_definition("policy_citation").version == 1
    with pytest.raises(WorkflowError, match="draft"):
        store.get_definition("policy_citation@2")
    listing = store.list_metrics()
    assert listing.loc[0, "latest_version"] == 2
    assert listing.loc[0, "latest_published"] == 1


def test_resolve_by_name_version_and_set(store: Any, fake_judge: Any) -> None:
    draft = _judge_metric(store)
    store.add_labels("policy_citation", LABELS)
    store.calibrate(draft.metric_version_id, fake_judge)
    store.publish(draft.metric_version_id, published_by="p", min_labels=2)
    store.create_metric("has_policy_number", "python_ref")
    ref = store.create_draft("has_policy_number", python_ref="contains_policy_number")
    store.publish(ref.metric_version_id, published_by="p")
    store.create_metric_set("claims-rag-v1", ["policy_citation@1", "has_policy_number@1"])

    by_name = resolve_metrics(["policy_citation", "has_policy_number@1"], fake_judge, store=store)
    assert [m.name for m in by_name] == ["policy_citation@1", "has_policy_number@1"]
    by_set = resolve_metrics(["set:claims-rag-v1"], fake_judge, store=store)
    assert [m.name for m in by_set] == ["policy_citation@1", "has_policy_number@1"]
    with pytest.raises(MetricResolutionError, match="more than once"):
        resolve_metrics(["claims-rag-v1", "policy_citation"], fake_judge, store=store)
    with pytest.raises(NotFoundError):
        resolve_metrics(["unknown_metric"], store=store)
    with pytest.raises(ValidationError, match="exact versions"):
        store.create_metric_set("s2", ["policy_citation"])

    sets = store.list_metric_sets()
    assert sets.loc[0, "metrics"] == "policy_citation@1, has_policy_number@1"

    store.deprecate(ref.metric_version_id)
    with pytest.raises(WorkflowError, match="deprecated"):
        store.get_metric_set("claims-rag-v1")


def test_python_ref_must_be_allow_listed(store: Any) -> None:
    store.create_metric("evil", "python_ref")
    draft = store.create_draft("evil", python_ref="os:system")
    with pytest.raises(WorkflowError, match="allow-list"):
        store.publish(draft.metric_version_id, published_by="p")


def test_judge_prompt_lives_in_trulens_registry(store: Any, ctx: Any, fake_judge: Any) -> None:
    """Spec section 4.5: a metric version pins an exact PromptVersion id, never a label."""
    draft = _judge_metric(store)
    session = ctx.session
    versions = session.get_prompt_versions("trulens-pg.policy_citation")
    assert list(versions["version_id"]) == [draft.prompt_version_id]
    assert list(draft.prompt.version.variables) == ["question", "answer"]

    store.add_labels("policy_citation", LABELS)
    store.calibrate(draft.metric_version_id, fake_judge)
    store.publish(draft.metric_version_id, published_by="p", min_labels=2)

    # A newer prompt version and a moved label do not change the published metric.
    newer = session.create_prompt_version(
        "trulens-pg.policy_citation", text="Changed: {{answer}}", change_note="experiment"
    )
    session.set_prompt_label("trulens-pg.policy_citation", "production", newer)
    pinned = store.get_definition("policy_citation@1")
    assert pinned.prompt_version_id == draft.prompt_version_id
    assert pinned.prompt.version.text == "Q: {{question}}\nA: {{answer}}"

    # A draft can pin a version picked from the registry.
    v2 = store.create_draft("policy_citation", prompt_version_id=newer.version_id)
    assert v2.prompt_variables() == ["answer"]
    diff = store.diff(v2.metric_version_id).set_index("field")
    assert diff.loc["prompt text", "draft"] == "Changed: {{answer}}"
    with pytest.raises(NotFoundError):
        store.create_draft("policy_citation", prompt_version_id="prompt_version_missing")


def test_invalid_judge_prompt_is_rejected(store: Any) -> None:
    store.create_metric("bad_prompt", "llm_judge")
    with pytest.raises(ValidationError, match="Invalid judge prompt"):
        store.create_draft("bad_prompt", prompt_text="{% if x %}{{x}}{% endif %}")
    with pytest.raises(ValidationError, match="no \\{\\{variable\\}\\}"):
        store.create_draft("bad_prompt", prompt_text="No placeholders at all")


def test_migration_0002_refuses_stored_prompt_templates(pg_url: str, ctx: Any) -> None:
    from trulens_pg import migrations
    from trulens_pg.session import create_bench_engine

    schema = "bench_migration_test"
    engine = create_bench_engine(pg_url)
    try:
        if migrations.current_revision(engine, schema) is not None:
            migrations.downgrade(engine, schema, "base")
        migrations.upgrade(engine, schema, "0001")
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {schema}.metric (metric_id, name, kind, higher_is_better) "
                    "VALUES (gen_random_uuid(), 'm', 'llm_judge', true)"
                )
            )
            conn.execute(
                text(
                    f"INSERT INTO {schema}.metric_version (metric_version_id, metric_id, "
                    "version, status, prompt_template, score_min, score_max, inputs, judge, "
                    "params, examples) SELECT gen_random_uuid(), metric_id, 1, 'draft', "
                    f"'{{question}}', 0, 3, '{{}}', '{{}}', '{{}}', '[]' FROM {schema}.metric"
                )
            )
        with pytest.raises(RuntimeError, match="prompt registry"):
            migrations.upgrade(engine, schema)
        assert migrations.current_revision(engine, schema) == "0001"
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE {schema}.metric_version, {schema}.metric CASCADE"))
        migrations.upgrade(engine, schema)
        assert migrations.current_revision(engine, schema) == migrations.head_revision()
        migrations.downgrade(engine, schema, "base")
    finally:
        engine.dispose()
