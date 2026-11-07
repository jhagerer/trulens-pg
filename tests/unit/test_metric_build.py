from __future__ import annotations

import inspect
from typing import Any

import pytest

from trulens_pg.errors import MetricResolutionError, ValidationError
from trulens_pg.metrics.build import (
    MetricDefinition,
    build_implementation,
    build_metric,
    render_user_prompt,
    template_placeholders,
    validate_definition,
)
from trulens_pg.metrics.judges import JudgeConfig, resolve_judge
from trulens_pg.metrics.store import (
    call_on_label,
    mean_abs_error,
    parse_metric_ref,
    spearman,
)

TEMPLATE = "Question: {question}\nAnswer: {answer}\nContexts: {contexts}"


def judge_defn(**kw: Any) -> MetricDefinition:
    base: dict[str, Any] = {
        "name": "policy_citation",
        "kind": "llm_judge",
        "version": 1,
        "prompt_template": TEMPLATE,
        "rubric": "3: cites the clause; 0: no citation.",
        "score_min": 0,
        "score_max": 3,
    }
    base.update(kw)
    return MetricDefinition(**base)


def test_placeholders_plain_names_only() -> None:
    assert template_placeholders("{a} {b} {a} {{literal}}") == ["a", "b"]
    for bad in ("{a.__class__}", "{a[0]}", "{a!r}", "{a:>10}", "{0}", "{"):
        with pytest.raises(ValidationError):
            template_placeholders(bad)


def test_render_user_prompt_joins_contexts() -> None:
    text = render_user_prompt(TEMPLATE, {"question": "q", "answer": "a", "contexts": ["x", "y"]})
    assert "Contexts: [1] x\n\n[2] y" in text


def test_default_inputs_from_placeholders() -> None:
    assert judge_defn().resolved_inputs() == {
        "question": "record_input",
        "answer": "record_output",
        "contexts": "context",
    }


def test_validation_messages() -> None:
    assert validate_definition(judge_defn()) == []
    errors = validate_definition(judge_defn(inputs={"question": "record_input"}))
    assert any("without a mapped input" in e for e in errors)
    errors = validate_definition(judge_defn(prompt_template="{foo}"))
    assert any("without a mapped input" in e for e in errors)
    assert validate_definition(judge_defn(score_min=3, score_max=3))
    assert validate_definition(MetricDefinition(name="x", kind="criteria", params={"method": "x"}))
    assert validate_definition(
        MetricDefinition(name="x", kind="python_ref", python_ref="os:system")
    )
    assert validate_definition(MetricDefinition(name="x", kind="unknown"))


def test_llm_judge_implementation_calls_judge(fake_judge: Any) -> None:
    impl, _extra, info = build_implementation(judge_defn(), fake_judge)
    assert list(inspect.signature(impl).parameters) == ["question", "answer", "contexts"]
    score, meta = impl(question="Is glass covered?", answer="Yes, KFZ-1001.", contexts=["c"])
    assert score == 1.0
    assert meta["judge_model"] == "fake-judge-1"
    assert meta["metric_version"] == 1
    assert "Is glass covered?" in fake_judge.calls[0]["user"]
    assert "integer score from 0 to 3" in fake_judge.calls[0]["system"]
    assert info["model"] == "fake-judge-1"


def test_build_metric_has_versioned_name_and_selectors(fake_judge: Any) -> None:
    metric = build_metric(judge_defn(), fake_judge)
    assert metric.name == "policy_citation@1"
    assert set(metric.selectors) == {"question", "answer", "contexts"}
    assert metric.higher_is_better is True


def test_judge_kind_without_judge_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRULENS_PG_JUDGE_PROVIDER", raising=False)
    with pytest.raises(MetricResolutionError, match="needs a judge"):
        build_metric(judge_defn())


def test_criteria_kind(fake_judge: Any) -> None:
    defn = MetricDefinition(
        name="guideline_correctness",
        kind="criteria",
        version=2,
        criteria="Follows the claims handling guideline.",
        params={"method": "correctness"},
    )
    metric = build_metric(defn, fake_judge)
    assert set(metric.selectors) == {"text"}
    score, _ = metric.imp(text="See KFZ-1")
    assert score == 1.0
    assert fake_judge.calls[-1]["criteria"] == "Follows the claims handling guideline."


def test_python_ref_kind_uses_entry_points() -> None:
    defn = MetricDefinition(
        name="has_policy", kind="python_ref", python_ref="contains_policy_number"
    )
    assert defn.resolved_inputs() == {"answer": "record_output"}
    metric = build_metric(defn)
    score, reason = metric.imp(answer="Clause KFZ-2024001 applies.")
    assert score == 1.0
    assert "KFZ-2024001" in reason["reason"]


def test_python_ref_with_bound_kwargs() -> None:
    defn = MetricDefinition(
        name="short",
        kind="python_ref",
        python_ref="answer_length_ok",
        params={"kwargs": {"max_words": 2}},
    )
    impl, _, _ = build_implementation(defn)
    assert list(inspect.signature(impl).parameters) == ["answer"]
    assert impl(answer="two words") == 1.0
    assert impl(answer="three words here") == 0.0


def test_ground_truth_kind_needs_golden_set() -> None:
    defn = MetricDefinition(
        name="context_recall_at_5", kind="ground_truth", params={"method": "recall_at_k", "k": 5}
    )
    with pytest.raises(MetricResolutionError, match="golden_set"):
        build_metric(defn)
    golden = [{"query": "q", "expected_response": "a", "expected_chunks": [{"text": "t1"}]}]
    impl, _, _ = build_implementation(defn, golden_set=golden)
    assert impl(query="q", retrieved_context_chunks=["t0", "t1"]) == 1.0
    assert impl(query="q", retrieved_context_chunks=["t0"]) == 0.0


def test_calibration_statistics() -> None:
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert mean_abs_error([0.0, 1.0], [0.5, 0.5]) == pytest.approx(0.5)


def test_call_on_label_per_context() -> None:
    def impl(context: str) -> float:
        return 1.0 if context == "good" else 0.0

    score, _ = call_on_label(impl, {"context": "context_each"}, {"contexts": ["good", "bad"]})
    assert score == 0.5


def test_parse_metric_ref() -> None:
    assert parse_metric_ref("policy_citation") == ("policy_citation", None)
    assert parse_metric_ref("policy_citation@2") == ("policy_citation", 2)
    with pytest.raises(ValidationError):
        parse_metric_ref("policy_citation@v2")


def test_judge_config() -> None:
    cfg = JudgeConfig.from_mapping({"provider": "openai", "model": "gpt-4o-mini", "temperature": 0})
    assert cfg.label == "openai:gpt-4o-mini@t=0"
    with pytest.raises(ValidationError):
        JudgeConfig(provider="cortex", model="x")


def test_resolve_judge_prefers_argument(fake_judge: Any) -> None:
    provider, info = resolve_judge(fake_judge, {"provider": "openai", "model": "gpt-4o"})
    assert provider is fake_judge
    assert info["model"] == "fake-judge-1"


def test_undefined_scores_are_skipped_not_written() -> None:
    """NaN scores would break the Postgres JSON insert of all evaluation spans."""
    from trulens.core.metric import SkipEval

    defn = MetricDefinition(
        name="context_recall_at_5", kind="ground_truth", params={"method": "recall_at_k", "k": 5}
    )
    golden = [{"query": "q", "expected_response": "a", "expected_chunks": [{"text": "t1"}]}]
    metric = build_metric(defn, golden_set=golden)
    assert metric.imp(query="q", retrieved_context_chunks=["t1"]) == 1.0
    with pytest.raises(SkipEval):
        metric.imp(query="unknown question", retrieved_context_chunks=["t1"])
