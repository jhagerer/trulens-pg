"""Contract tests: fail when TruLens changes the internals trulens-pg depends on.

Run them against every new TruLens release before widening the version pin.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract


def test_default_run_dao_interface() -> None:
    from trulens.core.dao.default_run import DefaultRunDao
    from trulens.core.dao.run import RunDaoBase

    assert issubclass(DefaultRunDao, RunDaoBase)
    assert list(inspect.signature(DefaultRunDao.__init__).parameters) == ["self", "db"]
    params = list(inspect.signature(DefaultRunDao.create_new_run).parameters)
    assert params == [
        "self",
        "object_name",
        "object_type",
        "run_name",
        "dataset_name",
        "source_type",
        "dataset_spec",
        "object_version",
        "description",
        "label",
        "llm_judge_name",
        "mode",
    ]
    assert list(inspect.signature(RunDaoBase.fetch_source_data).parameters) == [
        "self",
        "source_name",
    ]


def test_app_run_dao_fields() -> None:
    from trulens.core.app import App

    assert "run_dao" in App.model_fields
    private = App.__private_attributes__
    for name in ("_object_name", "_object_type", "_object_version"):
        assert name in private, name
    source = inspect.getsource(App.add_run)
    assert "self._object_name" in source
    assert "self.run_dao.create_new_run" in source


def test_run_start_fetches_source_without_input_df() -> None:
    from trulens.core.run import Run

    source = inspect.getsource(Run.start)
    assert "self.run_dao.fetch_source_data(self.source_info.name)" in source


def test_compute_metrics_accepts_metric_objects() -> None:
    from trulens.core.run import Run

    source = inspect.getsource(Run._compute_client_side_metrics_from_configs)
    assert "metric_module.Metric" in source


def test_selectors_and_span_attributes() -> None:
    from trulens.core.metric import Selector
    from trulens.otel.semconv.trace import SpanAttributes

    for name in ("select_record_input", "select_record_output", "select_context"):
        assert hasattr(Selector, name)
    assert SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT.endswith("ground_truth_output")
    assert "dataset_column" in inspect.signature(Selector.__init__).parameters


def test_judge_and_ground_truth_methods() -> None:
    from trulens.feedback import GroundTruthAgreement
    from trulens.feedback.llm_provider import LLMProvider

    from trulens_pg.metrics.build import CRITERIA_METHODS, GROUND_TRUTH_METHODS

    params = list(inspect.signature(LLMProvider.generate_score_and_reasons).parameters)
    assert params[:6] == [
        "self",
        "system_prompt",
        "user_prompt",
        "min_score_val",
        "max_score_val",
        "temperature",
    ]
    for method, args in CRITERIA_METHODS.items():
        sig = inspect.signature(getattr(LLMProvider, f"{method}_with_cot_reasons"))
        assert list(sig.parameters)[1 : 1 + len(args)] == list(args), method
        assert "criteria" in sig.parameters
    for method, args in GROUND_TRUTH_METHODS.items():
        sig = inspect.signature(getattr(GroundTruthAgreement, method))
        assert list(sig.parameters)[1 : 1 + len(args)] == list(args), method


def test_metric_constructor_arguments() -> None:
    from trulens.core.metric import Metric

    params = inspect.signature(Metric.__init__).parameters
    for name in ("implementation", "criteria", "min_score_val", "max_score_val", "description"):
        assert name in params
    assert "higher_is_better" in Metric.model_fields


def test_prompt_registry_api() -> None:
    """Judge prompts (spec section 4.5) use the TruLens prompt registry (TruLens >= 2.14)."""
    from trulens.core import prompt as core_prompt
    from trulens.core.schema import prompt as prompt_schema

    assert prompt_schema.VARIABLE_PATTERN.pattern == r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}"
    assert callable(prompt_schema.validate_template)
    for name, params in {
        "create_prompt": ["db", "slug", "name", "prompt_type", "description", "tags"],
        "create_prompt_version": ["db", "prompt", "text", "messages", "variables"],
        "resolve_prompt": ["db", "prompt", "label", "version_id"],
        "as_prompt": ["db", "prompt"],
    }.items():
        sig = list(inspect.signature(getattr(core_prompt, name)).parameters)
        assert sig[: len(params)] == params, name
    render = inspect.signature(prompt_schema.ResolvedPrompt.render).parameters
    assert "strict" in render


def test_dataset_api() -> None:
    """Published versions become TruLens datasets; traces are curated (spec sections 5, 6.1)."""
    from trulens.core import dataset as core_dataset
    from trulens.core.database.sqlalchemy import SQLAlchemyDB
    from trulens.core.schema.groundtruth import GroundTruth

    gt_params = inspect.signature(GroundTruth.__init__).parameters
    for name in ("dataset_id", "query", "query_id", "expected_response", "expected_chunks", "meta"):
        assert name in gt_params, name
    for method in (
        "insert_dataset",
        "batch_insert_ground_truth",
        "get_ground_truths_by_dataset",
        "get_prompt",
        "get_prompt_version",
        "get_prompts",
        "get_prompt_versions",
    ):
        assert hasattr(SQLAlchemyDB, method), method
    curate = inspect.signature(core_dataset.curate_records_to_dataset).parameters
    for name in ("dataset_name", "records", "db", "mapping", "on_error"):
        assert name in curate, name
    assert set(core_dataset.TraceDatasetMapping.model_fields) >= {
        "query",
        "query_id",
        "expected_response",
        "expected_chunks",
        "metadata",
    }
    assert core_dataset.PROVENANCE_COLUMNS["source_app_name"] == "app_name"
