"""`resolve_metrics()` and `compute_metrics()`: stored custom metrics by name."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from trulens_pg.errors import MetricResolutionError, NotFoundError, TrulensPgError
from trulens_pg.metrics.build import GoldenSet, MetricDefinition, build_metric
from trulens_pg.metrics.judges import judge_model_label
from trulens_pg.metrics.store import MetricStore

logger = logging.getLogger(__name__)

SET_PREFIX = "set:"
_READY = frozenset(
    {
        "INVOCATION_COMPLETED",
        "INVOCATION_PARTIALLY_COMPLETED",
        "COMPUTATION_IN_PROGRESS",
        "COMPLETED",
        "PARTIALLY_COMPLETED",
    }
)
_FAILED = frozenset({"FAILED", "CANCELLED"})


def _definitions_for(ref: str, store: MetricStore) -> list[MetricDefinition]:
    if ref.startswith(SET_PREFIX):
        return store.get_metric_set(ref.removeprefix(SET_PREFIX))
    if "@" in ref:
        return [store.get_definition(ref)]
    is_metric = store.has_metric(ref)
    is_set = store.has_metric_set(ref)
    if is_metric and is_set:
        raise MetricResolutionError(
            f"{ref!r} is both a metric and a metric set; write {ref}@<version> or set:{ref}."
        )
    if is_set:
        return store.get_metric_set(ref)
    if is_metric:
        return [store.get_definition(ref)]
    raise NotFoundError(f"No metric or metric set named {ref!r}.")


def resolve_definitions(
    metrics: Sequence[Any], *, store: MetricStore | None = None
) -> list[MetricDefinition | Any]:
    """Replace string references by stored definitions; keep other objects."""
    resolved: list[MetricDefinition | Any] = []
    store_ = store
    for metric in metrics:
        if isinstance(metric, str):
            store_ = store_ or MetricStore.from_context()
            resolved.extend(_definitions_for(metric, store_))
        else:
            resolved.append(metric)
    return resolved


def resolve_metrics(
    metrics: Sequence[Any],
    judge: Any = None,
    *,
    golden_set: GoldenSet | None = None,
    store: MetricStore | None = None,
) -> list[Any]:
    """Build TruLens ``Metric`` objects from stored metrics; pass ``Metric`` objects through.

    Args:
        metrics: Entries such as ``"policy_citation"`` (latest published version),
            ``"policy_citation@2"``, a metric set name (``"claims-rag-v1"`` or
            ``"set:claims-rag-v1"``), or TruLens ``Metric`` objects.
        judge: TruLens provider object or `JudgeConfig`; default: the judge
            stored with each metric version, then ``TRULENS_PG_JUDGE_*``.
        golden_set: Ground truth for ``ground_truth`` metrics in the TruLens
            ``GroundTruthAgreement`` format: a list of dicts or a DataFrame with
            ``query``, ``expected_response`` and ``expected_chunks`` (list of
            ``{"text": ...}``), for example read from a Postgres table or from
            ``TruSession.get_ground_truth(name)``.
        store: Metric store (default: from the current `connect()` context).

    Returns:
        List of ``Metric`` objects, in input order.
    """
    definitions = resolve_definitions(metrics, store=store)

    result = []
    for item in definitions:
        if isinstance(item, MetricDefinition):
            result.append(build_metric(item, judge, golden_set=golden_set))
        else:
            result.append(item)
    names = [getattr(m, "name", None) for m in result]
    duplicates = sorted({n for n in names if n is not None and names.count(n) > 1})
    if duplicates:
        raise MetricResolutionError(f"Metric(s) listed more than once: {duplicates}")
    return result


def wait_for_invocation(run: Any, *, timeout: float = 600.0, poll_interval: float = 2.0) -> str:
    """Wait until the run's invocation is complete (status ``INVOCATION_COMPLETED`` or later)."""
    deadline = time.monotonic() + timeout
    while True:
        status = run.get_status()
        name = getattr(status, "name", str(status))
        if name in _READY:
            return str(name)
        if name in _FAILED:
            raise TrulensPgError(f"Run {run.run_name} is {name}; metrics cannot be computed.")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Run {run.run_name} is still {name} after {timeout:.0f} s.")
        time.sleep(poll_interval)


def compute_metrics(
    run: Any,
    metrics: Sequence[Any],
    judge: Any = None,
    *,
    golden_set: GoldenSet | None = None,
    store: MetricStore | None = None,
    timeout: float = 600.0,
    poll_interval: float = 2.0,
) -> str:
    """``run.compute_metrics(resolve_metrics(metrics, judge))`` after the invocation completes.

    Stored metric versions and their judge are recorded in the ``run_metric`` table.
    """
    wait_for_invocation(run, timeout=timeout, poll_interval=poll_interval)
    if store is None and any(isinstance(x, str) for x in metrics):
        store = MetricStore.from_context()
    resolved = resolve_metrics(metrics, judge, golden_set=golden_set, store=store)
    status: str = run.compute_metrics(resolved)

    by_judge: dict[str | None, list[str]] = defaultdict(list)
    for metric in resolved:
        defn = getattr(metric, "_trulens_pg_definition", None)
        if isinstance(defn, MetricDefinition) and defn.metric_version_id:
            label = judge_model_label(getattr(metric, "_trulens_pg_judge", {}) or {})
            by_judge[label].append(defn.metric_version_id)
    if by_judge and store is not None:
        for judge_label, ids in by_judge.items():
            store.record_run_metrics(
                run_name=run.run_name,
                app_name=run.object_name,
                metric_version_ids=ids,
                judge_model=judge_label,
            )
    return status
