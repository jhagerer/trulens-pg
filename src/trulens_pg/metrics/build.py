"""Build TruLens ``Metric`` objects from stored metric definitions (spec section 4.5).

The database holds no executable code:

* ``llm_judge`` templates use ``str.format`` placeholders with plain names only
  (no attribute access, indexing, conversions or format specs);
* ``criteria`` metrics call one allow-listed TruLens provider method;
* ``ground_truth`` metrics call one ``GroundTruthAgreement`` method;
* ``python_ref`` metrics name an entry point in the group ``trulens_pg.metrics``
  of an installed, reviewed package.
"""

from __future__ import annotations

import functools
import inspect
import math
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from trulens_pg.errors import MetricResolutionError, ValidationError
from trulens_pg.metrics.judges import resolve_judge

ENTRY_POINT_GROUP = "trulens_pg.metrics"
KINDS = ("llm_judge", "criteria", "ground_truth", "python_ref")
JUDGE_KINDS = frozenset({"llm_judge", "criteria"})
"""Kinds scored by an LLM judge; they need calibration before publish."""

SELECTOR_NAMES = ("record_input", "record_output", "context", "context_each", "ground_truth")
DEFAULT_INPUTS: dict[str, str] = {
    "question": "record_input",
    "answer": "record_output",
    "contexts": "context",
    "ground_truth": "ground_truth",
}
"""Default mapping of llm_judge placeholders to selectors."""

CRITERIA_METHODS: dict[str, tuple[str, ...]] = {
    "correctness": ("text",),
    "coherence": ("text",),
    "conciseness": ("text",),
    "helpfulness": ("text",),
    "harmfulness": ("text",),
    "maliciousness": ("text",),
    "controversiality": ("text",),
    "insensitivity": ("text",),
    "relevance": ("prompt", "response"),
    "context_relevance": ("question", "context"),
}
"""Allow-list of provider methods for the ``criteria`` kind and their inputs.

The ``*_with_cot_reasons`` variant of each method is called, so results carry
the judge's reasons.
"""

GROUND_TRUTH_METHODS: dict[str, tuple[str, ...]] = {
    "recall_at_k": ("query", "retrieved_context_chunks"),
    "precision_at_k": ("query", "retrieved_context_chunks"),
    "ndcg_at_k": ("query", "retrieved_context_chunks"),
    "mrr": ("query", "retrieved_context_chunks"),
    "ir_hit_rate": ("query", "retrieved_context_chunks"),
    "agreement_measure": ("prompt", "response"),
}
GROUND_TRUTH_DEFAULT_INPUTS: dict[str, str] = {
    "query": "record_input",
    "prompt": "record_input",
    "response": "record_output",
    "retrieved_context_chunks": "context",
}
CRITERIA_DEFAULT_INPUTS: dict[str, str] = {
    "text": "record_output",
    "prompt": "record_input",
    "response": "record_output",
    "question": "record_input",
    "context": "context_each",
}


def make_selector(name: str) -> Any:
    """Return the TruLens ``Selector`` for a selector name."""
    from trulens.core.metric import Selector
    from trulens.otel.semconv.trace import SpanAttributes

    if name == "record_input":
        return Selector.select_record_input()
    if name == "record_output":
        return Selector.select_record_output()
    if name == "context":
        return Selector.select_context(collect_list=True)
    if name == "context_each":
        return Selector.select_context(collect_list=False)
    if name == "ground_truth":
        return Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
            ignore_none_values=True,
        )
    raise ValidationError(f"Unknown selector {name!r}; use one of {SELECTOR_NAMES}.")


def template_placeholders(template: str) -> list[str]:
    """Placeholder names in a template; raise for anything but plain names."""
    names: list[str] = []
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValidationError(f"Invalid prompt template: {exc}") from exc
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if not field_name.isidentifier() or format_spec or conversion:
            raise ValidationError(
                f"Placeholder {{{field_name}}} is not allowed: use plain names such as "
                "{question}, {answer}, {contexts}, {ground_truth}."
            )
        if field_name not in names:
            names.append(field_name)
    return names


@dataclass
class MetricDefinition:
    """A stored metric version, independent of the ORM."""

    name: str
    kind: str
    version: int | None = None
    description: str | None = None
    higher_is_better: bool = True
    prompt_template: str | None = None
    rubric: str | None = None
    criteria: str | None = None
    score_min: int = 0
    score_max: int = 3
    inputs: dict[str, str] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    python_ref: str | None = None
    examples: list[Any] = field(default_factory=list)
    metric_version_id: str | None = None
    status: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}" if self.version is not None else self.name

    def resolved_inputs(self) -> dict[str, str]:
        """Inputs with defaults filled in per kind."""
        if self.inputs:
            return dict(self.inputs)
        if self.kind == "llm_judge" and self.prompt_template:
            return {
                p: DEFAULT_INPUTS[p]
                for p in template_placeholders(self.prompt_template)
                if p in DEFAULT_INPUTS
            }
        if self.kind == "criteria":
            method = self.params.get("method", "")
            return {a: CRITERIA_DEFAULT_INPUTS[a] for a in CRITERIA_METHODS.get(method, ())}
        if self.kind == "ground_truth":
            method = self.params.get("method", "")
            return {a: GROUND_TRUTH_DEFAULT_INPUTS[a] for a in GROUND_TRUTH_METHODS.get(method, ())}
        if self.kind == "python_ref" and self.python_ref:
            ep = allowed_python_refs().get(self.python_ref)
            if ep is None:
                return {}
            bound = set((self.params.get("kwargs") or {}).keys())
            return {
                p: DEFAULT_INPUTS[p]
                for p in inspect.signature(ep.load()).parameters
                if p in DEFAULT_INPUTS and p not in bound
            }
        return {}

    def content(self) -> dict[str, Any]:
        """Fields that define the metric's behaviour (hashed on publish)."""
        return {
            "kind": self.kind,
            "prompt_template": self.prompt_template,
            "rubric": self.rubric,
            "criteria": self.criteria,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "inputs": self.resolved_inputs(),
            "judge": self.judge,
            "params": self.params,
            "python_ref": self.python_ref,
            "examples": self.examples,
        }


def allowed_python_refs() -> dict[str, Any]:
    """Entry points of the group ``trulens_pg.metrics`` (the python_ref allow-list)."""
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def validate_definition(defn: MetricDefinition) -> list[str]:
    """Return a list of problems; empty if the definition can be built."""
    errors: list[str] = []
    if defn.kind not in KINDS:
        return [f"Unknown kind {defn.kind!r}; use one of {KINDS}."]
    if defn.score_max <= defn.score_min:
        errors.append("score_max must be greater than score_min.")
    inputs = defn.resolved_inputs()
    for arg, sel in inputs.items():
        if not arg.isidentifier():
            errors.append(f"Input name {arg!r} is not a valid identifier.")
        if sel not in SELECTOR_NAMES:
            errors.append(f"Input {arg!r} uses unknown selector {sel!r}.")
    if defn.kind == "llm_judge":
        if not defn.prompt_template:
            errors.append("llm_judge needs a prompt_template.")
        else:
            try:
                placeholders = template_placeholders(defn.prompt_template)
            except ValidationError as exc:
                errors.append(str(exc))
            else:
                if not placeholders:
                    errors.append("The prompt template has no placeholders.")
                missing = [p for p in placeholders if p not in inputs]
                if missing:
                    errors.append(f"Placeholder(s) without a mapped input: {missing}.")
                unused = [a for a in inputs if a not in placeholders]
                if unused:
                    errors.append(f"Mapped input(s) not used in the template: {unused}.")
        for i, ex in enumerate(defn.examples):
            if not isinstance(ex, Mapping) or "score" not in ex:
                errors.append(f"Example {i} needs a 'score' and the template inputs.")
    elif defn.kind == "criteria":
        method = defn.params.get("method")
        if method not in CRITERIA_METHODS:
            errors.append(f"criteria needs params.method in {sorted(CRITERIA_METHODS)}.")
        elif set(inputs) != set(CRITERIA_METHODS[method]):
            errors.append(f"Inputs of {method} must be {list(CRITERIA_METHODS[method])}.")
        if not defn.criteria:
            errors.append("criteria needs criteria text.")
    elif defn.kind == "ground_truth":
        method = defn.params.get("method")
        if method not in GROUND_TRUTH_METHODS:
            errors.append(f"ground_truth needs params.method in {sorted(GROUND_TRUTH_METHODS)}.")
        k = defn.params.get("k")
        if k is not None and (not isinstance(k, int) or k < 1):
            errors.append("params.k must be a positive integer.")
    elif defn.kind == "python_ref":
        if not defn.python_ref:
            errors.append("python_ref needs the name of an entry point.")
        elif defn.python_ref not in allowed_python_refs():
            errors.append(
                f"python_ref {defn.python_ref!r} is not an entry point in the group "
                f"{ENTRY_POINT_GROUP!r} (allow-list)."
            )
    return errors


def _with_signature(func: Callable[..., Any], params: Sequence[str]) -> Callable[..., Any]:
    """Give ``func(**kwargs)`` an explicit signature so TruLens maps selectors to it."""
    func.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(p, inspect.Parameter.KEYWORD_ONLY) for p in params]
    )
    return func


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n\n".join(f"[{i + 1}] {v}" for i, v in enumerate(value))
    return str(value)


def render_system_prompt(defn: MetricDefinition) -> str:
    best = defn.score_max if defn.higher_is_better else defn.score_min
    parts = [
        "You are an impartial evaluator of answers produced by a retrieval-augmented "
        "generation (RAG) system.",
    ]
    if defn.rubric:
        parts.append(f"Rubric:\n{defn.rubric.strip()}")
    parts.append(
        f"Give an integer score from {defn.score_min} to {defn.score_max}; "
        f"{best} is the best score."
    )
    if defn.examples:
        lines = ["Scored examples:"]
        for i, ex in enumerate(defn.examples, start=1):
            fields = ", ".join(f"{k}: {_text(v)}" for k, v in ex.items() if k != "score")
            lines.append(f"Example {i}: {fields}\nScore: {ex['score']}")
        parts.append("\n\n".join(lines))
    parts.append("Respond with the criteria you applied, the supporting evidence, and the score.")
    return "\n\n".join(parts)


def render_user_prompt(template: str, values: Mapping[str, Any]) -> str:
    """Fill the template; values are converted to text first."""
    names = template_placeholders(template)
    return template.format_map({n: _text(values.get(n)) for n in names})


def _llm_judge_impl(
    defn: MetricDefinition, judge: Any, info: Mapping[str, Any]
) -> Callable[..., Any]:
    assert defn.prompt_template is not None
    template = defn.prompt_template
    system_prompt = render_system_prompt(defn)
    params = template_placeholders(template)
    temperature = float(info.get("temperature") or 0.0)

    def implementation(**kwargs: Any) -> tuple[float, dict[str, Any]]:
        user_prompt = render_user_prompt(template, kwargs)
        score, reasons = judge.generate_score_and_reasons(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            min_score_val=defn.score_min,
            max_score_val=defn.score_max,
            temperature=temperature,
        )
        meta = dict(reasons or {})
        meta.update(
            {
                "metric": defn.name,
                "metric_version": defn.version,
                "judge_provider": info.get("provider"),
                "judge_model": info.get("model"),
                "judge_temperature": temperature,
            }
        )
        return float(score), meta

    implementation.__name__ = f"{defn.name}_v{defn.version or 0}"
    return _with_signature(implementation, params)


@functools.cache
def _offline_provider() -> Any:
    """Provider for IR metrics (recall@k, ...) that need no LLM; any LLM call fails."""
    from trulens.feedback.llm_provider import LLMProvider

    class OfflineProvider(LLMProvider):  # type: ignore[misc]
        def _create_chat_completion(self, *args: Any, **kwargs: Any) -> Any:
            raise MetricResolutionError("This ground-truth metric has no judge configured.")

    return OfflineProvider(model_engine="none")


def build_implementation(
    defn: MetricDefinition,
    judge: Any = None,
    *,
    golden_set: list[dict[str, Any]] | None = None,
) -> tuple[Callable[..., Any], dict[str, Any], dict[str, Any]]:
    """Return ``(callable, Metric keyword arguments, judge info)`` for a definition.

    The callable takes the mapped inputs as keyword arguments. It is used both
    by `build_metric` and by calibration and the playground.
    """
    errors = validate_definition(defn)
    if errors:
        raise MetricResolutionError(f"Metric {defn.ref} is invalid: " + " ".join(errors))

    extra: dict[str, Any] = {}
    info: dict[str, Any] = {}
    if defn.kind in JUDGE_KINDS:
        provider, info = resolve_judge(judge, defn.judge)
        if provider is None:
            raise MetricResolutionError(
                f"Metric {defn.ref} needs a judge: pass judge=..., store one with the metric, "
                "or set TRULENS_PG_JUDGE_PROVIDER and TRULENS_PG_JUDGE_MODEL."
            )
        if defn.kind == "llm_judge":
            return _llm_judge_impl(defn, provider, info), extra, info
        method = defn.params["method"]
        func = getattr(provider, f"{method}_with_cot_reasons")
        extra.update(
            criteria=defn.criteria,
            min_score_val=defn.score_min,
            max_score_val=defn.score_max,
            temperature=float(info.get("temperature") or 0.0),
        )

        def criteria_impl(**kwargs: Any) -> Any:
            return func(
                **kwargs,
                criteria=defn.criteria,
                min_score_val=defn.score_min,
                max_score_val=defn.score_max,
                temperature=float(info.get("temperature") or 0.0),
            )

        return _with_signature(criteria_impl, CRITERIA_METHODS[method]), extra, info

    if defn.kind == "ground_truth":
        from trulens.feedback import GroundTruthAgreement

        method = defn.params["method"]
        if golden_set is None:
            raise MetricResolutionError(
                f"Metric {defn.ref} compares with ground truth: pass dataset='bench:<ds>@<ver>' "
                "or golden_set=[...] to resolve_metrics()."
            )
        if method == "agreement_measure":
            provider, info = resolve_judge(judge, defn.judge)
            if provider is None:
                raise MetricResolutionError(f"Metric {defn.ref} (agreement_measure) needs a judge.")
        else:  # IR metrics compare chunks and call no LLM
            provider = _offline_provider()
        gta = GroundTruthAgreement(golden_set, provider=provider)
        func = getattr(gta, method)
        args = GROUND_TRUTH_METHODS[method]
        k = defn.params.get("k")

        def gt_impl(**kwargs: Any) -> Any:
            call = {a: kwargs[a] for a in args}
            if k is not None and method.endswith("_at_k"):
                call["k"] = k
            return func(**call)

        return _with_signature(gt_impl, args), extra, info

    # python_ref
    assert defn.python_ref is not None
    target = allowed_python_refs()[defn.python_ref].load()
    params = dict(defn.params.get("kwargs") or {})
    impl = functools.partial(target, **params) if params else target
    mapped = defn.resolved_inputs()
    inputs = [
        name
        for name, param in inspect.signature(target).parameters.items()
        if name not in params
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and (name in mapped or param.default is inspect.Parameter.empty)
    ]

    def ref_impl(**kwargs: Any) -> Any:
        return impl(**kwargs)

    ref_impl.__name__ = defn.python_ref
    return _with_signature(ref_impl, inputs), extra, info


def _skip_undefined_scores(func: Callable[..., Any], ref: str) -> Callable[..., Any]:
    """Raise TruLens ``SkipEval`` instead of returning ``None`` or NaN.

    IR metrics return NaN for items without ground truth (for example
    out-of-scope questions). TruLens 2.16 would write the NaN into a JSON column,
    which Postgres rejects, so the whole batch of evaluation spans is lost.
    """
    from trulens.core.metric import SkipEval

    def guarded(**kwargs: Any) -> Any:
        result = func(**kwargs)
        score = result[0] if isinstance(result, tuple) else result
        if score is None or (isinstance(score, float) and math.isnan(score)):
            raise SkipEval(f"{ref}: no score for this record (for example no ground truth).")
        return result

    guarded.__name__ = getattr(func, "__name__", ref)
    return _with_signature(guarded, list(inspect.signature(func).parameters))


def build_metric(
    defn: MetricDefinition,
    judge: Any = None,
    *,
    golden_set: list[dict[str, Any]] | None = None,
) -> Any:
    """Build a TruLens ``Metric`` named ``<name>@<version>`` from a stored definition."""
    from trulens.core.metric import Metric

    raw_implementation, extra, info = build_implementation(defn, judge, golden_set=golden_set)
    implementation = _skip_undefined_scores(raw_implementation, defn.ref)
    sig_params = set(inspect.signature(implementation).parameters)
    inputs = {a: s for a, s in defn.resolved_inputs().items() if a in sig_params}
    missing = sig_params - set(inputs)
    if missing:
        raise MetricResolutionError(
            f"Metric {defn.ref}: no selector for argument(s) {sorted(missing)}; set inputs."
        )
    metric = Metric(
        implementation=implementation,
        name=defn.ref,
        selectors={arg: make_selector(sel) for arg, sel in inputs.items()},
        higher_is_better=defn.higher_is_better,
        description=defn.description,
        metric_type=f"trulens_pg.{defn.kind}",
        min_score_val=defn.score_min,
        max_score_val=defn.score_max,
        temperature=float(info.get("temperature") or 0.0),
        **({"criteria": extra["criteria"]} if "criteria" in extra else {}),
    )
    metric._trulens_pg_definition = defn
    metric._trulens_pg_judge = info
    return metric
