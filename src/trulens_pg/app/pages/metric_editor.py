"""Metric editor: create or edit a draft; placeholders are checked against mapped inputs."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from trulens_pg.app import common
from trulens_pg.app.auth import EDITOR
from trulens_pg.errors import ValidationError
from trulens_pg.metrics.build import (
    CRITERIA_METHODS,
    DEFAULT_INPUTS,
    GROUND_TRUTH_METHODS,
    SELECTOR_NAMES,
    MetricDefinition,
    allowed_python_refs,
    validate_definition,
)
from trulens_pg.metrics.judges import PROVIDERS
from trulens_pg.trulens_store import default_prompt_slug, make_resolved_prompt

st.title("Metric editor")
store = common.store()
user = common.require(EDITOR)

name = common.pick_metric()
if name is None:
    st.stop()
metric = store.get_metric(name)
versions = store.list_versions(name)
drafts = versions[versions["status"].isin(common.EDITABLE)] if not versions.empty else versions

if drafts.empty:
    st.info("No draft. Create one, copied from the latest version if there is one.")
    if st.button("Create draft", type="primary", disabled=not user.is_editor):
        if (
            common.run_write(
                lambda: store.create_draft(name, created_by=user.name),
                lambda d: f"Created draft {d.ref} (1 row).",
            )
            is not None
        ):
            st.rerun()
    st.stop()

row = drafts.iloc[0]
defn = store.definition(row.metric_version_id)
st.caption(f"Editing {defn.ref} · kind {defn.kind} · status {defn.status}")

fields: dict[str, Any] = {}
prompt_text: str | None = None
prompt_slug: str | None = None
candidate_prompt = defn.prompt
if defn.kind == "llm_judge":
    st.subheader("Judge prompt (TruLens prompt registry)")
    if defn.prompt is not None:
        st.caption(
            f"Pinned: {defn.prompt.prompt.slug} · version {defn.prompt_version_id} "
            "(a metric version pins an exact version, never a label)"
        )
    source = st.radio(
        "Judge prompt",
        ["Write a new prompt version", "Pick a version from the registry"],
        horizontal=True,
    )
    if source.startswith("Write"):
        prompt_slug = st.text_input(
            "Prompt slug",
            defn.prompt.prompt.slug if defn.prompt is not None else default_prompt_slug(name),
        )
        prompt_text = st.text_area(
            "Prompt text",
            defn.prompt.version.text
            if defn.prompt is not None and defn.prompt.version.text
            else "Question: {{question}}\n\nAnswer: {{answer}}\n\nContexts: {{contexts}}",
            height=180,
            help="Placeholders use the TruLens syntax {{name}}, e.g. {{question}}, {{answer}}, "
            "{{contexts}}, {{ground_truth}}. Saving creates a new prompt version only if the "
            "text changed.",
        )
        try:
            candidate_prompt = make_resolved_prompt(prompt_text, slug=prompt_slug)
        except ValidationError as exc:
            st.error(str(exc))
            candidate_prompt = None
    else:
        with common.handled():
            prompts = store.trulens.list_prompts()
            if prompts.empty:
                st.info("The registry has no prompts yet.")
            else:
                slugs = list(prompts["slug"])
                current = defn.prompt.prompt.slug if defn.prompt is not None else slugs[0]
                slug = st.selectbox(
                    "Prompt", slugs, index=slugs.index(current) if current in slugs else 0
                )
                versions = store.trulens.list_prompt_versions(str(slug)).iloc[::-1]
                labels = [
                    f"{v.version_id} · {v.created_at} · {v.change_note or ''}"
                    for v in versions.itertuples()
                ]
                pick = st.selectbox("Version", range(len(labels)), format_func=lambda i: labels[i])
                version_id = str(versions.iloc[int(pick)].version_id)
                candidate_prompt = store.trulens.resolve_prompt_version(version_id)
                st.code(candidate_prompt.version.text or str(candidate_prompt.version.messages))
                fields["prompt_version_id"] = version_id
    fields["rubric"] = st.text_area("Rubric", defn.rubric or "", height=140)
    placeholders = list(candidate_prompt.version.variables) if candidate_prompt else []
elif defn.kind == "criteria":
    methods = sorted(CRITERIA_METHODS)
    current = defn.params.get("method", methods[0])
    method = st.selectbox(
        "Provider method", methods, index=methods.index(current) if current in methods else 0
    )
    fields["params"] = {"method": method}
    fields["criteria"] = st.text_area("Criteria", defn.criteria or "")
    placeholders = list(CRITERIA_METHODS[str(method)])
elif defn.kind == "ground_truth":
    methods = sorted(GROUND_TRUTH_METHODS)
    current = defn.params.get("method", methods[0])
    method = st.selectbox(
        "GroundTruthAgreement method",
        methods,
        index=methods.index(current) if current in methods else 0,
    )
    params: dict[str, Any] = {"method": method}
    if str(method).endswith("_at_k"):
        params["k"] = int(st.number_input("k", 1, 100, int(defn.params.get("k") or 5)))
    fields["params"] = params
    placeholders = list(GROUND_TRUTH_METHODS[str(method)])
else:
    refs = sorted(allowed_python_refs())
    current = defn.python_ref if defn.python_ref in refs else (refs[0] if refs else None)
    fields["python_ref"] = st.selectbox(
        "Function (entry point in group trulens_pg.metrics)",
        refs,
        index=refs.index(current) if current in refs else 0,
    )
    kwargs_text = st.text_area(
        "Fixed keyword arguments (JSON)", json.dumps(defn.params.get("kwargs") or {})
    )
    try:
        fields["params"] = {"kwargs": json.loads(kwargs_text or "{}")}
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
    probe = MetricDefinition(
        name=name,
        kind="python_ref",
        python_ref=fields["python_ref"],
        params=fields.get("params", {}),
    )
    placeholders = list(probe.resolved_inputs()) or list(defn.inputs)

cols = st.columns(2)
fields["score_min"] = int(cols[0].number_input("Minimum score", value=int(defn.score_min), step=1))
fields["score_max"] = int(cols[1].number_input("Maximum score", value=int(defn.score_max), step=1))

st.subheader("Inputs")
inputs: dict[str, str] = {}
grid = st.columns(max(len(placeholders), 1))
for i, placeholder in enumerate(placeholders):
    default = defn.inputs.get(placeholder) or DEFAULT_INPUTS.get(placeholder) or "record_output"
    inputs[placeholder] = grid[i].selectbox(
        f"{{{placeholder}}}",
        SELECTOR_NAMES,
        index=SELECTOR_NAMES.index(default) if default in SELECTOR_NAMES else 0,
        key=f"input_{placeholder}",
    )
fields["inputs"] = inputs

if defn.kind in ("llm_judge", "criteria") or defn.params.get("method") == "agreement_measure":
    st.subheader("Judge")
    cols = st.columns(3)
    provider = cols[0].selectbox(
        "Provider",
        PROVIDERS,
        index=PROVIDERS.index(defn.judge.get("provider"))
        if defn.judge.get("provider") in PROVIDERS
        else 0,
    )
    model = cols[1].text_input("Model / deployment", defn.judge.get("model", ""))
    temperature = cols[2].number_input(
        "Temperature", 0.0, 2.0, float(defn.judge.get("temperature", 0.0)), 0.1
    )
    fields["judge"] = (
        {"provider": provider, "model": model, "temperature": temperature} if model else {}
    )

if defn.kind == "llm_judge":
    examples_text = st.text_area(
        "Few-shot examples (JSON list of objects with the inputs and a score)",
        json.dumps(defn.examples, indent=2, ensure_ascii=False),
        height=140,
    )
    try:
        fields["examples"] = json.loads(examples_text or "[]")
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")

candidate = MetricDefinition(**{**defn.__dict__, **fields})
if defn.kind == "llm_judge":
    candidate.prompt = candidate_prompt
    candidate.prompt_version_id = (
        candidate_prompt.version_id if candidate_prompt is not None else None
    )
problems = validate_definition(candidate)
if problems:
    for problem in problems:
        st.warning(problem)
else:
    st.success("Definition is valid: every placeholder has a mapped input.")

fields["changelog"] = st.text_input("Changelog", row.changelog or "")
if st.button("Save draft", type="primary", disabled=not user.is_editor):
    common.run_write(
        lambda: store.update_draft(
            row.metric_version_id,
            prompt_text=prompt_text,
            prompt_slug=prompt_slug,
            updated_by=user.name,
            **fields,
        ),
        lambda d: f"Saved {d.ref} (1 row).",
    )
