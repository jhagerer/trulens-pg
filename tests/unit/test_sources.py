from __future__ import annotations

import pandas as pd
import pytest

from trulens_pg.errors import SourceError
from trulens_pg.sources import (
    SourceOptions,
    apply_sampling,
    bench_source,
    parse_source_name,
    view_name,
)


def test_parse_plain_table() -> None:
    ref = parse_source_name("questions")
    assert (ref.schema, ref.table, ref.is_benchmark) == (None, "questions", False)


def test_parse_schema_table() -> None:
    ref = parse_source_name("bench_eval.questions")
    assert (ref.schema, ref.table) == ("bench_eval", "questions")
    assert ref.qualified == "bench_eval.questions"


def test_parse_benchmark() -> None:
    ref = parse_source_name("bench:claims_faq@1.2.0", bench_schema="evals")
    assert ref.is_benchmark
    assert (ref.schema, ref.table, ref.dataset, ref.version) == (
        "evals",
        "v_claims_faq__1_2_0",
        "claims_faq",
        "1.2.0",
    )


@pytest.mark.parametrize(
    "name",
    [
        "",
        "a.b.c",
        "questions; DROP TABLE x",
        'public."x"',
        "bench:claims_faq",
        "bench:claims_faq@1.2",
        "bench:Claims@1.0.0",
        "bench:claims_faq@1.0.0?unknown=1",
        "bench:claims_faq@1.0.0?limit=0",
        "bench:claims_faq@1.0.0?sample=1.5",
    ],
)
def test_parse_rejects_invalid(name: str) -> None:
    with pytest.raises(SourceError):
        parse_source_name(name)


def test_options_round_trip() -> None:
    src = bench_source("claims_faq", "1.0.0", split="test", tags=["a", "b"], limit=5)
    assert src == "bench:claims_faq@1.0.0?split=test&tags=a,b&limit=5"
    ref = parse_source_name(src)
    assert ref.options == SourceOptions(split="test", tags=("a", "b"), limit=5)


def test_view_name() -> None:
    assert view_name("claims_faq", "10.0.3") == "v_claims_faq__10_0_3"


def test_sampling_is_deterministic_and_seeded() -> None:
    df = pd.DataFrame({"external_id": [f"q{i}" for i in range(1000)]})
    opts = SourceOptions(sample=0.2, seed="s1")
    first = apply_sampling(df, opts)
    assert first.equals(apply_sampling(df, opts))
    assert 150 < len(first) < 250
    other = apply_sampling(df, SourceOptions(sample=0.2, seed="s2"))
    assert set(first["external_id"]) != set(other["external_id"])
    # Stable per item: a subset of rows keeps the same decisions.
    subset = apply_sampling(df.iloc[:500], opts)
    assert set(subset["external_id"]) == {x for x in first["external_id"] if int(x[1:]) < 500}


def test_sampling_needs_id_column() -> None:
    with pytest.raises(SourceError):
        apply_sampling(pd.DataFrame({"x": [1]}), SourceOptions(sample=0.5))
