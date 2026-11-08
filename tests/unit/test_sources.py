from __future__ import annotations

import pandas as pd
import pytest

from trulens_pg.errors import SourceError
from trulens_pg.sources import (
    SourceOptions,
    apply_sampling,
    parse_source_name,
)


def test_parse_plain_table() -> None:
    ref = parse_source_name("questions")
    assert (ref.schema, ref.table) == (None, "questions")


def test_parse_schema_table() -> None:
    ref = parse_source_name("eval.questions")
    assert (ref.schema, ref.table) == ("eval", "questions")
    assert ref.qualified == "eval.questions"


@pytest.mark.parametrize(
    "name",
    [
        "",
        "a.b.c",
        "questions; DROP TABLE x",
        'public."x"',
        "bench:claims_faq@1.0.0",
        "eval.questions?unknown=1",
        "eval.questions?limit=0",
        "eval.questions?sample=1.5",
    ],
)
def test_parse_rejects_invalid(name: str) -> None:
    with pytest.raises(SourceError):
        parse_source_name(name)


def test_options_are_parsed() -> None:
    ref = parse_source_name("eval.questions?split=test&tags=a,b&limit=5")
    assert ref.options == SourceOptions(split="test", tags=("a", "b"), limit=5)


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
