from __future__ import annotations

import io

import pandas as pd
import pytest

from trulens_pg import hashing
from trulens_pg.bench import io as bench_io
from trulens_pg.bench.repository import next_version, validate_items
from trulens_pg.bench.schemas import ItemIn, default_external_id
from trulens_pg.errors import ValidationError


def test_item_parsing_of_lists_and_flags() -> None:
    item = ItemIn.model_validate(
        {
            "query": "  What is covered?  ",
            "ground_truth_context_ids": "c1; c2",
            "tags": '["motor", "glass"]',
            "split": "TEST",
            "contains_pii": "yes",
            "metadata": '{"source": "faq"}',
        }
    )
    assert item.query == "What is covered?"
    assert item.ground_truth_context_ids == ["c1", "c2"]
    assert item.tags == ["motor", "glass"]
    assert item.split == "test"
    assert item.contains_pii is True
    assert item.metadata == {"source": "faq"}
    assert item.external_id == default_external_id("What is covered?")


def test_item_rejects_bad_split_and_empty_query() -> None:
    with pytest.raises(ValueError, match="split"):
        ItemIn.model_validate({"query": "q", "split": "train"})
    with pytest.raises(ValueError, match="query"):
        ItemIn.model_validate({"query": "   "})


def test_validate_items_reports_rows_and_duplicates() -> None:
    report = validate_items(
        [
            {"external_id": "a", "query": "q1"},
            {"external_id": "a", "query": "q2"},
            {"external_id": "b", "query": ""},
        ]
    )
    assert not report.ok
    assert report.duplicate_ids == ["a"]
    assert [e["row"] for e in report.errors] == [2]


def test_next_version() -> None:
    assert next_version(None) == "1.0.0"
    assert next_version("1.2.3") == "1.3.0"
    assert next_version("1.2.3", "patch") == "1.2.4"
    assert next_version("1.2.3", "major") == "2.0.0"


def test_hash_is_order_and_whitespace_independent() -> None:
    a = [{"external_id": "1", "query": "x "}, {"external_id": "2", "query": "y"}]
    b = [{"external_id": "2", "query": "y"}, {"external_id": "1", "query": "x"}]
    assert hashing.hash_records(a, "external_id") == hashing.hash_records(b, "external_id")
    c = [{"external_id": "1", "query": "x!"}, {"external_id": "2", "query": "y"}]
    assert hashing.hash_records(a, "external_id") != hashing.hash_records(c, "external_id")


def test_guess_and_apply_mapping_moves_extra_columns_to_metadata() -> None:
    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "question": ["q1", "q2"],
            "answer": ["a1", "a2"],
            "doc_ids": ["c1;c2", ""],
            "source": ["faq", ""],
        }
    )
    mapping = bench_io.guess_mapping(list(df.columns))
    assert mapping == {
        "external_id": "id",
        "query": "question",
        "ground_truth_answer": "answer",
        "ground_truth_context_ids": "doc_ids",
    }
    report = bench_io.validate_frame(df, mapping)
    assert report.ok
    assert report.items[0].metadata == {"source": "faq"}
    assert report.items[1].metadata == {}
    assert report.items[0].ground_truth_context_ids == ["c1", "c2"]


def test_apply_mapping_errors() -> None:
    df = pd.DataFrame({"question": ["q"]})
    with pytest.raises(ValidationError, match="query"):
        bench_io.apply_mapping(df, {"ground_truth_answer": "question"})
    with pytest.raises(ValidationError, match="not in the file"):
        bench_io.apply_mapping(df, {"query": "missing"})


@pytest.mark.parametrize("fmt", ["csv", "jsonl", "parquet"])
def test_export_formats_round_trip(fmt: bench_io.FileFormat) -> None:
    df = pd.DataFrame(
        {
            "external_id": ["a", "b"],
            "input": ["q1", "q2"],
            "tags": [["x"], []],
        }
    )
    data = bench_io.to_bytes(df, fmt)
    back = bench_io.read_table(io.BytesIO(data), fmt)
    assert list(back["external_id"]) == ["a", "b"]
    assert list(back["input"]) == ["q1", "q2"]


def test_detect_format() -> None:
    assert bench_io.detect_format("x.CSV") == "csv"
    assert bench_io.detect_format("x.ndjson") == "jsonl"
    assert bench_io.detect_format("x.parquet") == "parquet"
    with pytest.raises(ValidationError):
        bench_io.detect_format("x.xlsx")


def test_parquet_export_with_empty_metadata() -> None:
    df = pd.DataFrame({"external_id": ["a", "b"], "metadata": [{}, {}], "tags": [["x"], []]})
    back = bench_io.read_table(io.BytesIO(bench_io.to_bytes(df, "parquet")), "parquet")
    assert list(back["metadata"]) == ["{}", "{}"]
    assert list(back["tags"][0]) == ["x"]
