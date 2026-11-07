from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import TEST_SCHEMA
from trulens_pg.cli import main

pytestmark = pytest.mark.pg

EXAMPLES = Path(__file__).parents[2] / "examples" / "claims_faq"


def _cli(pg_url: str, *args: str) -> int:
    return main([*args, "--database-url", pg_url, "--schema", TEST_SCHEMA])


def test_db_current(pg_url: str, ctx: Any, capsys: pytest.CaptureFixture[str]) -> None:
    assert _cli(pg_url, "db", "current") == 0
    assert "current=0001" in capsys.readouterr().out
    assert _cli(pg_url, "db", "upgrade") == 0


def test_import_publish_export_example(
    pg_url: str, ctx: Any, repo: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _cli(
            pg_url,
            "bench",
            "import-corpus",
            str(EXAMPLES / "corpus.jsonl"),
            "--corpus",
            "policies",
            "--create",
        )
        == 0
    )
    assert (
        _cli(
            pg_url,
            "bench",
            "import",
            str(EXAMPLES / "items.jsonl"),
            "--dataset",
            "claims_faq",
            "--dry-run",
        )
        == 0
    )
    assert "50 valid item(s)" in capsys.readouterr().out
    assert (
        _cli(
            pg_url,
            "bench",
            "import",
            str(EXAMPLES / "items.jsonl"),
            "--dataset",
            "claims_faq",
            "--create",
            "--corpus",
            "policies",
            "--version",
            "1.0.0",
            "--publish",
            "--changelog",
            "example",
            "--user",
            "ci",
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Published claims_faq@1.0.0 (50 items)" in out

    version = repo.get_version("claims_faq", "1.0.0")
    assert version.published_by == "ci"
    assert repo.missing_context_ids(version.version_id).empty

    target = tmp_path / "export.jsonl"
    assert _cli(pg_url, "bench", "export", "claims_faq@1.0.0", "--out", str(target)) == 0
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(rows) == 49  # one item contains personal data
    assert {"input", "ground_truth_output", "ground_truth_context_ids"} <= set(rows[0])
    assert (
        _cli(pg_url, "bench", "export", "claims_faq@1.0.0", "--out", str(target), "--include-pii")
        == 0
    )
    assert len(target.read_text().splitlines()) == 50

    assert _cli(pg_url, "bench", "list") == 0
    assert "claims_faq" in capsys.readouterr().out


def test_import_reports_invalid_rows(
    pg_url: str, ctx: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("question,split\nok?,test\n,test\nfine?,train\n")
    assert _cli(pg_url, "bench", "import", str(bad), "--dataset", "x", "--create") == 2
    err = capsys.readouterr().err
    assert "2 invalid row(s)" in err


def test_errors_are_reported(pg_url: str, ctx: Any, capsys: pytest.CaptureFixture[str]) -> None:
    assert _cli(pg_url, "bench", "export", "missing@1.0.0") == 1
    assert "does not exist" in capsys.readouterr().err
