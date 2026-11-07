"""Smoke tests of the Streamlit pages with Streamlit's AppTest (no browser)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.conftest import TEST_SCHEMA

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.pg

APP = Path(__file__).parents[2] / "src" / "trulens_pg" / "app"
PAGES = sorted(p.name for p in (APP / "pages").glob("*.py") if p.name != "__init__.py")


@pytest.fixture
def app_env(monkeypatch: pytest.MonkeyPatch, pg_url: str, ctx: Any) -> None:
    monkeypatch.setenv("TRULENS_PG_DATABASE_URL", pg_url)
    monkeypatch.setenv("TRULENS_PG_BENCH_SCHEMA", TEST_SCHEMA)
    monkeypatch.setenv("TRULENS_PG_AUTH", "none")
    monkeypatch.setenv("TRULENS_PG_USER", "tester")
    monkeypatch.delenv("TRULENS_PG_ROLES", raising=False)


def _run(path: Path) -> AppTest:
    at = AppTest.from_file(str(path), default_timeout=60)
    at.run()
    return at


def test_main_renders(app_env: None, published: Any) -> None:
    at = _run(APP / "main.py")
    assert not at.exception
    assert any("tester" in c.value for c in at.sidebar.caption)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_empty(app_env: None, page: str) -> None:
    at = _run(APP / "pages" / page)
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_data(app_env: None, published: Any, store: Any, page: str) -> None:
    store.create_metric("has_policy_number", "python_ref")
    draft = store.create_draft("has_policy_number", python_ref="contains_policy_number")
    store.publish(draft.metric_version_id, published_by="tester")
    store.create_draft("has_policy_number")
    at = _run(APP / "pages" / page)
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]


def test_create_dataset_form(app_env: None, repo: Any) -> None:
    at = _run(APP / "pages" / "datasets.py")
    at.text_input[1].input("new_ds")
    at.button[0].click().run()
    assert not at.exception
    assert any("Created dataset new_ds" in s.value for s in at.success)
    assert repo.get_dataset("new_ds").owner == "tester"


def test_viewer_cannot_publish(
    app_env: None, monkeypatch: pytest.MonkeyPatch, repo: Any, published: Any
) -> None:
    monkeypatch.setenv("TRULENS_PG_ROLES", "someone=publisher; *=editor")
    draft = repo.create_draft("claims_faq")
    repo.submit_for_review(draft.version_id)
    at = _run(APP / "pages" / "publish.py")
    assert not at.exception
    assert any("publisher role" in i.value for i in at.info)
    publish = next(b for b in at.button if b.label == "Publish")
    assert publish.disabled
