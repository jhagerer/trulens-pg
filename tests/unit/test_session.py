from __future__ import annotations

import pytest

from trulens_pg.errors import NotConnectedError, UnsupportedDatabaseError
from trulens_pg.session import get_context, normalize_database_url, set_context


def test_default_driver_is_psycopg3() -> None:
    url = normalize_database_url("postgresql://u:p@h/db")
    assert url.drivername == "postgresql+psycopg"


@pytest.mark.parametrize("driver", ["psycopg", "psycopg2"])
def test_sync_drivers_are_accepted(driver: str) -> None:
    assert normalize_database_url(f"postgresql+{driver}://h/db").drivername.endswith(driver)


def test_asyncpg_is_rejected() -> None:
    with pytest.raises(UnsupportedDatabaseError, match="synchronous"):
        normalize_database_url("postgresql+asyncpg://h/db")


def test_other_dialects_are_rejected() -> None:
    with pytest.raises(UnsupportedDatabaseError):
        normalize_database_url("sqlite:///x.db")


def test_get_context_without_connect() -> None:
    set_context(None)
    with pytest.raises(NotConnectedError):
        get_context()


def test_connect_needs_a_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from trulens_pg import connect

    monkeypatch.delenv("TRULENS_PG_DATABASE_URL", raising=False)
    with pytest.raises(NotConnectedError, match="TRULENS_PG_DATABASE_URL"):
        connect()
