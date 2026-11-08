from __future__ import annotations

import pytest

from trulens_pg.config import Settings, validate_schema_name
from trulens_pg.errors import NotConnectedError, UnsupportedDatabaseError, ValidationError
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


def test_schema_name_validation() -> None:
    assert validate_schema_name("bench") == "bench"
    for bad in ("Bench", "bench;drop", "1bench", ""):
        with pytest.raises(ValidationError):
            validate_schema_name(bad)


def test_settings_from_env() -> None:
    settings = Settings.from_env(
        {
            "TRULENS_PG_DATABASE_URL": "postgresql://h/db",
            "TRULENS_PG_SCHEMA": "evals",
            "TRULENS_PG_MIN_CALIBRATION_LABELS": "5",
            "TRULENS_PG_MIN_SPEARMAN": "0.6",
            "TRULENS_PG_JUDGE_PROVIDER": "openai",
            "TRULENS_PG_JUDGE_MODEL": "gpt-4o-mini",
        }
    )
    assert settings.schema == "evals"
    assert settings.min_calibration_labels == 5
    assert settings.min_spearman == 0.6
    assert settings.judge == {"provider": "openai", "model": "gpt-4o-mini"}


def test_get_context_without_connect() -> None:
    set_context(None)
    with pytest.raises(NotConnectedError):
        get_context()
