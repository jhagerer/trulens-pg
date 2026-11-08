"""`PgRunDao`: the TruLens OSS run backend plus Postgres table sources (closes G2)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import Engine
from trulens.core.dao.default_run import DefaultRunDao

from trulens_pg import sources

logger = logging.getLogger(__name__)

TABLE_SOURCE = "TABLE"


class PgRunDao(DefaultRunDao):  # type: ignore[misc]
    """Run backend for Postgres.

    Run metadata stays in the TruLens ``runs`` table (inherited from
    ``DefaultRunDao``). This class adds:

    * `fetch_source_data`, so ``run.start()`` without ``input_df`` reads Postgres
      tables and views;
    * an early column check against ``dataset_spec`` when a ``TABLE`` run is
      created.
    """

    def __init__(self, db: Any, engine: Engine | None = None) -> None:
        super().__init__(db=db)
        self._engine: Engine = engine if engine is not None else db.engine

    @property
    def engine(self) -> Engine:
        return self._engine

    def fetch_source_data(self, source_name: str) -> pd.DataFrame:
        """Read the run source; called by ``Run.start()`` when no ``input_df`` is given."""
        df = sources.read_source(self._engine, sources.parse_source_name(source_name))
        logger.info("Read %d rows from source %s", len(df), source_name)
        return df

    def create_new_run(
        self,
        object_name: str,
        object_type: str,
        run_name: str,
        dataset_name: str,
        source_type: str,
        dataset_spec: dict[str, str],
        object_version: str | None = None,
        description: str | None = "",
        label: str | None = "",
        llm_judge_name: str | None = "",
        mode: Any = None,
    ) -> pd.DataFrame:
        if (source_type or "").upper() == TABLE_SOURCE:
            table = sources.reflect_table(self._engine, sources.parse_source_name(dataset_name))
            sources.check_columns(table, dataset_spec, dataset_name)

        kwargs: dict[str, Any] = {}
        if mode is not None:
            kwargs["mode"] = mode
        result: pd.DataFrame = super().create_new_run(
            object_name=object_name,
            object_type=object_type,
            run_name=run_name,
            dataset_name=dataset_name,
            source_type=source_type,
            dataset_spec=dataset_spec,
            object_version=object_version,
            description=description,
            label=label,
            llm_judge_name=llm_judge_name,
            **kwargs,
        )
        return result
