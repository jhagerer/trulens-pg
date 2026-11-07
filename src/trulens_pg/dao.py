"""`PgRunDao`: the TruLens OSS run backend plus table sources (closes G2)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from trulens.core.dao.default_run import DefaultRunDao

from trulens_pg import sources
from trulens_pg.config import DEFAULT_BENCH_SCHEMA, validate_schema_name

logger = logging.getLogger(__name__)

TABLE_SOURCE = "TABLE"


class PgRunDao(DefaultRunDao):  # type: ignore[misc]
    """Run backend for Postgres.

    Run metadata stays in the TruLens ``runs`` table (inherited from
    ``DefaultRunDao``). This class adds:

    * `fetch_source_data`, so ``run.start()`` without ``input_df`` reads
      tables, views and published benchmark versions;
    * an early column check against ``dataset_spec`` when a ``TABLE`` run is
      created;
    * a ``bench.run_link`` row that connects a run to the benchmark version it
      uses.
    """

    def __init__(
        self, db: Any, engine: Engine | None = None, bench_schema: str = DEFAULT_BENCH_SCHEMA
    ) -> None:
        super().__init__(db=db)
        self._engine: Engine = engine if engine is not None else db.engine
        self._bench_schema = validate_schema_name(bench_schema)

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def bench_schema(self) -> str:
        return self._bench_schema

    def parse_source(self, source_name: str) -> sources.SourceRef:
        return sources.parse_source_name(source_name, bench_schema=self._bench_schema)

    def fetch_source_data(self, source_name: str) -> pd.DataFrame:
        """Read the run source; called by ``Run.start()`` when no ``input_df`` is given."""
        ref = self.parse_source(source_name)
        df = sources.read_source(self._engine, ref)
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
        ref: sources.SourceRef | None = None
        if (source_type or "").upper() == TABLE_SOURCE:
            ref = self.parse_source(dataset_name)
            table = sources.reflect_table(self._engine, ref)
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
        if ref is not None and ref.is_benchmark:
            self._link_run(ref, run_name, object_name, object_version, llm_judge_name or None)
        return result

    def _link_run(
        self,
        ref: sources.SourceRef,
        run_name: str,
        app_name: str,
        app_version: str | None,
        judge_model: str | None,
    ) -> None:
        from trulens_pg.bench import models

        engine = self._engine.execution_options(
            schema_translate_map={models.SCHEMA_PLACEHOLDER: self._bench_schema}
        )
        try:
            with engine.begin() as conn:
                version_id = conn.execute(
                    select(models.DatasetVersion.version_id)
                    .join(models.Dataset)
                    .where(
                        models.Dataset.name == ref.dataset,
                        models.DatasetVersion.version == ref.version,
                    )
                ).scalar_one_or_none()
                if version_id is None:
                    return
                conn.execute(
                    insert(models.RunLink)
                    .values(
                        run_name=run_name,
                        app_name=app_name,
                        app_version=app_version,
                        version_id=version_id,
                        platform="postgres",
                        judge_model=judge_model,
                    )
                    .on_conflict_do_nothing(constraint="uq_run_link")
                )
        except SQLAlchemyError:
            logger.warning("Could not record bench.run_link for run %s", run_name, exc_info=True)
