"""Command line interface: ``trulens-pg db | bench``."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from trulens_pg import migrations
from trulens_pg.config import Settings
from trulens_pg.errors import TrulensPgError, ValidationError

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from trulens_pg.bench import Repository


def _user(args: argparse.Namespace) -> str:
    return str(args.user or os.environ.get("TRULENS_PG_USER") or getpass.getuser())


def _engine(args: argparse.Namespace) -> Engine:
    from trulens_pg.session import create_bench_engine

    url = args.database_url or Settings.from_env().database_url
    if not url:
        raise ValidationError("Pass --database-url or set TRULENS_PG_DATABASE_URL.")
    return create_bench_engine(url)


def _schema(args: argparse.Namespace) -> str:
    return str(args.schema or Settings.from_env().bench_schema)


def _repo(args: argparse.Namespace) -> Repository:
    from trulens_pg.bench import Repository

    return Repository(_engine(args), _schema(args))


def _parse_mapping(pairs: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pair in pairs:
        field, sep, column = pair.partition("=")
        if not sep or not field or not column:
            raise ValidationError(f"Invalid --map {pair!r}; use field=column, e.g. query=question.")
        mapping[field.strip()] = column.strip()
    return mapping


# ------------------------------------------------------------------------- db


def cmd_db_upgrade(args: argparse.Namespace) -> int:
    engine, schema = _engine(args), _schema(args)
    migrations.upgrade(engine, schema, args.revision)
    from trulens_pg.bench import Repository

    views = Repository(engine, schema).refresh_views()
    print(f"Schema {schema} is at revision {migrations.current_revision(engine, schema)}.")
    print(f"Refreshed {len(views)} benchmark view(s).")
    return 0


def cmd_db_current(args: argparse.Namespace) -> int:
    engine, schema = _engine(args), _schema(args)
    current = migrations.current_revision(engine, schema)
    head = migrations.head_revision()
    print(f"schema={schema} current={current} head={head}")
    return 0 if current == head else 1


def cmd_db_downgrade(args: argparse.Namespace) -> int:
    migrations.downgrade(_engine(args), _schema(args), args.revision)
    print(f"Downgraded {_schema(args)} to {args.revision}.")
    return 0


# ---------------------------------------------------------------------- bench


def cmd_bench_list(args: argparse.Namespace) -> int:
    repo = _repo(args)
    frame = repo.list_versions(args.dataset) if args.dataset else repo.list_datasets()
    print(frame.to_string(index=False) if not frame.empty else "(none)")
    return 0


def cmd_bench_import(args: argparse.Namespace) -> int:
    from trulens_pg.bench import io as bench_io

    repo = _repo(args)
    if args.hf:
        df = bench_io.load_huggingface(args.hf, split=args.hf_split, name=args.hf_name)
    else:
        df = bench_io.read_table(Path(args.file), args.format)
    mapping = _parse_mapping(args.map) or bench_io.guess_mapping(list(df.columns))
    report = bench_io.validate_frame(df, mapping)
    print(f"Read {len(df)} rows; mapping: {json.dumps(mapping)}")
    if not report.ok:
        for error in report.errors[:20]:
            print(f"  row {error['row']}: {error['error']}", file=sys.stderr)
        if report.duplicate_ids:
            print(f"  duplicate external_id: {report.duplicate_ids[:20]}", file=sys.stderr)
        print(
            f"{len(report.errors)} invalid row(s), {len(report.duplicate_ids)} duplicate(s).",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        print(f"{len(report.items)} valid item(s); nothing written (--dry-run).")
        return 0

    from trulens_pg.errors import NotFoundError

    try:
        repo.get_dataset(args.dataset)
    except NotFoundError:
        if not args.create:
            raise
        repo.create_dataset(args.dataset, domain=args.domain, owner=_user(args))
    draft = repo.create_draft(
        args.dataset,
        args.version,
        copy_from="latest" if args.append else None,
        created_by=_user(args),
        changelog=args.changelog,
    )
    if args.corpus:
        repo.set_corpus(draft.version_id, args.corpus)
    n = repo.add_items(draft.version_id, report.items, upsert=args.append)
    print(f"Wrote {n} item(s) to draft {args.dataset}@{draft.version}.")
    if args.publish:
        result = repo.publish_version(
            draft.version_id,
            published_by=_user(args),
            changelog=args.changelog,
            require_review=False,
        )
        print(
            f"Published {result.dataset}@{result.version} "
            f"({result.item_count} items) as {result.view}."
        )
        print(f"Wrote the TruLens dataset {result.trulens_dataset}.")
        if result.duplicate_of:
            print(f"Warning: same content as version(s) {result.duplicate_of}.")
    return 0


def cmd_bench_import_corpus(args: argparse.Namespace) -> int:
    from trulens_pg.bench import io as bench_io

    repo = _repo(args)
    df = bench_io.read_table(Path(args.file), args.format)
    if args.create:
        repo.create_corpus(args.corpus, source_uri=args.source_uri or str(args.file))
    mapping = _parse_mapping(args.map)
    if mapping:
        df = df.rename(columns={col: field for field, col in mapping.items()})
    keep = [c for c in ("chunk_id", "text", "title", "source_uri", "metadata") if c in df.columns]
    records = [{str(k): v for k, v in r.items()} for r in df[keep].to_dict(orient="records")]
    n = repo.add_documents(args.corpus, records)
    print(f"Wrote {n} document(s) to corpus {args.corpus}.")
    return 0


def cmd_bench_sync_trulens(args: argparse.Namespace) -> int:
    repo = _repo(args)
    names = []
    if args.source:
        names.append(repo.sync_trulens_dataset(repo.resolve_source(args.source).version_id))
    else:
        datasets = repo.list_datasets()
        for dataset in [] if datasets.empty else datasets["name"]:
            for row in repo.list_versions(str(dataset)).itertuples():
                if row.status in ("published", "deprecated"):
                    names.append(repo.sync_trulens_dataset(str(row.version_id)))
    print(f"Wrote {len(names)} TruLens dataset(s): {', '.join(names) or '-'}.")
    return 0


def cmd_bench_export(args: argparse.Namespace) -> int:
    from trulens_pg.bench import io as bench_io

    repo = _repo(args)
    version = repo.resolve_source(args.source)
    if version.status not in ("published", "deprecated") and not args.allow_draft:
        raise ValidationError(
            f"Version {args.source} is {version.status}; export published versions."
        )
    frame = repo.version_frame(version.version_id, include_pii=args.include_pii)
    if args.target == "snowflake":
        if not args.snowflake_table:
            raise ValidationError("--target snowflake needs --snowflake-table.")
        params = {"connection_name": args.snowflake_connection} if args.snowflake_connection else {}
        n = bench_io.export_snowflake(frame, args.snowflake_table, connection_params=params)
        print(f"Exported {n} item(s) to Snowflake table {args.snowflake_table}.")
        return 0
    out = Path(args.out or f"{version.dataset.name}-{version.version}.{args.target}")
    bench_io.write_file(frame, out, args.target)
    print(f"Exported {len(frame)} item(s) to {out}.")
    return 0


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--database-url", help="default: TRULENS_PG_DATABASE_URL")
    common.add_argument("--schema", help="bench schema; default: TRULENS_PG_BENCH_SCHEMA or bench")
    common.add_argument("--user", help="user name for created_by/published_by")

    parser = argparse.ArgumentParser(prog="trulens-pg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    db = sub.add_parser("db", help="bench schema migrations").add_subparsers(
        dest="db_command", required=True
    )
    p = db.add_parser("upgrade", parents=[common], help="create or upgrade the bench schema")
    p.add_argument("revision", nargs="?", default="head")
    p.set_defaults(func=cmd_db_upgrade)
    p = db.add_parser("current", parents=[common], help="show the schema revision")
    p.set_defaults(func=cmd_db_current)
    p = db.add_parser("downgrade", parents=[common], help="downgrade (base removes all tables)")
    p.add_argument("revision")
    p.set_defaults(func=cmd_db_downgrade)

    bench = sub.add_parser("bench", help="benchmark datasets").add_subparsers(
        dest="bench_command", required=True
    )
    p = bench.add_parser("list", parents=[common], help="list datasets or versions")
    p.add_argument("dataset", nargs="?")
    p.set_defaults(func=cmd_bench_list)

    p = bench.add_parser("import", parents=[common], help="import items into a new draft version")
    p.add_argument("file", nargs="?", help="CSV, JSONL or Parquet file")
    p.add_argument("--dataset", required=True)
    p.add_argument("--version", help="new version (default: next minor version)")
    p.add_argument("--format", choices=["csv", "jsonl", "parquet"])
    p.add_argument("--map", action="append", default=[], help="field=column, repeatable")
    p.add_argument("--hf", help="Hugging Face dataset path instead of a file (extra [hf])")
    p.add_argument("--hf-split", default="test")
    p.add_argument("--hf-name")
    p.add_argument("--create", action="store_true", help="create the dataset if missing")
    p.add_argument("--domain")
    p.add_argument("--corpus", help="corpus of the version")
    p.add_argument("--append", action="store_true", help="copy the latest version, upsert rows")
    p.add_argument("--changelog")
    p.add_argument("--publish", action="store_true", help="publish without review (CI use)")
    p.add_argument("--dry-run", action="store_true", help="validate only")
    p.set_defaults(func=cmd_bench_import)

    p = bench.add_parser("import-corpus", parents=[common], help="import corpus documents")
    p.add_argument("file")
    p.add_argument("--corpus", required=True)
    p.add_argument("--create", action="store_true")
    p.add_argument("--source-uri")
    p.add_argument("--format", choices=["csv", "jsonl", "parquet"])
    p.add_argument("--map", action="append", default=[], help="field=column, repeatable")
    p.set_defaults(func=cmd_bench_import_corpus)

    p = bench.add_parser(
        "sync-trulens",
        parents=[common],
        help="write published versions as TruLens datasets <dataset>@<version>",
    )
    p.add_argument("source", nargs="?", help="<dataset>@<version>; default: all published")
    p.set_defaults(func=cmd_bench_sync_trulens)

    p = bench.add_parser("export", parents=[common], help="export a version")
    p.add_argument("source", help="<dataset>@<version> or bench:<dataset>@<version>")
    p.add_argument("--target", choices=["jsonl", "csv", "parquet", "snowflake"], default="jsonl")
    p.add_argument("--out", help="output file")
    p.add_argument("--include-pii", action="store_true")
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--snowflake-table", help="DATABASE.SCHEMA.TABLE")
    p.add_argument(
        "--snowflake-connection", help="connection name from ~/.snowflake/connections.toml"
    )
    p.set_defaults(func=cmd_bench_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "bench_command", None) == "import" and not (args.file or args.hf):
        print("bench import needs a file or --hf.", file=sys.stderr)
        return 2
    try:
        return int(args.func(args))
    except TrulensPgError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
