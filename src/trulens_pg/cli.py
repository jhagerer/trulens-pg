"""Command line interface: ``trulens-pg db upgrade | current | downgrade``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from trulens_pg import migrations
from trulens_pg.config import Settings
from trulens_pg.errors import TrulensPgError, ValidationError

if TYPE_CHECKING:
    from sqlalchemy import Engine


def _engine(args: argparse.Namespace) -> Engine:
    from trulens_pg.session import create_store_engine

    url = args.database_url or Settings.from_env().database_url
    if not url:
        raise ValidationError("Pass --database-url or set TRULENS_PG_DATABASE_URL.")
    return create_store_engine(url)


def _schema(args: argparse.Namespace) -> str:
    return str(args.schema or Settings.from_env().schema)


def cmd_db_upgrade(args: argparse.Namespace) -> int:
    engine, schema = _engine(args), _schema(args)
    migrations.upgrade(engine, schema, args.revision)
    print(f"Schema {schema} is at revision {migrations.current_revision(engine, schema)}.")
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


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--database-url", help="default: TRULENS_PG_DATABASE_URL")
    common.add_argument("--schema", help="default: TRULENS_PG_SCHEMA or trulens_pg")

    parser = argparse.ArgumentParser(prog="trulens-pg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    db = sub.add_parser("db", help="metric store schema migrations").add_subparsers(
        dest="db_command", required=True
    )
    p = db.add_parser("upgrade", parents=[common], help="create or upgrade the schema")
    p.add_argument("revision", nargs="?", default="head")
    p.set_defaults(func=cmd_db_upgrade)
    p = db.add_parser("current", parents=[common], help="show the schema revision")
    p.set_defaults(func=cmd_db_current)
    p = db.add_parser("downgrade", parents=[common], help="downgrade (base removes all tables)")
    p.add_argument("revision")
    p.set_defaults(func=cmd_db_downgrade)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except TrulensPgError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
