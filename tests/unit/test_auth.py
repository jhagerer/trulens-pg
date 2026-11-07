from __future__ import annotations

import pytest

from trulens_pg.app import auth


def test_parse_roles() -> None:
    table = auth.parse_roles("Alice@x.com=publisher; bob=editor|publisher, *=editor")
    assert table == {
        "alice@x.com": frozenset({"publisher"}),
        "bob": frozenset({"editor", "publisher"}),
        "*": frozenset({"editor"}),
    }
    with pytest.raises(ValueError, match="Unknown role"):
        auth.parse_roles("a=admin")
    with pytest.raises(ValueError, match="user=role"):
        auth.parse_roles("alice")


def test_roles_for() -> None:
    spec = "alice=publisher; *=editor"
    assert auth.roles_for("ALICE", spec) == frozenset({"publisher"})
    assert auth.roles_for("carol", spec) == frozenset({"editor"})
    assert auth.roles_for("carol", "alice=publisher") == frozenset()
    assert auth.roles_for("anyone", "") == frozenset(auth.ROLES)


def test_publisher_is_editor() -> None:
    user = auth.User(name="a", roles=frozenset({"publisher"}), auth="none")
    assert user.is_editor
    assert user.is_publisher
    viewer = auth.User(name="b", roles=frozenset(), auth="none")
    assert not viewer.is_editor


def test_user_from_headers() -> None:
    assert auth.user_from_headers({"x-forwarded-email": "a@x.com"}) == "a@x.com"
    assert auth.user_from_headers({"X-Custom": "b"}, "x-custom") == "b"
    assert auth.user_from_headers({}) is None
