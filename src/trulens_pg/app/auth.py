"""User and role resolution for the Streamlit app (no own password store).

``TRULENS_PG_AUTH`` selects the login:

* ``oidc`` (default): Streamlit's ``st.login`` with an OIDC provider configured in
  ``.streamlit/secrets.toml`` (section ``[auth]``), Streamlit 1.42 or later;
* ``proxy``: an authenticating reverse proxy passes the user in a header
  (``TRULENS_PG_AUTH_HEADER``, default: ``X-Forwarded-Email``, then
  ``X-Forwarded-User``, ``X-Auth-Request-Email``, ``X-Auth-Request-User``);
* ``none``: local development only; the user is ``TRULENS_PG_USER`` or the OS user.

``TRULENS_PG_ROLES`` assigns roles, for example
``alice@example.com=publisher; bob@example.com=editor; *=editor``. A publisher
is also an editor. Without the variable, every logged-in user has both roles.
"""

from __future__ import annotations

import getpass
import os
from collections.abc import Mapping
from dataclasses import dataclass

EDITOR = "editor"
PUBLISHER = "publisher"
ROLES = (EDITOR, PUBLISHER)
PROXY_HEADERS = (
    "X-Forwarded-Email",
    "X-Forwarded-User",
    "X-Auth-Request-Email",
    "X-Auth-Request-User",
)


@dataclass(frozen=True)
class User:
    name: str
    roles: frozenset[str]
    auth: str

    @property
    def is_editor(self) -> bool:
        return EDITOR in self.roles or PUBLISHER in self.roles

    @property
    def is_publisher(self) -> bool:
        return PUBLISHER in self.roles


def parse_roles(spec: str) -> dict[str, frozenset[str]]:
    """Parse ``user=role|role; user2=role`` (``,`` also separates entries)."""
    result: dict[str, frozenset[str]] = {}
    for raw in spec.replace(",", ";").split(";"):
        entry = raw.strip()
        if not entry:
            continue
        user, sep, roles = entry.partition("=")
        if not sep:
            raise ValueError(f"Invalid TRULENS_PG_ROLES entry {entry!r}; use user=role.")
        parsed = frozenset(r.strip().lower() for r in roles.split("|") if r.strip())
        unknown = parsed - set(ROLES)
        if unknown:
            raise ValueError(f"Unknown role(s) {sorted(unknown)} in TRULENS_PG_ROLES.")
        result[user.strip().lower()] = parsed
    return result


def roles_for(user: str, spec: str) -> frozenset[str]:
    if not spec.strip():
        return frozenset(ROLES)
    table = parse_roles(spec)
    return table.get(user.lower(), table.get("*", frozenset()))


def user_from_headers(headers: Mapping[str, str], header: str | None = None) -> str | None:
    lowered = {k.lower(): v for k, v in headers.items()}
    for name in (header,) if header else PROXY_HEADERS:
        value = lowered.get(name.lower())
        if value:
            return value.strip()
    return None


def local_user() -> str:
    return os.environ.get("TRULENS_PG_USER") or getpass.getuser()
