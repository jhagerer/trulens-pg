"""Exception types raised by trulens-pg."""

from __future__ import annotations


class TrulensPgError(Exception):
    """Base class for all trulens-pg errors."""


class UnsupportedDatabaseError(TrulensPgError):
    """The database URL uses a dialect or driver that trulens-pg does not support."""


class NotConnectedError(TrulensPgError):
    """A function needs `trulens_pg.connect()` to be called first."""


class SourceError(TrulensPgError):
    """A run source (table or view) cannot be read."""
