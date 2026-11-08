"""Exception types raised by trulens-pg."""

from __future__ import annotations


class TrulensPgError(Exception):
    """Base class for all trulens-pg errors."""


class UnsupportedDatabaseError(TrulensPgError):
    """The database URL uses a dialect or driver that trulens-pg does not support."""


class SchemaNotReadyError(TrulensPgError):
    """The trulens_pg schema is missing or not at the latest Alembic revision."""


class NotConnectedError(TrulensPgError):
    """A function needs `trulens_pg.connect()` to be called first."""


class SourceError(TrulensPgError):
    """A run source (table or view) cannot be read."""


class NotFoundError(TrulensPgError):
    """A dataset, version, metric or metric set does not exist."""


class ReadOnlyError(TrulensPgError):
    """A published or deprecated version cannot be changed."""


class WorkflowError(TrulensPgError):
    """A lifecycle step is not allowed in the current state (for example publish)."""


class ValidationError(TrulensPgError):
    """Input data or a metric definition is invalid."""


class MetricResolutionError(TrulensPgError):
    """A metric reference cannot be resolved to a TruLens Metric."""
