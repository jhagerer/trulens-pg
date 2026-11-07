"""Content hashes for dataset versions and metric versions."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any


def normalize(value: Any) -> Any:
    """Normalise a value for hashing: NFC text without outer whitespace, sorted keys."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value).strip()
    if isinstance(value, Mapping):
        return {str(k): normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        normalize(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )


def hash_records(records: Iterable[Mapping[str, Any]], sort_key: str) -> str:
    """SHA-256 over records sorted by ``sort_key`` after normalisation."""
    lines = sorted(
        (canonical_json(r) for r in records),
        key=lambda line: (json.loads(line).get(sort_key, ""), line),
    )
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_object(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
