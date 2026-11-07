"""Example python_ref metrics, registered as entry points in ``trulens_pg.metrics``.

A python_ref metric is a plain function in an installed, reviewed package. Its
parameter names map to selectors: ``question``, ``answer``, ``contexts``,
``ground_truth``. It returns a score between 0 and 1, optionally with a dict
of reasons.
"""

from __future__ import annotations

import re

POLICY_NUMBER = re.compile(r"\b[A-Z]{2,4}-\d{4,10}\b")


def contains_policy_number(answer: str) -> tuple[float, dict[str, str]]:
    """1.0 if the answer cites a policy or clause number such as ``KFZ-2024001``."""
    found = POLICY_NUMBER.findall(answer or "")
    return (1.0 if found else 0.0), {"reason": f"Found: {', '.join(found) or 'none'}"}


def answer_length_ok(answer: str, max_words: int = 150) -> float:
    """1.0 if the answer has between 1 and ``max_words`` words."""
    words = len((answer or "").split())
    return 1.0 if 0 < words <= max_words else 0.0
