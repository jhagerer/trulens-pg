"""Deterministic stand-in for an LLM judge, so the examples run without an API key.

Replace it with a real provider, for example::

    from trulens.providers.openai import OpenAI

    judge = OpenAI(model_engine="gpt-4o-mini")
"""

from __future__ import annotations

import re

CLAUSE = re.compile(r"\b[A-Z]{2,4}-\d{4}\b")


class OfflineJudge:
    model_engine = "offline-keyword-judge"

    def generate_score_and_reasons(
        self,
        system_prompt: str,
        user_prompt: str | None = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> tuple[float, dict[str, str]]:
        text = user_prompt or ""
        answer = text.split("Answer:", 1)[-1]
        found = CLAUSE.findall(answer)
        raw = max_score_val if found else min_score_val
        reason = f"Clause references in the answer: {found or 'none'}"
        return (raw - min_score_val) / (max_score_val - min_score_val), {"reason": reason}
