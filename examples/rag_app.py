"""A small keyword-retrieval RAG app over the example corpus (no LLM, no network)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

CORPUS = Path(__file__).parent / "claims_faq" / "corpus.jsonl"
_WORD = re.compile(r"[a-zäöüß]+")


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3}


class ClaimsRag:
    def __init__(self, top_k: int = 2) -> None:
        self.docs = [json.loads(line) for line in CORPUS.read_text().splitlines() if line]
        self.top_k = top_k

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> list[str]:
        q = _words(query)
        ranked = sorted(
            self.docs, key=lambda d: len(q & _words(d["title"] + " " + d["text"])), reverse=True
        )
        return [d["text"] for d in ranked[: self.top_k]]

    @instrument()
    def answer(self, query: str) -> str:
        contexts = self.retrieve(query)
        if not contexts:
            return "I cannot answer this from the policy terms."
        clause = next(d["chunk_id"] for d in self.docs if d["text"] == contexts[0])
        return f"{contexts[0]} (clause {clause})"
