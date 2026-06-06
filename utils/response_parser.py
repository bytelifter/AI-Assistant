from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable


@dataclass(frozen=True)
class ParsedResponse:
    answer: str
    sources: list[str]


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def parse_model_response(raw_content: str) -> ParsedResponse:
    """Normalize the model response and extract optional sources.

    Supports either plain text or JSON responses with keys like:
    - answer
    - response
    - message
    - sources / citations
    """

    if not raw_content:
        return ParsedResponse(answer="", sources=[])

    content = _strip_code_fences(raw_content)

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return ParsedResponse(answer=normalize_whitespace(content), sources=[])

    if isinstance(payload, dict):
        answer = (
            payload.get("answer")
            or payload.get("response")
            or payload.get("message")
            or payload.get("content")
            or ""
        )
        sources = payload.get("sources") or payload.get("citations") or []
        if isinstance(sources, str):
            sources = [sources]
        if not isinstance(sources, list):
            sources = []
        return ParsedResponse(answer=normalize_whitespace(str(answer)), sources=_unique(map(str, sources)))

    return ParsedResponse(answer=normalize_whitespace(content), sources=[])


def collect_sources(documents) -> list[str]:
    sources: list[str] = []
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("file_path") or metadata.get("path")
        if source:
            sources.append(str(source))
    return _unique(sources)
