from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import logging
import json

import httpx
from langchain_core.documents import Document

from core.config import AppConfig
from core.exceptions import KnowledgeBaseEmptyError, ModelRequestError, ResponseParseError
from core.vector_store import KnowledgeBaseManager
from utils.response_parser import collect_sources, parse_model_response

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[str]
    raw_response: str | None = None


class KnowledgeService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.vector_store = KnowledgeBaseManager(config).load_index()

    def _trim_history(self, history: Sequence[dict[str, str]] | None) -> list[dict[str, str]]:
        if not history:
            return []
        trimmed = list(history)[-self.config.max_history_messages :]
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in trimmed
            if msg.get("role") in {"user", "assistant"} and msg.get("content")
        ]

    def _format_context(self, documents: Sequence[Document]) -> str:
        parts: list[str] = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata or {}
            source = metadata.get("source", "sorgente sconosciuta")
            snippet = document.page_content.strip()
            parts.append(f"[Fonte {index}] {source}\n{snippet}")
        return "\n\n".join(parts)

    def ask(self, question: str, history: Sequence[dict[str, str]] | None = None) -> ChatResult:
        question = question.strip()
        if not question:
            raise ValueError("La domanda non può essere vuota.")

        logger.info("Nuova domanda: %s", question)
        documents = self.vector_store.similarity_search(question, k=self.config.top_k)
        if not documents:
            raise KnowledgeBaseEmptyError("Nessun contenuto rilevante trovato nel database vettoriale.")

        context = self._format_context(documents)
        messages = [
            {"role": "system", "content": f"{self.config.system_prompt}\n\nContesto:\n{context}"},
            *self._trim_history(history),
            {"role": "user", "content": question},
        ]

        endpoint = f"{self.config.openrouter_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model_name,
            "messages": messages,
        }

        try:
            response = httpx.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise ModelRequestError(f"Timeout nella chiamata al modello: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            details = ""
            try:
                error_payload = exc.response.json()
                details = error_payload.get("error", {}).get("message") or str(error_payload)
            except json.JSONDecodeError:
                details = exc.response.text
            raise ModelRequestError(
                f"Errore HTTP {exc.response.status_code} da OpenRouter: {details}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelRequestError(f"Errore di rete nella chiamata al modello: {exc}") from exc
        except ValueError as exc:
            raise ModelRequestError(f"Risposta JSON non valida da OpenRouter: {exc}") from exc

        choices = data.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        raw_response = message.get("content", "")
        if not raw_response:
            raise ResponseParseError("Risposta vuota dal modello.")

        parsed = parse_model_response(raw_response)
        answer = parsed.answer or raw_response.strip()
        if not answer:
            raise ResponseParseError("Impossibile interpretare la risposta del modello.")

        sources = parsed.sources or collect_sources(documents)
        return ChatResult(answer=answer, sources=sources, raw_response=raw_response)
