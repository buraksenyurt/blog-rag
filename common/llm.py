"""LM Studio üzerindeki sohbet modeli için ince sarmalayıcı."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from openai import OpenAI

from common.config import Settings, get_settings


class LLM:
    def __init__(self, settings: Settings | None = None, model: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.llm_model
        # Thinking modelleri düşünme metnini ayrı bir alanda (reasoning_content) akıtır.
        # Cevap boş kalırsa sebebini raporlayabilmek için uzunluğu tutulur.
        self.last_reasoning_chars = 0
        self._client = OpenAI(
            base_url=self.settings.lmstudio_url,
            api_key=self.settings.lmstudio_api_key,
            timeout=self.settings.llm_timeout,
            # Uzun bir üretimi zaman aşımında baştan tekrarlamak sadece süreyi katlar.
            max_retries=0,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict,
        name: str = "result",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict:
        """JSON şemasına uyan cevap ister (LM Studio structured output) ve ayrıştırır."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        )
        return parse_json_answer(response.choices[0].message.content or "")

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Cevabı parça parça döner (konsolda canlı göstermek için)."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        self.last_reasoning_chars = 0
        for event in response:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None) or getattr(
                delta, "reasoning", None
            )
            if isinstance(reasoning, str):
                self.last_reasoning_chars += len(reasoning)
            if delta.content:
                yield delta.content


JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_answer(text: str) -> dict:
    """Model çıktısındaki JSON nesnesini ayrıştırır (```json çitlerini ve önsözü tolere eder)."""
    cleaned = JSON_FENCE_RE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"JSON bulunamadı: {text[:120]!r}")
    return json.loads(cleaned[start : end + 1])
