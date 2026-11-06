"""LM Studio üzerindeki sohbet modeli için ince sarmalayıcı."""

from __future__ import annotations

from openai import OpenAI

from common.config import Settings, get_settings


class LLM:
    def __init__(self, settings: Settings | None = None, model: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.llm_model
        self._client = OpenAI(
            base_url=self.settings.lmstudio_url,
            api_key=self.settings.lmstudio_api_key,
            timeout=300,
            max_retries=2,
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
