"""Profil tabanlı embedding istemcisi (LM Studio, OpenAI uyumlu /v1/embeddings)."""

from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from common.config import EmbeddingProfile, Settings, get_settings, load_embedding_profile


class Embedder:
    def __init__(
        self,
        settings: Settings | None = None,
        profile: EmbeddingProfile | None = None,
        batch_size: int = 32,
    ) -> None:
        self.settings = settings or get_settings()
        self.profile = profile or load_embedding_profile(self.settings)
        self.batch_size = batch_size
        self._client = OpenAI(
            base_url=self.settings.lmstudio_url,
            api_key=self.settings.lmstudio_api_key,
            timeout=120,
            max_retries=3,
        )

    def format_document(self, text: str, title: str | None = None) -> str:
        return self.profile.doc_template.format(text=text, title=title or "none")

    def format_query(self, text: str) -> str:
        return self.profile.query_template.format(text=text)

    def embed_documents(
        self, texts: Sequence[str], titles: Sequence[str | None] | None = None
    ) -> list[list[float]]:
        if titles is not None and len(titles) != len(texts):
            raise ValueError("texts ve titles aynı uzunlukta olmalı")
        titles = titles or [None] * len(texts)
        inputs = [self.format_document(t, h) for t, h in zip(texts, titles, strict=True)]
        return self.embed_raw(inputs)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_raw([self.format_query(text)])[0]

    def embed_raw(self, inputs: Sequence[str]) -> list[list[float]]:
        """Şablonu zaten uygulanmış metinleri embed eder."""
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            batch = list(inputs[start : start + self.batch_size])
            response = self._client.embeddings.create(model=self.profile.model, input=batch)
            for item in sorted(response.data, key=lambda d: d.index):
                if len(item.embedding) != self.profile.dim:
                    raise ValueError(
                        f"{self.profile.name}: beklenen boyut {self.profile.dim}, "
                        f"gelen {len(item.embedding)}. embedding_profiles.yaml'ı kontrol edin."
                    )
                vectors.append(item.embedding)
        return vectors
