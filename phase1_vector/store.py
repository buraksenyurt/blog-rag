"""Qdrant collection yönetimi (Faz 1)."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from common.config import EmbeddingProfile

COLLECTION_PREFIX = "blog_posts"
KEYWORD_FIELDS = ("tags", "categories", "code_langs", "post_slug")
INTEGER_FIELDS = ("year", "chunk_index")
BOOL_FIELDS = ("has_code",)


def collection_name(profile: EmbeddingProfile) -> str:
    return f"{COLLECTION_PREFIX}__{profile.collection_suffix}"


def point_id(chunk_id: str) -> str:
    """Chunk id'sinden deterministik UUID (aynı chunk tekrar eklenince üzerine yazılır)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"blog-rag:{chunk_id}"))


def ensure_collection(
    client: QdrantClient, profile: EmbeddingProfile, recreate: bool = False
) -> str:
    name = collection_name(profile)
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config=models.VectorParams(size=profile.dim, distance=models.Distance.COSINE),
        )
        schemas = (
            [(f, models.PayloadSchemaType.KEYWORD) for f in KEYWORD_FIELDS]
            + [(f, models.PayloadSchemaType.INTEGER) for f in INTEGER_FIELDS]
            + [(f, models.PayloadSchemaType.BOOL) for f in BOOL_FIELDS]
        )
        for field, schema in schemas:
            client.create_payload_index(name, field_name=field, field_schema=schema)
    else:
        size = client.get_collection(name).config.params.vectors.size
        if size != profile.dim:
            raise ValueError(
                f"{name} boyutu {size}, profil {profile.dim} bekliyor. --recreate kullanın."
            )
    return name


def existing_ids(client: QdrantClient, name: str) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            name, limit=1000, offset=offset, with_payload=False, with_vectors=False
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            return ids
