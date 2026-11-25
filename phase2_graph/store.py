"""Neo4j şeması ve graph yazma işlemleri (Faz 2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from neo4j import Driver, GraphDatabase

from common.chunk_store import batched
from common.config import EmbeddingProfile, Settings
from phase2_graph.graph_model import GraphData

VECTOR_INDEX = "chunk_embedding"
META_KEY = "blog-rag"
BATCH = 500
CHUNK_BATCH = 200  # her satır 768 boyutlu vektör taşıdığı için daha küçük

CONSTRAINTS = [
    "CREATE CONSTRAINT post_slug IF NOT EXISTS FOR (p:Post) REQUIRE p.slug IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT language_name IF NOT EXISTS FOR (l:Language) REQUIRE l.name IS UNIQUE",
    "CREATE CONSTRAINT series_name IF NOT EXISTS FOR (s:Series) REQUIRE s.name IS UNIQUE",
    "CREATE INDEX post_year IF NOT EXISTS FOR (p:Post) ON (p.year)",
]

Q_POSTS = """
UNWIND $rows AS r
MERGE (p:Post {slug: r.slug})
SET p.title = r.title, p.date = date(r.date), p.year = r.year, p.url = r.url
"""
# Yeniden kurulumda eski etiket/kategori/seri/link ilişkileri kalmasın.
Q_CLEAR_POST_RELS = """
UNWIND $slugs AS s
MATCH (:Post {slug: s})-[r:TAGGED|IN_CATEGORY|IN_SERIES|LINKS_TO]->()
DELETE r
"""
Q_TAGGED = """
UNWIND $rows AS r
MATCH (p:Post {slug: r.slug})
MERGE (t:Tag {name: r.tag})
MERGE (p)-[:TAGGED]->(t)
"""
Q_TAG_VARIANTS = """
UNWIND $rows AS r
MATCH (t:Tag {name: r.tag})
SET t.variants = r.variants
"""
Q_IN_CATEGORY = """
UNWIND $rows AS r
MATCH (p:Post {slug: r.slug})
MERGE (c:Category {name: r.category})
MERGE (p)-[:IN_CATEGORY]->(c)
"""
Q_SERIES = """
UNWIND $rows AS r
MATCH (p:Post {slug: r.slug})
MERGE (s:Series {name: r.series})
MERGE (p)-[rel:IN_SERIES]->(s)
SET rel.position = r.position
"""
Q_LINKS = """
UNWIND $rows AS r
MATCH (a:Post {slug: r.src}), (b:Post {slug: r.dst})
MERGE (a)-[:LINKS_TO]->(b)
"""
Q_CHUNKS = """
UNWIND $rows AS r
MATCH (p:Post {slug: r.post_slug})
MERGE (c:Chunk {id: r.id})
SET c.post_slug = r.post_slug, c.text = r.text, c.section_path = r.section_path,
    c.chunk_index = r.chunk_index, c.n_tokens = r.n_tokens, c.has_code = r.has_code,
    c.code_langs = r.code_langs, c.embedding = r.embedding
MERGE (p)-[:HAS_CHUNK]->(c)
"""
Q_CLEAR_CHUNK_RELS = """
UNWIND $ids AS id
MATCH (:Chunk {id: id})-[r:NEXT|HAS_CODE_IN]->()
DELETE r
"""
Q_CODE_LANGS = """
UNWIND $rows AS r
MATCH (c:Chunk {id: r.chunk_id})
MERGE (l:Language {name: r.lang})
MERGE (c)-[:HAS_CODE_IN]->(l)
"""
Q_NEXT = """
UNWIND $rows AS r
MATCH (a:Chunk {id: r.a}), (b:Chunk {id: r.b})
MERGE (a)-[:NEXT]->(b)
"""
Q_DELETE_STALE_CHUNKS = """
MATCH (c:Chunk) WHERE NOT c.id IN $ids
WITH c LIMIT 10000
DETACH DELETE c
RETURN count(*) AS deleted
"""
Q_DELETE_STALE_POSTS = """
MATCH (p:Post) WHERE NOT p.slug IN $slugs
DETACH DELETE p
RETURN count(*) AS deleted
"""
Q_DELETE_ORPHANS = """
MATCH (n) WHERE (n:Tag OR n:Category OR n:Language OR n:Series) AND NOT (n)--()
DELETE n
RETURN count(*) AS deleted
"""
Q_WIPE = "MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS"


class Runner(Protocol):
    def __call__(self, query: str, **params: Any) -> list[dict]: ...


def connect(settings: Settings) -> Driver:
    driver = GraphDatabase.driver(
        settings.neo4j_url, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    driver.verify_connectivity()
    return driver


def make_runner(driver: Driver, database: str) -> Runner:
    def run(query: str, **params: Any) -> list[dict]:
        records, _, _ = driver.execute_query(query, params, database_=database)
        return [r.data() for r in records]

    return run


def vector_index_query(dim: int, quantization: bool = False) -> str:
    """Vector index tanımı.

    Quantization varsayılan olarak kapalı: embeddinggemma skorları çok dar bir aralıkta
    (ör. 0,73-0,77) toplandığı için sıkıştırmanın getirdiği küçük sapmalar sıralamayı
    değiştirebiliyor. 10 bin chunk için tam hassasiyetin bellek maliyeti önemsiz.
    """
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"geçersiz vektör boyutu: {dim}")
    return (
        f"CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, "
        f"`vector.similarity_function`: 'cosine', "
        f"`vector.quantization.enabled`: {str(quantization).lower()}}}}}"
    )


def vector_index_options(run: Runner) -> dict | None:
    rows = run(
        "SHOW INDEXES YIELD name, options, state WHERE name = $name RETURN options, state",
        name=VECTOR_INDEX,
    )
    if not rows:
        return None
    config = (rows[0]["options"] or {}).get("indexConfig", {})
    return {"state": rows[0]["state"], **config}


def rebuild_vector_index(run: Runner, profile: EmbeddingProfile, quantization: bool) -> None:
    """Veriye dokunmadan vector index'i silip yeniden kurar ve hazır olmasını bekler."""
    run(f"DROP INDEX {VECTOR_INDEX} IF EXISTS")
    run(vector_index_query(profile.dim, quantization))
    run("CALL db.awaitIndexes(1800)")


def check_meta(run: Runner, profile: EmbeddingProfile) -> None:
    """Graph başka bir embedding profiliyle kurulduysa karışık vektörleri önlemek için durdur."""
    rows = run(
        "MATCH (m:GraphMeta {key: $key}) RETURN m.embedding_profile AS profile, m.dim AS dim",
        key=META_KEY,
    )
    if rows and (rows[0]["profile"] != profile.name or rows[0]["dim"] != profile.dim):
        raise ValueError(
            f"Graph '{rows[0]['profile']}' ({rows[0]['dim']}) profiliyle kurulmuş, "
            f"şimdiki profil '{profile.name}' ({profile.dim}). --recreate kullanın."
        )


def ensure_schema(run: Runner, profile: EmbeddingProfile, quantization: bool = False) -> None:
    for query in CONSTRAINTS:
        run(query)
    run(vector_index_query(profile.dim, quantization))


def write_posts(run: Runner, data: GraphData) -> None:
    for rows in batched(data.posts, BATCH):
        run(Q_POSTS, rows=rows)
        run(Q_CLEAR_POST_RELS, slugs=[r["slug"] for r in rows])
    for rows in batched(data.tagged, BATCH):
        run(Q_TAGGED, rows=rows)
    variants = [{"tag": t, "variants": v} for t, v in data.tag_variants.items()]
    for rows in batched(variants, BATCH):
        run(Q_TAG_VARIANTS, rows=rows)
    for rows in batched(data.in_category, BATCH):
        run(Q_IN_CATEGORY, rows=rows)
    for rows in batched(data.series, BATCH):
        run(Q_SERIES, rows=rows)
    for rows in batched(data.links, BATCH):
        run(Q_LINKS, rows=rows)


def write_chunks(
    run: Runner,
    data: GraphData,
    vectors_for: Callable[[list[dict]], list[list[float]]],
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Chunk'ları vektörleriyle yazar. vectors_for bir chunk partisi için vektörleri döner."""
    total = len(data.chunks)
    done = 0
    for rows in batched(data.chunks, CHUNK_BATCH):
        vectors = vectors_for(rows)
        payload = [{**r, "embedding": v} for r, v in zip(rows, vectors, strict=True)]
        run(Q_CHUNKS, rows=payload)
        run(Q_CLEAR_CHUNK_RELS, ids=[r["id"] for r in rows])
        done += len(rows)
        if progress:
            progress(done, total)
    for rows in batched(data.code_langs, BATCH):
        run(Q_CODE_LANGS, rows=rows)
    for rows in batched(data.next_pairs, BATCH):
        run(Q_NEXT, rows=rows)


def delete_stale(run: Runner, data: GraphData) -> dict[str, int]:
    ids = [c["id"] for c in data.chunks]
    chunks = 0
    while True:
        deleted = run(Q_DELETE_STALE_CHUNKS, ids=ids)[0]["deleted"]
        chunks += deleted
        if deleted == 0:
            break
    posts = run(Q_DELETE_STALE_POSTS, slugs=[p["slug"] for p in data.posts])[0]["deleted"]
    orphans = run(Q_DELETE_ORPHANS)[0]["deleted"]
    return {"chunks": chunks, "posts": posts, "orphans": orphans}


def write_meta(run: Runner, profile: EmbeddingProfile, data: GraphData) -> None:
    run(
        """
        MERGE (m:GraphMeta {key: $key})
        SET m.embedding_profile = $profile, m.embedding_model = $model, m.dim = $dim,
            m.posts = $posts, m.chunks = $chunks, m.built_at = datetime()
        """,
        key=META_KEY,
        profile=profile.name,
        model=profile.model,
        dim=profile.dim,
        posts=len(data.posts),
        chunks=len(data.chunks),
    )


def wipe(driver: Driver, database: str) -> None:
    """Graph'ı tamamen siler (--recreate). IN TRANSACTIONS otomatik commit'li oturum ister."""
    with driver.session(database=database) as session:
        session.run(Q_WIPE).consume()
        session.run(f"DROP INDEX {VECTOR_INDEX} IF EXISTS").consume()


# --- Adım 9: kavramlar (LLM çıkarımı, pilot) -------------------------------------------

Q_CLEAR_CONCEPTS = "MATCH (k:Concept) DETACH DELETE k"
Q_CONCEPTS = """
UNWIND $rows AS r
MERGE (k:Concept {name: r.name})
SET k.type = r.type, k.aliases = r.aliases, k.df = r.df, k.mention_terms = r.mention_terms
"""
Q_ABOUT = """
UNWIND $rows AS r
MATCH (p:Post {slug: r.slug}), (k:Concept {name: r.concept})
MERGE (p)-[:ABOUT]->(k)
"""
Q_MENTIONS = """
UNWIND $rows AS r
MATCH (c:Chunk {id: r.chunk_id}), (k:Concept {name: r.concept})
MERGE (c)-[:MENTIONS]->(k)
"""
Q_CONCEPT_RELATIONS = """
UNWIND $rows AS r
MATCH (a:Concept {name: r.source}), (b:Concept {name: r.target})
MERGE (a)-[x:RELATED_TO {kind: r.kind}]->(b)
SET x.count = r.count
"""


def write_concepts(run: Runner, graph) -> None:
    """Kavram katmanını baştan yazar (pilot: her çalıştırmada temiz kurulum)."""
    run("CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (k:Concept) REQUIRE k.name IS UNIQUE")
    run(Q_CLEAR_CONCEPTS)
    for rows in batched(graph.concepts, BATCH):
        run(Q_CONCEPTS, rows=rows)
    for rows in batched(graph.about, BATCH):
        run(Q_ABOUT, rows=rows)
    for rows in batched(graph.mentions, BATCH):
        run(Q_MENTIONS, rows=rows)
    for rows in batched(graph.relations, BATCH):
        run(Q_CONCEPT_RELATIONS, rows=rows)
