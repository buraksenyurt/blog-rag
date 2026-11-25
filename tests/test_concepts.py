from common.llm import parse_json_answer
from phase2_graph.concepts import (
    PostConcepts,
    build_concept_graph,
    build_post_excerpt,
    canonical_names,
    clean_post_concepts,
    mention_pattern,
    normalize_name,
    shorten_code,
)


def test_parse_json_answer_tolerates_fences() -> None:
    assert parse_json_answer('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_answer('İşte sonuç: {"a": [1, 2]} umarım işe yarar') == {"a": [1, 2]}


def test_shorten_code_and_excerpt_budget() -> None:
    code = "```rust\n" + "\n".join(f"let x{i} = {i};" for i in range(20)) + "\n```"
    short = shorten_code(code, max_lines=3)
    assert "let x2 = 2;" in short and "let x3" not in short and "// ..." in short
    chunks = [
        {"chunk_index": 1, "title": "T", "tags": ["rust"], "section_path": "B", "text": "ikinci"},
        {"chunk_index": 0, "title": "T", "tags": ["rust"], "section_path": "A", "text": "birinci"},
    ]
    excerpt = build_post_excerpt(chunks, budget_chars=1000)
    assert excerpt.startswith("Başlık: T\nBölümler: A; B")  # tag'ler bilerek verilmez
    assert excerpt.index("birinci") < excerpt.index("ikinci")
    assert len(build_post_excerpt(chunks, budget_chars=20)) == 20


def test_clean_post_concepts_normalizes_and_filters() -> None:
    raw = {
        "concepts": [
            {"name": " `Ownership` ", "type": "concept", "aliases": ["Sahiplik", "ownership"]},
            {"name": "ownership", "type": "concept", "aliases": ["sahiplenme"]},
            {"name": "Kod", "type": "concept", "aliases": []},  # genel kelime
            {"name": "tokio", "type": "weird", "aliases": []},  # geçersiz tür -> concept
        ],
        "relations": [
            {"source": "Ownership", "target": "tokio", "kind": "uses"},
            {"source": "ownership", "target": "yok", "kind": "uses"},  # bilinmeyen hedef
        ],
    }
    post = clean_post_concepts("p", raw)
    assert [c["name"] for c in post.concepts] == ["ownership", "tokio"]
    assert post.concepts[0]["aliases"] == ["sahiplenme", "sahiplik"]
    assert post.concepts[1]["type"] == "concept"
    assert post.relations == [{"source": "ownership", "target": "tokio", "kind": "uses"}]
    assert normalize_name("Trait_Object.") == "trait object"


def test_canonical_names_merges_aliases_and_plurals() -> None:
    posts = [
        PostConcepts("a", [{"name": "ownership", "type": "concept", "aliases": ["sahiplik"]}]),
        PostConcepts("b", [{"name": "sahiplik", "type": "concept", "aliases": []}]),
        PostConcepts("c", [{"name": "ownership", "type": "concept", "aliases": []}]),
        PostConcepts("d", [{"name": "trait objects", "type": "concept", "aliases": []}]),
        PostConcepts("e", [{"name": "trait object", "type": "concept", "aliases": []}]),
    ]
    canon = canonical_names(posts)
    assert canon["sahiplik"] == "ownership" and canon["ownership"] == "ownership"
    assert canon["trait objects"] == canon["trait object"]


def test_mention_pattern_matches_word_boundaries_and_aliases() -> None:
    pattern = mention_pattern(["trait object", "dyn"])
    assert pattern.search("Burada Trait-Object kullandık")
    assert not pattern.search("dynamic dispatch")  # 'dyn' kelime sınırı
    assert mention_pattern(["ab"]) is None  # 3 karakterden kısa


def test_build_concept_graph() -> None:
    posts = [
        PostConcepts(
            "p1",
            [
                {"name": "shadowing", "type": "concept", "aliases": ["gölgeleme"]},
                {"name": "mutability", "type": "concept", "aliases": []},
            ],
            [{"source": "shadowing", "target": "mutability", "kind": "related_to"}],
        ),
        PostConcepts("p2", [{"name": "mutability", "type": "concept", "aliases": []}]),
    ]
    posts[1].concepts[0]["aliases"] = ["değişkenlik"]
    chunks = [
        {"id": "p1#0000", "post_slug": "p1", "text": "Gölgeleme (shadowing) ve mutability"},
        {"id": "p2#0000", "post_slug": "p2", "text": "Mutability ve let mut"},
        {"id": "p2#0001", "post_slug": "p2", "text": "Burada değişkenlik kelimesi geçiyor"},
        {"id": "x#0000", "post_slug": "x", "text": "shadowing burada da geçiyor"},  # pilot dışı
    ]
    graph = build_concept_graph(posts, chunks)
    df = {c["name"]: c["df"] for c in graph.concepts}
    assert df == {"mutability": 2, "shadowing": 1}
    assert {"chunk_id": "p1#0000", "concept": "shadowing"} in graph.mentions
    assert all(m["chunk_id"] != "x#0000" for m in graph.mentions)
    shadowing = next(c for c in graph.concepts if c["name"] == "shadowing")
    assert shadowing["mention_terms"] == ["shadowing"]
    assert shadowing["aliases"] == ["gölgeleme"]  # sorgu eşleştirmesi için tutulur
    # takma ad ('değişkenlik') metinde geçse bile MENTIONS kurulmaz
    assert {"chunk_id": "p2#0001", "concept": "mutability"} not in graph.mentions
    assert graph.relations == [
        {"source": "shadowing", "target": "mutability", "kind": "related_to", "count": 1}
    ]


def test_unverified_aliases_are_kept_but_not_used_for_mentions() -> None:
    posts = [
        PostConcepts("p1", [{"name": "heap", "type": "concept", "aliases": ["memory allocation"]}])
    ]
    chunks = [
        {"id": "p1#0000", "post_slug": "p1", "text": "Heap üzerinde yer ayırma"},
        {"id": "p1#0001", "post_slug": "p1", "text": "Memory pool"},
    ]
    graph = build_concept_graph(posts, chunks)
    heap = graph.concepts[0]
    assert heap["aliases"] == ["memory allocation"] and heap["mention_terms"] == ["heap"]
    assert graph.mentions == [{"chunk_id": "p1#0000", "concept": "heap"}]


def test_mention_pattern_applies_boundaries_to_every_alternative() -> None:
    pattern = mention_pattern(["struct", "yapı türü"])
    assert pattern.search("bir struct tanımı") and pattern.search("Yapı türü olarak")
    assert not pattern.search("structure")  # son alternatifte de kelime sınırı
    assert not pattern.search("altyapı türü")  # ilk alternatifte de kelime sınırı


def test_merged_names_are_mention_terms_but_aliases_are_not() -> None:
    posts = [
        PostConcepts("p1", [{"name": "lifetime", "type": "concept", "aliases": ["kopyalama"]}]),
        PostConcepts("p2", [{"name": "lifetimes", "type": "concept", "aliases": []}]),
    ]
    chunks = [
        {"id": "p1#0000", "post_slug": "p1", "text": "lifetime ve kopyalama"},
        {"id": "p2#0000", "post_slug": "p2", "text": "Lifetimes mevzusu"},
        {"id": "p2#0001", "post_slug": "p2", "text": "sadece kopyalama"},
    ]
    graph = build_concept_graph(posts, chunks)
    assert graph.concepts[0]["mention_terms"] == ["lifetime", "lifetimes"]
    assert {m["chunk_id"] for m in graph.mentions} == {"p1#0000", "p2#0000"}


def test_relations_to_dropped_concepts_are_ignored() -> None:
    posts = [
        PostConcepts(
            "p1",
            [{"name": "ownership", "type": "concept", "aliases": []}],
            [{"source": "rust", "target": "ownership", "kind": "part_of"}],
        )
    ]
    graph = build_concept_graph(posts, [{"id": "p1#0000", "post_slug": "p1", "text": "x"}])
    assert graph.relations == []


def test_code_expressions_are_not_concepts() -> None:
    raw = {
        "concepts": [
            {"name": n, "type": "concept", "aliases": []}
            for n in ["println!", "String::from", "main", "Vec<T>", "foo()", "message passing"]
        ],
        "relations": [],
    }
    assert [c["name"] for c in clean_post_concepts("p", raw).concepts] == ["message passing"]


def test_concepts_not_in_post_text_are_dropped() -> None:
    posts = [
        PostConcepts(
            "p1",
            [
                {"name": "ownership", "type": "concept", "aliases": ["sahiplik"]},
                {"name": "shadowing", "type": "concept", "aliases": []},  # metinde yok
            ],
            [{"source": "ownership", "target": "shadowing", "kind": "related_to"}],
        )
    ]
    chunks = [{"id": "p1#0000", "post_slug": "p1", "text": "Sahiplik kuralları"}]
    graph = build_concept_graph(posts, chunks)
    assert [c["name"] for c in graph.concepts] == ["ownership"]  # Türkçe karşılıkla kanıtlı
    assert graph.unsupported == 1 and graph.relations == []
    assert graph.about == [{"slug": "p1", "concept": "ownership"}]
