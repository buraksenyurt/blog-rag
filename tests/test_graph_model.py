import pytest

from phase2_graph.graph_model import build_graph_data, detect_series, normalize_tag, series_prefix


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Rust Pratikleri - Lifetimes Mevzusu", "Rust Pratikleri"),
        ("Ruby Kod Parçacıkları 16 - (JSON Serileştirme)", "Ruby Kod Parçacıkları"),
        ("Tasarım Desenleri – Template Method", "Tasarım Desenleri"),
        ("GoLang - Slice ve Map Kavramları", "GoLang"),
        ("EF Core : Testlerde InMemory Context Kullanımı", "EF Core"),
        ("Oyun Programlamaya Giriş (Çarpışma Teknikleri - 1)", "Oyun Programlamaya Giriş"),
        ("Rust ile Basit Bir Key-Value Store Uygulaması", None),  # ayraç yok
        ("C# - Delegates", None),  # önek seri adı için çok kısa
    ],
)
def test_series_prefix(title: str, expected: str | None) -> None:
    assert series_prefix(title) == expected


def test_detect_series_orders_by_date_and_requires_min_size() -> None:
    posts = [
        {"slug": "b", "title": "Seri A - İkinci", "date": "2022-02-01"},
        {"slug": "a", "title": "Seri A - Birinci", "date": "2022-01-01"},
        {"slug": "c", "title": "Seri A - Üçüncü", "date": "2022-03-01"},
        {"slug": "x", "title": "Seri B - Tek", "date": "2022-01-01"},
        {"slug": "y", "title": "Seri B - İki", "date": "2022-01-02"},
    ]
    rows = detect_series(posts)
    assert [(r["slug"], r["position"]) for r in rows] == [("a", 1), ("b", 2), ("c", 3)]
    assert {r["series"] for r in rows} == {"Seri A"}


def chunk(slug: str, index: int, **extra) -> dict:
    data = {
        "id": f"{slug}#{index:04d}",
        "post_slug": slug,
        "title": slug.title(),
        "date": "2022-01-01",
        "year": 2022,
        "url": f"u/{slug}",
        "text": f"metin {index}",
        "section_path": "",
        "chunk_index": index,
        "n_tokens": 10,
        "has_code": False,
        "code_langs": [],
        "tags": ["Rust", "rust-lang"],
        "categories": ["Programlama Dilleri"],
    }
    data.update(extra)
    return data


def test_build_graph_data() -> None:
    chunks = [
        chunk("p1", 1, has_code=True, code_langs=["rust"]),
        chunk("p1", 0),
        chunk("p2", 0, tags=["rust"]),
    ]
    links = {"p1": ["p2", "foto-ipucu", "p1"], "absent": ["p1"]}
    data = build_graph_data(chunks, links)

    assert [p["slug"] for p in data.posts] == ["p1", "p2"]
    assert len(data.chunks) == 3 and "embedding" not in data.chunks[0]
    assert data.next_pairs == [{"a": "p1#0000", "b": "p1#0001"}]
    assert data.code_langs == [{"chunk_id": "p1#0001", "lang": "rust"}]
    assert data.links == [{"src": "p1", "dst": "p2"}] and data.missing_links == 2
    assert data.tag_variants["rust"] == ["Rust", "rust"]
    assert {"slug": "p2", "tag": "rust"} in data.tagged
    assert sum(1 for t in data.tagged if t["slug"] == "p1") == 2  # tag'ler yazı başına bir kez
    assert data.in_category == [
        {"slug": "p1", "category": "Programlama Dilleri"},
        {"slug": "p2", "category": "Programlama Dilleri"},
    ]


def test_normalize_tag() -> None:
    assert normalize_tag("  PostgreSQL ") == "postgresql"
