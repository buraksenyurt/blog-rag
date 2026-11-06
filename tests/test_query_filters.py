import pytest

from phase1_vector.query_filters import detect_filter, normalize_query


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Rust dilinde değişken kullanımları ile ilgili örnekler yaz.", ["rust"]),
        ("Rust'ta ownership nedir?", ["rust"]),
        ("C#'ta LINQ ile gruplama", ["csharp", "linq"]),
        ("Go'da goroutine örneği", ["go"]),
        ("Node.js ile REST servisi", ["nodejs"]),
        ("WCF servislerinde binding türleri", ["wcf"]),
        ("Entity Framework ile migration", ["entity-framework"]),
    ],
)
def test_detects_technologies(query: str, expected: list[str]) -> None:
    assert detect_filter(query).technologies == expected


@pytest.mark.parametrize(
    "query",
    [
        "Yazılımda iyi isimlendirme nasıl yapılır?",
        "as ve is operatörleri arasındaki fark",  # serbest tag eşleştirmesi yapılmaz
        "trust ve rustik kelimeleri",  # kelime sınırı
        "javafx",
    ],
)
def test_no_false_positives(query: str) -> None:
    assert not detect_filter(query)


def test_filter_contents_and_match() -> None:
    qf = detect_filter("Rust örnekleri")
    assert qf.tags == ["rust", "rust-lang"] and qf.code_langs == ["rust"]
    assert qf.matches({"tags": ["csharp"], "code_langs": ["rust"]})
    assert not qf.matches({"tags": ["csharp"], "code_langs": ["csharp"]})


def test_normalize_query() -> None:
    assert normalize_query("Rust'ın ve C#’ta") == "rust ve c#"
