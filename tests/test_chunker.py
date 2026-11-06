import datetime as dt
from pathlib import Path

from common.chunker import ChunkerConfig, chunk_post, parse_blocks
from common.loader import Post

CFG = ChunkerConfig(
    target_tokens=60,
    max_tokens=200,
    min_tokens=5,
    overlap_tokens=15,
    intro_max_tokens=60,
    chars_per_token=4,
)


def make_post(body: str) -> Post:
    return Post(
        slug="2022-01-01-test",
        title="Test",
        date=dt.datetime(2022, 1, 1),
        tags=["rust"],
        categories=["Programlama Dilleri"],
        body=body,
        path=Path("x"),
    )


def code(lines: int, lang: str = "rust") -> str:
    return "```" + lang + "\n" + "\n".join(f"let x{i} = {i};" for i in range(lines)) + "\n```"


def test_heading_inside_code_is_not_a_heading() -> None:
    blocks = parse_blocks("## Başlık\n\n```csharp\n#region X\n#endregion\n```")
    assert [b.kind for b in blocks] == ["heading", "code"]
    assert blocks[1].lang == "csharp"


def test_section_path_follows_heading_levels() -> None:
    text = "Bu bölümde yeterince uzun bir açıklama var."
    body = f"## A\n\n{text}\n\n### B\n\n{text}\n\n## C\n\n{text}"
    paths = [c.section_path for c in chunk_post(make_post(body), CFG)]
    assert paths == ["A", "A > B", "C"]


def test_tiny_section_is_merged_into_previous() -> None:
    body = "## A\n\nBu bölümde yeterince uzun bir açıklama var.\n\n## B\n\nkısa"
    chunks = chunk_post(make_post(body), CFG)
    assert len(chunks) == 1 and chunks[0].text.endswith("kısa")


def test_code_block_is_never_split_and_keeps_intro() -> None:
    intro = "Aşağıdaki örnekte değişkenleri tanımlıyoruz."
    body = "Uzun bir giriş paragrafı. " * 8 + "\n\n" + intro + "\n\n" + code(20)
    chunks = chunk_post(make_post(body), CFG)
    with_code = [c for c in chunks if c.has_code]
    assert len(with_code) == 1
    assert with_code[0].text.count("```") == 2
    assert with_code[0].text.startswith(intro)
    assert with_code[0].code_langs == ["rust"]
    assert not with_code[0].code_split


def test_oversized_code_is_split_on_lines_and_refenced() -> None:
    chunks = chunk_post(make_post(code(120)), CFG)
    assert len(chunks) > 1
    assert all(c.code_split and c.text.startswith("```rust") for c in chunks)
    assert all(c.text.rstrip().endswith("```") for c in chunks)
    assert all(c.n_tokens <= CFG.max_tokens for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    assert all(f"let x{i} = {i};" in joined for i in range(120))


def test_large_table_is_split_with_header() -> None:
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i * 2} |" for i in range(200))
    chunks = chunk_post(make_post(table), CFG)
    assert len(chunks) > 1
    assert all(c.text.startswith("| a | b |\n|---|---|") for c in chunks)


def test_unclosed_fence_is_closed() -> None:
    blocks = parse_blocks("metin\n\n```rust\nlet x = 1;")
    assert blocks[-1].kind == "code" and blocks[-1].text.endswith("```")


def test_chunk_ids_are_deterministic() -> None:
    body = "## A\n\n" + "Bir cümle. " * 40
    first = [c.id for c in chunk_post(make_post(body), CFG)]
    second = [c.id for c in chunk_post(make_post(body), CFG)]
    assert first == second and first[0] == "2022-01-01-test#0000"
