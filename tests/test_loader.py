from pathlib import Path

from common.config import Settings
from common.loader import clean_body, extract_internal_links, iter_posts, load_post

POST = """---
layout: post
title: "Rust Değişkenler"
date: 2022-04-17 10:00:00
tags:
    - rust
categories:
    - Programlama Dilleri
---
Giriş paragrafı. ![ekran](/assets/images/x.png) Detay için
[önceki yazı](/2022/04/10/rust-pratikleri-wordle-oyunu/).

<br/>Generic tür `List<T>` korunmalı.

{% raw %}
```html
<div>{{ value }}</div>
```
{% endraw %}
"""


def write(tmp: Path, name: str, text: str) -> Path:
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_post_metadata_and_url(tmp_path: Path) -> None:
    post = load_post(write(tmp_path, "2022-04-17-rust-degiskenler.md", POST))
    assert post.title == "Rust Değişkenler"
    assert post.year == 2022
    assert post.tags == ["rust"]
    assert post.url == "https://buraksenyurt.github.io/2022/04/17/rust-degiskenler/"
    assert post.links == ["2022-04-10-rust-pratikleri-wordle-oyunu"]


def test_clean_body_keeps_code_and_link_text() -> None:
    body = clean_body(POST.split("---", 2)[2])
    assert "![ekran]" not in body and "/assets/images" not in body
    assert "önceki yazı" in body and "](/2022" not in body
    assert "<br/>" not in body and "`List<T>`" in body
    assert "{% raw %}" not in body
    assert "<div>{{ value }}</div>" in body  # kod içindeki HTML'e dokunulmaz


def test_extract_internal_links_dedup() -> None:
    md = "[a](/2009/04/22/koleksiyon/) [b](https://buraksenyurt.github.io/2009/04/22/koleksiyon/)"
    assert extract_internal_links(md) == ["2009-04-22-koleksiyon"]


def test_excluded_categories_are_skipped(tmp_path: Path) -> None:
    write(tmp_path, "2022-04-17-rust-degiskenler.md", POST)
    foto = POST.replace("Programlama Dilleri", "Foto İpucu")
    write(tmp_path, "2012-02-05-tek-fotoluk-ipucu-1.md", foto)
    settings = Settings(_env_file=None, posts_dir=tmp_path)
    assert [p.slug for p in iter_posts(settings)] == ["2022-04-17-rust-degiskenler"]
