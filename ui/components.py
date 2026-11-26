"""Tekrar kullanılan Streamlit gösterimleri: cevap kartı ve açılır paneller."""

from __future__ import annotations

import streamlit as st

from ui.service import MODES, QueryOutcome, ServiceStatus, source_rows, timing

BOOST_HELP = "F: teknoloji filtresi · k: kod niyeti · G: graph ilişkisi (link/seri/kavram)"


def render_status(status: ServiceStatus) -> None:
    def line(ok: bool, name: str, detail: str | None) -> None:
        color = "green" if ok else "red"
        text = f":{color}[●] **{name}**"
        if not ok and detail:
            text += f"  \n:gray[{detail}]"
        st.markdown(text)

    embed_detail = None if status.embedding_loaded else f"{status.embedding_model} yüklü değil"
    line(status.lmstudio, "LM Studio", status.lmstudio_error)
    if status.lmstudio:
        line(status.embedding_loaded, "Embedding modeli", embed_detail)
    line(status.qdrant, "Qdrant (Faz 1)", status.qdrant_error)
    line(status.neo4j, "Neo4j (Faz 2)", status.neo4j_error)


def render_sources(outcome: QueryOutcome) -> None:
    rows = source_rows(outcome.result)
    if not rows:
        return
    st.markdown("**Kaynaklar**")
    lines = []
    for r in rows:
        mark = "✓" if r["cited"] else "·"
        note = f" · _{r['note']}_" if r["note"] else ""
        link = f"[{r['title']}]({r['url']})"
        lines.append(f"{mark} **[#{r['n']}]** {link} ({r['date']}) — {r['section']}{note}")
    st.markdown("  \n".join(lines))
    if not outcome.result.cited:
        st.caption("Model cevapta [#n] atfı yapmadı.")
    elif outcome.result.invalid_citations:
        st.caption(f"Bağlamda olmayan atıf numaraları: {outcome.result.invalid_citations}")


def render_chunks(outcome: QueryOutcome) -> None:
    table = [
        {
            "#": r["rank"],
            "Bağlam": f"#{r['context']}" if r["context"] else "",
            "Final": r["final"],
            "Skor": r["score"],
            "Boost": r["boosts"],
            "Token": r["tokens"],
            "Yazı": r["title"],
            "Bölüm": r["section"],
            "Graph nedeni": ", ".join(r["reasons"]),
            "Eklenen komşu": ", ".join(r["extras"]),
            "Link": r["url"],
        }
        for r in outcome.rows
    ]
    st.caption(f"Bağlam sütunu dolu olanlar LLM'e gitti. {BOOST_HELP}")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="aç")},
    )


def render_prompt(outcome: QueryOutcome) -> None:
    for message in outcome.result.messages:
        st.markdown(f"**{message['role']}**")
        st.code(message["content"], language="markdown", wrap_lines=True)


def render_timing(outcome: QueryOutcome) -> None:
    t = timing(outcome.result)
    cols = st.columns(4)
    cols[0].metric("Retrieval", f"{t['retrieval_seconds']} sn")
    first = t["first_token_seconds"]
    cols[1].metric("İlk token", f"{first} sn" if first is not None else "–")
    cols[2].metric("Üretim", f"{t['generation_seconds']} sn")
    cols[3].metric("Bağlam", f"~{t['context_tokens']} token")
    st.caption(f"Model: {outcome.result.model}")


def render_outcome(outcome: QueryOutcome, show_answer: bool = True) -> None:
    """Bir sorgunun sonucunu gösterir: cevap, kaynaklar ve açılır paneller."""
    st.markdown(f"#### {MODES[outcome.mode]}")
    if outcome.error:
        st.error(outcome.error)
    if outcome.warning:
        st.warning(outcome.warning)
    if outcome.result and show_answer:
        st.markdown(outcome.result.answer or "_(boş cevap)_")
    if outcome.result:
        render_sources(outcome)
    if outcome.result and outcome.result.facts:
        with st.expander("Graph bilgisi (yapısal sorgu sonucu)", expanded=True):
            for fact in outcome.result.facts:
                st.markdown(f"- {fact}")
    if outcome.rows:
        with st.expander(f"Getirilen chunk'lar ({len(outcome.rows)} aday)"):
            render_chunks(outcome)
    if outcome.result:
        with st.expander("LLM'e giden prompt"):
            render_prompt(outcome)
        with st.expander("Süreler"):
            render_timing(outcome)
