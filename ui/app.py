"""Ders demo arabirimi: aynı soruya vektör RAG (Faz 1) ya da graph RAG (Faz 2) ile cevap.

Çalıştırma:
    uv run streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run ui/app.py` sys.path'e sadece ui/ klasörünü ekler; proje kökü olmadan
# common/, phase1_vector/ gibi paketler bulunamaz (ModuleNotFoundError: common).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from common.config import get_settings, load_embedding_profile  # noqa: E402
from ui.components import render_outcome, render_status  # noqa: E402
from ui.service import (  # noqa: E402
    MODES,
    Engine,
    UISettings,
    check_services,
    example_questions,
    run_query,
)

st.set_page_config(page_title="blog-rag · Vektör ve Graph RAG", layout="wide")


@st.cache_resource(show_spinner="Bağlantılar hazırlanıyor...")
def get_engine() -> Engine:
    settings = get_settings()
    return Engine(settings, load_embedding_profile(settings))


@st.cache_data(ttl=30, show_spinner="Servisler kontrol ediliyor...")
def get_status():
    engine = get_engine()
    return check_services(engine.settings, engine.profile)


@st.cache_data
def get_examples() -> list[dict]:
    return example_questions()


def sidebar(status) -> tuple[str | None, UISettings]:
    with st.sidebar:
        st.subheader("Servis durumu")
        render_status(status)
        if st.button("Durumu yenile", width="stretch"):
            get_status.clear()
            st.rerun()

        st.subheader("Model")
        models = status.chat_models
        default = get_engine().settings.llm_model
        model = st.selectbox(
            "LLM",
            models,
            index=models.index(default) if default in models else 0,
            disabled=not models,
            help="LM Studio'da yüklü sohbet modelleri",
        )

        with st.expander("Retrieval ayarları"):
            n = st.slider("LLM'e gidecek chunk sayısı", 2, 8, 6)
            budget = st.slider("Bağlam token bütçesi", 1000, 4000, 3500, step=250)
            temperature = st.slider("Sıcaklık", 0.0, 1.0, 0.2, step=0.1)
            max_tokens = st.select_slider("En fazla cevap token'ı", [512, 1024, 2048], 1024)
            st.markdown("**Graph RAG**")
            graph_expand = st.checkbox("Link/seri genişletmesi", value=True)
            structural = st.checkbox("Yapısal sorgular (sayma, listeleme)", value=True)
            concept_expand = st.checkbox(
                "Kavram katmanı (pilot)", value=False, help="Eval'de katkı sağlamadı"
            )

        st.subheader("Örnek sorular")
        for example in get_examples():
            label = f"{example['label']}: {example['question']}"
            if st.button(label, key=example["id"], width="stretch"):
                st.session_state.question = example["question"]
    ui = UISettings(n, budget, temperature, max_tokens, graph_expand, structural, concept_expand)
    return model, ui


def ask(mode: str, question: str, model: str, ui: UISettings) -> None:
    """Soruyu çalıştırır; retrieval tablosu ve cevabı canlı gösterir, sonucu saklar."""
    engine = get_engine()
    st.markdown(f"#### {MODES[mode]}")
    with st.status("Retrieval yapılıyor...", expanded=False) as progress:
        answer_box = st.empty()
        parts: list[str] = []

        def on_retrieval(result) -> None:
            facts = " · graph bilgisi var" if result.facts else ""
            progress.update(
                label=f"{len(result.hits)} chunk bağlama alındı{facts}. Cevap üretiliyor...",
                expanded=True,
            )

        def on_token(token: str) -> None:
            parts.append(token)
            answer_box.markdown("".join(parts) + " ▌")

        outcome = run_query(engine, mode, question, ui, model, on_token, on_retrieval)
        state = "error" if outcome.error else "complete"
        progress.update(label="Tamamlandı" if state == "complete" else "Hata", state=state)
    st.session_state.last = outcome


def main() -> None:
    status = get_status()
    model, ui = sidebar(status)

    st.title("Vektör RAG mı, Graph RAG mı?")
    st.caption(
        "Kaynak: buraksenyurt.github.io yazıları · Faz 1: Qdrant vektör araması · "
        "Faz 2: Neo4j graph (seri, link, tag ilişkileri ve yapısal sorgular)"
    )

    question = st.text_area(
        "Soru",
        key="question",
        height=90,
        placeholder="Ör. Rust Pratikleri serisinde hangi yazılar var?",
    )
    mode = st.radio("Mod", list(MODES), format_func=MODES.get, horizontal=True)
    available, reason = status.mode_available(mode)
    if not available:
        st.warning(f"Bu mod şu an kullanılamıyor. {reason}")

    if st.button("Sor", type="primary", disabled=not (available and question.strip() and model)):
        ask(mode, question.strip(), model, ui)
        st.rerun()

    last = st.session_state.get("last")
    if last is not None:
        st.divider()
        st.markdown(f"**Soru:** {last.question}")
        render_outcome(last)


# Streamlit (ve AppTest) betiği __main__ olarak çalıştırır; import edildiğinde sayfa kurulmaz.
if __name__ == "__main__":
    main()
