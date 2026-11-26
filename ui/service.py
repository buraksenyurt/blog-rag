"""Demo arabiriminin Streamlit'ten bağımsız servis katmanı.

Arayüz (ui/app.py) sadece gösterimden sorumludur; bağlantılar, servis durumu, retrieval,
cevap üretimi ve karşılaştırma burada yapılır. Böylece bu katman test edilebilir ve
ileride başka bir arayüze (ör. API) taşınabilir. Faz 1 ve Faz 2'yi birlikte kullanan iki
yerden biridir (diğeri eval/).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from common.answering import AskResult, answer_question, empty_answer_reason, log_run
from common.config import ROOT, EmbeddingProfile, Settings
from common.retrieval import RetrievalConfig, RetrievalResult, candidate_rows

MODES = {"vector": "Vektör RAG (Faz 1 · Qdrant)", "graph": "Graph RAG (Faz 2 · Neo4j)"}
QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
# Derste hızlı seçim için eval setinden örnekler: (etiket, soru id)
DEFAULT_EXAMPLES = [
    ("Normal", "rust-variables"),
    ("Normal", "go-defer-panic"),
    ("Zor", "hard-moved-value"),
    ("Zor", "hard-test-email-sending"),
    ("Yapısal", "count-rust-posts"),
    ("Yapısal", "rust-pratikleri-series"),
]


# --- Servis durumu ---------------------------------------------------------------------


@dataclass
class ServiceStatus:
    lmstudio: bool = False
    lmstudio_error: str | None = None
    models: list[str] = field(default_factory=list)
    embedding_model: str = ""
    qdrant: bool = False
    qdrant_error: str | None = None
    neo4j: bool = False
    neo4j_error: str | None = None

    @property
    def embedding_loaded(self) -> bool:
        return self.embedding_model in self.models

    @property
    def chat_models(self) -> list[str]:
        return [m for m in self.models if "embed" not in m.lower()]

    def mode_available(self, mode: str) -> tuple[bool, str]:
        """Mod kullanılabilir mi? Değilse sebebi (Türkçe) döner."""
        if not self.lmstudio:
            return False, f"LM Studio'ya ulaşılamıyor: {self.lmstudio_error}"
        if not self.embedding_loaded:
            return False, f"Embedding modeli yüklü değil: {self.embedding_model}"
        if not self.chat_models:
            return False, "LM Studio'da yüklü sohbet modeli yok"
        if mode == "vector" and not self.qdrant:
            return False, f"Qdrant hazır değil: {self.qdrant_error}"
        if mode == "graph" and not self.neo4j:
            return False, f"Neo4j hazır değil: {self.neo4j_error}"
        return True, ""


def short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}".splitlines()[0][:160]


def check_services(settings: Settings, profile: EmbeddingProfile) -> ServiceStatus:
    """LM Studio, Qdrant ve Neo4j'yi kısa zaman aşımlarıyla yoklar."""
    status = ServiceStatus(embedding_model=profile.model)
    try:
        client = OpenAI(
            base_url=settings.lmstudio_url, api_key=settings.lmstudio_api_key, timeout=3
        )
        status.models = sorted(m.id for m in client.models.list().data)
        status.lmstudio = True
    except Exception as exc:  # noqa: BLE001 - durum paneli her hatayı göstermeli
        status.lmstudio_error = short_error(exc)

    try:
        from qdrant_client import QdrantClient

        from phase1_vector.store import collection_name

        name = collection_name(profile)
        if QdrantClient(url=settings.qdrant_url, timeout=3).collection_exists(name):
            status.qdrant = True
        else:
            status.qdrant_error = f"'{name}' collection'ı yok (phase1_vector.ingest)"
    except Exception as exc:  # noqa: BLE001
        status.qdrant_error = short_error(exc)

    try:
        from phase2_graph.store import connect, make_runner

        driver = connect(settings)
        try:
            rows = make_runner(driver, settings.neo4j_database)(
                "MATCH (m:GraphMeta) RETURN m.embedding_profile AS profile LIMIT 1"
            )
        finally:
            driver.close()
        if not rows:
            status.neo4j_error = "graph kurulmamış (phase2_graph.build_graph)"
        elif rows[0]["profile"] != profile.name:
            status.neo4j_error = f"graph '{rows[0]['profile']}' profiliyle kurulmuş"
        else:
            status.neo4j = True
    except Exception as exc:  # noqa: BLE001
        status.neo4j_error = short_error(exc)
    return status


# --- Ayarlar ve motor -------------------------------------------------------------------


@dataclass(frozen=True)
class UISettings:
    n: int = 6
    budget: int = 3500
    temperature: float = 0.2
    max_tokens: int = 1024
    graph_expand: bool = True
    structural: bool = True
    concept_expand: bool = False


def configure(base: RetrievalConfig, ui: UISettings) -> RetrievalConfig:
    """Arayüz ayarlarını retriever'ın mevcut config'ine uygular (fazına uygun alanlarla)."""
    overrides: dict[str, Any] = {"n": ui.n, "token_budget": ui.budget}
    names = set(base.__dataclass_fields__)
    for key in ("graph_expand", "structural", "concept_expand"):
        if key in names:
            overrides[key] = getattr(ui, key)
    return replace(base, **overrides)


@dataclass
class QueryOutcome:
    mode: str
    question: str
    result: AskResult | None = None
    rows: list[dict] = field(default_factory=list)
    error: str | None = None
    warning: str | None = None


class Engine:
    """Uygulama ömrü boyunca bir kez kurulan bağlantılar, retriever'lar ve LLM istemcileri.

    Gerçek bağlantılar ilk kullanımda kurulur. Testlerde retriever_factory ve llm_factory
    ile sahte nesneler verilebilir.
    """

    def __init__(
        self,
        settings: Settings,
        profile: EmbeddingProfile,
        retriever_factory: Callable[[str], Any] | None = None,
        llm_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.profile = profile
        self._retriever_factory = retriever_factory or self._make_retriever
        self._llm_factory = llm_factory or self._make_llm
        self._retrievers: dict[str, Any] = {}
        self._llms: dict[str, Any] = {}
        self._embedder = None
        self._driver = None

    # Gerçek nesneler (sadece varsayılan fabrikalar kullanır)
    def _embedder_instance(self):
        if self._embedder is None:
            from common.embeddings import Embedder

            self._embedder = Embedder(self.settings, self.profile)
        return self._embedder

    def _make_retriever(self, mode: str):
        if mode == "vector":
            from qdrant_client import QdrantClient

            from phase1_vector.retrieve import Retriever

            return Retriever(
                self._embedder_instance(), QdrantClient(url=self.settings.qdrant_url, timeout=30)
            )
        if mode == "graph":
            from phase2_graph.retrieve import GraphRetriever
            from phase2_graph.store import connect, make_runner

            if self._driver is None:
                self._driver = connect(self.settings)
            return GraphRetriever(
                self._embedder_instance(), make_runner(self._driver, self.settings.neo4j_database)
            )
        raise ValueError(f"bilinmeyen mod: {mode}")

    def _make_llm(self, model: str):
        from common.llm import LLM

        return LLM(self.settings, model=model)

    def retriever(self, mode: str, ui: UISettings):
        if mode not in self._retrievers:
            self._retrievers[mode] = self._retriever_factory(mode)
        retriever = self._retrievers[mode]
        retriever.config = configure(retriever.config, ui)
        return retriever

    def llm(self, model: str):
        if model not in self._llms:
            self._llms[model] = self._llm_factory(model)
        return self._llms[model]

    def warm_up(self) -> None:
        """Embedding modelini bir kez çağırır (ilk sorudaki bekleme azalır)."""
        self._embedder_instance().embed_query("ısınma")

    def retrieve(self, mode: str, question: str, ui: UISettings) -> tuple[RetrievalResult, float]:
        started = time.perf_counter()
        result = self.retriever(mode, ui).retrieve(question)
        return result, round(time.perf_counter() - started, 2)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


def run_query(
    engine: Engine,
    mode: str,
    question: str,
    ui: UISettings,
    model: str,
    on_token: Callable[[str], None] | None = None,
    on_retrieval: Callable[[RetrievalResult], None] | None = None,
    log: bool = True,
) -> QueryOutcome:
    """Tek modda soru çalıştırır: retrieval -> (isteğe bağlı gösterim) -> cevap."""
    outcome = QueryOutcome(mode=mode, question=question)
    try:
        retrieval, seconds = engine.retrieve(mode, question, ui)
    except Exception as exc:  # noqa: BLE001 - arayüz hatayı göstermeli, çökmemeli
        outcome.error = f"Retrieval hatası: {short_error(exc)}"
        return outcome
    outcome.rows = candidate_rows(retrieval)
    if on_retrieval is not None:
        on_retrieval(retrieval)

    llm = engine.llm(model)
    try:
        outcome.result = answer_question(
            question,
            llm,
            None,
            temperature=ui.temperature,
            max_tokens=ui.max_tokens,
            on_token=on_token or (lambda _t: None),
            retrieval=retrieval,
            retrieval_seconds=seconds,
        )
    except APITimeoutError:
        outcome.error = (
            f"LLM {engine.settings.llm_timeout:.0f} saniyede cevap vermedi. "
            "Daha küçük bir model ya da daha az chunk deneyin."
        )
        return outcome
    except (APIConnectionError, APIError) as exc:
        outcome.error = f"LLM hatası: {short_error(exc)}"
        return outcome

    if not outcome.result.answer.strip():
        outcome.warning = empty_answer_reason(llm)
    if log:
        log_run(
            outcome.result,
            engine.settings,
            engine.profile.name,
            engine.retriever(mode, ui).config,
            phase=f"ui-{mode}",
        )
    return outcome


# --- Gösterim için yardımcı veriler -----------------------------------------------------


def source_rows(result: AskResult) -> list[dict]:
    """Cevaptaki [#n] numaralarının karşılığı olan kaynak yazılar."""
    rows = []
    for number, hit in enumerate(result.hits, start=1):
        p = hit.payload
        rows.append(
            {
                "n": number,
                "title": p.get("title", ""),
                "date": p.get("date", ""),
                "section": p.get("section_path") or "Giriş",
                "url": p.get("url", ""),
                "cited": number in result.cited,
                "note": p.get("context_note"),
            }
        )
    return rows


def timing(result: AskResult) -> dict:
    return {
        "retrieval_seconds": result.retrieval_seconds,
        "first_token_seconds": result.first_token_seconds,
        "generation_seconds": result.generation_seconds,
        "context_tokens": sum(h.total_tokens for h in result.hits),
    }


def context_posts(outcome: QueryOutcome) -> list[tuple[str, str]]:
    """Bağlama giren yazılar (slug, başlık), sırasıyla ve tekrarsız."""
    if not outcome.result:
        return []
    seen: dict[str, str] = {}
    for hit in outcome.result.hits:
        seen.setdefault(hit.post_slug, hit.payload.get("title", hit.post_slug))
    return list(seen.items())


def compare_contexts(vector: QueryOutcome, graph: QueryOutcome) -> dict:
    """İki modun bağlamlarını karşılaştırır: ortak ve yalnızca bir tarafta olan yazılar."""
    a, b = dict(context_posts(vector)), dict(context_posts(graph))
    return {
        "common": [(s, a[s]) for s in a if s in b],
        "only_vector": [(s, a[s]) for s in a if s not in b],
        "only_graph": [(s, b[s]) for s in b if s not in a],
        "graph_facts": list(graph.result.facts) if graph.result else [],
    }


def example_questions(
    path: Path = QUESTIONS_PATH, picks: list[tuple[str, str]] | None = None
) -> list[dict]:
    """Eval setinden örnek sorular: [{label, id, question}]."""
    questions = {q["id"]: q for q in yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]}
    return [
        {"label": label, "id": qid, "question": questions[qid]["question"]}
        for label, qid in (picks or DEFAULT_EXAMPLES)
        if qid in questions
    ]
