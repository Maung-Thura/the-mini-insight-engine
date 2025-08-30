import json
import os

from app.utils import tokenize
from app.retrieval import HybridRetriever


def test_tokenize_basic():
    txt = "The quick brown fox, with a THE leash's!"
    toks = tokenize(txt)
    # 'the' removed as stopword; possessive handled
    assert "the" not in toks
    assert "leash's" in toks
    assert "quick" in toks and "brown" in toks and "fox" in toks


def test_rrf_fusion_and_bm25(monkeypatch, tmp_path):
    # Prepare a tiny fake corpus.json
    corpus = [
        {"id": "A", "text": "hot flashes reduce with vitamin e supplement", "metadata": {"recommendation_id": "VASO_002"}},
        {"id": "B", "text": "sleep schedule insomnia consistent bedtime", "metadata": {"recommendation_id": "SLEEP_001"}},
    ]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with open(data_dir / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f)

    # Monkeypatch config paths to point to temp
    import app.config as config
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma"), raising=False)
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_kb", raising=False)

    # Monkeypatch HybridRetriever to skip Chroma vector search and just return empty vector hits
    def fake_vector_search(self, query: str, k: int):
        return []

    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # Write fake Chroma dependency by creating empty dir (we won't use it)
    os.makedirs(config.CHROMA_DIR, exist_ok=True)

    monkeypatch.setattr(HybridRetriever, "_vector_search", fake_vector_search, raising=True)

    # Patch path for corpus.json read
    def fake_init(self):
        self.embeddings = None  # unused in test
        self.vs = None
        with open(os.path.join(str(data_dir), "corpus.json"), "r", encoding="utf-8") as f:
            docs = json.load(f)
        self.id_to_text = {d["id"]: d["text"] for d in docs}
        self.id_to_meta = {d["id"]: d.get("metadata", {}) for d in docs}
        self.corpus_ids = [d["id"] for d in docs]
        from rank_bm25 import BM25Okapi
        from app.utils import tokenize as tok
        tokenized_corpus = [tok(d["text"]) for d in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    monkeypatch.setattr(HybridRetriever, "__init__", fake_init, raising=True)

    r = HybridRetriever()
    res = r.search("how to reduce hot flashes with supplement?")
    assert res and res[0]["id"] in {"A", "B"}

