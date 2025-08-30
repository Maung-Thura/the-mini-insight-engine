import json
import logging
from typing import List, Dict, Tuple

from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

from .llm import get_embeddings
from . import config
from .utils import tokenize

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vs = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_DIR,
        )
        # Load corpus for BM25
        with open("data/corpus.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        self.id_to_text: Dict[str, str] = {d["id"]: d["text"] for d in docs}
        self.id_to_meta: Dict[str, Dict] = {d["id"]: d.get("metadata", {}) for d in docs}
        corpus_texts: List[str] = [d["text"] for d in docs]
        self.corpus_ids: List[str] = [d["id"] for d in docs]
        tokenized_corpus = [self._tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Improved tokenizer: word regex + lowercase + basic stopword removal
        return tokenize(text)

    def _vector_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        results = self.vs.similarity_search_with_score(query, k=k)
        pairs: List[Tuple[str, float]] = []
        for doc, score in results:
            # Chroma returns smaller score for closer distance in some configs; convert to similarity
            sim = 1.0 / (1.0 + score)
            doc_id = doc.metadata.get("recommendation_id") or doc.metadata.get("id")
            if not doc_id:
                # fallback using page_content mapping (rare)
                # search corpus for exact text to map id
                for id_, txt in self.id_to_text.items():
                    if txt == doc.page_content:
                        doc_id = id_
                        break
            pairs.append((doc_id, sim))
        return pairs

    def _bm25_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        toks = self._tokenize(query)
        scores = self.bm25.get_scores(toks)
        # Pair with ids
        id_scores = list(zip(self.corpus_ids, scores))
        id_scores.sort(key=lambda x: x[1], reverse=True)
        return id_scores[:k]

    def _rrf_fuse(self, ranked_lists: List[List[Tuple[str, float]]], k: int, rrf_k: int) -> List[Tuple[str, float]]:
        # Convert each list to ranking positions by id
        agg: Dict[str, float] = {}
        for lst in ranked_lists:
            for rank, (doc_id, _score) in enumerate(lst, start=1):
                agg[doc_id] = agg.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        fused = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        return fused[:k]

    def search(self, query: str) -> List[Dict]:
        vect = self._vector_search(query, k=config.VECTOR_TOP_K)
        kw = self._bm25_search(query, k=config.BM25_TOP_K)
        logger.debug({"vector": vect[:3], "bm25": kw[:3]})
        fused = self._rrf_fuse([vect, kw], k=config.FUSION_K, rrf_k=config.RRF_K)
        logger.debug({"fused": fused[:5]})
        results: List[Dict] = []
        for doc_id, score in fused:
            results.append({
                "id": doc_id,
                "score": score,
                "text": self.id_to_text[doc_id],
                "metadata": self.id_to_meta[doc_id],
            })
        return results

