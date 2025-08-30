import json
import os
import shutil
from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from .llm import get_embeddings
from . import config


def load_knowledge_base(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_documents(kb: List[Dict]) -> List[Document]:
    docs: List[Document] = []
    for entry in kb:
        symptom = entry.get("symptom")
        category = entry.get("category")
        for rec in entry.get("recommendations", []):
            rec_id = rec.get("recommendation_id")
            base_text = (
                f"Symptom: {symptom}\n"
                f"Category: {category}\n"
                f"Recommendation: {rec.get('recommendation_text')}\n"
                f"Explanation: {rec.get('explanation')}\n"
            )
            metadata = {
                "symptom": symptom,
                "category": category,
                "recommendation_id": rec_id,
            }
            if config.CHUNK_SIZE and len(base_text) > config.CHUNK_SIZE:
                start = 0
                while start < len(base_text):
                    end = start + config.CHUNK_SIZE
                    chunk = base_text[start:end]
                    docs.append(Document(page_content=chunk, metadata=metadata))
                    if not config.CHUNK_OVERLAP:
                        start = end
                    else:
                        start = end - config.CHUNK_OVERLAP
                        if start <= 0:
                            start = end
            else:
                docs.append(Document(page_content=base_text, metadata=metadata))
    return docs


def persist_corpus_json(docs: List[Document], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    serializable = [
        {
            "id": getattr(d, "id", None) or d.metadata.get("recommendation_id"),
            "text": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def ingest(reset: bool = True) -> Dict[str, int]:
    # Optionally reset persistent stores
    if reset and os.path.isdir(config.CHROMA_DIR):
        shutil.rmtree(config.CHROMA_DIR)

    kb = load_knowledge_base(config.KNOWLEDGE_JSON_PATH)
    docs = build_documents(kb)

    # Vector store (Chroma via LangChain)
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )

    # Add with explicit ids for fusion
    ids = [getattr(d, "id", None) or d.metadata.get("recommendation_id") for d in docs]
    vectorstore.add_documents(docs, ids=ids)
    vectorstore.persist()

    # Persist a BM25 corpus snapshot for runtime construction
    persist_corpus_json(docs, os.path.join("data", "corpus.json"))

    return {"documents": len(docs)}

