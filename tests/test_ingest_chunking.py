from app.ingest import build_documents
from app import config


def test_build_documents_chunking(monkeypatch):
    kb = [{
        "symptom": "Sleep",
        "category": "Sleep",
        "recommendations": [{
            "recommendation_id": "SLEEP_001",
            "recommendation_text": "A" * 10,
            "explanation": "B" * 200,
        }]
    }]

    monkeypatch.setattr(config, "CHUNK_SIZE", 50, raising=False)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 10, raising=False)

    docs = build_documents(kb)
    # Expect multiple chunks
    assert len(docs) >= 3
    # Each doc includes recommendation_id metadata
    assert all(d.metadata.get("recommendation_id") == "SLEEP_001" for d in docs)

