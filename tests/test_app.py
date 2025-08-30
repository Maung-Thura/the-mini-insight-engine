import json
import os

# Make sure OPENAI_API_KEY is not required for basic endpoints
os.environ.pop("OPENAI_API_KEY", None)

from app.server import app

def test_health():
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("status") == "ok"


def test_qa_basic():
    client = app.test_client()
    # Ensure indexes exist (server lazy-loads, but this warms paths)
    payload = {"question": "I have trouble sleeping; what should I try?"}
    resp = client.post("/qa", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "result" in data
    result = data["result"]
    assert "answer" in result
    # contexts should be returned
    assert isinstance(result.get("contexts", []), list)

