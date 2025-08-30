from flask import Flask, request, jsonify
import os

from .ingest import ingest as run_ingest
from .graph import QAGraph
from . import config
from .utils import setup_logging
from .web import ui as ui_blueprint

app = Flask(__name__, template_folder="templates")
app.register_blueprint(ui_blueprint)
setup_logging()

# Ensure OPENAI_API_KEY is available for libraries like RAGAS that may not honor alternative names
if not os.getenv("OPENAI_API_KEY") and os.getenv("OPEN_AI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY")  # mirror into expected var name

_graph = None

def ensure_indexes():
    # Build indexes if missing
    corpus_path = os.path.join("data", "corpus.json")
    if not os.path.exists(corpus_path) or not os.path.isdir(config.CHROMA_DIR):
        run_ingest(reset=False)


def get_graph() -> QAGraph:
    global _graph
    if _graph is None:
        ensure_indexes()
        _graph = QAGraph()
    return _graph


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest_endpoint():
    body = request.get_json(silent=True) or {}
    reset = bool(body.get("reset", True))
    try:
        stats = run_ingest(reset=reset)
        # reset cached graph so it rebuilds with fresh stores
        global _graph
        _graph = None
        return jsonify({"status": "ok", "stats": stats})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.post("/ask")
def ask():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    filters = body.get("filters") or None
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400
    try:
        graph = get_graph()
        result = graph.run(question, filters=filters)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT)

