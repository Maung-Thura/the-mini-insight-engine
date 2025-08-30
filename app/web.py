from flask import Blueprint, render_template, request, jsonify, make_response
import os

from .graph import QAGraph
from .metrics import compute_ragas_metrics, ragas_available
from .ingest import ingest as run_ingest
from . import config

ui = Blueprint("ui", __name__, static_folder="static", template_folder="templates")

_graph = None

def ensure_indexes():
    corpus_path = os.path.join("data", "corpus.json")
    if not os.path.exists(corpus_path) or not os.path.isdir(config.CHROMA_DIR):
        run_ingest(reset=False)


def _get_graph():
    global _graph
    if _graph is None:
        ensure_indexes()
        _graph = QAGraph()
    return _graph


@ui.get("/")
def index():
    resp = make_response(render_template("index.html"))
    # Disable caching to avoid stale JS after deployments
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@ui.post("/qa")
def qa_endpoint():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    filters = data.get("filters") or None
    if not question:
        return jsonify({"error": "Missing question"}), 400
    g = _get_graph()
    result = g.run(question, filters=filters)

    # Compute RAGAS metrics (optional if package is installed and configured)
    ok, reason = ragas_available()
    metrics = compute_ragas_metrics(question, result.get("answer", ""), result.get("contexts", [])) if ok else None
    status = "ok" if (metrics is not None) else (reason or "unavailable")
    return jsonify({"result": result, "metrics": metrics, "metrics_status": status })

