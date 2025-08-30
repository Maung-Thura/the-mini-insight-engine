## Mini Insight Engine - Hybrid RAG

This project implements a hybrid Retrieval Augmented Generation (RAG) question answering system using:
- OpenAI LLM (langchain-openai)
- Chroma vector DB (semantic search)
- BM25 keyword search (top 25) and rank fusion (RRF)
- LangGraph for an agentic actor-critic evaluation loop
- Flask API and Docker packaging

The retrieval architecture follows the "Introducing Contextual Retrieval" approach [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) with query rewriting, hybrid retrieval, generation, critique, and optional revision.

### Repo layout
- knowledge_base/knowledge_base.json — sample KB to ingest
- app/ — Python modules (ingest, retrieval, prompts, graph, server)
- requirements.txt — Python dependencies
- Dockerfile — container image

### Prerequisites
- Python 3.11+
- An OpenAI API key

Quickstart: create a .env at the project root (same folder as docker-compose.yml) and add:

OPENAI_API_KEY=sk-...

Notes:
- .env is gitignored; do not commit secrets
- Optional OPEN_AI_API_KEY is also accepted; the server mirrors it into OPENAI_API_KEY for compatibility
- A starter .env.example is provided; copy it: cp .env.example .env and fill in your key

Optional envs (defaults in parentheses):
- CHROMA_DIR (data/chroma)
- CHROMA_COLLECTION (knowledge_base)
- KNOWLEDGE_JSON_PATH (knowledge_base/knowledge_base.json)
- MAX_CONTEXT_CHUNKS (6)
- BM25_TOP_K (25)
- VECTOR_TOP_K (10)
- FUSION_K (8)
- RRF_K (60)
- MAX_GRAPH_ITERS (2)
- PORT (8080)
- HOST (0.0.0.0)
- CHUNK_SIZE (0 disables)
- CHUNK_OVERLAP (0)
- LOG_LEVEL (INFO)

### Install and run (Python)
1) pip install -r requirements.txt
2) Option A: Pre-ingest
   python -c "from app.ingest import ingest; print(ingest())"
   Option B: Let the server auto-ingest on first run
3) Start API
   python -m app.server

Health check:
- GET http://localhost:8080/health

Ingest/reingest:
- POST http://localhost:8080/ingest
  {"reset": true}

Ask a question:
- POST http://localhost:8080/ask
  {"question": "I'm having trouble sleeping; what should I try?"}

Ask with metadata filter (optional):
- POST http://localhost:8080/ask
  {"question": "Recommendations for sleep?", "filters": {"category": "Sleep"}}

### Docker
Build:
- docker build -t mini-insight-engine .

Run (Option B: named volume - recommended on macOS):
- docker volume create mini_insight_data
- docker run --rm -p 8080:8080 --env-file .env -v mini_insight_data:/app/data mini-insight-engine

Alternative (no persistence):
- docker run --rm -p 8080:8080 --env-file .env mini-insight-engine
- Quickstart: cp .env.example .env; edit OPENAI_API_KEY in .env; then run: docker compose up -d --build


Alternative (bind mount; requires Docker Desktop File Sharing for your project path):
- docker run --rm -p 8080:8080 --env-file .env -v "$(pwd)/data:/app/data" mini-insight-engine

### Docker Compose (recommended)
- Ensure .env contains your key(s). OPENAI_API_KEY is preferred; OPEN_AI_API_KEY is also accepted. If you only have OPEN_AI_API_KEY, the server will mirror it into OPENAI_API_KEY at runtime for compatibility (e.g., RAGAS).
- Start in background:
  docker compose up -d --build
- Follow logs:
  docker compose logs -f app
- Stop and remove container:
  docker compose down

UI: http://localhost:8080/
Health: curl http://localhost:8080/health
Ingest: curl -X POST http://localhost:8080/ingest -H 'Content-Type: application/json' -d '{"reset": true}'
Ask: curl -X POST http://localhost:8080/qa -H 'Content-Type: application/json' -d '{"question":"I have trouble sleeping"}'

### How it works
- Ingestion (app/ingest.py): loads knowledge_base.json into Chroma (OpenAI embeddings) and stores a plain corpus.json snapshot for BM25.
- Retrieval (app/retrieval.py): runs semantic search and BM25 (top 25); fuses results with Reciprocal Rank Fusion (RRF), returning top FUSION_K.
- Graph (app/graph.py): LangGraph pipeline
  1) rewrite_query — improves retrievability
  2) retrieve — hybrid retrieval
  3) generate — LLM answers using only provided context, citing recommendation IDs
  4) critic — LLM checks faithfulness and missing citations
  5) revise — optional revision if critique requests it (actor-critic loop)
- API (app/server.py): /health, /ingest, /ask

### Notes
- The assistant is constrained to the provided context and should cite recommendation_id tokens, e.g., [SLEEP_001].
- BM25_TOP_K is set to 25 to satisfy the "batch match 25" requirement.
- For a larger corpus, consider adding chunking and additional metadata filters.


### RAGAS metrics (optional)
If RAGAS is installed and your OpenAI API key is configured, the /qa endpoint will include a metrics object with automatic evaluation scores.

Quick curl to see the scores:

curl --location 'http://127.0.0.1:8080/qa' \
--header 'Content-Type: application/json' \
--data '{"question":"I have trouble sleeping; what should I try?"}'

Interpreting the scores (0.0–1.0 range; higher is better):
- faithfulness: How well the answer is grounded in the retrieved context.
  • >0.8: strong grounding
  • 0.5–0.8: needs human review
  • <0.5: likely ungrounded or hallucinated
- response_relevancy or context_utilization: How relevant the answer is to the question and/or how well it used the provided context. Higher is better.

If metrics are missing, ensure your .env contains OPENAI_API_KEY and that ragas is installed (the Docker setup in this repo already includes it). The UI will still return the answer even when metrics are unavailable.


### Design Questions

1) My Approach
- Retrieval: Hybrid RAG with dense + keyword fusion. I store the KB in Chroma (OpenAI embeddings) for semantic search and also build a BM25 index over the same corpus. At query time I run both and fuse results via Reciprocal Rank Fusion (RRF). This keeps strong recall on short or OOV terms (BM25) and semantic paraphrases (dense).
- Generation: The LLM answers strictly “from context” and is prompted to cite recommendation_id tokens present in the retrieved chunks.
- Query Rewriting: I include a rewrite step to improve retrievability where needed (e.g., expand acronyms, add salient terms).
- Trade-offs considered:
  - Simplicity vs. accuracy: Hybrid + RRF gives robust performance without requiring a heavy cross-encoder reranker; acceptable for small corpora and a take‑home scope.
  - Latency vs. quality: I cap BM25/VECTOR top‑k and the fusion size to keep latency low.
  - Chunking: Disabled by default for the small sample set; for larger corpora we’d chunk with overlap and store metadata.

2) Improving the System (two things with one more day)
- Add a reranking and abstention layer: Use a cross‑encoder reranker (e.g., bge‑reranker) on the fused top‑N to boost precision, and implement calibrated “low confidence” abstention (refuse or ask a clarifying question if top score/faithfulness is below threshold).
- Clarifying interactions and UX polish: If the query is ambiguous or broad, ask a short, targeted follow‑up question before running full retrieval; add inline citation links to the exact context passages.

3) Evaluating Performance
- Retrieval metrics:
  - Recall@k and nDCG@k using a held‑out set of QA pairs with labeled relevant passages.
  - MRR/Precision@k for top‑k relevance.
- Generation metrics:
  - Faithfulness (RAGAS) to ensure grounding to retrieved contexts.
  - Response Relevancy / Context Utilization (RAGAS) to capture topical appropriateness and use of evidence.
  - Citation accuracy: fraction of cited IDs that appear in the used contexts.
- System metrics: latency (p50/p95), cost per question, and failure/abstention rate.

4) Scaling to ~1M medical documents
- Embeddings: Prefer high‑quality but cost‑aware model (e.g., text‑embedding-3-large for best accuracy; text‑embedding-3-small for lower cost/latency). Batch and cache embeddings; periodically refresh.
- Vector DB: A production ANN store like Qdrant/Weaviate (HNSW) or pgvector for simpler ops. Tune efSearch/M for latency/recall; shard by specialty/topic; compress with PQ/IVF‑PQ if needed.
- Keyword index: Dedicated inverted index (OpenSearch/Elasticsearch) for BM25 + filters; keep hybrid fusion (RRF or learned fusion).
- Indexing strategy: Chunk docs with windowed overlap; attach rich metadata (symptom, category, rec_id). Maintain hierarchical/semantic routing (route query to likely shards), and optionally add a cross‑encoder reranker over top‑100.
- Graph signals (optional): Maintain lightweight edges (co‑citation, entity links) and use graph‑aware reweighting for recall on rarer medical terms.
- Trade‑offs: For lowest latency, favor smaller embeddings and tighter HNSW search; for highest accuracy, add reranking and larger embeddings with caching.

5) Handling Ambiguity & Failure
- Ambiguity: Detect broad/underspecified queries (short length + low retrieval confidence) and ask a clarifying follow‑up (e.g., “sleep difficulty: falling asleep or staying asleep?”).
- Out‑of‑scope: If no strong matches, the system abstains: explain it cannot answer from the KB and suggest next steps or categories to explore; never fabricate.
- Guardrails: Keep “answer strictly from provided context,” require citations, and refuse if context is empty or below confidence thresholds.

### Evaluation Criteria: How this submission meets them
- Product Thinking & Safety: The LLM is grounded to retrieved context and required to cite recommendation IDs; optional RAGAS metrics help quantify grounding; abstention path is described.
- Code Quality & Structure: Modular files (ingest, retrieval, graph, server, web UI); clear configuration; Dockerized. Simple to extend with reranking/chunking.
- Communication: Setup instructions, curl examples (including metrics), and design answers are documented here.
- Technical Implementation: The system ingests, retrieves (hybrid RRF), answers with citations, and exposes HTTP endpoints and a minimal UI. Optional metrics are integrated via RAGAS.


### Tests
- Install dev deps (optional): pytest
- Run tests:
  pytest -q

The basic suite validates that:
- /health returns {"status":"ok"}
- /qa returns an answer and retrieved contexts for a sleep-related question
