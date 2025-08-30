import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "knowledge_base")
KNOWLEDGE_JSON_PATH = os.getenv("KNOWLEDGE_JSON_PATH", "knowledge_base/knowledge_base.json")
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "25"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
FUSION_K = int(os.getenv("FUSION_K", "8"))
RRF_K = int(os.getenv("RRF_K", "60"))
MAX_GRAPH_ITERS = int(os.getenv("MAX_GRAPH_ITERS", "2"))
PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "0.0.0.0")

# Optional enhancements
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "0"))  # 0 disables chunking
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

