"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", BASE_DIR / "docs"))
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = DATA_DIR / "chroma"
QUERY_LOG = DATA_DIR / "query_log.jsonl"
EVAL_SET = Path(os.getenv("RAG_EVAL_SET", BASE_DIR / "eval" / "questions.json"))
EVAL_RESULTS = DATA_DIR / "eval_results.json"

# --- Embeddings -----------------------------------------------------------
# "default" = ChromaDB's built-in all-MiniLM-L6-v2 (ONNX, runs on CPU, no server)
# "ollama"  = any Ollama embedding model, e.g. nomic-embed-text
EMBED_BACKEND = os.getenv("RAG_EMBED_BACKEND", "default")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")

# --- LLM ------------------------------------------------------------------
# "ollama" = local model through Ollama (default, fully offline)
# "openai" = any OpenAI-compatible API (OpenAI, Groq, LM Studio, ...)
LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", "ollama")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Chunking & retrieval -------------------------------------------------
CHUNK_LINES = int(os.getenv("RAG_CHUNK_LINES", 8))      # lines per chunk
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 2))  # lines shared by neighbours
TOP_K = int(os.getenv("RAG_TOP_K", 5))
# Minimum cosine similarity (0..1) for a chunk to count as evidence.
# If nothing clears this bar the assistant refuses WITHOUT calling the LLM.
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", 0.30))

# --- Security ---------------------------------------------------------------
USERS_FILE = DATA_DIR / "users.json"          # salted PBKDF2 hashes only, never committed
AUDIT_LOG = DATA_DIR / "audit.jsonl"
MAX_UPLOAD_MB = int(os.getenv("RAG_MAX_UPLOAD_MB", 25))
MAX_PAGES = int(os.getenv("RAG_MAX_PAGES", 500))
LOGIN_MAX_ATTEMPTS = int(os.getenv("RAG_LOGIN_MAX_ATTEMPTS", 5))
LOCKOUT_MINUTES = int(os.getenv("RAG_LOCKOUT_MINUTES", 5))
SESSION_TIMEOUT_MINUTES = int(os.getenv("RAG_SESSION_TIMEOUT_MINUTES", 30))
REDACT_LOGS = os.getenv("RAG_REDACT_LOGS", "1") != "0"  # mask emails / phone numbers in query log
INJECTION_GUARD = os.getenv("RAG_INJECTION_GUARD", "1") != "0"

for _d in (DOCS_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)
