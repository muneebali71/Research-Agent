from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

# ── API keys 
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── PostgreSQL (runs locally, NOT in Docker) 
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/research_agent",
)
# Sync URL used by the LangGraph Postgres checkpointer (psycopg / psycopg2).
DATABASE_URL_SYNC = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql://postgres:postgres@localhost:5432/research_agent",
)

# ── Qdrant (runs in Docker)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_chunks")

# ── Embeddings (local HuggingFace model) 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  