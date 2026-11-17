"""Centralized configuration.

All settings are read from environment variables or from the `backend/.env` file.
Import the shared `settings` object anywhere you need configuration:

    from app.config import settings
    print(settings.gemini_model)
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ directory (one level above this file's folder)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings. Field names map to UPPER_CASE env variables."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- App ----
    app_name: str = "Enterprise Knowledge Assistant"
    environment: str = "development"
    debug: bool = False

    # Frontend origins allowed to call the API (comma-separated in .env).
    cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
    )

    # ---- Gemini LLM ----
    # Optional for now so the server can start without a key.
    # SecretStr keeps the key hidden in logs and error messages.
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_temperature: float = 0.1     # low = factual, less creative
    gemini_max_output_tokens: int = 1024  # caps cost and rambling answers
    gemini_timeout_seconds: int = 60
    gemini_max_retries: int = 2         # retries for temporary errors (429/503/504)
    gemini_retry_base_delay: float = 1.0  # seconds; doubles each retry

    # ---- Embeddings ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- ChromaDB ----
    chroma_persist_dir: Path = BASE_DIR / "chroma_data"
    chroma_collection_name: str = "enterprise_knowledge"

    # ---- Chunking (measured in characters) ----
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ---- Uploads ----
    max_upload_size_mb: int = 10

    # ---- Retrieval ----
    default_top_k: int = 3              # small context = fewer Gemini tokens
    max_top_k: int = 20
    # Drop chunks whose cosine distance is above this (0 = identical, 1 = unrelated).
    # Lenient on purpose: Gemini still decides whether evidence is sufficient.
    retrieval_max_distance: float = 0.9

    # ---- Multi-agent routing ----
    routing_threshold: float = 0.35     # min cosine similarity for a specialist agent
    routing_margin: float = 0.08        # must be within this of the clause's best agent


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (the .env file is read only once)."""
    return Settings()


settings = get_settings()
