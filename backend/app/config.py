"""Application configuration from environment."""
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # LLM: auto = OpenAI → Hugging Face → Ollama (free local, no keys)
    llm_backend: str = "auto"  # auto | openai | huggingface | ollama
    # Hugging Face Inference API (https://huggingface.co/settings/tokens)
    huggingfacehub_api_token: str = ""
    hf_chat_repo_id: str = "HuggingFaceH4/zephyr-7b-beta"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Ollama — free local LLM (https://ollama.com/) — no API cost
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    # Vision model for image ingestion (ollama pull llava or llava-phi3, etc.)
    ollama_vision_model: str = "llava"
    # Stop slow vision calls (CPU can take minutes); falls back to OCR / placeholder
    ollama_vision_timeout_sec: int = 90
    # Set false to skip llava on images (fast upload; use OCR only if tesseract installed)
    use_ollama_vision: bool = True

    # SentenceTransformers fallback when no OpenAI key for embeddings (dev only)
    use_sentence_transformers: bool = False
    st_model_name: str = "all-MiniLM-L6-v2"

    # Override filesystem root for tenant data (FAISS, uploads, registry). Env: DATA_ROOT.
    # Default: repo-root `data/` next to `backend/`. On Render, set to /tmp/... or a mounted disk.
    data_root: Path | None = None

    @field_validator("data_root", mode="before")
    @classmethod
    def _empty_data_root(cls, v: object) -> object:
        if v is None or v == "":
            return None
        return v

    @property
    def data_dir(self) -> Path:
        if self.data_root is not None:
            return Path(self.data_root).expanduser().resolve()
        return Path(__file__).resolve().parent.parent.parent / "data"

    # When set, all mutating routes require X-API-Key or Authorization: Bearer
    api_key: str = ""

    default_top_k: int = 5
    max_upload_mb: int = 100

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
