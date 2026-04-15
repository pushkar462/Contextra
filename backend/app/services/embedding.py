"""Embeddings: Groq (uses OpenAI-compatible, but Groq doesn't provide embeddings),
so for embeddings we use:
  1. OpenAI embeddings  (if OPENAI_API_KEY set)
  2. HuggingFace Inference API (if HF_TOKEN set)
  3. Sentence Transformers locally (USE_SENTENCE_TRANSFORMERS=true or fallback)

When using Groq for LLM, embeddings fall through to option 2 or 3.
Set USE_SENTENCE_TRANSFORMERS=true on Render for zero-cost local embeddings.
"""
from functools import lru_cache
import os

from langchain_core.embeddings import Embeddings

from app.config import get_settings


class _STEmbeddings(Embeddings):
    """SentenceTransformers wrapper for LangChain."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()


def _hf_token() -> str:
    s = get_settings()
    return (s.huggingfacehub_api_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()


@lru_cache
def get_embeddings() -> Embeddings:
    settings = get_settings()

    # 1. OpenAI embeddings
    if settings.openai_api_key and not settings.use_sentence_transformers:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url,
        )

    # 2. HuggingFace Inference API embeddings
    token = _hf_token()
    if token and not settings.use_sentence_transformers:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        return HuggingFaceEndpointEmbeddings(
            model=settings.hf_embedding_model,
            task="feature-extraction",
            huggingfacehub_api_token=token,
        )

    # 3. Local SentenceTransformers (free, works on Render, no API key needed)
    # Triggered by USE_SENTENCE_TRANSFORMERS=true  OR  when no API keys at all
    # (Groq doesn't offer embeddings, so this is the right fallback)
    return _STEmbeddings(settings.st_model_name)
