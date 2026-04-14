"""Embeddings: OpenAI, Hugging Face Inference API, or (dev-only) SentenceTransformers."""
from functools import lru_cache
import os

from langchain_core.embeddings import Embeddings

from app.config import get_settings


class _STEmbeddings(Embeddings):
    """SentenceTransformers wrapper for LangChain."""

    def __init__(self, model_name: str) -> None:
        # Heavy dependency (torch); keep behind a flag so production deploys can avoid it.
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
    if settings.openai_api_key and not settings.use_sentence_transformers:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url,
        )

    token = _hf_token()
    if token and not settings.use_sentence_transformers:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        return HuggingFaceEndpointEmbeddings(
            model=settings.hf_embedding_model,
            task="feature-extraction",
            huggingfacehub_api_token=token,
        )

    if settings.use_sentence_transformers:
        return _STEmbeddings(settings.st_model_name)

    raise RuntimeError(
        "No embeddings configured. Set OPENAI_API_KEY (OpenAI embeddings) or "
        "HUGGINGFACEHUB_API_TOKEN/HF_TOKEN (Hugging Face embeddings), or set "
        "USE_SENTENCE_TRANSFORMERS=true for dev-only local embeddings."
    )
