"""ONNX-based embeddings via fastembed — no torch, fits in 512MB.

Uses BAAI/bge-small-en-v1.5: 384 dims, ~50MB model, ~80MB in RAM.
Lazy-loaded on first embed call so /health passes instantly.

NOTE: Switching embedding models invalidates existing FAISS indexes.
      Re-upload your sources after deploying this change.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
from typing import Optional

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class _FastEmbedWrapper(Embeddings):
    """LangChain Embeddings interface over fastembed. Lazy-loaded, thread-safe."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            logger.info("Loading fastembed model: %s", self._model_name)
            from fastembed import TextEmbedding
            cache_dir = os.getenv("FASTEMBED_CACHE_DIR", "/tmp/fastembed_cache")
            self._model = TextEmbedding(
                model_name=self._model_name,
                cache_dir=cache_dir,
                threads=1,
            )
            logger.info("fastembed model ready (cache=%s).", cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        batch_size = int(os.getenv("EMBED_BATCH_SIZE", "16"))
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            vecs = list(self._model.embed(chunk))
            out.extend(v.tolist() for v in vecs)
        gc.collect()
        return out

    def embed_query(self, text: str) -> list[float]:
        self._ensure_model()
        vec = next(iter(self._model.embed([text])))
        return vec.tolist()


_singleton: Optional[_FastEmbedWrapper] = None
_singleton_lock = threading.Lock()


def get_embeddings() -> Embeddings:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                model_name = os.getenv(
                    "FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"
                )
                _singleton = _FastEmbedWrapper(model_name)
    return _singleton