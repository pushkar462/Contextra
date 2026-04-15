"""LLM factory: Groq, OpenAI, Hugging Face text-generation, or free local Ollama.

Priority (auto mode):
  1. Groq  (fast, free tier — set GROQ_API_KEY)
  2. OpenAI (set OPENAI_API_KEY)
  3. Hugging Face (set HUGGINGFACEHUB_API_TOKEN)
  4. Ollama (local, no key needed)
"""
from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.config import get_settings


def _hf_token() -> str:
    s = get_settings()
    return (s.huggingfacehub_api_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()


def _groq_key() -> str:
    return (os.getenv("GROQ_API_KEY") or "").strip()


def is_llm_configured() -> bool:
    s = get_settings()
    mode = (s.llm_backend or "auto").lower()
    if mode == "groq":
        return bool(_groq_key())
    if mode == "openai":
        return bool(s.openai_api_key)
    if mode == "huggingface":
        return bool(_hf_token())
    if mode == "ollama":
        return True
    return True


def get_llm() -> tuple[str, Runnable]:
    """
    Returns (kind, runnable) where kind is:
    - "chat"  for chat-models (Groq, OpenAI, Ollama)
    - "text"  for text-generation models (Hugging Face)
    """
    settings = get_settings()
    mode = (settings.llm_backend or "auto").lower()

    if mode == "auto":
        if _groq_key():
            mode = "groq"
        elif settings.openai_api_key:
            mode = "openai"
        elif _hf_token():
            mode = "huggingface"
        else:
            mode = "ollama"

    if mode == "groq":
        key = _groq_key()
        if not key:
            raise RuntimeError("GROQ_API_KEY is required when LLM_BACKEND=groq.")
        from langchain_openai import ChatOpenAI
        groq_model = os.getenv("GROQ_CHAT_MODEL") or "llama-3.3-70b-versatile"
        return (
            "chat",
            ChatOpenAI(
                model=groq_model,
                api_key=key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.2,
                max_tokens=1024,
            ),
        )

    if mode == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_BACKEND=openai.")
        from langchain_openai import ChatOpenAI
        return (
            "chat",
            ChatOpenAI(
                model=settings.openai_chat_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                temperature=0.2,
            ),
        )

    if mode == "huggingface":
        token = _hf_token()
        if not token:
            raise RuntimeError(
                "Hugging Face token required: set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN."
            )
        from langchain_huggingface.llms import HuggingFaceEndpoint
        llm = HuggingFaceEndpoint(
            repo_id=settings.hf_chat_repo_id,
            huggingfacehub_api_token=token,
            task="text-generation",
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=True,
        )
        return ("text", llm)

    if mode == "ollama":
        from langchain_ollama import ChatOllama
        return (
            "chat",
            ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0.2,
            ),
        )

    raise RuntimeError(
        f"Unknown LLM_BACKEND: {settings.llm_backend!r} (use auto, groq, openai, huggingface, ollama)"
    )


def get_chat_llm() -> BaseChatModel:
    kind, llm = get_llm()
    if kind != "chat":
        raise RuntimeError(
            "This operation requires a chat model. Set LLM_BACKEND=groq, openai, or ollama."
        )
    return llm  # type: ignore[return-value]
