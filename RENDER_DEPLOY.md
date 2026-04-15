# Deploying Contextra on Render — Quick Fix Guide

## What was broken & what was fixed

| Issue | Root Cause | Fix |
|---|---|---|
| `Failed to fetch` on query | No LLM or embeddings configured — backend crashed silently | Added **Groq** support; embeddings auto-fall to SentenceTransformers |
| CORS errors | `CORS_ORIGINS` defaulted to `localhost` only | Default is now `*`; Render env var confirmed |
| Embeddings crash | Required OpenAI or HF token — neither set | SentenceTransformers now auto-activates as fallback |

---

## Step 1 — Get a free Groq API key

1. Go to https://console.groq.com/keys
2. Create a new key (free tier: 14,400 tokens/min, no credit card)
3. Copy the key — you'll add it to Render next

---

## Step 2 — Set Render environment variables (Backend service)

In your **`contextra-b`** Render service → **Environment**:

```
LLM_BACKEND=auto
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx   ← your key from Step 1
GROQ_CHAT_MODEL=llama-3.3-70b-versatile
CORS_ORIGINS=*
DATA_ROOT=/tmp/contextra_data
USE_SENTENCE_TRANSFORMERS=true
```

> **Note:** `USE_SENTENCE_TRANSFORMERS=true` makes embeddings work for free on Render 
> without any API key. The model (`all-MiniLM-L6-v2`, ~90 MB) downloads on first boot.

### Optional — use a different Groq model

| Model | Best for |
|---|---|
| `llama-3.3-70b-versatile` | Best quality (default) |
| `llama-3.1-8b-instant` | Fastest, lowest latency |
| `mixtral-8x7b-32768` | Long context |
| `gemma2-9b-it` | Lightweight |

---

## Step 3 — Frontend env (no change needed)

Your `frontend/.env` already points to `https://contextra-b.onrender.com`. ✅

---

## Step 4 — Push & redeploy

```bash
git add -A
git commit -m "fix: add Groq LLM support + fix embeddings + fix CORS"
git push
```

Render auto-deploys on push. Backend cold start takes ~60s the first time 
(SentenceTransformers model download). Subsequent starts are fast.

---

## Local development with Ollama

You already have Ollama installed on your Mac. For local dev:

```bash
# Pull models (one-time)
ollama pull llama3.2
ollama pull nomic-embed-text   # optional — for ollama embeddings

# In backend/.env, set:
# LLM_BACKEND=ollama
# OLLAMA_BASE_URL=http://127.0.0.1:11434
# OLLAMA_MODEL=llama3.2
```

Then start the backend normally — it'll use your local Ollama.

---

## Architecture after fix

```
Frontend (Render)
    │
    │ HTTPS  →  CORS_ORIGINS=*  ✅
    ▼
Backend (Render)
    ├── LLM:        Groq API  (llama-3.3-70b-versatile)
    │               via OpenAI-compatible endpoint
    ├── Embeddings: SentenceTransformers (local, all-MiniLM-L6-v2)
    │               no API key needed
    └── Vector DB:  FAISS (in-memory + /tmp persistence)
```
