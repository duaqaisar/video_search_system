"""
Video Q&A chatbot: retrieval-augmented Q&A grounded in a specific video's
transcript and visual captions, powered by an LLM via OpenRouter.
"""
import os
import time
import requests
from dotenv import load_dotenv
load_dotenv()
from app.core.qdrant_indexing import search

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# $0 free-tier model on OpenRouter -- no per-token cost, rate-limited
# (20 req/min, 50-1000 req/day depending on account credit history).
# Swap for a paid model (e.g. "anthropic/claude-sonnet-4.5") when budget allows.
CHAT_MODEL = "openai/gpt-oss-20b:free"

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a specific video's content.
You will be given excerpts from the video's transcript and visual descriptions, each with a timestamp.
Answer the student's question using ONLY the provided excerpts. If the excerpts don't contain
enough information to answer, say so honestly rather than guessing.
Always cite the timestamp(s) your answer is based on, like this: (at 45s).
Keep answers concise and directly focused on what was asked.
"""


def _build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks (from qdrant_indexing.search()) into a plain
    text block for the LLM prompt: one line per chunk, labeled Speech or
    Visual with its timestamp, in whatever order search() returned them
    (already ranked by relevance, not necessarily chronological).
    """
    lines = []
    for c in chunks:
        source = "Speech" if c["type"] == "speech" else "Visual"
        lines.append(f"[{c['timestamp']}s] {source}: {c['text']}")
    return "\n".join(lines)


def ask_about_video(question: str, video_id: str, owner_id: str, n_context_chunks: int = 6) -> dict:
    """
    Answer a question about a specific video using RAG: retrieve the most
    relevant transcript/caption chunks for this video (scoped to video_id +
    owner_id, same access-control mechanism as the main /search endpoint --
    see qdrant_indexing.search()), then ask an LLM to answer grounded only
    in those chunks.

    Returns {"answer": str, "sources": [{"timestamp", "type", "text"}, ...]}.
    If nothing has been indexed for this video yet, returns a friendly
    fallback message instead of an empty/confusing answer.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set.")

    chunks = search(question, n_results=n_context_chunks, video_id=video_id, owner_id=owner_id)
    if not chunks:
        return {
            "answer": "I couldn't find any indexed content for this video yet. "
                      "Make sure processing has completed.",
            "sources": [],
        }

    context = _build_context(chunks)
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"VIDEO EXCERPTS:\n{context}\n\nQUESTION: {question}"},
        ],
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Retry-on-429 loop. Unlike summary.py's _get_summary_json (which tries
    # two different models with fixed backoff), this only retries the same
    # single CHAT_MODEL, and uses the rate-limit's own suggested wait time
    # (retry_after_seconds from the error response) instead of a fixed
    # schedule -- OpenRouter tells us how long to wait, so we use that
    # directly rather than guessing. Falls back to 5s if that field is
    # missing from the error payload.
    max_retries = 3
    for attempt in range(max_retries):
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 429 and attempt < max_retries - 1:
            retry_after = resp.json().get("error", {}).get("metadata", {}).get("retry_after_seconds", 5)
            print(f"Rate limited, retrying in {retry_after:.0f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(retry_after)
            continue
        # Either succeeded, or this was the last attempt -- raise_for_status()
        # will raise on a 429 here too if we've exhausted all retries, same
        # as any other non-2xx status.
        resp.raise_for_status()
        break

    answer_text = resp.json()["choices"][0]["message"]["content"].strip()
    return {
        "answer": answer_text,
        "sources": [
            {"timestamp": c["timestamp"], "type": c["type"], "text": c["text"]}
            for c in chunks
        ],
    }
