"""
Pipeline stage: Summary & chapter generation.
Merges transcript + visual captions into a single chronological timeline,
then asks a free-tier text LLM (via OpenRouter) to produce a JSON summary
and chapter markers from it.
"""
import os
import json
import time
import requests
from pathlib import Path
from app.config import PROCESSED_DIR
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Two free-tier OpenRouter models: SUMMARY_MODEL is tried first, and if it
# fails entirely (all its attempts exhausted), SUMMARY_FALLBACK_MODEL is
# tried next. Both are ":free" tier -- expect occasional rate limiting.
SUMMARY_MODEL = "openai/gpt-oss-20b:free"
SUMMARY_FALLBACK_MODEL = "tencent/hy3:free"

SUMMARY_PROMPT_TEMPLATE = """You are given a chronological timeline of a video, combining
speech transcript segments and visual scene descriptions, each with a timestamp in seconds.
Produce a JSON object with exactly this shape and nothing else (no markdown fences, no preamble):
{{
  "summary": "a 3-5 sentence overview of the whole video",
  "chapters": [
    {{"start": <seconds:int>, "title": "<short chapter title>"}}
  ]
}}
Guidelines:
- Chapters should mark meaningful topic/scene changes, not every timestamp.
- Aim for roughly 1 chapter per 1-3 minutes of content, fewer for short videos.
- Titles should be short (3-8 words) and specific enough to be searchable.
TIMELINE:
{timeline}
"""


def _build_timeline(transcript: list[dict], captions: list[dict]) -> str:
    """
    Interleave transcript segments (SPEECH) and keyframe captions (VISUAL)
    into a single chronological, timestamped text block -- this is what
    gets dropped into SUMMARY_PROMPT_TEMPLATE's {timeline} placeholder so
    the LLM sees speech and visuals in the order they actually occurred.
    """
    events = []
    for seg in transcript:
        events.append((seg["start"], f"[{seg['start']:.0f}s] SPEECH: {seg['text']}"))
    for cap in captions:
        events.append((cap["timestamp"], f"[{cap['timestamp']}s] VISUAL: {cap['caption']}"))
    events.sort(key=lambda e: e[0])
    return "\n".join(line for _, line in events)


def _call_model(model: str, prompt: str):
    """Single OpenRouter chat-completion request. Returns the raw response
    object (caller inspects status_code/JSON) rather than raising here, so
    _get_summary_json can decide how to react per status code."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        # Low reasoning effort + excluded from output: we just want the
        # final JSON, not the model's chain-of-thought, and reasoning
        # tokens would eat into the free tier's usage for no benefit here.
        "reasoning": {"effort": "low", "exclude": True},
    }
    return requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)


def _get_summary_json(prompt: str) -> dict:
    """
    Try SUMMARY_MODEL, then SUMMARY_FALLBACK_MODEL, up to 2 attempts each,
    with different retry behavior depending on how a given attempt fails:

      - 429 (rate limited): sleep with linear backoff (5s, 10s) and retry
        the SAME model for the next attempt.
      - 200 but the JSON in the response body doesn't parse: give up on
        this model immediately (break) and move to the fallback model --
        a malformed response is treated as unlikely to fix itself on retry.
      - 200 but no usable `choices[0].message.content` at all (e.g. empty
        completion): retry the SAME model again (continue) before giving up.
      - any other non-200 status: give up on this model immediately (break)
        and move to the fallback model.

    Note the asymmetry between the two "got a 200" failure modes above --
    invalid JSON does NOT retry the same model, but empty content DOES.
    This is intentional in the existing logic (not a bug introduced here),
    but worth knowing if you're debugging why a bad response wasn't retried.

    Raises RuntimeError if every model/attempt combination fails, with the
    last error message included.
    """
    last_error = "unknown error"
    for model in (SUMMARY_MODEL, SUMMARY_FALLBACK_MODEL):
        for attempt in range(2):
            resp = _call_model(model, prompt)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices")
                if choices and choices[0].get("message", {}).get("content"):
                    content = choices[0]["message"]["content"].strip()
                    # Strip markdown code fences in case the model wraps
                    # its JSON in ```json ... ``` despite being told not to.
                    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        last_error = f"{model} -> 200 but invalid JSON: {content[:300]}"
                        print(f"  {model} gave invalid JSON, trying next: {last_error}")
                        break  # don't retry this model again -- move to fallback
                last_error = f"{model} -> 200 but no usable content. Raw response: {json.dumps(data)[:500]}"
                print(f"  {last_error}")
                continue  # retry same model (uses up the 2nd attempt)
            last_error = f"{model} -> {resp.status_code}: {resp.text[:300]}"
            if resp.status_code == 429:
                # Rate limited -- back off and retry the same model rather
                # than immediately burning the fallback model too.
                wait = 5 * (attempt + 1)
                print(f"  {model} rate-limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  {model} failed: {last_error}")
            break  # any other error status -- move straight to fallback

    raise RuntimeError(f"All free summary models failed. Last error: {last_error}")


def generate_summary(transcript: list[dict], captions: list[dict], video_stem: str) -> dict:
    """
    Produce (or load cached) {"summary": str, "chapters": [...]} for a video.
    Cached to processed/<video_stem>/summary.json -- same caching pattern as
    the earlier pipeline stages, so re-running the pipeline on an
    already-processed video skips the LLM call entirely.
    """
    out_path = PROCESSED_DIR / video_stem / "summary.json"
    if out_path.exists():
        print(f"Using cached summary at {out_path}")
        return json.loads(out_path.read_text())
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set.")

    timeline = _build_timeline(transcript, captions)
    prompt = SUMMARY_PROMPT_TEMPLATE.format(timeline=timeline)
    result = _get_summary_json(prompt)

    out_path.write_text(json.dumps(result, indent=2))
    return result
