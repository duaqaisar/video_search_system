# Video Search System

A backend that ingests videos (uploaded files, URLs, or live streams),
understands their content end-to-end — what's said, who said it, and what's
shown on screen — and makes all of it semantically searchable and
chat-able. Built with FastAPI, Celery, and Qdrant.

This README is written to also work as a walkthrough of the implementation:
if you want to understand *why* each model/tool was chosen and how data
flows through the system, read the "Pipeline stages explained" section.

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Pipeline stages explained](#pipeline-stages-explained)
- [API reference](#api-reference)
- [Setup](#setup)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)

## What it does

1. **Ingest** a video — upload a file directly, or submit a URL (YouTube or
   anything [yt-dlp](https://github.com/yt-dlp/yt-dlp) supports).
2. **Process** it through a pipeline that extracts everything searchable
   from both the audio and the visuals (details below).
3. **Search or chat** — semantically search across a video's (or all your
   videos') transcript and visual content, or ask natural-language
   questions grounded in the actual video content (RAG).
4. **Live sessions** — the same transcription + indexing pipeline also runs
   on a live/streaming source (e.g. an RTMP feed), producing a rolling,
   searchable transcript in near-real-time as the stream plays.

Everything is scoped per authenticated user. One user can never search or
see another user's videos, even though all videos share the same underlying
Postgres database and Qdrant vector collection — isolation is enforced
entirely through query-time filters (more on this below, since it's the
single most important security property of the system).

## Architecture

    upload / URL
         |
         v
    +-------------+
    |   FastAPI   |----> Postgres   (videos, jobs, users)
    |  app/api/   |----> S3 / MinIO (raw video storage)
    +------+------+
           | enqueues a Celery task
           v
    +-----------------+
    |  Celery worker  |
    | app/workers/    |
    +--------+--------+
             |
             v
    ================== Processing pipeline (app/core/) ==================
      preprocessing.py   -> audio.wav + periodic keyframes (FFmpeg)
      transcription.py   -> timestamped transcript (Whisper)
      diarization.py     -> "who spoke when" (pyannote)
      merge_speakers.py  -> transcript + speaker labels merged
      vision.py          -> keyframe captions (BLIP)
      summary.py         -> overall summary + chapters (LLM via OpenRouter)
      qdrant_indexing.py -> embed everything, index into Qdrant
    ========================================================================
             |
             v
    +-------------+
    |   Qdrant    |  <--- queried by /search, /chat, /live/.../search
    | (vector DB) |       (all search/chat traffic goes through here)
    +-------------+

Live sessions run a parallel, rolling version of this: `live_ingest.py` uses
ffmpeg to segment a live source into fixed-length audio chunks as they
arrive, and `streaming_transcription.py` transcribes and indexes each chunk
as soon as it's ready — same Whisper model, same Qdrant collection, just
incremental instead of run-once-on-a-complete-file.

## Pipeline stages explained

This section is the "why," not just the "what" — useful if you want to
understand the actual design decisions, not just the file list.

### 1. Preprocessing (`app/core/preprocessing.py`)

Uses FFmpeg to pull two things out of the raw video:

- **Audio**, resampled to mono 16kHz 16-bit PCM. This exact format matches
  what Whisper expects internally — matching it up front avoids Whisper
  silently resampling on every call.
- **Keyframes**, one JPEG every N seconds (configurable via
  `KEYFRAME_INTERVAL_SECONDS` in `app/config.py`, default 10s). These feed
  the visual captioning stage later. A fixed time interval is a simple,
  predictable choice compared to scene-detection-based sampling, at the
  cost of possibly missing very short visual moments between samples.

Every video's outputs are cached under `data/processed/<video_id>/` — the
`video_id` (a UUID assigned at upload time, not the original filename) is
what keeps two different videos from ever colliding in the cache, even if
a user uploads two files with the same name.

### 2. Transcription (`app/core/transcription.py`)

Runs [OpenAI Whisper](https://github.com/openai/whisper) locally (not via
API — no per-request cost, but does require CPU/GPU capacity) to produce
timestamped speech segments. Model size defaults to `base` (see
`app/config.py`) — a deliberate speed/accuracy tradeoff for local
development; `small` or `medium` would give better accuracy at the cost of
slower processing and more memory.

Results are cached to `transcript.json` per video, so re-running the
pipeline on an already-processed video skips Whisper entirely.

### 3. Speaker diarization (`app/core/diarization.py`)

Runs [pyannote.audio](https://github.com/pyannote/pyannote-audio)'s
pretrained speaker-diarization pipeline to figure out *who* was speaking at
each point in time — independent of *what* was said (that's Whisper's job).
This produces speaker-labeled time segments like `SPEAKER_00: 12.4s-18.1s`,
which don't yet have any text attached.

This requires a Hugging Face token with access to the gated
`pyannote/speaker-diarization-3.1` model (see Setup below).

### 4. Merging transcript + speakers (`app/core/merge_speakers.py`)

Whisper's transcript and pyannote's diarization are two independent
outputs on two independent timelines. This stage combines them: for every
transcript segment, it finds which diarization "turn" overlaps it the
most, and assigns that speaker's label to the segment. If a segment
doesn't overlap any speaker turn at all, it's labeled `UNKNOWN` rather than
dropped, so no transcript text is ever silently lost.

### 5. Visual captioning (`app/core/vision.py`)

Runs [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
(a vision-language model) locally on each extracted keyframe, producing a
short caption describing what's visible — on-screen text, slides, actions,
diagrams. This is what makes purely visual content (a slide with no
narration, a diagram someone points at silently) searchable at all, since
Whisper alone would have nothing to transcribe in those moments.

Like Whisper, this runs fully offline after the first model download — no
API cost, no rate limits, at the cost of needing local compute.

### 6. Summarization (`app/core/summary.py`)

Interleaves the transcript and visual captions into one chronological
timeline, then sends that timeline to a free-tier LLM via
[OpenRouter](https://openrouter.ai/) to produce a short overall summary and
a set of chapter markers. Uses a primary model with a fallback model, and
retries on rate-limiting — since free-tier models can be rate-limited more
aggressively than paid ones.

### 7. Indexing (`app/core/qdrant_indexing.py`)

This is the stage that makes everything actually searchable. Every
transcript segment and every visual caption becomes a "chunk," gets
embedded with a sentence-embedding model
([BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)),
and is upserted into a single shared Qdrant collection alongside every
other video from every other user.

Two design decisions worth calling out:

- **Single shared collection, not one collection per user/video.** This
  keeps the system simple to operate, but means the *only* thing
  preventing one user from seeing another user's content is a
  payload filter (`owner_id`, `video_id`) applied at query time, not
  physical separation of data. `search()` in this file explicitly
  requires a non-empty `owner_id` and raises rather than silently
  searching across all users if one isn't provided.
- **Two-stage retrieval.** A search first does fast approximate
  nearest-neighbor search over embeddings to get ~20 candidates, then a
  cross-encoder ([ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2))
  re-scores just those candidates by looking at the query and each
  candidate's text together. This two-stage approach is standard practice
  in retrieval-augmented systems: embeddings alone are fast but
  approximate; cross-encoders are much more accurate but far too slow to
  run over an entire collection, so they're used only to re-rank a small
  shortlist.

### 8. Search & chat (`app/api/main.py`, `app/core/video_chat.py`)

`/search` runs the two-stage retrieval above directly. `/chat` does the
same retrieval, then feeds the top results plus the user's question to an
LLM (again via OpenRouter) with a system prompt instructing it to answer
*only* from the provided excerpts and to cite timestamps — this is a
standard RAG (retrieval-augmented generation) pattern, which keeps answers
grounded in the actual video rather than the model's general knowledge.

### 9. Live sessions (`app/core/live_ingest.py`, `live_sessions.py`, `streaming_transcription.py`)

Runs the same underlying ideas (Whisper transcription + Qdrant indexing) on
a continuously growing audio source instead of one finished file:

- `live_ingest.py` runs ffmpeg to segment a live/RTMP source into
  fixed-length WAV chunks on disk as they arrive, and watches for each
  chunk to finish writing before handing it off.
- `streaming_transcription.py` transcribes each chunk as it lands, keeping
  a running, timestamped transcript, and indexes each new chunk into the
  same Qdrant collection used by uploaded videos — under the live
  session's ID as the `video_id`. This means live sessions are searchable
  with the exact same `search()` function as uploaded videos, with no
  separate search path needed.

## API reference

All endpoints except `/health`, `/auth/signup`, and `/auth/login` require a
Bearer token obtained from signup/login.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/auth/signup` | Create an account, returns an access token |
| POST | `/auth/login` | Log in, returns an access token |
| POST | `/videos` | Upload a video file to process |
| POST | `/videos/from-url` | Submit a video URL to download and process |
| GET | `/videos` | List your videos with latest processing status |
| GET | `/jobs/{job_id}` | Check a processing job's status/stage/result |
| GET | `/search?q=...` | Semantic search across your videos (optionally scoped to one `video_id`) |
| GET | `/chat?video_id=...&question=...` | Ask a question about a specific video (RAG) |
| POST | `/live/start` | Start a live transcription session from a source URL |
| GET | `/live` | List your active live sessions |
| GET | `/live/{session_id}/transcript` | Get a live session's running transcript |
| GET | `/live/{session_id}/search` | Semantic search within a live session |
| POST | `/live/{session_id}/stop` | Stop a live session |

## Setup

### Option A: Docker (recommended)

Requires Docker and Docker Compose.

    cp .env.example .env
    # fill in HF_TOKEN, OPENROUTER_API_KEY, JWT_SECRET in .env
    docker-compose up --build

This starts Postgres, Redis, Qdrant, MinIO, the FastAPI app, and a Celery
worker. The API is then available at `http://localhost:8000`.

Run migrations once, after the containers are up:

    docker-compose exec api alembic upgrade head

### Option B: Local (without Docker)

Requires Python 3.12+, plus local instances of Postgres, Redis, Qdrant, and
an S3-compatible store (e.g. MinIO) already running.

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    cp .env.example .env
    # fill in all values, including DATABASE_URL / S3_* / REDIS_URL /
    # QDRANT_URL if not using the docker-compose defaults

    alembic upgrade head

    # terminal 1
    uvicorn app.api.main:app --reload

    # terminal 2
    celery -A app.workers.celery_app worker --loglevel=info

### Required environment variables

Full explanations are in `.env.example`. At minimum:

- `HF_TOKEN` — Hugging Face token, needed for the gated pyannote
  diarization model
- `OPENROUTER_API_KEY` — used for summary generation and video Q&A
- `JWT_SECRET` — used to sign auth tokens
- `DATABASE_URL`, `S3_*`, `REDIS_URL`, `QDRANT_URL` — service connection
  info (defaults provided match `docker-compose.yml`)

## Project layout

    app/
    +-- api/main.py       FastAPI app: all HTTP endpoints
    +-- auth/             Password hashing, JWT issuing, current-user dependency
    +-- core/             Pipeline stages (see above) + live session logic
    +-- db/               SQLAlchemy models + Alembic migrations
    +-- storage/          S3-compatible object storage client
    +-- workers/          Celery app + task definitions (the real production pipeline)
    scripts/              Standalone dev/utility scripts, not part of the API
    data/                 Runtime output: raw videos, transcripts, embeddings
                          (gitignored -- regenerable by re-running the pipeline)

## Known limitations

- Live sessions are tracked in an in-memory dict
  (`app/core/live_sessions.py`), which works for a single-process
  deployment but would need a shared store (e.g. Redis) to support
  multiple API/worker processes.
- All videos and users share a single Qdrant collection; per-user
  isolation is enforced entirely through query-time payload filters, not
  physical separation of data.
- `app/workers/tasks.py`'s two processing entrypoints (file upload vs.
  URL) share ~90% identical pipeline logic; this is a known, intentionally
  deferred refactor (see the TODO in that file).
