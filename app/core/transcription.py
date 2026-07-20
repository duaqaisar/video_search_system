"""
Pipeline stage: Transcription
Runs OpenAI Whisper on extracted audio and returns timestamped segments.
"""
import json
from pathlib import Path
import whisper
from app.config import WHISPER_MODEL, PROCESSED_DIR

# Process-local cache so repeated calls within the same worker process don't
# reload the model from disk every time. Note: under Celery's default fork
# pool, each worker process gets its OWN copy of this cache -- meaning the
# Whisper model is loaded into memory once per worker process, not once
# globally. Factor this into memory sizing when scaling worker concurrency.
_model_cache = {}


def _get_model(model_size: str = WHISPER_MODEL):
    """Load (and cache) a Whisper model by size. First call per process
    downloads the weights if not already present locally."""
    if model_size not in _model_cache:
        print(f"Loading Whisper model '{model_size}' (first run downloads weights)...")
        _model_cache[model_size] = whisper.load_model(model_size)
    return _model_cache[model_size]


def transcribe(audio_path: Path, video_stem: str, model_size: str = WHISPER_MODEL) -> list[dict]:
    """
    Transcribe audio to timestamped segments.
    Returns: [{"start": float, "end": float, "text": str}, ...]

    Results are cached to disk as transcript.json, keyed only by video_stem --
    re-running transcribe() for the same video_stem returns the cached file
    immediately without touching Whisper at all, even if model_size differs
    from the run that produced the cache. This makes iterating on later
    pipeline stages (diarization, summary, etc.) fast, but means changing
    WHISPER_MODEL for an already-processed video won't have any effect
    unless the cached transcript.json is deleted first.
    """
    audio_path = Path(audio_path)
    out_path = PROCESSED_DIR / video_stem / "transcript.json"

    if out_path.exists():
        print(f"Using cached transcript at {out_path}")
        return json.loads(out_path.read_text())

    model = _get_model(model_size)
    result = model.transcribe(str(audio_path), verbose=False)

    # Whisper returns richer segment data (token ids, confidence, etc.) --
    # we only keep start/end/text, which is all downstream stages need.
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
    ]

    out_path.write_text(json.dumps(segments, indent=2))
    return segments


if __name__ == "__main__":
    # Manual CLI usage for testing transcription in isolation:
    #   python -m app.core.transcription path/to/audio.wav [video_stem]
    import sys
    audio = Path(sys.argv[1])
    stem = sys.argv[2] if len(sys.argv) > 2 else audio.parent.name
    segs = transcribe(audio, stem)
    for s in segs[:5]:
        print(f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}")
    print(f"... {len(segs)} segments total")
