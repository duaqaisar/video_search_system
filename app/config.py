"""
Central configuration: shared paths and pipeline constants used across the
app (preprocessing, transcription, API, workers, etc.).
"""
from pathlib import Path

# Resolve the project root regardless of where this module is imported from
# (two .parent calls: app/config.py -> app/ -> project root).
BASE_DIR = Path(__file__).resolve().parent.parent

# All runtime data (videos, extracted audio/keyframes, transcripts, etc.)
# lives under data/. This whole directory is gitignored -- it's regenerable
# output, not source code (see .gitignore).
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"        # raw uploaded/downloaded video files
PROCESSED_DIR = DATA_DIR / "processed"  # per-video output: audio, keyframes,
                                         # transcript.json, summary.json, etc.

# Ensure these directories exist at import time, so downstream code
# (preprocessing, transcription, etc.) can assume they're always present
# without each module having to check/create them itself.
for d in (VIDEOS_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Whisper model size used for transcription (see app/core/transcription.py).
# Options: tiny, base, small, medium, large -- larger = more accurate but
# slower and more memory-hungry. "base" is a reasonable speed/accuracy
# tradeoff for local/dev use.
WHISPER_MODEL = "base"

# How often to sample a frame from the video for visual captioning
# (see app/core/preprocessing.py). Smaller interval = more keyframes =
# more detail captured but more vision-model calls (slower, costs more).
KEYFRAME_INTERVAL_SECONDS = 10
