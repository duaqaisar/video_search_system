"""
Pipeline stage: Visual understanding.
Captions video keyframes using a local vision-language model (BLIP).
Fully offline after the first model download -- no API keys, no rate
limits, no cost (unlike summary.py, which uses a paid/free API model).
"""
import json
from pathlib import Path
from PIL import Image
from app.config import PROCESSED_DIR

# Guides BLIP toward concrete, searchable captions (on-screen text, slides,
# actions, diagrams) rather than generic scene descriptions. Passed as
# conditional text into the processor in _caption_frame() below.
CAPTION_PROMPT = (
    "Describe what is visible in this video frame in 1-2 concise sentences. "
    "Focus on concrete, searchable details: on-screen text, slides, people, "
    "actions, diagrams, or UI shown. Skip generic filler."
)

# Process-local model cache, same pattern as transcription.py/diarization.py.
_model = None
_processor = None


def _load_model():
    """Loads the BLIP model once, lazily, on first use (not at import time),
    so importing this module doesn't trigger a ~1GB download/load unless
    captioning is actually invoked."""
    global _model, _processor
    if _model is not None:
        return
    print("Loading BLIP vision model (first run downloads weights, ~1GB)...")
    from transformers import BlipProcessor, BlipForConditionalGeneration
    model_id = "Salesforce/blip-image-captioning-base"
    _processor = BlipProcessor.from_pretrained(model_id)
    _model = BlipForConditionalGeneration.from_pretrained(model_id)
    print("BLIP loaded.")


def _caption_frame(image_path: Path) -> str:
    """Generate a caption for a single keyframe image.
    FIXED: previously called _processor(image, return_tensors="pt") with no
    text prompt, so BLIP produced generic unconditional captions instead of
    the concrete/searchable style CAPTION_PROMPT describes. Now passes
    CAPTION_PROMPT as conditional text, which steers BLIP toward describing
    on-screen text, slides, actions, and diagrams -- the actual point of
    this prompt existing in the first place."""
    _load_model()
    image = Image.open(image_path).convert("RGB")
    inputs = _processor(image, CAPTION_PROMPT, return_tensors="pt")
    out = _model.generate(**inputs, max_new_tokens=50)
    return _processor.decode(out[0], skip_special_tokens=True).strip()


def caption_keyframes(keyframes: list[dict], video_stem: str) -> list[dict]:
    """
    Caption every keyframe for a video.
    Returns [{"timestamp": int, "path": Path, "caption": str}, ...].
    Cached to processed/<stem>/captions.json.
    """
    out_path = PROCESSED_DIR / video_stem / "captions.json"

    if out_path.exists():
        print(f"Using cached captions at {out_path}")
        cached = json.loads(out_path.read_text())
        # Paths are stored as strings in JSON; convert back to Path objects
        # to match the return type of the non-cached branch below.
        for c in cached:
            c["path"] = Path(c["path"])
        return cached

    captioned = []
    for i, kf in enumerate(keyframes):
        print(f"  Captioning frame {i+1}/{len(keyframes)} (t={kf['timestamp']}s)...")
        caption = _caption_frame(kf["path"])
        captioned.append({
            "timestamp": kf["timestamp"],
            "path": str(kf["path"]),  # stringify for JSON serialization
            "caption": caption,
        })

    out_path.write_text(json.dumps(captioned, indent=2))

    # Convert paths back to Path objects before returning, so callers get
    # the same type whether this came from cache or was just computed.
    for c in captioned:
        c["path"] = Path(c["path"])
    return captioned
