"""
Stage 2: Visual Understanding + Summary Generation
Captions video keyframes using a vision-language model (Qwen2.5-VL via
OpenRouter), then combines those captions with the transcript to produce
an overall summary and timestamped chapter markers.

Requires OPENROUTER_API_KEY to be set (see .env).

Usage:
    python generate_video_summary.py data/videos/client_test_clip.mp4
"""
import sys
import json
from pathlib import Path
from app.core.preprocessing import extract_keyframes
from app.core.vision import caption_keyframes
from app.core.summary import generate_summary
from app.config import PROCESSED_DIR


def main(video_path: str):
    video = Path(video_path)
    if not video.exists():
        print(f"Error: video file not found at {video}")
        sys.exit(1)

    stem = video.stem
    transcript_path = PROCESSED_DIR / stem / "transcript.json"

    if not transcript_path.exists():
        print(f"Error: no transcript found for '{stem}'.")
        print("Run process_video_transcript.py on this video first.")
        sys.exit(1)

    transcript = json.loads(transcript_path.read_text())

    print("=== Visual Understanding ===")
    keyframes = extract_keyframes(video)
    print(f"Captioning {len(keyframes)} keyframes via Qwen2.5-VL...")
    captions = caption_keyframes(keyframes, stem)
    for c in captions:
        print(f"  [{c['timestamp']}s] {c['caption']}")

    print("\n=== Summary & Chapters ===")
    result = generate_summary(transcript, captions, stem)
    print(f"\nSummary:\n{result['summary']}\n")
    print("Chapters:")
    for ch in result["chapters"]:
        print(f"  [{ch['start']}s] {ch['title']}")

    print(f"\nSaved to: data/processed/{stem}/captions.json")
    print(f"Saved to: data/processed/{stem}/summary.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_video_summary.py <path_to_video>")
        sys.exit(1)
    main(sys.argv[1])
