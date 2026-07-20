"""
Stage 1: Preprocessing + Transcription
Extracts audio and keyframes from a video, then transcribes the audio
with timestamps using Whisper.

Usage:
    python process_video_transcript.py data/videos/sample_video.mp4
"""
import sys
from pathlib import Path
from app.core.preprocessing import extract_audio, extract_keyframes
from app.core.transcription import transcribe


def main(video_path: str):
    video = Path(video_path)
    if not video.exists():
        print(f"Error: video file not found at {video}")
        sys.exit(1)

    stem = video.stem

    print("=== Preprocessing ===")
    print("Extracting audio...")
    audio_path = extract_audio(video)
    print(f"  -> {audio_path}")

    print("Extracting keyframes...")
    keyframes = extract_keyframes(video)
    print(f"  -> {len(keyframes)} keyframes extracted")

    print("\n=== Transcription ===")
    print("Running Whisper (first run downloads the model — may take a minute)...")
    segments = transcribe(audio_path, stem)
    print(f"{len(segments)} segments transcribed\n")

    for s in segments[:5]:
        print(f"  [{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}")
    if len(segments) > 5:
        print(f"  ... and {len(segments) - 5} more segments")

    print(f"\nFull transcript saved to: data/processed/{stem}/transcript.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_video_transcript.py <path_to_video>")
        sys.exit(1)
    main(sys.argv[1])
