"""
Pipeline stage: Preprocessing
Extracts the audio track and periodic keyframes from a video using FFmpeg.
These are the raw inputs for transcription (audio) and visual captioning
(keyframes) later in the pipeline.

Note on caching keys: this file, transcription.py, and diarization.py all
cache their output under PROCESSED_DIR / video_stem / ..., derived from
video_path.stem. This is safe from collisions because every caller (see
app/workers/tasks.py) saves the local video file as "{video_id}{ext}"
before passing it in -- so video_stem is always the video's UUID, never
the user's original filename. Don't call these functions directly with an
arbitrary user-named file path without being aware of this assumption.
"""
import subprocess
from pathlib import Path
from app.config import PROCESSED_DIR, KEYFRAME_INTERVAL_SECONDS


def extract_audio(video_path: Path) -> Path:
    """
    Extract mono 16kHz WAV audio from the video.
    This format (mono, 16kHz, 16-bit PCM) is Whisper's preferred input --
    matching it exactly avoids implicit resampling before transcription.
    """
    video_path = Path(video_path)
    out_dir = PROCESSED_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "audio.wav"

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",                  # strip video stream, audio only
        "-ac", "1",             # downmix to mono
        "-ar", "16000",         # resample to 16kHz
        "-acodec", "pcm_s16le", # uncompressed 16-bit PCM
        str(audio_path),
    ]
    # capture_output=True suppresses ffmpeg's verbose stderr logging unless
    # it actually fails (check=True raises CalledProcessError on non-zero exit).
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def extract_keyframes(video_path: Path, interval: int = KEYFRAME_INTERVAL_SECONDS) -> list[dict]:
    """
    Extract one JPEG frame every `interval` seconds for visual captioning.
    Returns a list of {"timestamp": seconds, "path": Path} dicts, one per
    extracted frame.

    Note: "timestamp" is computed as index * interval rather than read from
    the actual frame metadata. This assumes ffmpeg's fps filter samples
    starting at 0s with no drift, which holds for standard video files.
    """
    video_path = Path(video_path)
    out_dir = PROCESSED_DIR / video_path.stem / "keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)

    # %04d gives frame_0001.jpg, frame_0002.jpg, ... (sortable, up to 9999 frames)
    pattern = str(out_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps=1/{interval}",  # sample 1 frame every `interval` seconds
        "-q:v", "2",                  # high JPEG quality (2 = near-lossless, scale is 2-31)
        pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Glob + sort to get frames back in chronological order (string sort
    # works here because %04d zero-pads consistently).
    frames = sorted(out_dir.glob("frame_*.jpg"))
    keyframes = []
    for i, frame_path in enumerate(frames):
        keyframes.append({
            "timestamp": i * interval,
            "path": frame_path,
        })
    return keyframes


if __name__ == "__main__":
    # Manual CLI usage for testing preprocessing in isolation:
    #   python -m app.core.preprocessing path/to/video.mp4
    import sys
    video = Path(sys.argv[1])
    print(f"Extracting audio from {video}...")
    audio_path = extract_audio(video)
    print(f"  -> {audio_path}")

    print(f"Extracting keyframes every {KEYFRAME_INTERVAL_SECONDS}s...")
    kf = extract_keyframes(video)
    print(f"  -> {len(kf)} keyframes in {kf[0]['path'].parent if kf else 'N/A'}")
