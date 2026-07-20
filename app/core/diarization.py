"""
Pipeline stage: Speaker diarization ("who spoke when").
Uses pyannote.audio's pretrained pipeline to produce speaker-labeled time
segments, which merge_speakers.py later combines with the Whisper transcript.
"""
import os
import json
from pathlib import Path
from app.config import PROCESSED_DIR

# Process-local cache, same rationale as transcription.py's _model_cache:
# avoids reloading the (large) pyannote pipeline on every call within a
# worker process, at the cost of one copy in memory per forked worker.
_pipeline_cache = None


def _get_pipeline():
    """Load (and cache) the pyannote speaker-diarization pipeline.
    Requires a Hugging Face token with access to the gated
    pyannote/speaker-diarization-3.1 model (see .env.example)."""
    global _pipeline_cache
    if _pipeline_cache is None:
        from pyannote.audio import Pipeline
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN not set. See setup instructions.")
        print("Loading pyannote speaker-diarization pipeline...")
        _pipeline_cache = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    return _pipeline_cache


def diarize(
    audio_path: Path,
    video_stem: str,
    num_speakers: int | None = None,
    min_speakers: int = 1,
    max_speakers: int = 6,
) -> list[dict]:
    """
    Identify speaker turns in the audio.
    Returns: [{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]
    Cached to processed/<stem>/diarization.json.

    If num_speakers is known ahead of time, pass it for more accurate
    results. Otherwise the pipeline estimates the speaker count within
    [min_speakers, max_speakers].
    """
    audio_path = Path(audio_path)
    out_path = PROCESSED_DIR / video_stem / "diarization.json"

    if out_path.exists():
        print(f"Using cached diarization at {out_path}")
        return json.loads(out_path.read_text())

    pipeline = _get_pipeline()

    # Load the waveform manually and pass it as a tensor dict rather than
    # a file path -- gives explicit control over the audio format fed to
    # the model instead of relying on pyannote's own (ffmpeg-based) file
    # loading, and lets us feed it audio.wav's known format directly.
    import soundfile as sf
    import torch as _torch
    waveform, sample_rate = sf.read(str(audio_path), dtype='float32')

    # pyannote expects shape (channels, samples). soundfile returns a 1D
    # array for mono audio and 2D (samples, channels) for multi-channel --
    # handle both cases to end up with the right shape either way.
    waveform = _torch.from_numpy(waveform).unsqueeze(0) if waveform.ndim == 1 else _torch.from_numpy(waveform.T)

    if num_speakers is not None:
        diarization = pipeline({'waveform': waveform, 'sample_rate': sample_rate}, num_speakers=num_speakers)
    else:
        diarization = pipeline(
            {'waveform': waveform, 'sample_rate': sample_rate},
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    # NOTE: pyannote/speaker-diarization-3.1's pipeline returns a
    # multi-output object, not the annotation directly -- the actual
    # diarization result lives on the .speaker_diarization attribute.
    # (Older pyannote versions returned the annotation itself here.)
    segments = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(segments, indent=2))
    return segments


if __name__ == "__main__":
    # Manual CLI usage for testing diarization in isolation:
    #   python -m app.core.diarization path/to/audio.wav [video_stem]
    import sys
    audio = Path(sys.argv[1])
    stem = sys.argv[2] if len(sys.argv) > 2 else audio.parent.name
    segs = diarize(audio, stem)
    for s in segs[:20]:
        print(f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['speaker']}")
    print(f"... {len(segs)} turns total")
