"""
Pipeline stage: Merge transcript + diarization.
Combines Whisper's transcript segments (what was said) with pyannote's
diarization turns (who was speaking when) into a single speaker-labeled
transcript. For each transcript segment, assigns whichever speaker turn
overlaps it the most.

Unlike transcribe() and diarize() (see transcription.py, diarization.py),
this function does NOT check for a cached output file before running --
it always recomputes and overwrites transcript_with_speakers.json. This is
fine in practice since the pipeline (app/workers/tasks.py) only calls it
once per video, but means re-running this function manually will always
redo the merge, even if nothing changed.
"""
import json
from pathlib import Path
from app.config import PROCESSED_DIR


def _overlap(a_start, a_end, b_start, b_end):
    """
    Compute the overlap (in seconds) between two time intervals [a_start,
    a_end] and [b_start, b_end]. Returns 0.0 if they don't overlap at all
    (clamped with max(0.0, ...) rather than going negative).
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge_transcript_with_speakers(video_stem: str) -> list[dict]:
    """
    Read the cached transcript.json and diarization.json for a video
    (produced earlier in the pipeline by transcribe() and diarize()) and
    merge them into a single list of speaker-labeled segments.

    For each transcript segment, every diarization turn is checked and the
    turn with the greatest time overlap "wins" -- its speaker label is
    assigned to that segment. This is an O(segments x turns) scan; fine
    for typical video lengths, but would need a smarter approach (e.g.
    sorting + a sliding window) if videos got long enough to have
    thousands of segments/turns.

    If a segment doesn't overlap any diarization turn at all (e.g. a gap
    pyannote didn't attribute to any speaker), it's labeled "UNKNOWN"
    rather than dropped, so no transcript text is ever silently lost.

    Returns: [{"start": float, "end": float, "speaker": str, "text": str}, ...]
    Writes the result to processed/<video_stem>/transcript_with_speakers.json
    and also returns it directly.
    """
    transcript_path = PROCESSED_DIR / video_stem / "transcript.json"
    diarization_path = PROCESSED_DIR / video_stem / "diarization.json"
    out_path = PROCESSED_DIR / video_stem / "transcript_with_speakers.json"

    transcript = json.loads(transcript_path.read_text())
    turns = json.loads(diarization_path.read_text())

    merged = []
    for seg in transcript:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for turn in turns:
            ov = _overlap(seg["start"], seg["end"], turn["start"], turn["end"])
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = turn["speaker"]
        merged.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": best_speaker,
            "text": seg["text"],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2))
    return merged


if __name__ == "__main__":
    # Manual CLI usage for testing the merge in isolation, assuming
    # transcribe() and diarize() have already been run for this video_stem:
    #   python -m app.core.merge_speakers <video_stem>
    import sys
    stem = sys.argv[1]
    merged = merge_transcript_with_speakers(stem)
    for m in merged[:25]:
        print(f"[{m['start']:.1f}s] {m['speaker']}: {m['text']}")
    print(f"... {len(merged)} segments total")
