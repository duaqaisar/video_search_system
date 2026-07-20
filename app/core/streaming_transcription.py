"""
Streaming transcription: processes a continuously growing audio source in
rolling chunks instead of one complete file, for live video / meeting use.

Works with any audio source that can hand us fixed-duration WAV chunks --
a live video source (ffmpeg tailing an RTMP stream), a meeting bot's
captured audio, or a mic feed can all plug into this the same way.
"""
import time
import json
from pathlib import Path
from datetime import datetime, timezone

from app.core.transcription import _get_model
from app.core.qdrant_indexing import index_video
from app.config import WHISPER_MODEL, PROCESSED_DIR

CHUNK_SECONDS = 20  # how much audio to accumulate before transcribing


class StreamingTranscriber:
    """
    Feed it audio chunks (raw WAV file paths) as they become available;
    it transcribes each chunk and keeps a running, timestamped transcript.

    Usage:
        st = StreamingTranscriber(session_id="meeting_123")
        st.add_chunk(wav_path)   # call this every ~CHUNK_SECONDS as new audio lands
        st.get_transcript()      # full transcript so far
    """

    def __init__(
        self,
        session_id: str,
        model_size: str = WHISPER_MODEL,
        language: str | None = None,
        owner_id: str | None = None,
    ):
        self.session_id = session_id
        self.model = _get_model(model_size)
        self.segments: list[dict] = []
        self._elapsed_seconds = 0.0
        self._out_dir = PROCESSED_DIR / "live_sessions" / session_id
        self._out_dir.mkdir(parents=True, exist_ok=True)
        # Lock language after the first chunk (or if given upfront) so later
        # chunks don't mis-detect language on ambiguous short audio and
        # produce garbled output in the wrong script.
        self._language = language
        self._prev_tail_text = ""  # last bit of previous chunk, for context continuity
        # Live sessions currently have no per-user auth/ownership wired in
        # (unlike uploaded videos, which get owner_id from the authenticated
        # user -- see app/api/main.py). Defaults to None until that's added;
        # passed through to index_video() below so chunks are still indexed
        # (just without owner-based access scoping).
        self.owner_id = owner_id

    def add_chunk(self, wav_chunk_path: Path) -> list[dict]:
        """
        Transcribe one chunk of audio, offset its timestamps by how much
        audio has already been processed in this session, and append to
        the running transcript. Returns just the new segments from this chunk.
        """
        wav_chunk_path = Path(wav_chunk_path)
        result = self.model.transcribe(
            str(wav_chunk_path),
            verbose=False,
            language=self._language,          # None on first chunk = auto-detect once
            initial_prompt=self._prev_tail_text or None,  # carries context across chunks
        )

        if self._language is None:
            self._language = result.get("language")
            print(f"  Locked transcription language to '{self._language}' after first chunk")

        if result["segments"]:
            self._prev_tail_text = result["segments"][-1]["text"].strip()

        new_segments = []
        chunk_duration = 0.0
        for seg in result["segments"]:
            offset_seg = {
                "start": seg["start"] + self._elapsed_seconds,
                "end": seg["end"] + self._elapsed_seconds,
                "text": seg["text"].strip(),
                "received_at": datetime.now(timezone.utc).isoformat(),
            }
            new_segments.append(offset_seg)
            chunk_duration = max(chunk_duration, seg["end"])

        self.segments.extend(new_segments)
        # Advance the running clock by this chunk's actual transcribed
        # duration, or CHUNK_SECONDS as a fallback if Whisper returned no
        # segments at all (e.g. silence) -- keeps later chunks' timestamps
        # roughly aligned with real elapsed time either way.
        self._elapsed_seconds += chunk_duration if chunk_duration else CHUNK_SECONDS
        self._persist()

        if new_segments:
            try:
                # FIXED: previously called as index_video(new_segments, [],
                # self.session_id) -- missing the required owner_id argument,
                # which made every call raise TypeError, silently caught by
                # this except block. Live chunks were never actually being
                # indexed into Qdrant. Now passes self.owner_id explicitly
                # (see __init__ note above about live sessions' owner_id).
                index_video(new_segments, [], self.session_id, self.owner_id)
            except Exception as e:
                print(f"  Warning: live indexing failed for this chunk: {e}")

        return new_segments

    def get_transcript(self) -> list[dict]:
        return self.segments

    def _persist(self):
        # Cache progressively so a crash mid-session doesn't lose everything
        out_path = self._out_dir / "transcript.json"
        out_path.write_text(json.dumps(self.segments, indent=2))


if __name__ == "__main__":
    # Smoke test: simulate a "live" source by feeding an existing file in chunks.
    # Usage: python -m app.core.streaming_transcription path/to/audio.wav session_name
    import sys
    import subprocess
    import tempfile

    audio_path = Path(sys.argv[1])
    session_id = sys.argv[2] if len(sys.argv) > 2 else "smoke_test"

    st = StreamingTranscriber(session_id)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        i = 0
        while True:
            chunk_path = tmp / f"chunk_{i}.wav"
            start = i * CHUNK_SECONDS
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ss", str(start), "-t", str(CHUNK_SECONDS),
                str(chunk_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            if chunk_path.stat().st_size < 1000:  # essentially empty = end of audio
                break

            print(f"--- Chunk {i} (t={start}s) ---")
            new_segs = st.add_chunk(chunk_path)
            for s in new_segs:
                print(f"  [{s['start']:.1f}s] {s['text']}")
            i += 1

    print(f"\nFull running transcript: {len(st.get_transcript())} segments")
