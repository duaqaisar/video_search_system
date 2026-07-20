"""
Live audio ingestion: tails an RTMP (or any ffmpeg-readable) live source,
splits it into rolling WAV chunks as they arrive, and feeds each chunk into
a StreamingTranscriber as soon as it's complete.

Works with real RTMP URLs, or -- for local testing without a live stream --
any file played back in real time via ffmpeg's `-re` flag (see __main__).
"""
import subprocess
import time
import threading
from pathlib import Path

from app.core.streaming_transcription import StreamingTranscriber, CHUNK_SECONDS
from app.config import PROCESSED_DIR


class LiveIngestSession:
    """
    Starts an ffmpeg process that segments a live source into rolling WAV
    chunks on disk, then polls for completed chunks and transcribes them
    as they land.

    Usage:
        session = LiveIngestSession(source_url="rtmp://...", session_id="stream_1")
        session.start()
        ...
        session.stop()
        transcript = session.transcriber.get_transcript()
    """

    def __init__(
        self,
        source_url: str,
        session_id: str,
        language: str | None = None,
        owner_id: str | None = None,
    ):
        self.source_url = source_url
        self.session_id = session_id
        # owner_id identifies which authenticated user started this session
        # (see app/api/main.py's /live/start) -- used both for scoping
        # indexed chunks in Qdrant (via StreamingTranscriber) and for
        # access control in live_sessions.py (so one user can't view or
        # stop another user's live session).
        self.owner_id = owner_id
        self.transcriber = StreamingTranscriber(session_id, language=language, owner_id=owner_id)
        self._chunk_dir = PROCESSED_DIR / "live_sessions" / session_id / "audio_chunks"
        self._chunk_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg_proc = None
        self._watcher_thread = None
        self._stop_flag = threading.Event()

    def start(self):
        """Launch ffmpeg to segment the live source into fixed-length WAV
        chunks on disk, and start a background thread that watches for and
        transcribes each chunk as it becomes ready."""
        pattern = str(self._chunk_dir / "chunk_%05d.wav")
        cmd = [
            "ffmpeg", "-y",
            "-i", self.source_url,
            "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
            "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
            "-reset_timestamps", "1",  # each chunk's internal timestamps restart at 0
            pattern,
        ]
        print(f"Starting ffmpeg segmenter for session '{self.session_id}'...")
        self._ffmpeg_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._watcher_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher_thread.start()

    def _watch_loop(self):
        """
        ffmpeg only finishes writing chunk N once it starts writing chunk N+1
        (or exits). Poll for that so we never transcribe a half-written file.

        Runs in its own background thread for the lifetime of the session,
        checking every 2s for newly-completed chunks.
        """
        processed = set()
        while not self._stop_flag.is_set():
            chunks = sorted(self._chunk_dir.glob("chunk_*.wav"))
            ffmpeg_alive = self._ffmpeg_proc.poll() is None
            # While ffmpeg is still running, the LAST chunk on disk might
            # still be mid-write -- only treat it as "ready" once ffmpeg
            # has exited (meaning every chunk on disk is final).
            ready = chunks[:-1] if ffmpeg_alive else chunks

            for chunk_path in ready:
                if chunk_path in processed:
                    continue
                print(f"  New chunk ready: {chunk_path.name}")
                try:
                    new_segments = self.transcriber.add_chunk(chunk_path)
                    for s in new_segments:
                        print(f"    [{s['start']:.1f}s] {s['text']}")
                except Exception as e:
                    # Don't let one bad chunk kill the whole watch loop --
                    # log it and keep watching for future chunks.
                    print(f"  Failed to transcribe {chunk_path.name}: {e}")
                processed.add(chunk_path)

            if not ffmpeg_alive:
                break
            time.sleep(2)

    def stop(self):
        """Signal the watcher thread to stop and terminate the ffmpeg
        process. Note: terminate() sends SIGTERM but this doesn't
        explicitly wait() for ffmpeg's exit -- in practice the watcher
        thread's join() below covers the shutdown, but a stray ffmpeg
        process could theoretically linger briefly under unusual timing."""
        self._stop_flag.set()
        if self._ffmpeg_proc and self._ffmpeg_proc.poll() is None:
            self._ffmpeg_proc.terminate()
        if self._watcher_thread:
            self._watcher_thread.join(timeout=10)


if __name__ == "__main__":
    """
    Local test WITHOUT a real RTMP server: replay an existing file in real
    time using ffmpeg's -re flag as the "live" source, so chunks arrive at
    roughly the pace they would from an actual stream.

    Usage: python -m app.core.live_ingest data/videos/dua.mp4 live_rtmp_test
    """
    import sys

    source = sys.argv[1]
    session_id = sys.argv[2] if len(sys.argv) > 2 else "live_ingest_test"

    # NOTE: this duplicates LiveIngestSession.start() almost entirely --
    # only difference is the added "-re" flag (forces ffmpeg to read the
    # input at its native frame rate instead of as fast as possible, so a
    # regular file behaves like a live stream for testing purposes).
    # Same category of duplication as app/workers/tasks.py -- left as-is
    # since this is test/dev-only code, not the production path.
    class RealtimeFileSession(LiveIngestSession):
        def start(self):
            pattern = str(self._chunk_dir / "chunk_%05d.wav")
            cmd = [
                "ffmpeg", "-y", "-re", "-i", self.source_url,
                "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
                "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
                "-reset_timestamps", "1",
                pattern,
            ]
            print(f"Starting REAL-TIME simulated ffmpeg segmenter for '{self.session_id}'...")
            self._ffmpeg_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self._watcher_thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._watcher_thread.start()

    session = RealtimeFileSession(source, session_id)
    session.start()

    try:
        while session._ffmpeg_proc.poll() is None:
            time.sleep(1)
        session._watcher_thread.join(timeout=15)
    except KeyboardInterrupt:
        print("\nStopping...")
        session.stop()

    print(f"\nFull running transcript: {len(session.transcriber.get_transcript())} segments")
