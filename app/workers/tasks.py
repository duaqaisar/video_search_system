"""
Celery tasks: the full video processing pipeline, run asynchronously.

This is the REAL production pipeline (see Dockerfile CMD -> app.api.main:app,
which triggers these tasks). There are two entrypoints depending on how the
video arrived:

  - process_video_task           : video was uploaded as a file
  - process_video_from_url_task  : video was submitted as a URL (e.g. YouTube)

TODO: these two tasks share ~90% identical logic (preprocessing -> transcription
-> diarization -> captioning -> summarizing -> indexing). Only the first ~15
lines differ (how the local video file is obtained). Worth extracting the
shared steps into a helper function both tasks call, to avoid having to make
every pipeline change in two places. Left as-is for now to avoid touching
working production code during a cleanup pass -- refactor as a separate,
tested change.
"""
import json
from pathlib import Path

from app.workers.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import ProcessingJob, JobStatus, Video
from app.storage.s3_storage import download_file, upload_file
from app.core.preprocessing import extract_audio, extract_keyframes
from app.core.transcription import transcribe
from app.core.vision import caption_keyframes
from app.core.summary import generate_summary
from app.core.qdrant_indexing import index_video
from app.core.diarization import diarize
from app.core.merge_speakers import merge_transcript_with_speakers
from app.core.url_download import download_video_from_url

# Local scratch space for downloaded videos before/during processing.
# Cleaned up implicitly by the OS (or should be -- see note in process_video_task).
TEMP_DIR = Path("/tmp/video_processing")


def _update_job(db, job_id: str, **fields):
    """
    Small helper to update a ProcessingJob row's fields and commit.
    Used throughout both tasks to report progress (current_stage) as the
    pipeline advances, so the frontend can poll and show live status.
    """
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if job:
        for key, value in fields.items():
            setattr(job, key, value)
        db.commit()


@celery_app.task(bind=True)
def process_video_task(self, job_id: str, video_id: str, storage_key: str, filename: str):
    """
    Pipeline entrypoint for videos uploaded directly as a file.
    The file was already uploaded to S3 by the API layer (see app/api/main.py);
    this task downloads it locally, then runs it through the full pipeline.
    """
    db = SessionLocal()
    try:
        _update_job(db, job_id, status=JobStatus.PROCESSING, current_stage="downloading")

        # Pull the uploaded file down from S3 into local scratch space --
        # ffmpeg/whisper/etc. all need a local file path to work with.
        local_dir = TEMP_DIR / video_id
        local_dir.mkdir(parents=True, exist_ok=True)
        file_ext = Path(filename).suffix  # e.g. ".mp4"
        local_video_path = local_dir / f"{video_id}{file_ext}"
        download_file(storage_key, local_video_path)

        # --- Stage: Preprocessing ---
        # Extract the audio track (for transcription/diarization) and a set
        # of sampled keyframes (for visual captioning).
        _update_job(db, job_id, current_stage="preprocessing")
        audio_path = extract_audio(local_video_path)
        keyframes = extract_keyframes(local_video_path)

        # --- Stage: Transcription ---
        _update_job(db, job_id, current_stage="transcribing")
        transcript = transcribe(audio_path, video_id)

        # --- Stage: Speaker diarization ---
        # diarize() writes speaker segments to disk; merge_transcript_with_speakers
        # then combines those with the transcript so each line has a speaker label.
        _update_job(db, job_id, current_stage="diarizing")
        diarize(audio_path, video_id)
        transcript = merge_transcript_with_speakers(video_id)

        # --- Stage: Visual captioning ---
        _update_job(db, job_id, current_stage="captioning")
        captions = caption_keyframes(keyframes, video_id)

        # --- Stage: Summarization ---
        # Combines transcript + captions into an overall summary + chapters
        # via a free LLM on OpenRouter (see app/core/summary.py).
        _update_job(db, job_id, current_stage="summarizing")
        result = generate_summary(transcript, captions, video_id)

        # --- Stage: Indexing ---
        # Push transcript chunks into the vector DB (Qdrant) scoped to this
        # video's owner, so search/Q&A can retrieve them later.
        _update_job(db, job_id, current_stage="indexing")
        owner_video = db.query(Video).filter(Video.id == video_id).first()
        index_video(transcript, captions, video_id, owner_video.owner_id)

        _update_job(
            db, job_id,
            status=JobStatus.COMPLETED,
            current_stage="done",
            summary=result["summary"],
            chapters_json=json.dumps(result["chapters"]),
        )

    except Exception as e:
        # Any failure anywhere in the pipeline marks the job FAILED with the
        # error message, so the frontend can surface it instead of hanging.
        _update_job(db, job_id, status=JobStatus.FAILED, error_message=str(e))
        raise
    finally:
        db.close()


@celery_app.task(bind=True)
def process_video_from_url_task(self, job_id: str, video_id: str, video_url: str):
    """
    Pipeline entrypoint for videos submitted as a URL (e.g. a YouTube link).
    Unlike process_video_task, there's no pre-uploaded file -- this downloads
    the video from the source URL first, then uploads it to our own storage
    (so URL-sourced and file-uploaded videos end up consistent), before
    running the same pipeline stages as process_video_task.
    """
    db = SessionLocal()
    try:
        _update_job(db, job_id, status=JobStatus.PROCESSING, current_stage="downloading")

        # Download the video from the source URL (e.g. yt-dlp under the hood --
        # see app/core/url_download.py) into local scratch space.
        local_dir = TEMP_DIR / video_id
        local_video_path = download_video_from_url(video_url, local_dir, video_id)

        # Upload to our own storage so it's consistent with file-upload videos
        storage_key = f"videos/{video_id}/{local_video_path.name}"
        upload_file(local_video_path, storage_key)

        # Update the video record with the storage key now that we have it
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.storage_key = storage_key
            db.commit()

        # --- Stage: Preprocessing ---
        _update_job(db, job_id, current_stage="preprocessing")
        audio_path = extract_audio(local_video_path)
        keyframes = extract_keyframes(local_video_path)

        # --- Stage: Transcription ---
        _update_job(db, job_id, current_stage="transcribing")
        transcript = transcribe(audio_path, video_id)

        # --- Stage: Speaker diarization ---
        _update_job(db, job_id, current_stage="diarizing")
        diarize(audio_path, video_id)
        transcript = merge_transcript_with_speakers(video_id)

        # --- Stage: Visual captioning ---
        _update_job(db, job_id, current_stage="captioning")
        captions = caption_keyframes(keyframes, video_id)

        # --- Stage: Summarization ---
        _update_job(db, job_id, current_stage="summarizing")
        result = generate_summary(transcript, captions, video_id)

        # --- Stage: Indexing ---
        _update_job(db, job_id, current_stage="indexing")
        owner_video = db.query(Video).filter(Video.id == video_id).first()
        index_video(transcript, captions, video_id, owner_video.owner_id)

        _update_job(
            db, job_id,
            status=JobStatus.COMPLETED,
            current_stage="done",
            summary=result["summary"],
            chapters_json=json.dumps(result["chapters"]),
        )

    except Exception as e:
        _update_job(db, job_id, status=JobStatus.FAILED, error_message=str(e))
        raise
    finally:
        db.close()
