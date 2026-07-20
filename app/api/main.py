"""
FastAPI application: upload videos, trigger processing, check status,
retrieve results, and search.
"""
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.core.video_chat import ask_about_video
from app.db.database import get_db, engine, Base
from app.db.models import Video, ProcessingJob, JobStatus, User
from app.storage.s3_storage import upload_file
from app.workers.tasks import process_video_task
from app.core.qdrant_indexing import search as qdrant_search
from app.workers.tasks import process_video_from_url_task
from app.auth.users import hash_password, verify_password, create_access_token, get_current_user
from app.core.live_sessions import start_session, get_session, stop_session, list_sessions
from dotenv import load_dotenv
load_dotenv()
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Video Search API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-only: restrict this to your real frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_DIR = Path("/tmp/video_uploads")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/signup")
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with that email already exists")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user_id": user.id, "email": user.email}


@app.post("/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user_id": user.id, "email": user.email}


@app.post("/videos")
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    video_id = str(uuid.uuid4())
    local_path = TEMP_UPLOAD_DIR / f"{video_id}_{file.filename}"

    with open(local_path, "wb") as f:
        f.write(await file.read())

    storage_key = f"videos/{video_id}/{file.filename}"
    upload_file(local_path, storage_key)

    video = Video(id=video_id, owner_id=current_user.id, filename=file.filename, storage_key=storage_key)
    db.add(video)
    db.commit()

    job = ProcessingJob(id=str(uuid.uuid4()), video_id=video_id, status=JobStatus.PENDING)
    db.add(job)
    db.commit()

    process_video_task.delay(job.id, video_id, storage_key, file.filename)

    local_path.unlink(missing_ok=True)

    return {"video_id": video_id, "job_id": job.id, "status": job.status}


@app.post("/videos/from-url")
def upload_video_from_url(
    url: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    video_id = str(uuid.uuid4())

    video = Video(id=video_id, owner_id=current_user.id, filename=url, storage_key="pending")
    db.add(video)
    db.commit()

    job = ProcessingJob(id=str(uuid.uuid4()), video_id=video_id, status=JobStatus.PENDING)
    db.add(job)
    db.commit()

    process_video_from_url_task.delay(job.id, video_id, url)

    return {"video_id": video_id, "job_id": job.id, "status": job.status}


def _get_owned_video_or_404(db: Session, video_id: str, current_user: User) -> Video:
    video = db.query(Video).filter(Video.id == video_id, Video.owner_id == current_user.id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@app.get("/jobs/{job_id}")
def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = (
        db.query(ProcessingJob)
        .join(Video, Video.id == ProcessingJob.video_id)
        .filter(ProcessingJob.id == job_id, Video.owner_id == current_user.id)
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "video_id": job.video_id,
        "status": job.status,
        "current_stage": job.current_stage,
        "error_message": job.error_message,
        "summary": job.summary,
        "chapters": job.chapters_json,
    }


@app.get("/search")
def search_videos(
    q: str,
    limit: int = 5,
    video_id: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if video_id:
        _get_owned_video_or_404(db, video_id, current_user)
    results = qdrant_search(q, n_results=limit, video_id=video_id, owner_id=current_user.id)
    return {"query": q, "results": results}


@app.get("/chat")
def chat_about_video(
    video_id: str,
    question: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_video_or_404(db, video_id, current_user)
    result = ask_about_video(question, video_id, current_user.id)
    return result


@app.get("/videos")
def list_videos(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    videos = (
        db.query(Video)
        .filter(Video.owner_id == current_user.id)
        .order_by(Video.created_at.desc())
        .all()
    )

    results = []
    for video in videos:
        latest_job = (
            db.query(ProcessingJob)
            .filter(ProcessingJob.video_id == video.id)
            .order_by(ProcessingJob.created_at.desc())
            .first()
        )
        results.append({
            "video_id": video.id,
            "filename": video.filename,
            "created_at": video.created_at.isoformat() if video.created_at else None,
            "job_id": latest_job.id if latest_job else None,
            "status": latest_job.status if latest_job else None,
            "current_stage": latest_job.current_stage if latest_job else None,
            "summary": latest_job.summary if latest_job else None,
        })

    return {"videos": results, "count": len(results)}


class LiveStartRequest(BaseModel):
    source_url: str
    session_id: str | None = None
    language: str | None = None


@app.post("/live/start")
def start_live_session(
    payload: LiveStartRequest,
    current_user: User = Depends(get_current_user),
):
    session_id = payload.session_id or str(uuid.uuid4())
    try:
        start_session(
            session_id=session_id,
            source_url=payload.source_url,
            language=payload.language,
            owner_id=current_user.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"session_id": session_id, "status": "started"}


def _get_owned_session_or_404(session_id: str, current_user: User):
    session = get_session(session_id, owner_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Live session not found")
    return session


@app.get("/live")
def list_live_sessions(current_user: User = Depends(get_current_user)):
    return {"sessions": list_sessions(owner_id=current_user.id)}


@app.get("/live/{session_id}/transcript")
def get_live_transcript(session_id: str, current_user: User = Depends(get_current_user)):
    session = _get_owned_session_or_404(session_id, current_user)
    return {"session_id": session_id, "transcript": session.transcriber.get_transcript()}


@app.get("/live/{session_id}/search")
def search_live_session(
    session_id: str,
    q: str,
    limit: int = 5,
    current_user: User = Depends(get_current_user),
):
    # Ownership check first (raises 404 if this user doesn't own the session)
    _get_owned_session_or_404(session_id, current_user)
    # Live chunks are indexed into Qdrant under video_id=session_id (see
    # StreamingTranscriber.add_chunk -> index_video), so we can reuse the
    # same qdrant_search() the regular /search endpoint uses -- no need for
    # a separate search method on StreamingTranscriber.
    results = qdrant_search(q, n_results=limit, video_id=session_id, owner_id=current_user.id)
    return {"session_id": session_id, "query": q, "results": results}


@app.post("/live/{session_id}/stop")
def stop_live_session(session_id: str, current_user: User = Depends(get_current_user)):
    stopped = stop_session(session_id, owner_id=current_user.id)
    if not stopped:
        raise HTTPException(status_code=404, detail="Live session not found")
    return {"session_id": session_id, "status": "stopped"}
