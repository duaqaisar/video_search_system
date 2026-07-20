"""
SQLAlchemy models: users, videos, processing jobs, and (legacy) API keys.
"""
import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Text, Enum, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.database import Base


def _uuid():
    return str(uuid.uuid4())


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    videos = relationship("Video", back_populates="owner", cascade="all, delete-orphan")


class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    owner_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    storage_key = Column(String, nullable=False)  # path/key in S3/MinIO
    duration_seconds = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    owner = relationship("User", back_populates="videos")
    jobs = relationship("ProcessingJob", back_populates="video", cascade="all, delete-orphan")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    video_id = Column(UUID(as_uuid=False), ForeignKey("videos.id"), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    current_stage = Column(String, nullable=True)  # e.g. "transcribing", "captioning"
    error_message = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    chapters_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    video = relationship("Video", back_populates="jobs")


# Legacy single shared-key auth — no longer used for new requests once
# JWT-based user auth is wired in, kept only so old code paths don't explode.
class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    key_hash = Column(String, nullable=False, unique=True)
    label = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    revoked = Column(String, default="false")
