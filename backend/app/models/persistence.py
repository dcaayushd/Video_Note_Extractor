from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class VideoJobRecord(Base):
    __tablename__ = "video_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    current_step: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    runtime_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
