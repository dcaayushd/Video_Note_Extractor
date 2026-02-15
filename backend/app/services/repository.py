from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import UploadFile
from app.core.config import Settings
from app.models import (
    PipelineStatus,
    ProcessingStep,
    VideoAnalysisArtifact,
    VideoJobRecord,
)
from app.schemas.video import ProcessVideoRequest
from app.storage import DatabaseManager


class ArtifactRepository:
    def __init__(self, settings: Settings, database: DatabaseManager | None = None) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.database = database or DatabaseManager(settings)

    def _upload_dir(self, upload_id: str) -> Path:
        return self.settings.uploads_dir / upload_id

    def _serialize_artifact(self, artifact: VideoAnalysisArtifact) -> str:
        return artifact.model_dump_json(indent=2)

    def _record_to_artifact(self, record: VideoJobRecord) -> VideoAnalysisArtifact:
        return VideoAnalysisArtifact.model_validate_json(record.artifact_json)

    def _sync_record(self, record: VideoJobRecord, artifact: VideoAnalysisArtifact) -> None:
        record.title = artifact.title
        record.status = artifact.status.value
        record.source_type = artifact.source_type
        record.source_url = artifact.source_url
        record.uploaded_filename = artifact.uploaded_filename
        record.current_step = artifact.current_step
        record.progress_percent = artifact.progress_percent
        record.runtime_seconds = artifact.runtime_seconds
        record.error_message = artifact.error_message
        record.artifact_json = self._serialize_artifact(artifact)
        record.created_at = artifact.created_at
        record.updated_at = artifact.updated_at

    def _build_record(self, artifact: VideoAnalysisArtifact) -> VideoJobRecord:
        return VideoJobRecord(
            job_id=artifact.job_id,
            title=artifact.title,
            status=artifact.status.value,
            source_type=artifact.source_type,
            source_url=artifact.source_url,
            uploaded_filename=artifact.uploaded_filename,
            current_step=artifact.current_step,
            progress_percent=artifact.progress_percent,
            runtime_seconds=artifact.runtime_seconds,
            error_message=artifact.error_message,
            artifact_json=self._serialize_artifact(artifact),
            created_at=artifact.created_at,
            updated_at=artifact.updated_at,
        )

    def _classify_source_type(self, request: ProcessVideoRequest) -> str:
        if request.upload_id:
            return "upload"
        url = (request.youtube_url or request.source_url or "").strip().lower()
        host = urlparse(url).netloc.lower()
        if "youtube.com" in host or "youtu.be" in host:
            return "youtube"
        return "link"

    def create_job(self, request: ProcessVideoRequest) -> VideoAnalysisArtifact:
        job_id = str(uuid4())
        artifact = VideoAnalysisArtifact(
            job_id=job_id,
            title=request.title or "Untitled Video",
            source_type=self._classify_source_type(request),
            source_url=request.youtube_url or request.source_url,
            metadata={"upload_id": request.upload_id},
        )
        with self.database.new_session() as session:
            session.add(self._build_record(artifact))
            session.commit()
        return artifact

    def list_steps(self, current_status: PipelineStatus) -> list[ProcessingStep]:
        keys = [
            (PipelineStatus.downloading, "Downloading"),
            (PipelineStatus.transcribing, "Transcribing"),
            (PipelineStatus.chunking, "Chunking"),
            (PipelineStatus.summarizing, "Summarizing"),
            (PipelineStatus.indexing, "Indexing"),
            (PipelineStatus.completed, "Completed"),
        ]
        reached = False
        steps: list[ProcessingStep] = []
        for status, label in keys:
            active = status == current_status
            completed = reached or current_status == PipelineStatus.completed
            if current_status == status:
                completed = False
                reached = True
            elif not reached and current_status not in {PipelineStatus.queued, PipelineStatus.failed}:
                completed = True
            steps.append(
                ProcessingStep(
                    key=status.value,
                    label=label,
                    completed=completed,
                    active=active,
                )
            )
        return steps

    async def save_upload(self, upload: UploadFile) -> tuple[str, Path]:
        upload_id = str(uuid4())
        upload_dir = self._upload_dir(upload_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / (upload.filename or "uploaded-video.bin")
        content = await upload.read()
        target.write_bytes(content)
        metadata = {
            "filename": upload.filename,
            "content_type": upload.content_type,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        (upload_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return upload_id, target

    def resolve_upload(self, upload_id: str) -> Path:
        upload_dir = self._upload_dir(upload_id)
        metadata_path = upload_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Upload {upload_id} does not exist.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        filename = metadata.get("filename") or "uploaded-video.bin"
        target = upload_dir / filename
        if not target.exists():
            raise FileNotFoundError(f"Upload file for {upload_id} is missing.")
        return target

    def resolve_media(self, job_id: str) -> Path:
        artifact = self.load_artifact(job_id)
        local_media_path = artifact.metadata.get("local_media_path")
        if isinstance(local_media_path, str):
            candidate = Path(local_media_path)
            if candidate.exists():
                return candidate

        upload_id = artifact.metadata.get("upload_id")
        if isinstance(upload_id, str) and upload_id:
            return self.resolve_upload(upload_id)

        job_temp_dir = self.settings.temp_dir / job_id
        if job_temp_dir.exists():
            files = [
                path
                for path in job_temp_dir.iterdir()
                if path.is_file()
                and path.name != "info.json"
                and not path.name.endswith((".part", ".ytdl"))
            ]
            if files:
                files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
                return files[0]

        raise FileNotFoundError(f"Media for job {job_id} does not exist.")

    def load_upload_metadata(self, upload_id: str) -> dict[str, str]:
        metadata_path = self._upload_dir(upload_id) / "metadata.json"
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def save_artifact(self, artifact: VideoAnalysisArtifact) -> None:
        artifact.updated_at = datetime.now(timezone.utc)
        with self.database.new_session() as session:
            record = session.get(VideoJobRecord, artifact.job_id)
            if record is None:
                raise FileNotFoundError(f"Job {artifact.job_id} does not exist.")
            self._sync_record(record, artifact)
            session.commit()

    def load_artifact(self, job_id: str) -> VideoAnalysisArtifact:
        with self.database.new_session() as session:
            record = session.get(VideoJobRecord, job_id)
            if record is None:
                raise FileNotFoundError(f"Job {job_id} does not exist.")
            return self._record_to_artifact(record)

    def update_status(
        self,
        job_id: str,
        status: PipelineStatus,
        *,
        current_step: str,
        progress_percent: int,
        error_message: str | None = None,
    ) -> VideoAnalysisArtifact:
        artifact = self.load_artifact(job_id)
        artifact.status = status
        artifact.current_step = current_step
        artifact.progress_percent = progress_percent
        artifact.error_message = error_message
        self.save_artifact(artifact)
        return artifact
