from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from app.core.config import Settings
from app.models import (
    ChatMessage,
    PipelineStatus,
    TranscriptChunk,
    TranscriptSegment,
    VideoAnalysisArtifact,
)
from app.schemas.video import ChatRequest, ProcessVideoRequest
from app.services.repository import ArtifactRepository
from app.services.transcript_chunker import TranscriptChunker
from app.services.youtube_service import UnsupportedSourceError, YouTubeService
from app.summarization.summarizer import SummarizationService
from app.transcription.whisper_service import WhisperTranscriptionService
from app.rag.vector_store import TranscriptVectorStore

logger = logging.getLogger("uvicorn.error")


class VideoPipelineService:
    def __init__(
        self,
        settings: Settings,
        repository: ArtifactRepository,
        transcription_service: WhisperTranscriptionService,
        summarization_service: SummarizationService,
        vector_store: TranscriptVectorStore,
        youtube_service: YouTubeService,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.transcription_service = transcription_service
        self.summarization_service = summarization_service
        self.vector_store = vector_store
        self.youtube_service = youtube_service
        self.chunker = TranscriptChunker(settings)
        self._progress_lock = threading.Lock()

    def create_job(self, request: ProcessVideoRequest) -> VideoAnalysisArtifact:
        return self.repository.create_job(request)

    def get_summary(self, job_id: str) -> VideoAnalysisArtifact:
        return self.repository.load_artifact(job_id)

    async def process_video(self, job_id: str, request: ProcessVideoRequest) -> None:
        started = time.perf_counter()
        artifact = self.repository.load_artifact(job_id)
        last_download_bucket = -1
        last_transcription_bucket = -1
        logger.info(
            "Pipeline started | job_id=%s | title=%s | source=%s",
            job_id,
            artifact.title,
            artifact.source_type,
        )

        try:
            def on_download_progress(percent: int, detail: str) -> None:
                nonlocal last_download_bucket
                bucket = min(100, max(0, percent)) // 5
                if bucket == last_download_bucket:
                    return
                last_download_bucket = bucket
                self.repository.update_status(
                    job_id,
                    PipelineStatus.downloading,
                    current_step="downloading",
                    progress_percent=self._map_progress(percent, start=15, end=34),
                )
                self._emit_progress_line(
                    job_id,
                    stage="Downloading",
                    percent=percent,
                    detail=detail,
                )

            def on_transcription_progress(percent: int, detail: str) -> None:
                nonlocal last_transcription_bucket
                bucket = min(100, max(0, percent)) // 5
                if bucket == last_transcription_bucket:
                    return
                last_transcription_bucket = bucket
                self.repository.update_status(
                    job_id,
                    PipelineStatus.transcribing,
                    current_step="transcribing",
                    progress_percent=self._map_progress(percent, start=35, end=54),
                )
                self._emit_progress_line(
                    job_id,
                    stage="Transcribing",
                    percent=percent,
                    detail=detail,
                )

            media_path = await self._resolve_media_source(
                artifact,
                request,
                progress_callback=on_download_progress,
            )
            artifact = self.repository.load_artifact(job_id)
            self.repository.update_status(
                job_id,
                PipelineStatus.transcribing,
                current_step="transcribing",
                progress_percent=35,
            )
            self._emit_progress_line(
                job_id,
                stage="Transcribing",
                percent=0,
                detail="0% | preparing audio",
            )

            transcript_text, segments, transcript_source = await asyncio.to_thread(
                self._resolve_transcript,
                artifact,
                media_path,
                on_transcription_progress,
            )
            artifact = self.repository.load_artifact(job_id)
            artifact.transcript_text = transcript_text
            artifact.transcript_segments = segments
            artifact.metadata["transcript_source"] = transcript_source
            self.repository.save_artifact(artifact)

            self.repository.update_status(
                job_id,
                PipelineStatus.chunking,
                current_step="chunking",
                progress_percent=55,
            )
            self._emit_progress_line(
                job_id,
                stage="Chunking",
                percent=100,
                detail="100% | creating transcript chunks",
            )
            chunks = await asyncio.to_thread(self.chunker.chunk, segments)
            artifact = self.repository.load_artifact(job_id)
            artifact.transcript_chunks = chunks
            self.repository.save_artifact(artifact)

            self.repository.update_status(
                job_id,
                PipelineStatus.summarizing,
                current_step="summarizing",
                progress_percent=75,
            )
            self._emit_progress_line(
                job_id,
                stage="Summarizing",
                percent=100,
                detail="100% | generating notes",
            )
            artifact = self.repository.load_artifact(job_id)
            artifact = await asyncio.to_thread(
                self.summarization_service.summarize,
                artifact,
                chunks,
            )
            artifact.metadata.update(
                {
                    "analysis_version": "2.0",
                    "chapter_count": len(artifact.chapters),
                    "glossary_count": len(artifact.glossary),
                    "study_question_count": len(artifact.study_questions),
                    "transcript_word_count": artifact.analysis_metrics.transcript_word_count,
                }
            )
            self.repository.save_artifact(artifact)

            self.repository.update_status(
                job_id,
                PipelineStatus.indexing,
                current_step="indexing",
                progress_percent=90,
            )
            artifact = self.repository.load_artifact(job_id)
            artifact.status = PipelineStatus.completed
            artifact.current_step = "completed"
            artifact.progress_percent = 100
            artifact.runtime_seconds = round(time.perf_counter() - started, 2)
            artifact.searchable = False
            self.repository.save_artifact(artifact)
            self._finish_progress_line(
                job_id,
                stage="Ready",
                percent=100,
                detail=f"100% | completed in {artifact.runtime_seconds}s",
            )
            if self.settings.background_indexing:
                asyncio.create_task(self._index_in_background(job_id, chunks))
            else:
                await self._index_in_background(job_id, chunks)
            logger.info(
                "Pipeline completed | job_id=%s | runtime_seconds=%s",
                job_id,
                artifact.runtime_seconds,
            )
        except UnsupportedSourceError as exc:
            self._finish_progress_line(
                job_id,
                stage="Failed",
                percent=100,
                detail=str(exc),
            )
            logger.warning("Pipeline rejected source | job_id=%s | reason=%s", job_id, exc)
            self.repository.update_status(
                job_id,
                PipelineStatus.failed,
                current_step="failed",
                progress_percent=100,
                error_message=str(exc),
            )
        except Exception as exc:
            self._finish_progress_line(
                job_id,
                stage="Failed",
                percent=100,
                detail=str(exc),
            )
            logger.exception("Pipeline failed | job_id=%s", job_id)
            self.repository.update_status(
                job_id,
                PipelineStatus.failed,
                current_step="failed",
                progress_percent=100,
                error_message=str(exc),
            )

    async def _resolve_media_source(
        self,
        artifact: VideoAnalysisArtifact,
        request: ProcessVideoRequest,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> Path:
        self.repository.update_status(
            artifact.job_id,
            PipelineStatus.downloading,
            current_step="downloading",
            progress_percent=15,
        )
        source_url = request.youtube_url or request.source_url
        if request.upload_id:
            upload_id = request.upload_id
            self._emit_progress_line(
                artifact.job_id,
                stage="Uploading",
                percent=100,
                detail=f"100% | using upload {upload_id}",
            )
            media_path = self.repository.resolve_upload(upload_id)
            metadata = self.repository.load_upload_metadata(upload_id)
            artifact = self.repository.load_artifact(artifact.job_id)
            artifact.title = request.title or metadata.get("filename", artifact.title)
            artifact.uploaded_filename = metadata.get("filename")
            artifact.metadata["local_media_path"] = str(media_path)
            self.repository.save_artifact(artifact)
            return media_path

        if source_url:
            output_dir = self.settings.temp_dir / artifact.job_id
            self._emit_progress_line(
                artifact.job_id,
                stage="Downloading",
                percent=0,
                detail="0% | preparing source",
            )
            media_path, metadata = await asyncio.to_thread(
                self.youtube_service.download_media,
                source_url,
                output_dir,
                progress_callback,
            )
            artifact = self.repository.load_artifact(artifact.job_id)
            artifact.title = request.title or str(metadata.get("title") or artifact.title)
            artifact.source_url = str(metadata.get("source_url") or source_url)
            artifact.metadata.update(metadata)
            artifact.metadata["local_media_path"] = str(media_path)
            self.repository.save_artifact(artifact)
            return media_path
        raise RuntimeError("No media source was provided for processing.")

    def _resolve_transcript(
        self,
        artifact: VideoAnalysisArtifact,
        media_path: Path,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> tuple[str, list[TranscriptSegment], str]:
        source_transcript_path = artifact.metadata.get("source_transcript_path")
        if isinstance(source_transcript_path, str) and source_transcript_path:
            if progress_callback is not None:
                progress_callback(12, "12% | checking source transcript")
            source_result = self.transcription_service.load_source_transcript(
                Path(source_transcript_path),
            )
            if source_result is not None:
                source_transcript, source_segments = source_result
                if self.transcription_service.source_transcript_is_usable(
                    source_transcript,
                    source_segments,
                ):
                    if progress_callback is not None:
                        progress_callback(100, "100% | source transcript loaded")
                    transcript_source = str(
                        artifact.metadata.get("source_transcript_kind") or "source_transcript"
                    )
                    return source_transcript, source_segments, transcript_source
                if progress_callback is not None:
                    progress_callback(22, "22% | source transcript was weak, switching to whisper")

        transcript_text, segments = self.transcription_service.transcribe(
            media_path,
            progress_callback,
        )
        return transcript_text, segments, "whisper"

    def chat(self, request: ChatRequest) -> VideoAnalysisArtifact:
        response = self.vector_store.ask(request.job_id, request.question)
        artifact = self.repository.load_artifact(request.job_id)
        artifact.chat_history.extend(
            [
                ChatMessage(role="user", content=request.question),
                ChatMessage(
                    role="assistant",
                    content=response.answer,
                    citations=response.citations,
                ),
            ]
        )
        self.repository.save_artifact(artifact)
        return artifact

    def _map_progress(self, percent: int, *, start: int, end: int) -> int:
        percent = min(100, max(0, percent))
        span = max(0, end - start)
        return start + round((percent / 100) * span)

    def _emit_progress_line(
        self,
        job_id: str,
        *,
        stage: str,
        percent: int,
        detail: str,
    ) -> None:
        line = f"[{job_id[:8]}] {stage:<12} {percent:>3}% | {detail}"
        with self._progress_lock:
            sys.stdout.write(f"\r\033[2K{line}")
            sys.stdout.flush()

    def _finish_progress_line(
        self,
        job_id: str,
        *,
        stage: str,
        percent: int,
        detail: str,
    ) -> None:
        line = f"[{job_id[:8]}] {stage:<12} {percent:>3}% | {detail}"
        with self._progress_lock:
            sys.stdout.write(f"\r\033[2K{line}\n")
            sys.stdout.flush()

    async def _index_in_background(
        self,
        job_id: str,
        chunks: list[TranscriptChunk],
    ) -> None:
        try:
            self._emit_progress_line(
                job_id,
                stage="Indexing",
                percent=100,
                detail="100% | building search index",
            )
            await asyncio.to_thread(self.vector_store.index, job_id, chunks)
            artifact = self.repository.load_artifact(job_id)
            artifact.searchable = True
            artifact.metadata["index_status"] = "ready"
            self.repository.save_artifact(artifact)
            self._finish_progress_line(
                job_id,
                stage="Indexed",
                percent=100,
                detail="100% | search index ready",
            )
        except Exception:
            logger.exception("Background indexing failed | job_id=%s", job_id)
            artifact = self.repository.load_artifact(job_id)
            artifact.metadata["index_status"] = "failed"
            self.repository.save_artifact(artifact)
