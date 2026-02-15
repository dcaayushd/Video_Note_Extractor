from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.storage import DatabaseManager
from app.rag.vector_store import TranscriptVectorStore
from app.services.export_service import ExportService
from app.services.pipeline import VideoPipelineService
from app.services.repository import ArtifactRepository
from app.services.youtube_service import YouTubeService
from app.summarization.summarizer import SummarizationService
from app.transcription.whisper_service import WhisperTranscriptionService


@lru_cache
def get_database() -> DatabaseManager:
    settings = get_settings()
    return DatabaseManager(settings)


@lru_cache
def get_repository() -> ArtifactRepository:
    settings = get_settings()
    return ArtifactRepository(settings, database=get_database())


@lru_cache
def get_transcription_service() -> WhisperTranscriptionService:
    settings = get_settings()
    return WhisperTranscriptionService(settings)


@lru_cache
def get_summarization_service() -> SummarizationService:
    settings = get_settings()
    return SummarizationService(settings)


@lru_cache
def get_vector_store() -> TranscriptVectorStore:
    settings = get_settings()
    repository = get_repository()
    return TranscriptVectorStore(settings=settings, repository=repository)


@lru_cache
def get_youtube_service() -> YouTubeService:
    settings = get_settings()
    return YouTubeService(settings)


@lru_cache
def get_export_service() -> ExportService:
    return ExportService()


@lru_cache
def get_pipeline_service() -> VideoPipelineService:
    return VideoPipelineService(
        settings=get_settings(),
        repository=get_repository(),
        transcription_service=get_transcription_service(),
        summarization_service=get_summarization_service(),
        vector_store=get_vector_store(),
        youtube_service=get_youtube_service(),
    )
