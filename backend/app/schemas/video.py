from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.models import (
    ActionItem,
    AnalysisMetrics,
    ChatCitation,
    ChapterItem,
    GlossaryItem,
    MindMapNode,
    NoteSection,
    PipelineStatus,
    ProcessingStep,
    QuoteItem,
    StudyQuestion,
    TimestampItem,
    TranscriptChunk,
    TranscriptSegment,
    TranscriptWord,
    VideoAnalysisArtifact,
)


class ProcessVideoRequest(BaseModel):
    youtube_url: str | None = None
    source_url: str | None = None
    upload_id: str | None = None
    title: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "ProcessVideoRequest":
        if not self.youtube_url and not self.source_url and not self.upload_id:
            raise ValueError("Provide either source_url, youtube_url, or upload_id.")
        return self


class UploadVideoResponse(BaseModel):
    upload_id: str
    filename: str
    content_type: str | None = None


class ProcessVideoResponse(BaseModel):
    job_id: str
    status: PipelineStatus
    steps: list[ProcessingStep]
    message: str


class VideoSummaryResponse(BaseModel):
    job_id: str
    status: PipelineStatus
    artifact: VideoAnalysisArtifact


class TimestampsResponse(BaseModel):
    job_id: str
    status: PipelineStatus
    timestamps: list[TimestampItem]


class ChaptersResponse(BaseModel):
    job_id: str
    status: PipelineStatus
    chapters: list[ChapterItem]


class ActionItemsResponse(BaseModel):
    job_id: str
    status: PipelineStatus
    action_items: list[ActionItem]


class StudyGuideResponse(BaseModel):
    job_id: str
    status: PipelineStatus
    learning_objectives: list[str]
    glossary: list[GlossaryItem]
    study_questions: list[StudyQuestion]
    analysis_metrics: AnalysisMetrics


class ChatRequest(BaseModel):
    job_id: str
    question: str


class ChatResponse(BaseModel):
    job_id: str
    answer: str
    citations: list[ChatCitation] = Field(default_factory=list)


class ExportFormat(str, Enum):
    markdown = "markdown"
    pdf = "pdf"
    notion = "notion"
