from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PipelineStatus(str, Enum):
    queued = "queued"
    downloading = "downloading"
    transcribing = "transcribing"
    chunking = "chunking"
    summarizing = "summarizing"
    indexing = "indexing"
    completed = "completed"
    failed = "failed"


class TranscriptWord(BaseModel):
    word: str
    start_seconds: float
    end_seconds: float


class TranscriptSegment(BaseModel):
    segment_id: str = Field(default_factory=lambda: str(uuid4()))
    start_seconds: float
    end_seconds: float
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)


class TranscriptChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    start_seconds: float
    end_seconds: float
    semantic_focus: str
    text: str


class NoteSection(BaseModel):
    heading: str
    bullet_points: list[str] = Field(default_factory=list)
    detail: str
    start_seconds: float | None = None
    display_time: str | None = None
    jump_url: str | None = None


class TimestampItem(BaseModel):
    label: str
    description: str
    start_seconds: float
    end_seconds: float | None = None
    display_time: str
    jump_url: str | None = None


class ChapterItem(BaseModel):
    title: str
    summary: str
    start_seconds: float
    end_seconds: float | None = None
    display_time: str
    jump_url: str | None = None
    keywords: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ActionItem(BaseModel):
    title: str
    detail: str
    owner_hint: str | None = None
    due_hint: str | None = None
    completed: bool = False
    start_seconds: float | None = None
    display_time: str | None = None
    jump_url: str | None = None


class QuoteItem(BaseModel):
    quote: str
    context: str
    start_seconds: float
    display_time: str
    jump_url: str | None = None


class GlossaryItem(BaseModel):
    term: str
    definition: str
    evidence: str
    relevance: str = "medium"


class StudyQuestion(BaseModel):
    question: str
    answer: str
    question_type: Literal["concept", "application", "reflection", "exam"] = "concept"
    difficulty: Literal["introductory", "intermediate", "advanced"] = "intermediate"
    related_topic: str | None = None
    start_seconds: float | None = None
    display_time: str | None = None


class AnalysisMetrics(BaseModel):
    transcript_word_count: int = 0
    unique_word_count: int = 0
    sentence_count: int = 0
    chapter_count: int = 0
    action_item_count: int = 0
    question_count: int = 0
    estimated_reading_minutes: float = 0.0
    lexical_diversity: float = 0.0
    concept_density: float = 0.0
    academic_signal_score: float = 0.0


class MindMapNode(BaseModel):
    label: str
    children: list["MindMapNode"] = Field(default_factory=list)


class ChatCitation(BaseModel):
    label: str
    start_seconds: float
    display_time: str
    jump_url: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    citations: list[ChatCitation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ProcessingStep(BaseModel):
    key: str
    label: str
    completed: bool = False
    active: bool = False


class VideoAnalysisArtifact(BaseModel):
    job_id: str
    title: str
    status: PipelineStatus = PipelineStatus.queued
    source_type: Literal["youtube", "upload", "link"]
    source_url: str | None = None
    uploaded_filename: str | None = None
    current_step: str = "queued"
    progress_percent: int = 0
    runtime_seconds: float | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    transcript_text: str = ""
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    transcript_chunks: list[TranscriptChunk] = Field(default_factory=list)
    quick_summary: str = ""
    five_minute_summary: list[str] = Field(default_factory=list)
    key_topics: list[str] = Field(default_factory=list)
    note_sections: list[NoteSection] = Field(default_factory=list)
    timestamps: list[TimestampItem] = Field(default_factory=list)
    chapters: list[ChapterItem] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    key_quotes: list[QuoteItem] = Field(default_factory=list)
    learning_objectives: list[str] = Field(default_factory=list)
    glossary: list[GlossaryItem] = Field(default_factory=list)
    study_questions: list[StudyQuestion] = Field(default_factory=list)
    analysis_metrics: AnalysisMetrics = Field(default_factory=AnalysisMetrics)
    mind_map: MindMapNode | None = None
    chat_history: list[ChatMessage] = Field(default_factory=list)
    searchable: bool = False
    metadata: dict[str, str | int | float | None] = Field(default_factory=dict)


MindMapNode.model_rebuild()
