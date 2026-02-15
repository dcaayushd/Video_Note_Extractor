from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Video Note Extractor API"
    api_prefix: str = "/api"
    demo_mode: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    whisper_model_size: str = "base"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_best_of: int = 5
    whisper_temperature: float = 0.0
    whisper_condition_on_previous_text: bool = False
    transcription_language: str | None = None
    min_chunk_characters: int = 700
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    uploads_dir: Path = Field(default=Path("data/uploads"))
    database_path: Path = Field(default=Path("data/video_note_extractor.db"))
    chroma_persist_dir: Path = Field(default=Path("data/chroma"))
    temp_dir: Path = Field(default=Path("data/tmp"))
    chunk_size: int = 1800
    chunk_overlap: int = 250
    max_summary_chunks: int = 10
    summary_context_chunks: int = 6
    summary_context_characters: int = 320
    max_chapters: int = 8
    max_glossary_terms: int = 8
    max_study_questions: int = 6
    retrieval_k: int = 6
    background_indexing: bool = True
    allow_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        for path in (
            self.uploads_dir,
            self.chroma_persist_dir,
            self.temp_dir,
            self.database_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def use_local_llm(self) -> bool:
        return (not self.demo_mode) and bool(self.ollama_model)

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path}"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
