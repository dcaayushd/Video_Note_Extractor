from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routes.chat import router as chat_router
from app.routes.videos import router as video_router


class _SummaryPollingAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "GET /api/video-summary/" not in message


def _configure_logging() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, _SummaryPollingAccessFilter) for item in access_logger.filters):
        access_logger.addFilter(_SummaryPollingAccessFilter())


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging()
    application = FastAPI(
        title=settings.app_name,
        version="2.0.0",
        description="AI-powered video intelligence API for transcription, chapters, notes, study guides, and grounded chat.",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(video_router, prefix=settings.api_prefix)
    application.include_router(chat_router, prefix=settings.api_prefix)
    return application


app = create_app()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Video Note Extractor API is running."}


@app.get("/health")
async def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": "2.0.0",
        "demo_mode": str(settings.demo_mode).lower(),
        "llm_enabled": str(settings.use_local_llm).lower(),
        "ollama_model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
    }
