from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_pipeline_service
from app.schemas.video import ChatRequest, ChatResponse
from app.services.pipeline import VideoPipelineService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_with_video(
    request: ChatRequest,
    pipeline: VideoPipelineService = Depends(get_pipeline_service),
) -> ChatResponse:
    try:
        artifact = pipeline.chat(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    latest_message = artifact.chat_history[-1]
    return ChatResponse(
        job_id=request.job_id,
        answer=latest_message.content,
        citations=latest_message.citations,
    )

