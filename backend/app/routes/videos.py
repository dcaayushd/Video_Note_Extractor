from __future__ import annotations

import logging
import mimetypes

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from app.core.dependencies import (
    get_export_service,
    get_pipeline_service,
    get_repository,
    get_youtube_service,
)
from app.schemas.video import (
    ActionItemsResponse,
    ChaptersResponse,
    ExportFormat,
    ProcessVideoRequest,
    ProcessVideoResponse,
    StudyGuideResponse,
    TimestampsResponse,
    UploadVideoResponse,
    VideoSummaryResponse,
)
from app.services.export_service import ExportService
from app.services.pipeline import VideoPipelineService
from app.services.repository import ArtifactRepository
from app.services.youtube_service import YouTubeService

router = APIRouter(tags=["videos"])
logger = logging.getLogger("uvicorn.error")


@router.post("/upload-video", response_model=UploadVideoResponse)
async def upload_video(
    file: UploadFile = File(...),
    repository: ArtifactRepository = Depends(get_repository),
) -> UploadVideoResponse:
    upload_id, target = await repository.save_upload(file)
    logger.info(
        "Upload saved | upload_id=%s | filename=%s",
        upload_id,
        target.name,
    )
    return UploadVideoResponse(
        upload_id=upload_id,
        filename=target.name,
        content_type=file.content_type,
    )


@router.post("/process-video", response_model=ProcessVideoResponse)
async def process_video(
    request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    pipeline: VideoPipelineService = Depends(get_pipeline_service),
    repository: ArtifactRepository = Depends(get_repository),
    youtube_service: YouTubeService = Depends(get_youtube_service),
) -> ProcessVideoResponse:
    source_url = request.youtube_url or request.source_url
    if source_url and not request.upload_id:
        unsupported_message = youtube_service.unsupported_source_message(source_url)
        if unsupported_message is not None:
            raise HTTPException(status_code=400, detail=unsupported_message)

    artifact = pipeline.create_job(request)
    logger.info(
        "Job queued | job_id=%s | source=%s | url=%s | upload_id=%s",
        artifact.job_id,
        artifact.source_type,
        source_url,
        request.upload_id,
    )
    background_tasks.add_task(pipeline.process_video, artifact.job_id, request)
    steps = repository.list_steps(artifact.status)
    return ProcessVideoResponse(
        job_id=artifact.job_id,
        status=artifact.status,
        steps=steps,
        message="Video queued for processing.",
    )


@router.get("/video-summary/{job_id}", response_model=VideoSummaryResponse)
async def get_video_summary(
    job_id: str,
    pipeline: VideoPipelineService = Depends(get_pipeline_service),
) -> VideoSummaryResponse:
    try:
        artifact = pipeline.get_summary(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return VideoSummaryResponse(job_id=job_id, status=artifact.status, artifact=artifact)


@router.get("/timestamps/{job_id}", response_model=TimestampsResponse)
async def get_timestamps(
    job_id: str,
    pipeline: VideoPipelineService = Depends(get_pipeline_service),
) -> TimestampsResponse:
    try:
        artifact = pipeline.get_summary(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TimestampsResponse(job_id=job_id, status=artifact.status, timestamps=artifact.timestamps)


@router.get("/chapters/{job_id}", response_model=ChaptersResponse)
async def get_chapters(
    job_id: str,
    pipeline: VideoPipelineService = Depends(get_pipeline_service),
) -> ChaptersResponse:
    try:
        artifact = pipeline.get_summary(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChaptersResponse(job_id=job_id, status=artifact.status, chapters=artifact.chapters)


@router.get("/action-items/{job_id}", response_model=ActionItemsResponse)
async def get_action_items(
    job_id: str,
    pipeline: VideoPipelineService = Depends(get_pipeline_service),
) -> ActionItemsResponse:
    try:
        artifact = pipeline.get_summary(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ActionItemsResponse(
        job_id=job_id,
        status=artifact.status,
        action_items=artifact.action_items,
    )


@router.get("/study-guide/{job_id}", response_model=StudyGuideResponse)
async def get_study_guide(
    job_id: str,
    pipeline: VideoPipelineService = Depends(get_pipeline_service),
) -> StudyGuideResponse:
    try:
        artifact = pipeline.get_summary(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StudyGuideResponse(
        job_id=job_id,
        status=artifact.status,
        learning_objectives=artifact.learning_objectives,
        glossary=artifact.glossary,
        study_questions=artifact.study_questions,
        analysis_metrics=artifact.analysis_metrics,
    )


@router.get("/export/{job_id}")
async def export_notes(
    job_id: str,
    format: ExportFormat = Query(default=ExportFormat.markdown),
    pipeline: VideoPipelineService = Depends(get_pipeline_service),
    export_service: ExportService = Depends(get_export_service),
) -> Response:
    try:
        artifact = pipeline.get_summary(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    content, media_type, filename = export_service.render(artifact, format)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=content, media_type=media_type, headers=headers)


@router.get("/media/{job_id}")
async def stream_media(
    job_id: str,
    repository: ArtifactRepository = Depends(get_repository),
) -> FileResponse:
    try:
        media_path = repository.resolve_media(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    media_type, _ = mimetypes.guess_type(media_path.name)
    return FileResponse(
        media_path,
        media_type=media_type or "application/octet-stream",
        filename=media_path.name,
    )
