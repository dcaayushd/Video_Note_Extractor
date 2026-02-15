import os
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.dependencies import (
    get_database,
    get_export_service,
    get_pipeline_service,
    get_repository,
    get_summarization_service,
    get_transcription_service,
    get_vector_store,
    get_youtube_service,
)


def reset_caches() -> None:
    get_settings.cache_clear()
    get_database.cache_clear()
    get_repository.cache_clear()
    get_pipeline_service.cache_clear()
    get_transcription_service.cache_clear()
    get_summarization_service.cache_clear()
    get_vector_store.cache_clear()
    get_youtube_service.cache_clear()
    get_export_service.cache_clear()


def test_process_video_demo_flow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "video_note_extractor.db"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "tmp"))
    reset_caches()

    from app.main import app

    client = TestClient(app)
    process_response = client.post(
        "/api/process-video",
        json={"youtube_url": "https://www.youtube.com/watch?v=demo123", "title": "Demo"},
    )
    assert process_response.status_code == 200
    job_id = process_response.json()["job_id"]

    summary_response = client.get(f"/api/video-summary/{job_id}")
    assert summary_response.status_code == 200
    assert summary_response.json()["artifact"]["status"] == "completed"
    assert summary_response.json()["artifact"]["chapters"]
    assert summary_response.json()["artifact"]["study_questions"]

    export_response = client.get(f"/api/export/{job_id}?format=markdown")
    assert export_response.status_code == 200
    assert "# Demo" in export_response.text
    assert "## Chapters" in export_response.text

    chapters_response = client.get(f"/api/chapters/{job_id}")
    assert chapters_response.status_code == 200
    assert chapters_response.json()["chapters"]

    study_guide_response = client.get(f"/api/study-guide/{job_id}")
    assert study_guide_response.status_code == 200
    assert study_guide_response.json()["glossary"]

    os.environ.pop("DEMO_MODE", None)
    reset_caches()


def test_process_video_rejects_spotify_links(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "video_note_extractor.db"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "tmp"))
    reset_caches()

    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/process-video",
        json={"source_url": "https://open.spotify.com/episode/example123"},
    )

    assert response.status_code == 400
    assert "Spotify links are not supported" in response.json()["detail"]

    os.environ.pop("DEMO_MODE", None)
    reset_caches()
