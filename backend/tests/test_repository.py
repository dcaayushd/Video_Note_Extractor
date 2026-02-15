from pathlib import Path

from app.core.config import Settings
from app.schemas.video import ProcessVideoRequest
from app.services.repository import ArtifactRepository


def test_repository_persists_artifacts_in_sqlite(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        demo_mode=False,
        uploads_dir=tmp_path / "uploads",
        database_path=tmp_path / "video_note_extractor.db",
        chroma_persist_dir=tmp_path / "chroma",
        temp_dir=tmp_path / "tmp",
    )
    repository = ArtifactRepository(settings)

    artifact = repository.create_job(
        ProcessVideoRequest(
            youtube_url="https://www.youtube.com/watch?v=demo123",
            title="Architecture Walkthrough",
        )
    )
    artifact.quick_summary = "Stored in SQLite."
    repository.save_artifact(artifact)

    reloaded = ArtifactRepository(settings).load_artifact(artifact.job_id)

    assert reloaded.title == "Architecture Walkthrough"
    assert reloaded.quick_summary == "Stored in SQLite."
