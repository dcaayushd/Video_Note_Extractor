import json
from pathlib import Path
from unittest.mock import Mock

from app.core.config import Settings
from app.models import TranscriptSegment, VideoAnalysisArtifact
from app.services.pipeline import VideoPipelineService
from app.services.youtube_service import UnsupportedSourceError, YouTubeService
from app.transcription.whisper_service import WhisperTranscriptionService


def test_load_source_transcript_from_json3(tmp_path: Path) -> None:
    transcript_path = tmp_path / "captions.json3"
    transcript_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "tStartMs": 0,
                        "dDurationMs": 1800,
                        "segs": [{"utf8": "Product onboarding needs a clearer first step. "}],
                    },
                    {
                        "tStartMs": 2000,
                        "dDurationMs": 2200,
                        "segs": [{"utf8": "The team also assigns design review by Friday."}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    service = WhisperTranscriptionService(Settings(_env_file=None, demo_mode=False))
    result = service.load_source_transcript(transcript_path)

    assert result is not None
    transcript, segments = result
    assert "product onboarding" in transcript.lower()
    assert len(segments) == 2
    assert segments[0].start_seconds == 0
    assert segments[1].end_seconds >= 4


def test_load_source_transcript_from_vtt(tmp_path: Path) -> None:
    transcript_path = tmp_path / "captions.vtt"
    transcript_path.write_text(
        (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.500\n"
            "<c>We begin by defining the migration plan.</c>\n\n"
            "00:00:02.700 --> 00:00:05.000\n"
            "Next, the team maps risks and owners.\n"
        ),
        encoding="utf-8",
    )

    service = WhisperTranscriptionService(Settings(_env_file=None, demo_mode=False))
    result = service.load_source_transcript(transcript_path)

    assert result is not None
    transcript, segments = result
    assert "migration plan" in transcript.lower()
    assert len(segments) == 2
    assert segments[1].start_seconds > segments[0].start_seconds


def test_load_source_transcript_from_srv3_xml(tmp_path: Path) -> None:
    transcript_path = tmp_path / "captions.srv3"
    transcript_path.write_text(
        (
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<timedtext>\n"
            "  <body>\n"
            "    <p t=\"0\" d=\"1800\">We define the function.</p>\n"
            "    <p t=\"1900\" d=\"2200\">Then we add parameters and call it.</p>\n"
            "  </body>\n"
            "</timedtext>\n"
        ),
        encoding="utf-8",
    )

    service = WhisperTranscriptionService(Settings(_env_file=None, demo_mode=False))
    result = service.load_source_transcript(transcript_path)

    assert result is not None
    transcript, segments = result
    assert "define the function" in transcript.lower()
    assert "add parameters" in transcript.lower()
    assert len(segments) == 2
    assert segments[1].start_seconds > segments[0].start_seconds


def test_load_source_transcript_trims_overlapping_caption_prefixes(tmp_path: Path) -> None:
    transcript_path = tmp_path / "captions.vtt"
    transcript_path.write_text(
        (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "We define the function\n\n"
            "00:00:02.000 --> 00:00:04.500\n"
            "We define the function and call it with two inputs.\n"
        ),
        encoding="utf-8",
    )

    service = WhisperTranscriptionService(Settings(_env_file=None, demo_mode=False))
    result = service.load_source_transcript(transcript_path)

    assert result is not None
    transcript, segments = result
    assert "we define the function and call it with two inputs" in transcript.lower()
    assert "we define the function we define the function" not in transcript.lower()
    assert 1 <= len(segments) <= 2


def test_source_transcript_quality_rejects_repetitive_noise() -> None:
    service = WhisperTranscriptionService(Settings(_env_file=None, demo_mode=False))
    usable = service.source_transcript_is_usable(
        "Okay okay okay",
        [
            TranscriptSegment(start_seconds=0, end_seconds=1, text="Okay"),
            TranscriptSegment(start_seconds=1, end_seconds=2, text="Okay"),
            TranscriptSegment(start_seconds=2, end_seconds=3, text="Okay"),
        ],
    )

    assert usable is False


def test_youtube_service_prefers_manual_subtitles_over_automatic_captions() -> None:
    service = YouTubeService(Settings(_env_file=None, demo_mode=False))

    asset = service._select_source_transcript_asset(
        {
            "language": "en",
            "subtitles": {
                "en": [
                    {"ext": "vtt", "url": "https://example.com/manual.vtt"},
                    {"ext": "json3", "url": "https://example.com/manual.json3"},
                ]
            },
            "automatic_captions": {
                "en": [
                    {"ext": "json3", "url": "https://example.com/auto.json3"},
                ]
            },
        }
    )

    assert asset is not None
    assert asset["kind"] == "manual_subtitles"
    assert asset["ext"] == "json3"
    assert asset["language"] == "en"


def test_youtube_service_rejects_spotify_links_before_download(tmp_path: Path) -> None:
    service = YouTubeService(Settings(_env_file=None, demo_mode=False))

    try:
        service.download_media("https://open.spotify.com/episode/example123", tmp_path)
    except UnsupportedSourceError as exc:
        assert "Spotify links are not supported" in str(exc)
    else:
        raise AssertionError("Expected Spotify links to be rejected.")


def test_pipeline_prefers_source_transcript_before_whisper(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, demo_mode=False)
    transcription_service = WhisperTranscriptionService(settings)
    pipeline = VideoPipelineService(
        settings=settings,
        repository=Mock(),
        transcription_service=transcription_service,
        summarization_service=Mock(),
        vector_store=Mock(),
        youtube_service=Mock(),
    )
    transcript_path = tmp_path / "captions.vtt"
    transcript_path.write_text("WEBVTT\n", encoding="utf-8")
    artifact = VideoAnalysisArtifact(
        job_id="job-1",
        title="Planning Review",
        source_type="youtube",
        metadata={
            "source_transcript_path": str(transcript_path),
            "source_transcript_kind": "manual_subtitles",
        },
    )

    load_source_transcript = Mock(
        return_value=(
            "Source transcript text.",
            [
                TranscriptSegment(
                    start_seconds=0,
                    end_seconds=3,
                    text="Source transcript text.",
                )
            ],
        )
    )
    transcribe = Mock(
        return_value=(
            "Whisper transcript text.",
            [
                TranscriptSegment(
                    start_seconds=0,
                    end_seconds=3,
                    text="Whisper transcript text.",
                )
            ],
        )
    )
    monkeypatch.setattr(
        transcription_service,
        "load_source_transcript",
        load_source_transcript,
    )
    monkeypatch.setattr(transcription_service, "transcribe", transcribe)

    transcript, segments, source = pipeline._resolve_transcript(
        artifact,
        tmp_path / "video.mp4",
    )

    assert transcript == "Source transcript text."
    assert len(segments) == 1
    assert source == "manual_subtitles"
    load_source_transcript.assert_called_once()
    transcribe.assert_not_called()


def test_pipeline_falls_back_to_whisper_when_source_transcript_is_weak(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, demo_mode=False)
    transcription_service = WhisperTranscriptionService(settings)
    pipeline = VideoPipelineService(
        settings=settings,
        repository=Mock(),
        transcription_service=transcription_service,
        summarization_service=Mock(),
        vector_store=Mock(),
        youtube_service=Mock(),
    )
    transcript_path = tmp_path / "captions.vtt"
    transcript_path.write_text("WEBVTT\n", encoding="utf-8")
    artifact = VideoAnalysisArtifact(
        job_id="job-2",
        title="Planning Review",
        source_type="youtube",
        metadata={
            "source_transcript_path": str(transcript_path),
            "source_transcript_kind": "manual_subtitles",
        },
    )

    load_source_transcript = Mock(
        return_value=(
            "Okay okay okay",
            [
                TranscriptSegment(start_seconds=0, end_seconds=1, text="Okay"),
                TranscriptSegment(start_seconds=1, end_seconds=2, text="Okay"),
                TranscriptSegment(start_seconds=2, end_seconds=3, text="Okay"),
            ],
        )
    )
    transcribe = Mock(
        return_value=(
            "Whisper transcript text.",
            [
                TranscriptSegment(
                    start_seconds=0,
                    end_seconds=3,
                    text="Whisper transcript text.",
                )
            ],
        )
    )
    monkeypatch.setattr(
        transcription_service,
        "load_source_transcript",
        load_source_transcript,
    )
    monkeypatch.setattr(transcription_service, "transcribe", transcribe)

    transcript, segments, source = pipeline._resolve_transcript(
        artifact,
        tmp_path / "video.mp4",
    )

    assert transcript == "Whisper transcript text."
    assert len(segments) == 1
    assert source == "whisper"
    load_source_transcript.assert_called_once()
    transcribe.assert_called_once()
