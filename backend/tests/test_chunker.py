from app.core.config import Settings
from app.schemas.video import TranscriptSegment
from app.services.transcript_chunker import TranscriptChunker


def test_chunker_groups_segments() -> None:
    chunker = TranscriptChunker(
        Settings(
            _env_file=None,
            chunk_size=60,
            chunk_overlap=15,
            min_chunk_characters=20,
        )
    )
    segments = [
        TranscriptSegment(start_seconds=0, end_seconds=20, text="A" * 30),
        TranscriptSegment(start_seconds=20, end_seconds=40, text="B" * 30),
        TranscriptSegment(start_seconds=40, end_seconds=60, text="C" * 30),
    ]

    chunks = chunker.chunk(segments)

    assert len(chunks) == 2
    assert chunks[0].start_seconds == 0
    assert chunks[1].start_seconds == 20
    assert chunks[-1].end_seconds == 60
