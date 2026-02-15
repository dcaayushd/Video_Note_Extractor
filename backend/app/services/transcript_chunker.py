from __future__ import annotations

from app.core.config import Settings
from app.models import TranscriptChunk, TranscriptSegment
from app.utils.text_intelligence import (
    build_headline,
    contains_transition_cue,
    extract_keywords,
    normalize_whitespace,
    split_sentences,
)


class TranscriptChunker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chunk(self, segments: list[TranscriptSegment]) -> list[TranscriptChunk]:
        cleaned_segments = [
            segment.model_copy(update={"text": normalize_whitespace(segment.text)})
            for segment in segments
            if normalize_whitespace(segment.text)
        ]
        if not cleaned_segments:
            return []

        chunks: list[TranscriptChunk] = []
        current_segments: list[TranscriptSegment] = []
        current_length = 0

        for index, segment in enumerate(cleaned_segments):
            current_segments.append(segment)
            current_length += len(segment.text)
            next_segment = (
                cleaned_segments[index + 1]
                if index + 1 < len(cleaned_segments)
                else None
            )
            if not self._should_close_chunk(
                current_segments,
                next_segment=next_segment,
                current_length=current_length,
            ):
                continue

            chunks.append(self._build_chunk(current_segments, len(chunks)))
            current_segments = self._carry_overlap(current_segments)
            current_length = sum(len(item.text) for item in current_segments)

        if current_segments:
            final_end = current_segments[-1].end_seconds
            if not chunks or chunks[-1].end_seconds < final_end:
                chunks.append(self._build_chunk(current_segments, len(chunks)))

        return chunks

    def _should_close_chunk(
        self,
        current_segments: list[TranscriptSegment],
        *,
        next_segment: TranscriptSegment | None,
        current_length: int,
    ) -> bool:
        if not current_segments:
            return False
        if next_segment is None:
            return True
        effective_minimum = min(
            self.settings.min_chunk_characters,
            max(1, int(self.settings.chunk_size * 0.5)),
        )
        if current_length < effective_minimum:
            return False

        current_last = current_segments[-1]
        time_gap = max(0.0, next_segment.start_seconds - current_last.end_seconds)
        if current_length >= self.settings.chunk_size:
            return True
        if time_gap >= 12:
            return True
        if contains_transition_cue(next_segment.text) and current_length >= int(
            self.settings.chunk_size * 0.65
        ):
            return True
        if current_length >= int(self.settings.chunk_size * 0.8):
            return self._topic_shift(current_last.text, next_segment.text) >= 0.6
        return False

    def _carry_overlap(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        if self.settings.chunk_overlap <= 0 or not segments:
            return []
        overlap_segments: list[TranscriptSegment] = []
        carried_length = 0
        for segment in reversed(segments):
            overlap_segments.append(segment)
            carried_length += len(segment.text)
            if carried_length >= self.settings.chunk_overlap:
                break
        overlap_segments.reverse()
        return overlap_segments

    def _topic_shift(self, left_text: str, right_text: str) -> float:
        left_keywords = {item.lower() for item in extract_keywords(left_text, limit=5)}
        right_keywords = {item.lower() for item in extract_keywords(right_text, limit=5)}
        if not left_keywords or not right_keywords:
            return 0.0
        overlap = len(left_keywords & right_keywords)
        union = len(left_keywords | right_keywords)
        return 1 - (overlap / union)

    def _build_chunk(self, segments: list[TranscriptSegment], index: int) -> TranscriptChunk:
        text = " ".join(segment.text for segment in segments if segment.text.strip())
        keywords = extract_keywords(text, limit=3)
        sentences = split_sentences(text, minimum_words=4)
        focus_source = keywords[0] if keywords else (sentences[0] if sentences else text)
        return TranscriptChunk(
            chunk_id=f"chunk-{index}",
            start_seconds=segments[0].start_seconds,
            end_seconds=segments[-1].end_seconds,
            semantic_focus=build_headline(
                focus_source,
                fallback=f"Segment group {index + 1}",
                max_words=7,
            ),
            text=text,
        )
