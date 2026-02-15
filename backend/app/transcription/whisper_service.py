from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from html import unescape
from pathlib import Path
import xml.etree.ElementTree as ET

from app.core.config import Settings
from app.models import TranscriptSegment, TranscriptWord
from app.utils.text_intelligence import normalize_whitespace

_CAPTION_NOISE_TOKENS = {
    "ah",
    "hmm",
    "mm",
    "oh",
    "ok",
    "okay",
    "uh",
    "um",
    "yeah",
}


class WhisperTranscriptionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._model_lock = threading.Lock()

    def transcribe(
        self,
        media_path: Path,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> tuple[str, list[TranscriptSegment]]:
        if self.settings.demo_mode:
            sample_segments = self._demo_segments()
            if progress_callback is not None:
                progress_callback(100, "100% | sample transcript ready")
            return " ".join(segment.text for segment in sample_segments), sample_segments

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install faster-whisper to enable real transcription.") from exc

        model = self._get_model(WhisperModel)
        transcription_options: dict[str, object] = {
            "word_timestamps": True,
            "vad_filter": True,
            "beam_size": self.settings.whisper_beam_size,
            "best_of": self.settings.whisper_best_of,
            "temperature": self.settings.whisper_temperature,
            "condition_on_previous_text": self.settings.whisper_condition_on_previous_text,
        }
        if self.settings.transcription_language:
            transcription_options["language"] = self.settings.transcription_language

        segments, info = model.transcribe(str(media_path), **transcription_options)

        transcript_segments: list[TranscriptSegment] = []
        duration = float(info.duration or 0)
        last_percent = -1

        for index, segment in enumerate(segments):
            normalized_words = [
                TranscriptWord(
                    word=normalize_whitespace(word.word),
                    start_seconds=word.start,
                    end_seconds=word.end,
                )
                for word in (segment.words or [])
                if normalize_whitespace(word.word)
            ]
            cleaned_text = self._clean_text(segment.text)
            if not cleaned_text:
                continue
            transcript_segments.append(
                TranscriptSegment(
                    segment_id=f"segment-{index}",
                    start_seconds=segment.start,
                    end_seconds=segment.end,
                    text=cleaned_text,
                    words=normalized_words,
                )
            )
            if progress_callback is not None and duration > 0:
                percent = min(100, int((segment.end / duration) * 100))
                if percent > last_percent:
                    last_percent = percent
                    progress_callback(
                        percent,
                        f"{percent}% | segment {index + 1} | {int(segment.end)}s/{int(duration)}s",
                    )

        transcript_segments = self._post_process_segments(transcript_segments)
        transcript = " ".join(segment.text for segment in transcript_segments)
        if progress_callback is not None and last_percent < 100:
            progress_callback(100, "100% | transcription finished")
        return transcript, transcript_segments

    def load_source_transcript(
        self,
        transcript_path: Path,
    ) -> tuple[str, list[TranscriptSegment]] | None:
        if not transcript_path.exists():
            return None

        try:
            suffix = transcript_path.suffix.lower()
            if suffix == ".json3":
                segments = self._parse_json3_transcript(transcript_path)
            elif suffix in {".vtt", ".srt"}:
                segments = self._parse_timed_text_transcript(
                    transcript_path.read_text(encoding="utf-8", errors="ignore"),
                )
            elif suffix in {".ttml", ".xml", ".srv3", ".srv2", ".srv1"}:
                segments = self._parse_xml_transcript(
                    transcript_path.read_text(encoding="utf-8", errors="ignore"),
                )
            else:
                return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None

        normalized_segments = self._normalize_source_segments(segments)
        if not normalized_segments:
            return None
        transcript = " ".join(segment.text for segment in normalized_segments)
        return transcript, normalized_segments

    def source_transcript_is_usable(
        self,
        transcript: str,
        segments: list[TranscriptSegment],
    ) -> bool:
        cleaned_transcript = normalize_whitespace(transcript)
        if not cleaned_transcript or not segments:
            return False

        informative_segments = [
            segment
            for segment in segments
            if not self._looks_like_noise_text(segment.text)
        ]
        if not informative_segments:
            return False

        segment_word_counts = [
            len(self._alpha_tokens(segment.text))
            for segment in informative_segments
        ]
        if not segment_word_counts:
            return False
        if len(informative_segments) == 1:
            return segment_word_counts[0] >= 3

        transcript_tokens = self._alpha_tokens(cleaned_transcript)
        if len(transcript_tokens) < 6:
            return False

        average_words = sum(segment_word_counts) / len(segment_word_counts)
        unique_ratio = len(
            {
                self._text_signature(segment.text)
                for segment in informative_segments
            }
        ) / len(informative_segments)
        repeated_pairs = sum(
            1
            for previous, current in zip(informative_segments, informative_segments[1:])
            if self._texts_are_near_duplicates(previous.text, current.text)
        )
        repeated_ratio = repeated_pairs / max(1, len(informative_segments) - 1)
        return (
            average_words >= 2.2
            and unique_ratio >= 0.45
            and repeated_ratio <= 0.55
        )

    def _get_model(self, whisper_model_cls):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                self._model = whisper_model_cls(
                    self.settings.whisper_model_size,
                    device="auto",
                    compute_type=self.settings.whisper_compute_type,
                )
        return self._model

    def _demo_segments(self) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                start_seconds=0,
                end_seconds=52,
                text=(
                    "Welcome everyone. Today we will examine the central framework for extracting "
                    "high-value notes from long-form educational video."
                ),
                words=[
                    TranscriptWord(word="Welcome", start_seconds=0, end_seconds=0.4),
                    TranscriptWord(word="everyone", start_seconds=0.4, end_seconds=0.9),
                ],
            ),
            TranscriptSegment(
                start_seconds=52,
                end_seconds=126,
                text=(
                    "The first priority is reliable transcription because every downstream summary, "
                    "chapter boundary, and study guide depends on that signal quality."
                ),
            ),
            TranscriptSegment(
                start_seconds=126,
                end_seconds=214,
                text=(
                    "Next we connect transcript evidence to semantic grouping, grounded retrieval, "
                    "and note generation so the system explains why a topic matters."
                ),
            ),
            TranscriptSegment(
                start_seconds=214,
                end_seconds=318,
                text=(
                    "A deployment-ready pipeline should also surface action items, concepts, and "
                    "discussion questions that help a student review the material quickly."
                ),
            ),
            TranscriptSegment(
                start_seconds=318,
                end_seconds=448,
                text=(
                    "Finally, we evaluate the output for clarity, organization, and academic rigor "
                    "so that the results are useful in real coursework, research meetings, and labs."
                ),
            ),
        ]

    def _clean_text(self, text: str) -> str:
        cleaned = normalize_whitespace(text)
        cleaned = cleaned.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
        cleaned = cleaned.replace(" !", "!").replace(" :", ":").replace(" ;", ";")
        return cleaned

    def _post_process_segments(
        self,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        if not segments:
            return []

        return self._normalize_segments(
            segments,
            merge_gap_seconds=0.45,
            allow_overlap_trim=False,
        )

    def _normalize_source_segments(
        self,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        return self._normalize_segments(
            segments,
            merge_gap_seconds=None,
            allow_overlap_trim=True,
        )

    def _parse_json3_transcript(self, transcript_path: Path) -> list[TranscriptSegment]:
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        events = payload.get("events")
        if not isinstance(events, list):
            return []

        rows: list[tuple[float, float | None, str]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            start_ms = event.get("tStartMs")
            if not isinstance(start_ms, (int, float)):
                continue
            segments = event.get("segs")
            if not isinstance(segments, list):
                continue
            text = self._clean_caption_text(
                "".join(
                    str(segment.get("utf8") or "")
                    for segment in segments
                    if isinstance(segment, dict)
                )
            )
            if not text:
                continue
            duration_ms = event.get("dDurationMs")
            end_seconds = (
                (float(start_ms) + float(duration_ms)) / 1000
                if isinstance(duration_ms, (int, float))
                else None
            )
            rows.append((float(start_ms) / 1000, end_seconds, text))
        return self._build_caption_segments(rows)

    def _parse_timed_text_transcript(self, content: str) -> list[TranscriptSegment]:
        blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
        rows: list[tuple[float, float | None, str]] = []
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            timing_index = next(
                (index for index, line in enumerate(lines) if "-->" in line),
                -1,
            )
            if timing_index == -1:
                continue
            start_seconds, end_seconds = self._parse_timing_line(lines[timing_index])
            if start_seconds is None:
                continue
            text = self._clean_caption_text(" ".join(lines[timing_index + 1 :]))
            if not text:
                continue
            rows.append((start_seconds, end_seconds, text))
        return self._build_caption_segments(rows)

    def _parse_xml_transcript(self, content: str) -> list[TranscriptSegment]:
        try:
            root = ET.fromstring(content.lstrip("\ufeff"))
        except ET.ParseError as exc:
            raise ValueError("Unsupported XML transcript format.") from exc

        rows: list[tuple[float, float | None, str]] = []
        for element in root.iter():
            tag_name = self._xml_local_name(element.tag)
            if tag_name not in {"p", "text"}:
                continue

            text = self._clean_caption_text(" ".join(element.itertext()))
            if not text:
                continue

            attributes = element.attrib
            start_seconds = self._parse_time_value(
                attributes.get("begin") or attributes.get("start"),
            )
            if start_seconds is None and "t" in attributes:
                start_seconds = self._parse_time_value(
                    attributes.get("t"),
                    assume_milliseconds=True,
                )
            if start_seconds is None:
                continue

            end_seconds = self._parse_time_value(attributes.get("end"))
            duration_seconds = self._parse_time_value(
                attributes.get("dur"),
            )
            if duration_seconds is None and "d" in attributes:
                duration_seconds = self._parse_time_value(
                    attributes.get("d"),
                    assume_milliseconds=True,
                )
            if end_seconds is None and duration_seconds is not None:
                end_seconds = start_seconds + duration_seconds

            rows.append((start_seconds, end_seconds, text))
        return self._build_caption_segments(rows)

    def _build_caption_segments(
        self,
        rows: list[tuple[float, float | None, str]],
    ) -> list[TranscriptSegment]:
        rows = sorted(rows, key=lambda row: (row[0], row[1] or row[0]))
        segments: list[TranscriptSegment] = []
        for index, (start_seconds, end_seconds, text) in enumerate(rows):
            next_start = rows[index + 1][0] if index + 1 < len(rows) else None
            estimated_end = start_seconds + min(8.0, max(1.4, len(text.split()) * 0.42))
            resolved_end = end_seconds if end_seconds and end_seconds > start_seconds else None
            if resolved_end is None and next_start is not None and next_start > start_seconds:
                resolved_end = next_start
            if resolved_end is None:
                resolved_end = estimated_end
            segments.append(
                TranscriptSegment(
                    segment_id=f"segment-{index}",
                    start_seconds=start_seconds,
                    end_seconds=max(start_seconds, resolved_end),
                    text=text,
                )
            )
        return segments

    def _parse_timing_line(self, line: str) -> tuple[float | None, float | None]:
        start_text, _, end_text = line.partition("-->")
        start_seconds = self._parse_timestamp_value(start_text)
        end_seconds = self._parse_timestamp_value(end_text)
        return start_seconds, end_seconds

    def _parse_timestamp_value(self, value: str) -> float | None:
        return self._parse_time_value(value)

    def _clean_caption_text(self, text: str) -> str:
        cleaned = unescape(text)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = cleaned.replace("\\n", " ").replace("\n", " ")
        cleaned = normalize_whitespace(cleaned)
        if not cleaned:
            return ""
        if re.fullmatch(r"[\[(].*[\])]", cleaned) and len(cleaned.split()) <= 3:
            return ""
        if cleaned.startswith("WEBVTT") or cleaned in {"NOTE", "STYLE"}:
            return ""
        return self._clean_text(cleaned)

    def _normalize_segments(
        self,
        segments: list[TranscriptSegment],
        *,
        merge_gap_seconds: float | None,
        allow_overlap_trim: bool,
    ) -> list[TranscriptSegment]:
        normalized: list[TranscriptSegment] = []
        ordered_segments = sorted(
            segments,
            key=lambda segment: (segment.start_seconds, segment.end_seconds),
        )
        for segment in ordered_segments:
            cleaned_text = self._clean_text(segment.text)
            cleaned_text = self._collapse_repeated_phrases(cleaned_text)
            if normalized and allow_overlap_trim:
                cleaned_text = self._trim_repeated_prefix(
                    normalized[-1].text,
                    cleaned_text,
                )
            if not cleaned_text or self._looks_like_noise_text(cleaned_text):
                continue

            candidate = TranscriptSegment(
                segment_id=segment.segment_id,
                start_seconds=max(0.0, segment.start_seconds),
                end_seconds=max(segment.start_seconds, segment.end_seconds),
                text=cleaned_text,
                words=segment.words,
            )

            if normalized and self._segments_are_duplicates(normalized[-1], candidate):
                normalized[-1] = self._merge_segments(normalized[-1], candidate)
                continue
            if merge_gap_seconds is not None and normalized and self._should_merge_segments(
                normalized[-1],
                candidate,
                merge_gap_seconds=merge_gap_seconds,
            ):
                normalized[-1] = self._merge_segments(normalized[-1], candidate)
                continue

            normalized.append(candidate)

        return [
            TranscriptSegment(
                segment_id=f"segment-{index}",
                start_seconds=segment.start_seconds,
                end_seconds=max(segment.start_seconds, segment.end_seconds),
                text=self._clean_text(segment.text),
                words=segment.words,
            )
            for index, segment in enumerate(normalized)
        ]

    def _should_merge_segments(
        self,
        previous: TranscriptSegment,
        current: TranscriptSegment,
        *,
        merge_gap_seconds: float,
    ) -> bool:
        gap = max(0.0, current.start_seconds - previous.end_seconds)
        short_segment = len(current.text.split()) <= 5 or (
            current.end_seconds - current.start_seconds
        ) <= 2.5
        return (
            gap <= merge_gap_seconds
            and (short_segment or len(previous.text.split()) <= 8)
            and len((previous.text + " " + current.text).split()) <= 60
        )

    def _merge_segments(
        self,
        previous: TranscriptSegment,
        current: TranscriptSegment,
    ) -> TranscriptSegment:
        current_delta = self._trim_repeated_prefix(previous.text, current.text)
        merged_text = previous.text if not current_delta else f"{previous.text} {current_delta}"
        return TranscriptSegment(
            segment_id=previous.segment_id,
            start_seconds=previous.start_seconds,
            end_seconds=max(previous.end_seconds, current.end_seconds),
            text=self._collapse_repeated_phrases(self._clean_text(merged_text)),
            words=[*previous.words, *current.words],
        )

    def _segments_are_duplicates(
        self,
        previous: TranscriptSegment,
        current: TranscriptSegment,
    ) -> bool:
        return self._texts_are_near_duplicates(previous.text, current.text)

    def _texts_are_near_duplicates(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        left_key = self._text_signature(left)
        right_key = self._text_signature(right)
        if not left_key or not right_key:
            return False
        if left_key == right_key:
            return True
        left_tokens = left_key.split()
        right_tokens = right_key.split()
        overlap = len(set(left_tokens) & set(right_tokens))
        minimum = max(1, min(len(left_tokens), len(right_tokens)))
        return overlap / minimum >= 0.85

    def _trim_repeated_prefix(self, previous_text: str, current_text: str) -> str:
        previous_tokens = self._signature_tokens(previous_text)
        current_tokens = self._signature_tokens(current_text)
        if not previous_tokens or not current_tokens:
            return current_text

        max_overlap = min(8, len(previous_tokens), len(current_tokens))
        for overlap in range(max_overlap, 1, -1):
            if previous_tokens[-overlap:] == current_tokens[:overlap]:
                raw_tokens = current_text.split()
                if overlap >= len(raw_tokens):
                    return ""
                return self._clean_text(" ".join(raw_tokens[overlap:]))
        return current_text

    def _collapse_repeated_phrases(self, text: str) -> str:
        tokens = text.split()
        if len(tokens) < 3:
            return text

        collapsed: list[str] = []
        index = 0
        while index < len(tokens):
            collapsed_run = False
            max_size = min(4, (len(tokens) - index) // 2)
            for size in range(max_size, 0, -1):
                left = [self._normalize_token(token) for token in tokens[index : index + size]]
                right = [self._normalize_token(token) for token in tokens[index + size : index + (size * 2)]]
                if left != right or not any(left):
                    continue
                collapsed.extend(tokens[index : index + size])
                index += size * 2
                while index + size <= len(tokens):
                    next_run = [
                        self._normalize_token(token)
                        for token in tokens[index : index + size]
                    ]
                    if next_run != left:
                        break
                    index += size
                collapsed_run = True
                break
            if collapsed_run:
                continue
            collapsed.append(tokens[index])
            index += 1
        return self._clean_text(" ".join(collapsed))

    def _looks_like_noise_text(self, text: str) -> bool:
        normalized = normalize_whitespace(text)
        if not normalized:
            return True
        alpha_tokens = self._alpha_tokens(normalized)
        if not alpha_tokens:
            return True
        lowered_tokens = [token.lower() for token in alpha_tokens]
        if len(lowered_tokens) <= 2 and all(token in _CAPTION_NOISE_TOKENS for token in lowered_tokens):
            return True
        if len(set(lowered_tokens)) == 1 and len(lowered_tokens) >= 3:
            return True
        if re.fullmatch(r"[\[(][^\])]{1,32}[\])]", normalized):
            return True
        return False

    def _alpha_tokens(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z][A-Za-z0-9+#'\-]*", text)

    def _signature_tokens(self, text: str) -> list[str]:
        return [
            token
            for token in (self._normalize_token(part) for part in text.split())
            if token
        ]

    def _text_signature(self, text: str) -> str:
        return " ".join(self._signature_tokens(text))

    def _normalize_token(self, token: str) -> str:
        return re.sub(r"[^a-z0-9+#]+", "", token.lower())

    def _xml_local_name(self, tag: str) -> str:
        if "}" in tag:
            return tag.rsplit("}", 1)[-1].lower()
        return tag.lower()

    def _parse_time_value(
        self,
        value: str | None,
        *,
        assume_milliseconds: bool = False,
    ) -> float | None:
        if value is None:
            return None
        cleaned = normalize_whitespace(str(value))
        if not cleaned:
            return None
        cleaned = cleaned.split()[0]
        lowered = cleaned.lower()
        if lowered.endswith("ms"):
            try:
                return float(lowered[:-2]) / 1000
            except ValueError:
                return None
        if lowered.endswith("s"):
            try:
                return float(lowered[:-1])
            except ValueError:
                return None
        if ":" in lowered:
            parts = lowered.replace(",", ".").split(":")
            try:
                if len(parts) == 3:
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return (hours * 3600) + (minutes * 60) + seconds
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return (minutes * 60) + seconds
            except ValueError:
                return None
        try:
            numeric = float(lowered)
        except ValueError:
            return None
        if assume_milliseconds:
            return numeric / 1000
        return numeric
