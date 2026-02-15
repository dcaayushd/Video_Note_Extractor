from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.models import (
    ActionItem,
    AnalysisMetrics,
    ChapterItem,
    GlossaryItem,
    MindMapNode,
    NoteSection,
    QuoteItem,
    StudyQuestion,
    TimestampItem,
    TranscriptChunk,
    VideoAnalysisArtifact,
)
from app.utils.json_tools import extract_json_object
from app.utils.text_intelligence import (
    academic_signal_score,
    build_headline,
    contains_transition_cue,
    estimate_reading_minutes,
    extract_keywords,
    is_low_signal_topic,
    lexical_diversity,
    normalize_whitespace,
    sample_evenly,
    split_sentences,
    strip_timecodes,
    tokenize,
    truncate_text,
)
from app.utils.timecode import build_youtube_jump_url, format_timestamp

logger = logging.getLogger("uvicorn.error")


class SummarizationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llm: Any | None = None

    def summarize(
        self,
        artifact: VideoAnalysisArtifact,
        chunks: list[TranscriptChunk],
    ) -> VideoAnalysisArtifact:
        fallback_artifact = self._postprocess_artifact(
            self._heuristic_summary(
                artifact.model_copy(deep=True),
                chunks,
            )
        )
        if not self.settings.use_local_llm:
            return fallback_artifact

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError("Install langchain-ollama to enable local summarization.") from exc

        llm = self._llm
        if llm is None:
            llm = ChatOllama(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                temperature=0.15,
            )
            self._llm = llm

        llm_context = self._build_llm_context(fallback_artifact, chunks)

        prompt = [
            SystemMessage(
                content=(
                    "You are an academically rigorous transcript analyst. Return only valid JSON with the keys "
                    "quick_summary, five_minute_summary, key_topics, note_sections, chapters, timestamps, "
                    "action_items, key_quotes, learning_objectives, glossary, study_questions, "
                    "analysis_metrics, and mind_map. Keep everything grounded in the supplied transcript context. "
                    "Use concise, high-signal concept labels for key_topics, chapters, glossary terms, timeline "
                    "labels, and the mind_map. Avoid conversational fragments, filler words, pronouns, greetings, "
                    "timecodes, and sentence openings. Prefer academically precise noun phrases such as "
                    "'Remote SSH Connections', 'Dev Containers', or 'Pull Request Review Workflow'. For action_items, "
                    "capture explicit tasks when the speaker assigns them; otherwise synthesize strong study actions "
                    "tied to the chapter evidence. Avoid repeating the same concept across neighboring summary bullets, "
                    "chapter headings, notes, timeline items, actions, or mind-map leaves."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "title": artifact.title,
                        "source_url": artifact.source_url,
                        "analysis_draft": llm_context,
                    },
                    indent=2,
                )
            ),
        ]
        try:
            response = llm.invoke(prompt)
            data = extract_json_object(self._stringify_response(response.content))
        except Exception:
            return fallback_artifact
        try:
            llm_artifact = self._apply_llm_summary(
                artifact.model_copy(deep=True),
                data,
            )
            return self._postprocess_artifact(
                self._merge_with_fallback(llm_artifact, fallback_artifact)
            )
        except Exception:
            logger.exception("LLM summary normalization failed; falling back to heuristic summary.")
            return fallback_artifact

    def _stringify_response(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(item) for item in content)
        return str(content)

    def _build_llm_context(
        self,
        fallback_artifact: VideoAnalysisArtifact,
        chunks: list[TranscriptChunk],
    ) -> dict[str, Any]:
        sampled_chunks = sample_evenly(chunks, self.settings.summary_context_chunks)
        sampled_chapters = sorted(
            sample_evenly(
                fallback_artifact.chapters,
                min(len(fallback_artifact.chapters), self.settings.max_chapters),
            ),
            key=lambda chapter: chapter.start_seconds,
        )
        sampled_sections = sorted(
            sample_evenly(
                fallback_artifact.note_sections,
                min(len(fallback_artifact.note_sections), 4),
            ),
            key=lambda section: float(section.start_seconds or 0.0),
        )
        transcript_topics = self._sanitize_topic_list(
            [
                *fallback_artifact.key_topics,
                *extract_keywords(" ".join(chunk.text for chunk in sampled_chunks), limit=10),
            ],
            limit=8,
        )
        compact_chunks = [
            {
                "focus": chunk.semantic_focus,
                "start_seconds": chunk.start_seconds,
                "end_seconds": chunk.end_seconds,
                "keywords": extract_keywords(chunk.text, limit=4),
                "text_excerpt": truncate_text(
                    chunk.text,
                    self.settings.summary_context_characters,
                ),
            }
            for chunk in sampled_chunks
        ]
        compact_sections = [
            {
                "heading": section.heading,
                "detail": truncate_text(section.detail, 180),
                "start_seconds": section.start_seconds,
            }
            for section in sampled_sections
        ]
        compact_chapters = [
            {
                "title": chapter.title,
                "summary": truncate_text(chapter.summary, 180),
                "start_seconds": chapter.start_seconds,
                "end_seconds": chapter.end_seconds,
                "keywords": chapter.keywords,
            }
            for chapter in sampled_chapters
        ]
        concept_evidence = [
            {
                "topic": topic,
                "supporting_chapters": [
                    chapter.title
                    for chapter in fallback_artifact.chapters
                    if topic.lower() in chapter.title.lower()
                    or topic.lower() in chapter.summary.lower()
                    or any(topic.lower() in keyword.lower() for keyword in chapter.keywords)
                ][:3],
            }
            for topic in transcript_topics[:5]
        ]
        return {
            "quick_summary": fallback_artifact.quick_summary,
            "five_minute_summary": fallback_artifact.five_minute_summary[:5],
            "key_topics": transcript_topics,
            "learning_objectives": fallback_artifact.learning_objectives[:5],
            "chapters": compact_chapters,
            "note_sections": compact_sections,
            "action_items": [
                {
                    "title": item.title,
                    "detail": truncate_text(item.detail, 180),
                    "display_time": item.display_time,
                }
                for item in fallback_artifact.action_items[:4]
            ],
            "glossary": [
                {
                    "term": item.term,
                    "definition": truncate_text(item.definition, 140),
                }
                for item in fallback_artifact.glossary[:5]
            ],
            "chunk_evidence": compact_chunks,
            "transcript_topics": transcript_topics,
            "concept_evidence": concept_evidence,
            "quality_rules": {
                "topic_labels": "Use 2-5 word concept labels and avoid partial sentences.",
                "timeline": "Timeline labels should describe the idea discussed in that segment, not quote the transcript.",
                "mind_map": "Create 3-5 concept branches with 2-3 grounded child concepts when evidence exists.",
                "actions": "If no explicit tasks are spoken, create study/review actions from the chapter evidence.",
                "coverage": "Prefer distinct concepts and phases instead of repeating the same idea with slightly different wording.",
            },
        }

    def _apply_llm_summary(
        self,
        artifact: VideoAnalysisArtifact,
        data: dict[str, Any],
    ) -> VideoAnalysisArtifact:
        artifact.quick_summary = self._coerce_text(
            data.get("quick_summary"),
            fallback="",
        )
        artifact.five_minute_summary = self._coerce_string_list(
            data.get("five_minute_summary"),
        )
        artifact.key_topics = self._sanitize_topic_list(
            self._coerce_string_list(data.get("key_topics")),
            limit=8,
        )
        artifact.note_sections = self._coerce_note_sections(data.get("note_sections"))
        artifact.chapters = self._coerce_chapters(
            artifact.source_url,
            data.get("chapters"),
        )
        artifact.timestamps = self._coerce_timestamps(
            artifact.source_url,
            data.get("timestamps"),
        )
        artifact.action_items = self._coerce_action_items(data.get("action_items"))
        artifact.key_quotes = self._coerce_quotes(data.get("key_quotes"))
        artifact.learning_objectives = self._coerce_string_list(
            data.get("learning_objectives"),
        )
        artifact.glossary = self._coerce_glossary(data.get("glossary"))
        artifact.study_questions = self._coerce_study_questions(
            data.get("study_questions"),
        )
        artifact.analysis_metrics = self._coerce_analysis_metrics(
            data.get("analysis_metrics"),
        )
        artifact.mind_map = self._coerce_mind_map(
            data.get("mind_map"),
            fallback_label=artifact.title,
        )
        return artifact

    def _merge_with_fallback(
        self,
        artifact: VideoAnalysisArtifact,
        fallback: VideoAnalysisArtifact,
    ) -> VideoAnalysisArtifact:
        if not artifact.quick_summary or self._text_is_low_quality(artifact.quick_summary):
            artifact.quick_summary = fallback.quick_summary
        if self._summary_points_are_low_quality(artifact.five_minute_summary):
            artifact.five_minute_summary = self._merge_string_lists(
                artifact.five_minute_summary,
                fallback.five_minute_summary,
                limit=5,
            )
        artifact.key_topics = self._sanitize_topic_list(
            self._merge_string_lists(
                artifact.key_topics,
                fallback.key_topics,
                limit=8,
            ),
            limit=8,
        )
        if not artifact.note_sections or self._note_sections_are_low_quality(artifact.note_sections):
            artifact.note_sections = fallback.note_sections
        if not artifact.chapters or self._chapters_are_low_quality(artifact.chapters):
            artifact.chapters = fallback.chapters
        if not artifact.timestamps or self._timestamps_are_low_quality(artifact.timestamps):
            artifact.timestamps = fallback.timestamps
        if not artifact.action_items or self._actions_are_low_quality(artifact.action_items):
            artifact.action_items = fallback.action_items
        if not artifact.key_quotes:
            artifact.key_quotes = fallback.key_quotes
        artifact.learning_objectives = self._merge_string_lists(
            artifact.learning_objectives,
            fallback.learning_objectives,
            limit=6,
        )
        if not artifact.glossary:
            artifact.glossary = fallback.glossary
        if not artifact.study_questions:
            artifact.study_questions = fallback.study_questions
        if self._analysis_metrics_is_empty(artifact.analysis_metrics):
            artifact.analysis_metrics = fallback.analysis_metrics
        if artifact.mind_map is None or self._mind_map_is_low_quality(artifact.mind_map):
            artifact.mind_map = fallback.mind_map
        return artifact

    def _heuristic_summary(
        self,
        artifact: VideoAnalysisArtifact,
        chunks: list[TranscriptChunk],
    ) -> VideoAnalysisArtifact:
        sentence_records = self._build_sentence_records(artifact, chunks)
        transcript_text = artifact.transcript_text or " ".join(chunk.text for chunk in chunks)
        title_keywords = self._title_topic_candidates(artifact.title)
        keywords = self._sanitize_topic_list(
            [
                *title_keywords,
                *extract_keywords(transcript_text, limit=8),
            ],
            limit=8,
        )
        chapters = self._build_chapters(artifact, chunks, sentence_records)
        key_topics = self._build_key_topics(
            title=artifact.title,
            transcript_text=transcript_text,
            sentence_records=sentence_records,
            chapters=chapters,
            chunks=chunks,
            keywords=keywords,
        )
        chapters = self._refine_chapters(chapters, chunks, key_topics)
        preliminary_content_format = self._infer_content_format(
            title=artifact.title,
            transcript_text=transcript_text,
            chapters=chapters,
            action_items=[],
        )
        action_items = self._build_action_items(
            artifact.source_url,
            sentence_records,
            chapters,
            content_format=preliminary_content_format,
        )
        content_format = self._infer_content_format(
            title=artifact.title,
            transcript_text=transcript_text,
            chapters=chapters,
            action_items=action_items,
        )
        note_sections = self._build_note_sections(
            chapters,
            sentence_records,
            content_format=content_format,
        )
        glossary = self._build_glossary(sentence_records, key_topics)
        learning_objectives = self._build_learning_objectives(
            chapters,
            key_topics,
            content_format=content_format,
            title=artifact.title,
            transcript_text=transcript_text,
        )
        study_questions = self._build_study_questions(
            chapters,
            action_items,
            glossary,
            learning_objectives,
        )

        artifact.quick_summary = self._build_quick_summary(
            title=artifact.title,
            transcript_text=transcript_text,
            chapters=chapters,
            keywords=keywords,
            action_items=action_items,
            content_format=content_format,
            sentence_records=sentence_records,
        )
        artifact.five_minute_summary = self._build_five_minute_summary(
            sentence_records=sentence_records,
            chapters=chapters,
            action_items=action_items,
            content_format=content_format,
        )
        artifact.key_topics = key_topics
        artifact.note_sections = note_sections
        artifact.chapters = chapters
        artifact.timestamps = self._build_timestamps(
            artifact,
            artifact.source_url,
            chapters,
            sentence_records,
            content_format=content_format,
        )
        artifact.action_items = action_items
        artifact.key_quotes = self._build_quotes(
            artifact.source_url,
            sentence_records,
            chapters,
        )
        artifact.learning_objectives = learning_objectives
        artifact.glossary = glossary
        artifact.study_questions = study_questions
        artifact.analysis_metrics = self._build_analysis_metrics(
            transcript_text=transcript_text,
            sentence_records=sentence_records,
            chapters=chapters,
            action_items=action_items,
            glossary=glossary,
            study_questions=study_questions,
        )
        artifact.mind_map = self._build_mind_map(
            title=artifact.title,
            transcript_text=transcript_text,
            key_topics=key_topics,
            chapters=chapters,
            glossary=glossary,
            action_items=action_items,
        )
        return artifact

    def _build_sentence_records(
        self,
        artifact: VideoAnalysisArtifact,
        chunks: list[TranscriptChunk],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        source_blocks = [
            {
                "text": segment.text,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "focus": "",
            }
            for segment in artifact.transcript_segments
        ]
        if not source_blocks:
            source_blocks = [
                {
                    "text": chunk.text,
                    "start_seconds": chunk.start_seconds,
                    "end_seconds": chunk.end_seconds,
                    "focus": chunk.semantic_focus,
                }
                for chunk in chunks
            ]

        global_keywords = extract_keywords(
            artifact.transcript_text or " ".join(block["text"] for block in source_blocks),
            limit=10,
        )

        for block in source_blocks:
            for sentence in split_sentences(str(block["text"]), minimum_words=5):
                cleaned_sentence = self._normalize_sentence_text(sentence)
                if cleaned_sentence is None:
                    continue
                key = cleaned_sentence.lower()
                if key in seen:
                    continue
                seen.add(key)
                sentence_keywords = extract_keywords(cleaned_sentence, limit=4)
                records.append(
                    {
                        "text": cleaned_sentence,
                        "start_seconds": float(block["start_seconds"]),
                        "end_seconds": float(block["end_seconds"]),
                        "focus": block["focus"],
                        "keywords": sentence_keywords,
                        "score": self._sentence_score(
                            cleaned_sentence,
                            sentence_keywords,
                            global_keywords,
                        ),
                    }
                )

        records.sort(
            key=lambda item: (float(item["start_seconds"]), -float(item["score"])),
        )
        return records

    def _sentence_score(
        self,
        sentence: str,
        sentence_keywords: list[str],
        global_keywords: list[str],
    ) -> float:
        lowered = sentence.lower()
        word_count = len(sentence.split())
        keyword_hits = sum(1 for keyword in global_keywords if keyword.lower() in lowered)
        actionable_hits = sum(
            1
            for marker in (
                "should",
                "need to",
                "must",
                "important",
                "key",
                "because",
                "therefore",
                "next",
                "remember",
                "evidence",
                "framework",
            )
            if marker in lowered
        )
        reasoning_hits = sum(
            1
            for marker in (
                "because",
                "which means",
                "explains",
                "highlights",
                "shows how",
                "reveals",
                "demonstrates",
                "compares",
                "agrees that",
                "decides to",
                "the key point",
            )
            if marker in lowered
        )
        sentence_focus = len(set(sentence_keywords)) * 0.32
        length_bonus = 0.8 if 9 <= word_count <= 28 else 0.3
        lexical_density = len(set(tokenize(sentence, minimum_length=4))) / max(1, word_count)
        density_bonus = min(0.75, lexical_density * 2.2)
        filler_penalty = 0.75 if re.search(r"\b(?:kind of|sort of|you know|i mean)\b", lowered) else 0.0
        repetition_penalty = 0.45 if lexical_density < 0.38 else 0.0
        return round(
            keyword_hits * 0.55
            + actionable_hits * 0.7
            + reasoning_hits * 0.6
            + sentence_focus
            + length_bonus
            + density_bonus
            - filler_penalty
            - repetition_penalty,
            4,
        )

    def _normalize_sentence_text(
        self,
        text: str,
        *,
        minimum_words: int = 5,
    ) -> str | None:
        cleaned = strip_timecodes(normalize_whitespace(text)).strip(" -•")
        if not cleaned:
            return None
        cleaned = re.sub(
            r"^(?:and|but|so|well|okay|right|now|then|of course|you know|i mean|uh|um)\b[\s,.\-:;]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        words = cleaned.split()
        if len(words) < minimum_words:
            return None
        if words[0].lower() in {"because", "even", "which", "while"} and len(words) < 10:
            return None
        if cleaned and cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        return cleaned[:1].upper() + cleaned[1:]

    def _join_labels(
        self,
        labels: list[str],
        *,
        conjunction: str = "and",
    ) -> str:
        cleaned = [normalize_whitespace(label).strip(" .") for label in labels if label.strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} {conjunction} {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, {conjunction} {cleaned[-1]}"

    def _strip_scaffold_prefix(self, text: str) -> str:
        cleaned = strip_timecodes(normalize_whitespace(text))
        return re.sub(
            r"^(?:decision|context|alignment|follow[\s-]?up|foundation|core idea|example|synthesis|"
            r"step(?:\s+\d+)?|outcome|theme|insight|takeaway|question|response|decision focus|"
            r"discussion detail|next step|core concept|evidence|why it matters|step focus|how it works)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    def _concept_signature(self, text: str) -> tuple[str, ...]:
        cleaned = self._strip_scaffold_prefix(text)
        if not cleaned:
            return ()
        generic_tokens = {
            "action",
            "actions",
            "branch",
            "chapter",
            "chapters",
            "concept",
            "concepts",
            "context",
            "decision",
            "detail",
            "details",
            "discussion",
            "example",
            "examples",
            "focus",
            "idea",
            "ideas",
            "insight",
            "insights",
            "item",
            "items",
            "moment",
            "moments",
            "note",
            "notes",
            "outcome",
            "phase",
            "point",
            "points",
            "question",
            "questions",
            "response",
            "responses",
            "section",
            "sections",
            "step",
            "steps",
            "summary",
            "takeaway",
            "takeaways",
            "theme",
            "themes",
            "timeline",
            "topic",
            "topics",
            "video",
            "videos",
        }
        concepts: list[str] = []
        for source in [*extract_keywords(cleaned, limit=5), cleaned]:
            for token in tokenize(source.lower(), minimum_length=3):
                if token in generic_tokens or token in concepts:
                    continue
                concepts.append(token)
                if len(concepts) == 6:
                    return tuple(concepts)
        return tuple(concepts)

    def _texts_are_redundant(self, left: str, right: str) -> bool:
        left_clean = self._strip_scaffold_prefix(left)
        right_clean = self._strip_scaffold_prefix(right)
        if not left_clean or not right_clean:
            return False
        left_lower = left_clean.lower()
        right_lower = right_clean.lower()
        if left_lower == right_lower:
            return True
        if left_lower in right_lower or right_lower in left_lower:
            return True
        left_signature = set(self._concept_signature(left_clean))
        right_signature = set(self._concept_signature(right_clean))
        if left_signature and right_signature:
            overlap = len(left_signature & right_signature)
            minimum_size = min(len(left_signature), len(right_signature))
            maximum_size = max(len(left_signature), len(right_signature))
            if overlap >= 3:
                return True
            if minimum_size >= 2 and overlap == minimum_size and overlap / max(1, maximum_size) >= 0.75:
                return True
        return False

    def _text_is_low_quality(self, text: str | None) -> bool:
        if text is None:
            return True
        cleaned = strip_timecodes(normalize_whitespace(text))
        if len(cleaned.split()) < 4:
            return True
        if is_low_signal_topic(cleaned) and len(cleaned.split()) <= 6:
            return True
        lowered = cleaned.lower()
        generic_starts = (
            "this video",
            "this section",
            "the video",
            "the section",
            "the transcript",
        )
        if any(lowered.startswith(prefix) for prefix in generic_starts):
            keyword_count = len(extract_keywords(cleaned, limit=3))
            if keyword_count <= 1:
                return True
        return False

    def _summary_points_are_low_quality(self, items: list[str]) -> bool:
        cleaned_items = [item for item in items if not self._text_is_low_quality(item)]
        return len(cleaned_items) < 3

    def _note_sections_are_low_quality(self, sections: list[NoteSection]) -> bool:
        if len(sections) < 2:
            return True
        strong_sections = [
            section
            for section in sections
            if not is_low_signal_topic(section.heading)
            and not self._text_is_low_quality(section.detail)
        ]
        return len(strong_sections) < 2

    def _chapters_are_low_quality(self, chapters: list[ChapterItem]) -> bool:
        if len(chapters) < 2:
            return True
        strong_chapters = [
            chapter
            for chapter in chapters
            if not is_low_signal_topic(chapter.title)
            and not self._text_is_low_quality(chapter.summary)
        ]
        return len(strong_chapters) < 2

    def _timestamps_are_low_quality(self, timestamps: list[TimestampItem]) -> bool:
        if len(timestamps) < 3:
            return True
        strong_timestamps = [
            item
            for item in timestamps
            if not is_low_signal_topic(item.label)
            and not self._text_is_low_quality(item.description)
        ]
        return len(strong_timestamps) < 3

    def _actions_are_low_quality(self, actions: list[ActionItem]) -> bool:
        if not actions:
            return True
        strong_actions = [
            item
            for item in actions
            if not is_low_signal_topic(item.title)
            and not self._text_is_low_quality(item.detail)
        ]
        return not strong_actions

    def _build_quick_summary(
        self,
        *,
        title: str,
        transcript_text: str,
        chapters: list[ChapterItem],
        keywords: list[str],
        action_items: list[ActionItem],
        content_format: str | None,
        sentence_records: list[dict[str, Any]],
    ) -> str:
        sampled_chapters = sorted(
            sample_evenly(chapters, min(len(chapters), 3)),
            key=lambda chapter: chapter.start_seconds,
        )
        chapter_summaries = [
            summary
            for chapter in sampled_chapters
            if (
                summary := self._normalize_sentence_text(
                    chapter.summary,
                    minimum_words=6,
                )
            )
            and not self._text_is_low_quality(summary)
        ]
        chapter_summaries = self._dedupe_exact_strings(chapter_summaries)
        topic_candidates = self._sanitize_topic_list(
            [
                *[chapter.title for chapter in sampled_chapters],
                *[
                    keyword
                    for chapter in sampled_chapters[:2]
                    for keyword in chapter.keywords[:2]
                ],
                *keywords[:4],
            ],
            limit=4,
        )
        leading_topic = topic_candidates[0] if topic_candidates else self._best_topic_label(
            [title],
            fallback=title,
        )
        supporting_topics = [
            topic
            for topic in topic_candidates[1:]
            if not self._labels_share_topic(topic, leading_topic)
        ][:2]
        if chapters:
            if chapter_summaries:
                summary_parts = chapter_summaries[:2]
                if action_items and content_format == "Meeting":
                    summary_parts.append(
                        f"Follow-ups include {self._join_labels([item.title for item in action_items[:2]])}."
                    )
                return truncate_text(" ".join(self._dedupe_strings(summary_parts)), 320)
            lead_chapter = chapters[0]
            if content_format == "Meeting":
                summary_parts = [f"This meeting centers on {lead_chapter.title}."]
                if supporting_topics:
                    summary_parts.append(
                        f"It also aligns the discussion around {self._join_labels(supporting_topics)}."
                    )
                if action_items:
                    summary_parts.append(
                        f"It closes with follow-ups on {self._join_labels([item.title for item in action_items[:2]])}."
                    )
                return truncate_text(" ".join(summary_parts), 320)
            if content_format in {"Lecture", "Workshop"}:
                summary_parts = [f"This {content_format.lower()} focuses on {leading_topic}."]
                if supporting_topics:
                    summary_parts.append(
                        f"It then develops the topic through {self._join_labels(supporting_topics)}."
                    )
                return truncate_text(" ".join(summary_parts), 320)
            if content_format in {"Podcast", "Interview", "Talk"}:
                summary_parts = [f"This {content_format.lower()} is anchored by {leading_topic}."]
                if supporting_topics:
                    summary_parts.append(
                        f"The discussion branches into {self._join_labels(supporting_topics)}."
                    )
                return truncate_text(" ".join(summary_parts), 320)
            summary_parts = [f"The video focuses on {leading_topic}."]
            if supporting_topics:
                summary_parts.append(
                    f"It also covers {self._join_labels(supporting_topics)}."
                )
            return truncate_text(" ".join(summary_parts), 320)
        top_sentences = self._top_sentence_records(sentence_records, limit=2)
        if top_sentences:
            return truncate_text(
                " ".join(record["text"] for record in top_sentences),
                320,
            )
        if keywords:
            intro = (
                f"This {content_format.lower()} centers on"
                if content_format
                else "The video centers on"
            )
            return f"{intro} {self._join_labels(keywords[:4])}."
        if transcript_text:
            transcript_keywords = extract_keywords(transcript_text, limit=4)
            if transcript_keywords:
                return truncate_text(
                    f"The discussion revolves around {self._join_labels(transcript_keywords)}.",
                    320,
                )
        return "The transcript was processed successfully."

    def _build_five_minute_summary(
        self,
        *,
        sentence_records: list[dict[str, Any]],
        chapters: list[ChapterItem],
        action_items: list[ActionItem],
        content_format: str | None,
    ) -> list[str]:
        if chapters:
            sampled_chapters = sorted(
                sample_evenly(chapters, min(len(chapters), 4)),
                key=lambda chapter: chapter.start_seconds,
            )
            summary_points = self._dedupe_exact_strings([
                truncate_text(
                    self._format_summary_point(
                        self._summary_point_from_chapter(chapter),
                        content_format=content_format,
                        index=index,
                    ),
                    220,
                )
                for index, chapter in enumerate(sampled_chapters)
            ])[:4]
            if action_items and content_format == "Meeting":
                follow_up_point = truncate_text(
                    f"Close with the follow-ups: {self._join_labels([item.title for item in action_items[:2]])}.",
                    220,
                )
                if follow_up_point.lower() not in {item.lower() for item in summary_points}:
                    summary_points.append(follow_up_point)
            return summary_points[:5]
        return [
            truncate_text(record["text"], 200)
            for record in self._top_sentence_records(sentence_records, limit=5)
        ]

    def _format_summary_point(
        self,
        point: str,
        *,
        content_format: str | None,
        index: int,
    ) -> str:
        stage_label = self._summary_stage_label(content_format, index)
        if not stage_label or point.lower().startswith(stage_label.lower()):
            return point
        return f"{stage_label}: {point}"

    def _summary_stage_label(
        self,
        content_format: str | None,
        index: int,
    ) -> str | None:
        labels = {
            "Meeting": ("Decision", "Context", "Alignment", "Follow-up"),
            "Lecture": ("Foundation", "Core Idea", "Example", "Synthesis"),
            "Workshop": ("Step 1", "Step 2", "Step 3", "Outcome"),
            "Podcast": ("Theme", "Insight", "Example", "Takeaway"),
            "Talk": ("Opening Idea", "Key Insight", "Example", "Takeaway"),
            "Interview": ("Question", "Response", "Example", "Takeaway"),
        }.get(content_format or "")
        if not labels:
            return None
        return labels[min(index, len(labels) - 1)]

    def _chapter_brief(self, chapter: ChapterItem) -> str:
        focus = ", ".join(keyword.lower() for keyword in chapter.keywords[:3])
        if focus:
            return truncate_text(
                f"This section focuses on {focus}.",
                180,
            )
        return truncate_text(chapter.summary, 180)

    def _summary_point_from_chapter(self, chapter: ChapterItem) -> str:
        if chapter.summary and not self._text_is_low_quality(chapter.summary):
            cleaned_summary = self._normalize_sentence_text(
                chapter.summary,
                minimum_words=6,
            )
            if cleaned_summary and self._labels_share_topic(chapter.title, cleaned_summary):
                return truncate_text(cleaned_summary, 180)
            if cleaned_summary:
                if len(extract_keywords(cleaned_summary, limit=3)) >= 2:
                    return truncate_text(cleaned_summary, 180)
                return truncate_text(f"{chapter.title}: {cleaned_summary}", 180)
        supporting = self._chapter_supporting_topics(chapter, limit=2)
        if supporting:
            return truncate_text(
                f"{chapter.title} covers {self._join_labels(supporting)}.",
                180,
            )
        return f"{chapter.title} as a major section of the discussion."

    def _chapter_supporting_topics(
        self,
        chapter: ChapterItem,
        *,
        limit: int,
    ) -> list[str]:
        candidates = [
            keyword
            for keyword in chapter.keywords
            if not self._labels_share_topic(keyword, chapter.title)
            and self._topic_noise_penalty(keyword) < 1.0
        ]
        return self._sanitize_topic_list(candidates, limit=limit)

    def _build_key_topics(
        self,
        *,
        title: str,
        transcript_text: str,
        sentence_records: list[dict[str, Any]],
        chapters: list[ChapterItem],
        chunks: list[TranscriptChunk],
        keywords: list[str],
    ) -> list[str]:
        scored_topics: dict[str, dict[str, Any]] = {}

        def add_candidates(candidates: list[str], *, weight: float, evidence: str | None = None) -> None:
            cleaned_candidates = self._sanitize_topic_list(candidates, limit=max(1, len(candidates)))
            for candidate in cleaned_candidates:
                cleaned = self._clean_topic_label(candidate)
                if not cleaned:
                    continue
                key = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
                if not key:
                    continue
                noise_penalty = self._topic_noise_penalty(cleaned)
                if noise_penalty >= 1.8:
                    continue
                support_score = self._topic_support_score(
                    cleaned,
                    evidence_text=evidence or transcript_text,
                    sentence_records=sentence_records,
                    chunks=chunks,
                )
                if support_score <= 0:
                    continue
                word_count = len(cleaned.split())
                if 2 <= word_count <= 4:
                    specificity_bonus = 0.9
                elif word_count == 1:
                    specificity_bonus = -0.2 if len(cleaned) < 6 else 0.1
                else:
                    specificity_bonus = 0.25
                chapter_bonus = 0.35 if any(cleaned.lower() in chapter.title.lower() for chapter in chapters) else 0.0
                record = scored_topics.setdefault(
                    key,
                    {"label": cleaned, "score": 0.0, "mentions": 0},
                )
                record["score"] += weight + support_score + specificity_bonus + chapter_bonus - noise_penalty
                record["mentions"] += 1

        title_candidates = self._title_topic_candidates(title)
        programming_candidates = self._programming_focus_candidates(
            title=title,
            transcript_text=transcript_text,
            topic_candidates=[
                *title_candidates,
                *keywords,
                *[chapter.title for chapter in chapters[:4]],
            ],
        )
        add_candidates(programming_candidates, weight=3.4, evidence=transcript_text)
        add_candidates(title_candidates, weight=3.0, evidence=transcript_text)
        add_candidates(keywords, weight=2.6, evidence=transcript_text)
        for chapter in chapters[: self.settings.max_chapters]:
            add_candidates([chapter.title], weight=2.4, evidence=chapter.summary)
            add_candidates(
                [
                    *chapter.keywords[:5],
                    *extract_keywords(chapter.summary, limit=5),
                    *extract_keywords(f"{chapter.title}. {chapter.summary}", limit=4),
                ],
                weight=1.9,
                evidence=chapter.summary,
            )
        for chunk in sample_evenly(chunks, min(len(chunks), 10)):
            add_candidates([chunk.semantic_focus], weight=2.0, evidence=chunk.text)
            add_candidates(extract_keywords(chunk.text, limit=5), weight=1.6, evidence=chunk.text)
        for record in self._top_sentence_records(sentence_records, limit=12):
            add_candidates(
                [str(record["focus"]), *record["keywords"], *extract_keywords(str(record["text"]), limit=4)],
                weight=1.0,
                evidence=str(record["text"]),
            )

        ranked_topics = [
            item["label"]
            for item in sorted(
                scored_topics.values(),
                key=lambda item: (-float(item["score"]), -int(item["mentions"]), len(item["label"])),
            )
        ]
        topics = self._sanitize_topic_list(ranked_topics, limit=8)
        if topics:
            return topics
        return self._sanitize_topic_list([chapter.title for chapter in chapters], limit=6)

    def _build_chapters(
        self,
        artifact: VideoAnalysisArtifact,
        chunks: list[TranscriptChunk],
        sentence_records: list[dict[str, Any]],
    ) -> list[ChapterItem]:
        if not chunks and sentence_records:
            chunks = [
                TranscriptChunk(
                    start_seconds=record["start_seconds"],
                    end_seconds=record["end_seconds"],
                    semantic_focus=build_headline(
                        record["text"],
                        fallback=f"Topic {index + 1}",
                    ),
                    text=record["text"],
                )
                for index, record in enumerate(sentence_records)
            ]
        if not chunks:
            return []

        source_chapters = self._build_source_chapters(artifact, chunks, sentence_records)
        if source_chapters:
            return source_chapters

        groups = self._chapter_groups(chunks)
        chapters: list[ChapterItem] = []
        for index, group in enumerate(groups):
            start_seconds = group[0].start_seconds
            end_seconds = group[-1].end_seconds
            supporting_records = [
                record
                for record in sentence_records
                if self._in_span(
                    float(record["start_seconds"]),
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
            ]
            chapter_summary_records = self._top_sentence_records(supporting_records, limit=2)
            chapter_text = self._chapter_text_for_span(
                chunks,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
            chapter_keywords = self._chapter_keywords(
                chapter_text,
                supporting_records=chapter_summary_records,
                focus_candidates=[group[0].semantic_focus],
            )
            preferred_focus = self._clean_topic_label(
                group[0].semantic_focus,
                max_words=6,
            )
            provisional_summary = self._chapter_summary(
                chapter_title=preferred_focus or group[0].semantic_focus,
                chapter_text=chapter_text,
                chapter_keywords=chapter_keywords,
                sentence_records=chapter_summary_records,
            )
            if (
                len(group) == 1
                and preferred_focus
                and not is_low_signal_topic(preferred_focus)
                and self._topic_noise_penalty(preferred_focus) < 0.8
            ):
                title_source = preferred_focus
            else:
                title_source = self._chapter_title(
                    [
                        group[0].semantic_focus,
                        *chapter_keywords,
                        *extract_keywords(provisional_summary, limit=4),
                        *[
                            str(record["focus"])
                            for record in chapter_summary_records
                            if str(record["focus"]).strip()
                        ],
                        provisional_summary,
                    ],
                    chapter_text=chapter_text,
                    supporting_records=chapter_summary_records,
                    fallback=f"Chapter {index + 1}",
                )
            summary = self._chapter_summary(
                chapter_title=title_source,
                chapter_text=chapter_text,
                chapter_keywords=chapter_keywords,
                sentence_records=chapter_summary_records,
                fallback_summary=provisional_summary,
            )
            chapters.append(
                ChapterItem(
                    title=title_source,
                    summary=summary,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    display_time=format_timestamp(start_seconds),
                    jump_url=build_youtube_jump_url(artifact.source_url, start_seconds),
                    keywords=chapter_keywords,
                    confidence=round(min(0.98, 0.6 + (len(chapter_keywords) * 0.07)), 2),
                )
            )
        return chapters

    def _build_source_chapters(
        self,
        artifact: VideoAnalysisArtifact,
        chunks: list[TranscriptChunk],
        sentence_records: list[dict[str, Any]],
    ) -> list[ChapterItem]:
        markers = self._load_source_chapter_markers(artifact)
        if not markers:
            return []

        chapters: list[ChapterItem] = []
        transcript_end = max(
            [chunk.end_seconds for chunk in chunks] or [float(markers[-1]["start_seconds"])],
        )
        for index, marker in enumerate(markers):
            start_seconds = float(marker["start_seconds"])
            end_seconds = marker.get("end_seconds")
            if end_seconds is None and index + 1 < len(markers):
                end_seconds = float(markers[index + 1]["start_seconds"])
            if end_seconds is None:
                end_seconds = transcript_end
            if end_seconds <= start_seconds:
                continue

            supporting_records = [
                record
                for record in sentence_records
                if self._in_span(
                    float(record["start_seconds"]),
                    start_seconds=start_seconds,
                    end_seconds=float(end_seconds),
                )
            ]
            chapter_summary_records = self._top_sentence_records(supporting_records, limit=2)
            chapter_text = self._chapter_text_for_span(
                chunks,
                start_seconds=start_seconds,
                end_seconds=float(end_seconds),
            )
            raw_title = str(marker.get("title") or "").strip()
            chapter_keywords = self._chapter_keywords(
                chapter_text,
                supporting_records=chapter_summary_records,
                focus_candidates=[raw_title],
            )
            title = self._clean_topic_label(raw_title, max_words=8)
            title_for_summary = title or raw_title or f"Chapter {index + 1}"
            summary = self._chapter_summary(
                chapter_title=title_for_summary,
                chapter_text=chapter_text,
                chapter_keywords=chapter_keywords,
                sentence_records=chapter_summary_records,
                fallback_summary=raw_title if not is_low_signal_topic(raw_title) else None,
            )
            if not title or is_low_signal_topic(title):
                title = self._chapter_title(
                    [
                        raw_title,
                        *chapter_keywords,
                        *extract_keywords(summary, limit=4),
                        summary,
                    ],
                    chapter_text=chapter_text,
                    supporting_records=chapter_summary_records,
                    fallback=f"Chapter {index + 1}",
                )
            chapters.append(
                ChapterItem(
                    title=title,
                    summary=summary,
                    start_seconds=start_seconds,
                    end_seconds=float(end_seconds),
                    display_time=format_timestamp(start_seconds),
                    jump_url=build_youtube_jump_url(artifact.source_url, start_seconds),
                    keywords=chapter_keywords,
                    confidence=0.92 if raw_title and not is_low_signal_topic(raw_title) else 0.79,
                )
            )
        return chapters

    def _load_source_chapter_markers(
        self,
        artifact: VideoAnalysisArtifact,
    ) -> list[dict[str, float | str | None]]:
        local_media_path = artifact.metadata.get("local_media_path")
        if not isinstance(local_media_path, str) or not local_media_path:
            return []
        info_path = Path(local_media_path).parent / "info.json"
        if not info_path.exists():
            return []
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        raw_markers = info.get("chapters")
        if not isinstance(raw_markers, list):
            return []

        markers: list[dict[str, float | str | None]] = []
        seen_starts: set[float] = set()
        for item in raw_markers:
            if not isinstance(item, dict):
                continue
            start_seconds = self._coerce_seconds(
                item["start_time"]
                if "start_time" in item
                else item["start_seconds"]
                if "start_seconds" in item
                else item["time"]
                if "time" in item
                else item.get("seconds"),
            )
            if start_seconds is None:
                continue
            rounded_start = round(float(start_seconds), 3)
            if rounded_start in seen_starts:
                continue
            seen_starts.add(rounded_start)
            markers.append(
                {
                    "title": self._coerce_optional_text(item.get("title") or item.get("label")),
                    "start_seconds": float(start_seconds),
                    "end_seconds": self._coerce_seconds(
                        item["end_time"]
                        if "end_time" in item
                        else item["end_seconds"]
                        if "end_seconds" in item
                        else item.get("stop_time"),
                    ),
                }
            )
        markers.sort(key=lambda item: float(item["start_seconds"] or 0))
        return markers

    def _chapter_groups(self, chunks: list[TranscriptChunk]) -> list[list[TranscriptChunk]]:
        if not chunks:
            return []
        total_duration = max(0.0, chunks[-1].end_seconds - chunks[0].start_seconds)
        duration_target = max(2, round(total_duration / 240)) if total_duration else 2
        desired_chapters = min(
            self.settings.max_chapters,
            len(chunks),
            max(2, duration_target),
        )
        if len(chunks) <= desired_chapters:
            return [[chunk] for chunk in chunks]

        scored_boundaries = [
            (
                self._chapter_boundary_score(chunks[index], chunks[index + 1]),
                index,
            )
            for index in range(len(chunks) - 1)
        ]
        selected_boundaries = sorted(
            index
            for _, index in sorted(scored_boundaries, reverse=True)[: desired_chapters - 1]
        )

        groups: list[list[TranscriptChunk]] = []
        start = 0
        for boundary in selected_boundaries:
            groups.append(chunks[start : boundary + 1])
            start = boundary + 1
        groups.append(chunks[start:])
        return self._merge_short_chapter_groups([group for group in groups if group])

    def _merge_short_chapter_groups(
        self,
        groups: list[list[TranscriptChunk]],
    ) -> list[list[TranscriptChunk]]:
        if len(groups) <= 2:
            return groups

        minimum_duration = 75.0
        minimum_characters = max(260, int(self.settings.min_chunk_characters * 0.4))
        merged: list[list[TranscriptChunk]] = []
        for group in groups:
            if not merged:
                merged.append(list(group))
                continue
            duration = max(0.0, group[-1].end_seconds - group[0].start_seconds)
            character_count = sum(len(chunk.text) for chunk in group)
            if duration < minimum_duration or character_count < minimum_characters:
                merged[-1].extend(group)
                continue
            merged.append(list(group))

        if len(merged) > 1:
            last_group = merged[-1]
            last_duration = max(0.0, last_group[-1].end_seconds - last_group[0].start_seconds)
            last_characters = sum(len(chunk.text) for chunk in last_group)
            if last_duration < minimum_duration * 0.85 or last_characters < minimum_characters * 0.75:
                merged[-2].extend(merged.pop())
        return merged

    def _chapter_boundary_score(
        self,
        left: TranscriptChunk,
        right: TranscriptChunk,
    ) -> float:
        left_keywords = {item.lower() for item in extract_keywords(left.text, limit=5)}
        right_keywords = {item.lower() for item in extract_keywords(right.text, limit=5)}
        overlap = len(left_keywords & right_keywords)
        union = len(left_keywords | right_keywords) or 1
        shift = 1 - (overlap / union)
        time_gap = max(0.0, right.start_seconds - left.end_seconds)
        time_bonus = min(0.35, time_gap / 30)
        focus_bonus = 0.18 if left.semantic_focus.lower() != right.semantic_focus.lower() else 0.0
        transition_bonus = 0.22 if contains_transition_cue(right.text) else 0.0
        return round((shift * 0.68) + time_bonus + focus_bonus + transition_bonus, 4)

    def _refine_chapters(
        self,
        chapters: list[ChapterItem],
        chunks: list[TranscriptChunk],
        key_topics: list[str],
    ) -> list[ChapterItem]:
        refined: list[ChapterItem] = []
        for chapter in chapters:
            chapter_text = (
                self._chapter_text_for_span(
                    chunks,
                    start_seconds=chapter.start_seconds,
                    end_seconds=chapter.end_seconds,
                )
                or chapter.summary
            )
            supporting_topics = [
                topic
                for topic in key_topics
                if topic.lower() in strip_timecodes(chapter_text).lower()
            ]
            keywords = self._chapter_keywords(
                chapter_text,
                supporting_records=[],
                focus_candidates=[chapter.title, *supporting_topics, *chapter.keywords],
            )
            title = (
                chapter.title
                if chapter.confidence >= 0.8 and not is_low_signal_topic(chapter.title)
                else self._chapter_title(
                    [
                        chapter.title,
                        *supporting_topics,
                        *extract_keywords(chapter_text, limit=5),
                        *keywords,
                    ],
                    chapter_text=chapter_text,
                    supporting_records=[],
                    fallback=chapter.title,
                )
            )
            summary = self._chapter_summary(
                chapter_title=title,
                chapter_text=chapter_text,
                chapter_keywords=keywords or chapter.keywords,
                sentence_records=[],
                fallback_summary=chapter.summary,
            )
            refined.append(
                chapter.model_copy(
                    update={
                        "title": title,
                        "keywords": keywords or chapter.keywords,
                        "summary": summary,
                    }
                )
            )
        return refined

    def _chapter_text_for_span(
        self,
        chunks: list[TranscriptChunk],
        *,
        start_seconds: float,
        end_seconds: float | None,
    ) -> str:
        relevant_chunks = [
            chunk.text
            for chunk in chunks
            if self._spans_overlap(
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                span_start=start_seconds,
                span_end=end_seconds,
            )
        ]
        return " ".join(relevant_chunks)

    def _chapter_keywords(
        self,
        chapter_text: str,
        *,
        supporting_records: list[dict[str, Any]],
        focus_candidates: list[str],
    ) -> list[str]:
        supporting_keywords = [
            keyword
            for record in supporting_records
            for keyword in (
                record.get("keywords")
                if isinstance(record.get("keywords"), list)
                else []
            )
            if isinstance(keyword, str)
        ]
        return self._sanitize_topic_list(
            [
                *focus_candidates,
                *extract_keywords(chapter_text, limit=6),
                *supporting_keywords,
            ],
            limit=5,
        )

    def _chapter_title(
        self,
        candidates: list[str],
        *,
        chapter_text: str,
        supporting_records: list[dict[str, Any]],
        fallback: str,
    ) -> str:
        best_label = fallback
        best_score = float("-inf")
        for index, candidate in enumerate(candidates):
            cleaned = self._clean_topic_label(candidate, max_words=6)
            if not cleaned:
                continue
            score = self._topic_candidate_score(
                cleaned,
                chapter_text=chapter_text,
                supporting_records=supporting_records,
            )
            score += max(0.0, 0.18 - (index * 0.03))
            if score > best_score:
                best_score = score
                best_label = cleaned
        return self._best_topic_label(
            [
                best_label,
                build_headline(
                    chapter_text,
                    fallback=fallback,
                    max_words=6,
                ),
            ],
            fallback=fallback,
        )

    def _topic_candidate_score(
        self,
        label: str,
        *,
        chapter_text: str,
        supporting_records: list[dict[str, Any]],
    ) -> float:
        tokens = [
            token
            for token in tokenize(label.lower(), minimum_length=2)
            if len(token) > 1
        ]
        if not tokens:
            return float("-inf")

        lowered_label = label.lower()
        lowered_text = strip_timecodes(chapter_text).lower()
        support = 1.5 if lowered_label in lowered_text else 0.0
        sentence_support = 0.0
        for record in supporting_records:
            record_text = strip_timecodes(str(record["text"])).lower()
            if lowered_label in record_text:
                sentence_support += 0.9
                continue
            record_tokens = set(tokenize(record_text, minimum_length=2))
            if tokens and len(record_tokens & set(tokens)) >= max(1, min(2, len(tokens))):
                sentence_support += 0.45

        token_count = len(tokens)
        if 2 <= token_count <= 4:
            specificity_bonus = 0.95
        elif token_count == 1:
            specificity_bonus = -0.25 if len(tokens[0]) < 6 else 0.05
        else:
            specificity_bonus = 0.3
        question_penalty = 0.15 if label.endswith("?") else 0.0
        low_signal_penalty = 1.2 if is_low_signal_topic(label) else 0.0
        noise_penalty = self._topic_noise_penalty(label)
        return round(
            support + sentence_support + specificity_bonus - question_penalty - low_signal_penalty - noise_penalty,
            4,
        )

    def _spans_overlap(
        self,
        *,
        start_seconds: float,
        end_seconds: float | None,
        span_start: float,
        span_end: float | None,
    ) -> bool:
        effective_end = end_seconds if end_seconds is not None else start_seconds
        effective_span_end = span_end if span_end is not None else span_start
        return start_seconds < effective_span_end and effective_end > span_start

    def _chapter_summary(
        self,
        *,
        chapter_title: str | None = None,
        chapter_text: str,
        chapter_keywords: list[str],
        sentence_records: list[dict[str, Any]],
        fallback_summary: str | None = None,
    ) -> str:
        cleaned_text = strip_timecodes(chapter_text)
        chapter_label = self._clean_topic_label(chapter_title, max_words=7)
        supporting_topics = self._sanitize_topic_list(
            [
                *(chapter_keywords or []),
                *extract_keywords(cleaned_text, limit=5),
            ],
            limit=4,
        )
        supporting_topics = [
            topic
            for topic in supporting_topics
            if not chapter_label or not self._labels_share_topic(topic, chapter_label)
        ][:2]
        high_value_supporting_topics = [
            topic
            for topic in supporting_topics
            if len(topic.split()) >= 2 and self._topic_noise_penalty(topic) < 0.8
        ]
        if sentence_records:
            ranked_records = sorted(
                sentence_records,
                key=lambda record: (
                    -float(record.get("score", 0.0)),
                    float(record.get("start_seconds", 0.0)),
                ),
            )
            high_signal_records = [
                record
                for record in ranked_records
                if (
                    record.get("keywords")
                    or float(record.get("score", 0.0)) >= 1.2
                )
                and not self._text_is_low_quality(str(record["text"]))
            ]
            lead_sentence = self._normalize_sentence_text(
                str((high_signal_records or ranked_records[:1])[0]["text"]),
                minimum_words=6,
            )
            if lead_sentence:
                if self._sentence_contains_topic_noise(lead_sentence):
                    lead_sentence = None
            if lead_sentence:
                mentions_support = any(
                    topic.lower() in lead_sentence.lower()
                    for topic in high_value_supporting_topics[:1]
                )
                if high_value_supporting_topics and not mentions_support:
                    return truncate_text(
                        f"{lead_sentence.rstrip('.')} It focuses on {self._join_labels(high_value_supporting_topics)}.",
                        220,
                    )
                return truncate_text(lead_sentence, 220)
        if fallback_summary:
            cleaned_fallback = self._normalize_sentence_text(
                fallback_summary,
                minimum_words=4,
            )
            if (
                cleaned_fallback
                and not self._text_is_low_quality(cleaned_fallback)
                and not self._sentence_contains_topic_noise(cleaned_fallback)
            ):
                return truncate_text(cleaned_fallback, 220)
        if chapter_label and high_value_supporting_topics:
            return truncate_text(
                f"{chapter_label} covers {self._join_labels(high_value_supporting_topics)}.",
                220,
            )
        if chapter_label:
            return truncate_text(
                f"This section explains {chapter_label} as a central idea in the discussion.",
                220,
            )
        if chapter_keywords:
            lead_topics = self._join_labels(chapter_keywords[:3]).lower()
            return truncate_text(
                f"This section focuses on {lead_topics} and explains why those ideas matter in the overall video.",
                220,
            )
        return truncate_text(cleaned_text, 220)

    def _build_note_sections(
        self,
        chapters: list[ChapterItem],
        sentence_records: list[dict[str, Any]],
        *,
        content_format: str | None,
    ) -> list[NoteSection]:
        sections: list[NoteSection] = []
        for chapter in chapters[: self.settings.max_chapters]:
            supporting_records = [
                record
                for record in sentence_records
                if self._in_span(
                    float(record["start_seconds"]),
                    start_seconds=chapter.start_seconds,
                    end_seconds=chapter.end_seconds,
                )
            ]
            bullets = self._note_bullets_for_chapter(
                chapter=chapter,
                supporting_records=supporting_records,
                content_format=content_format,
            )
            sections.append(
                NoteSection(
                    heading=chapter.title,
                    bullet_points=bullets[:3],
                    detail=chapter.summary,
                    start_seconds=chapter.start_seconds,
                    display_time=chapter.display_time,
                    jump_url=chapter.jump_url,
                )
            )

        if sections:
            return sections

        for index, record in enumerate(self._top_sentence_records(sentence_records, limit=3)):
            sections.append(
                NoteSection(
                    heading=self._best_topic_label(
                        [str(record["text"])],
                        fallback=f"Section {index + 1}",
                    ),
                    bullet_points=[],
                    detail=str(record["text"]),
                    start_seconds=float(record["start_seconds"]),
                    display_time=format_timestamp(float(record["start_seconds"])),
                )
            )
        return sections

    def _note_bullets_for_chapter(
        self,
        *,
        chapter: ChapterItem,
        supporting_records: list[dict[str, Any]],
        content_format: str | None,
    ) -> list[str]:
        bullets: list[str] = []
        focus_label, detail_label, wrapup_label = self._note_bullet_labels(content_format)

        def add_bullet(text: str) -> None:
            normalized = normalize_whitespace(text)
            if (
                not normalized
                or self._text_is_low_quality(normalized)
                or any(self._texts_are_redundant(normalized, existing) for existing in bullets)
            ):
                return
            bullets.append(normalized)

        supporting_topics = self._chapter_supporting_topics(chapter, limit=3)
        if content_format:
            supporting_topics = [
                topic
                for topic in supporting_topics
                if not self._labels_share_topic(topic, content_format)
            ][:3]
        if supporting_topics:
            add_bullet(
                truncate_text(
                    f"{focus_label}: {self._join_labels(supporting_topics)}.",
                    140,
                )
            )

        for record in self._top_sentence_records(supporting_records, limit=2):
            candidate = self._normalize_sentence_text(str(record["text"]), minimum_words=6)
            if candidate is None:
                continue
            shortened = truncate_text(f"{detail_label}: {candidate}", 140)
            add_bullet(shortened)

        if len(bullets) < 2 and chapter.summary and not self._text_is_low_quality(chapter.summary):
            add_bullet(
                truncate_text(
                    f"{detail_label}: {chapter.summary}",
                    140,
                )
            )

        if len(bullets) < 3 and supporting_topics:
            add_bullet(
                truncate_text(
                    f"{wrapup_label}: it connects back to {self._join_labels(supporting_topics[:2])}.",
                    140,
                )
            )
        return bullets[:3]

    def _note_bullet_labels(
        self,
        content_format: str | None,
    ) -> tuple[str, str, str]:
        return {
            "Meeting": ("Decision focus", "Discussion detail", "Next step"),
            "Lecture": ("Core concept", "Evidence", "Why it matters"),
            "Workshop": ("Step focus", "How it works", "Outcome"),
            "Podcast": ("Theme", "Example", "Takeaway"),
            "Talk": ("Key insight", "Example", "Takeaway"),
            "Interview": ("Question", "Response", "Takeaway"),
        }.get(content_format or "", ("Focus", "Detail", "Why it matters"))

    def _build_timestamps(
        self,
        artifact: VideoAnalysisArtifact,
        source_url: str | None,
        chapters: list[ChapterItem],
        sentence_records: list[dict[str, Any]],
        *,
        content_format: str | None,
    ) -> list[TimestampItem]:
        if chapters:
            if self._has_source_chapters(artifact):
                return [
                    TimestampItem(
                        label=chapter.title,
                        description=chapter.summary,
                        start_seconds=chapter.start_seconds,
                        end_seconds=chapter.end_seconds,
                        display_time=chapter.display_time,
                        jump_url=chapter.jump_url
                        or build_youtube_jump_url(source_url, chapter.start_seconds),
                    )
                    for chapter in chapters
                ]

            timestamps: list[TimestampItem] = []
            seen: set[tuple[int, str]] = set()

            def add_timestamp(
                *,
                label: str,
                description: str,
                start_seconds: float,
                end_seconds: float | None,
                jump_url: str | None,
            ) -> None:
                key = (round(start_seconds), label.lower())
                if key in seen:
                    return
                if any(
                    (
                        self._texts_are_redundant(label, existing.label)
                        and abs(start_seconds - existing.start_seconds) < 120
                    )
                    or (
                        self._texts_are_redundant(label, existing.label)
                        and self._texts_are_redundant(description, existing.description)
                    )
                    for existing in timestamps
                ):
                    return
                seen.add(key)
                timestamps.append(
                    TimestampItem(
                        label=label,
                        description=description,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        display_time=format_timestamp(start_seconds),
                        jump_url=jump_url or build_youtube_jump_url(source_url, start_seconds),
                    )
                )

            for chapter in chapters:
                add_timestamp(
                    label=chapter.title,
                    description=chapter.summary,
                    start_seconds=chapter.start_seconds,
                    end_seconds=chapter.end_seconds,
                    jump_url=chapter.jump_url,
                )
                supporting_records = [
                    record
                    for record in self._top_sentence_records(
                        [
                            record
                            for record in sentence_records
                            if self._in_span(
                                float(record["start_seconds"]),
                                start_seconds=chapter.start_seconds,
                                end_seconds=chapter.end_seconds,
                            )
                        ],
                        limit=3,
                    )
                    if abs(float(record["start_seconds"]) - chapter.start_seconds) >= 10
                ]
                for record in supporting_records[:2]:
                    label = self._timestamp_label(
                        chapter=chapter,
                        record=record,
                        content_format=content_format,
                    )
                    if self._labels_share_topic(label, chapter.title):
                        continue
                    add_timestamp(
                        label=label,
                        description=self._timestamp_description(
                            chapter=chapter,
                            record=record,
                            content_format=content_format,
                        ),
                        start_seconds=float(record["start_seconds"]),
                        end_seconds=float(record["end_seconds"]),
                        jump_url=build_youtube_jump_url(
                            source_url,
                            float(record["start_seconds"]),
                        ),
                    )
            timestamps.sort(key=lambda item: (item.start_seconds, item.label.lower()))
            return timestamps[: max(self.settings.max_chapters * 2, 10)]

        return [
            TimestampItem(
                label=self._best_topic_label(
                    [str(record["text"])],
                    fallback=f"Moment {index + 1}",
                ),
                description=truncate_text(str(record["text"]), 180),
                start_seconds=float(record["start_seconds"]),
                end_seconds=float(record["end_seconds"]),
                display_time=format_timestamp(float(record["start_seconds"])),
                jump_url=build_youtube_jump_url(source_url, float(record["start_seconds"])),
            )
            for index, record in enumerate(self._top_sentence_records(sentence_records, limit=4))
        ]

    def _has_source_chapters(self, artifact: VideoAnalysisArtifact) -> bool:
        source_chapter_count = artifact.metadata.get("source_chapter_count")
        if isinstance(source_chapter_count, int):
            return source_chapter_count > 0
        if isinstance(source_chapter_count, float):
            return source_chapter_count > 0
        return bool(self._load_source_chapter_markers(artifact))

    def _timestamp_label(
        self,
        *,
        chapter: ChapterItem,
        record: dict[str, Any],
        content_format: str | None,
    ) -> str:
        candidates = self._sanitize_topic_list(
            [
                str(record.get("focus") or ""),
                *[
                    keyword
                    for keyword in record.get("keywords", [])
                    if isinstance(keyword, str)
                ],
                *extract_keywords(str(record["text"]), limit=5),
                build_headline(
                    str(record["text"]),
                    fallback=chapter.title,
                    max_words=6,
                ),
            ],
            limit=8,
        )
        ranked = sorted(
            candidates,
            key=lambda candidate: -self._mind_map_concept_score(
                candidate,
                branch_label=chapter.title,
                root_label="",
                evidence_text=strip_timecodes(str(record["text"])).lower(),
            ),
        )
        for candidate in ranked:
            if len(candidate.split()) >= 2 and not self._labels_share_topic(candidate, chapter.title):
                concept = candidate
                break
        else:
            concept = None
        for candidate in ranked:
            if concept is not None:
                break
            if not self._labels_share_topic(candidate, chapter.title):
                concept = candidate
        resolved_concept = concept or chapter.title
        moment_type = self._timeline_moment_type(
            str(record["text"]),
            content_format=content_format,
        )
        if moment_type in {None, "Concept", "Theme", "Discussion", "Perspective"}:
            return truncate_text(resolved_concept, 48)
        return truncate_text(f"{moment_type}: {resolved_concept}", 48)

    def _timestamp_description(
        self,
        *,
        chapter: ChapterItem,
        record: dict[str, Any],
        content_format: str | None,
    ) -> str:
        cleaned_sentence = self._normalize_sentence_text(
            str(record["text"]),
            minimum_words=6,
        )
        moment_type = self._timeline_moment_type(
            str(record["text"]),
            content_format=content_format,
        )
        if cleaned_sentence and not self._text_is_low_quality(cleaned_sentence):
            if self._sentence_contains_topic_noise(cleaned_sentence):
                cleaned_sentence = None
        if cleaned_sentence and not self._text_is_low_quality(cleaned_sentence):
            if moment_type not in {None, "Concept", "Theme", "Discussion", "Perspective"}:
                return truncate_text(
                    f"{moment_type}: {cleaned_sentence}",
                    170,
                )
            return truncate_text(cleaned_sentence, 170)
        supporting_topics = self._chapter_supporting_topics(chapter, limit=2)
        if supporting_topics:
            return truncate_text(
                f"This {((moment_type or 'moment').lower())} expands on {self._join_labels(supporting_topics)} within {chapter.title}.",
                170,
            )
        return truncate_text(chapter.summary, 170)

    def _timeline_moment_type(
        self,
        text: str,
        *,
        content_format: str | None,
    ) -> str | None:
        lowered = text.lower()
        if content_format == "Meeting":
            if self._is_actionable_sentence(
                lowered,
                triggers=(
                    "need to",
                    "should",
                    "must",
                    "follow up",
                    "next step",
                    "send",
                    "confirm",
                    "assign",
                ),
            ):
                return "Next Step"
            if any(marker in lowered for marker in ("we agreed", "decided", "decision", "we will", "we'll")):
                return "Decision"
            if any(marker in lowered for marker in ("risk", "blocker", "dependency", "timeline", "owner")):
                return "Risk"
            return "Discussion"
        if content_format == "Lecture":
            if any(marker in lowered for marker in ("is defined as", "refers to", "means", "define")):
                return "Definition"
            if any(marker in lowered for marker in ("compare", "contrast", "versus", "whereas")):
                return "Comparison"
            if any(marker in lowered for marker in ("example", "for instance", "for example", "case")):
                return "Example"
            return "Concept"
        if content_format == "Workshop":
            if any(marker in lowered for marker in ("step", "first", "next", "then")):
                return "Step"
            if any(marker in lowered for marker in ("demo", "configure", "install", "build", "run")):
                return "Demonstration"
            if any(marker in lowered for marker in ("check", "verify", "test", "validate")):
                return "Check"
            return "Practice"
        if content_format == "Interview":
            if any(marker in lowered for marker in ("question", "asked", "i asked", "you asked")):
                return "Question"
            if any(marker in lowered for marker in ("because", "i think", "i believe", "my view")):
                return "Response"
            return "Perspective"
        if content_format in {"Podcast", "Talk"}:
            if any(marker in lowered for marker in ("story", "journey", "experience")):
                return "Story"
            if any(marker in lowered for marker in ("because", "means", "shows", "reveals", "important")):
                return "Insight"
            return "Theme"
        return None

    def _build_action_items(
        self,
        source_url: str | None,
        sentence_records: list[dict[str, Any]],
        chapters: list[ChapterItem],
        *,
        content_format: str | None,
    ) -> list[ActionItem]:
        actions: list[ActionItem] = []
        seen: set[str] = set()
        triggers = (
            "need to",
            "needs to",
            "should",
            "must",
            "follow up",
            "next step",
            "next steps",
            "review",
            "confirm",
            "send",
            "share",
            "schedule",
            "plan",
            "assign",
            "todo",
        )
        decision_markers = (
            "we agreed",
            "agreed to",
            "decided to",
            "decision",
            "we will",
            "we'll",
        )

        def action_is_redundant(title: str, detail: str) -> bool:
            return any(
                self._texts_are_redundant(title, existing.title)
                or self._texts_are_redundant(detail, existing.detail)
                or self._texts_are_redundant(title, existing.detail)
                for existing in actions
            )

        ranked_records = sorted(
            sentence_records,
            key=lambda item: (
                -self._action_priority(str(item["text"])),
                -float(item["score"]),
                float(item["start_seconds"]),
            ),
        )
        for record in ranked_records:
            sentence = str(record["text"])
            lowered = sentence.lower()
            is_explicit_action = self._is_actionable_sentence(lowered, triggers=triggers)
            is_decision = content_format == "Meeting" and any(
                marker in lowered for marker in decision_markers
            )
            if not is_explicit_action and not is_decision:
                continue
            title = self._action_title(sentence)
            key = title.lower()
            if key in seen or action_is_redundant(title, sentence):
                continue
            seen.add(key)
            actions.append(
                ActionItem(
                    title=title if is_explicit_action else f"Follow through on {title}",
                    detail=sentence,
                    due_hint=self._infer_due_hint(lowered),
                    start_seconds=float(record["start_seconds"]),
                    display_time=format_timestamp(float(record["start_seconds"])),
                    jump_url=build_youtube_jump_url(
                        source_url,
                        float(record["start_seconds"]),
                    ),
                )
            )
            if len(actions) == 5:
                break

        synthesized_actions: list[ActionItem] = []
        templates = self._action_templates(content_format)
        for chapter, template in zip(chapters[:3], templates, strict=False):
            title_template, detail_template = template
            supporting_topics = self._chapter_supporting_topics(chapter, limit=2)
            detail = detail_template.format(topic_lower=chapter.title.lower())
            if supporting_topics:
                detail = truncate_text(
                    f"{detail} Focus especially on {self._join_labels([topic.lower() for topic in supporting_topics])}.",
                    180,
                )
            synthesized_actions.append(
                ActionItem(
                    title=title_template.format(topic=chapter.title),
                    detail=detail,
                    start_seconds=chapter.start_seconds,
                    display_time=chapter.display_time,
                    jump_url=chapter.jump_url or build_youtube_jump_url(source_url, chapter.start_seconds),
                )
            )
        if len(actions) < min(3, max(1, len(chapters))):
            for item in synthesized_actions:
                if action_is_redundant(item.title, item.detail):
                    continue
                actions.append(item)
                if len(actions) == 5:
                    break
        if actions:
            return actions[:5]
        return synthesized_actions[:5]

    def _action_templates(
        self,
        content_format: str | None,
    ) -> tuple[tuple[str, str], ...]:
        if content_format == "Meeting":
            return (
                (
                    "Confirm {topic}",
                    "Capture the final decision on {topic_lower}, assign an owner, and note the next deadline or dependency.",
                ),
                (
                    "Share {topic}",
                    "Summarize the agreed direction for {topic_lower} and circulate the key changes to the people involved.",
                ),
                (
                    "Track {topic}",
                    "Monitor progress on {topic_lower} and record blockers, timeline changes, or follow-up decisions.",
                ),
            )
        if content_format == "Lecture":
            return (
                (
                    "Review {topic}",
                    "Revisit {topic_lower} and capture the main claim, strongest evidence, and one practical takeaway.",
                ),
                (
                    "Explain {topic}",
                    "Write a short explanation of {topic_lower} using the chapter summary and one concrete example from the transcript.",
                ),
                (
                    "Compare {topic}",
                    "Compare {topic_lower} with the surrounding chapter ideas and note the most important distinction or tradeoff.",
                ),
            )
        if content_format == "Workshop":
            return (
                (
                    "Practice {topic}",
                    "Repeat the workflow for {topic_lower} step by step and note the exact setup, command, or sequence used in the transcript.",
                ),
                (
                    "Apply {topic}",
                    "Use {topic_lower} in a small example of your own and capture the result, output, or behavior you observed.",
                ),
                (
                    "Check {topic}",
                    "Verify the key checks, troubleshooting steps, or validation criteria tied to {topic_lower}.",
                ),
            )
        if content_format == "Interview":
            return (
                (
                    "Capture {topic}",
                    "Write down the speaker's position on {topic_lower} and the reasoning they used to support it.",
                ),
                (
                    "Reflect On {topic}",
                    "Summarize what makes {topic_lower} persuasive, debatable, or surprising in the conversation.",
                ),
                (
                    "Connect {topic}",
                    "Explain how {topic_lower} links to the broader themes, examples, or contrasts in the interview.",
                ),
            )
        if content_format in {"Podcast", "Talk"}:
            return (
                (
                    "Capture {topic}",
                    "Write down the strongest idea behind {topic_lower} and the example or story used to make it memorable.",
                ),
                (
                    "Reflect On {topic}",
                    "Summarize why {topic_lower} matters and what practical implication or perspective it adds to the discussion.",
                ),
                (
                    "Connect {topic}",
                    "Relate {topic_lower} to the broader themes in the episode and note the insight it reinforces.",
                ),
            )
        return (
            (
                "Review {topic}",
                "Revisit {topic_lower} and capture the main claim, strongest evidence, and one practical takeaway.",
            ),
            (
                "Explain {topic}",
                "Write a short explanation of {topic_lower} using the chapter summary and one concrete example from the transcript.",
            ),
            (
                "Connect {topic}",
                "Summarize how {topic_lower} fits into the overall video and note the decision, framework, or implication it supports.",
            ),
        )

    def _build_quotes(
        self,
        source_url: str | None,
        sentence_records: list[dict[str, Any]],
        chapters: list[ChapterItem],
    ) -> list[QuoteItem]:
        candidates: list[tuple[float, str, str, dict[str, Any]]] = []
        for record in sentence_records:
            sentence = self._normalize_sentence_text(
                str(record["text"]),
                minimum_words=8,
            )
            if sentence is None:
                continue
            if not 8 <= len(sentence.split()) <= 24:
                continue
            if sentence.endswith("?"):
                continue
            if self._text_is_low_quality(sentence):
                continue
            context = self._chapter_context(chapters, float(record["start_seconds"]))
            score = self._quote_candidate_score(
                sentence,
                record=record,
                context=context,
            )
            if score <= 0:
                continue
            candidates.append((score, sentence, context, record))

        candidates.sort(
            key=lambda item: (
                -item[0],
                float(item[3]["start_seconds"]),
            )
        )
        quotes: list[QuoteItem] = []
        seen: set[str] = set()
        used_contexts: set[str] = set()

        def append_quote(
            sentence: str,
            context: str,
            record: dict[str, Any],
        ) -> None:
            seen.add(sentence.lower())
            quotes.append(
                QuoteItem(
                    quote=sentence,
                    context=context,
                    start_seconds=float(record["start_seconds"]),
                    display_time=format_timestamp(float(record["start_seconds"])),
                    jump_url=build_youtube_jump_url(
                        source_url,
                        float(record["start_seconds"]),
                    ),
                )
            )

        for _, sentence, context, record in candidates:
            key = sentence.lower()
            if key in seen or context in used_contexts:
                continue
            append_quote(sentence, context, record)
            used_contexts.add(context)
            if len(quotes) == 4:
                return quotes

        for _, sentence, context, record in candidates:
            if sentence.lower() in seen:
                continue
            append_quote(sentence, context, record)
            if len(quotes) == 4:
                break
        return quotes

    def _build_learning_objectives(
        self,
        chapters: list[ChapterItem],
        key_topics: list[str],
        *,
        content_format: str | None,
        title: str,
        transcript_text: str,
    ) -> list[str]:
        objectives: list[str] = []
        seen: set[str] = set()
        title_topics = self._title_topic_candidates(title)
        chapter_topics = [
            candidate
            for chapter in chapters[:4]
            for candidate in (chapter.title, *chapter.keywords[:3])
        ]
        topic_candidates = self._sanitize_topic_list(
            [*title_topics, *key_topics[:6], *chapter_topics],
            limit=8,
        )
        programming_language = self._programming_language(title, topic_candidates)
        programming_context = self._is_programming_context(
            title=title,
            topic_candidates=topic_candidates,
            chapters=chapters,
        )

        def add_objective(text: str) -> None:
            normalized = normalize_whitespace(text)
            if (
                not normalized
                or normalized.lower() in seen
                or self._text_is_low_quality(normalized)
            ):
                return
            seen.add(normalized.lower())
            objectives.append(normalized)

        if programming_context:
            programming_topics = self._programming_focus_candidates(
                title=title,
                transcript_text=transcript_text,
                topic_candidates=topic_candidates,
            )
            combined_programming_topics = self._sanitize_topic_list(
                [*programming_topics, *topic_candidates],
                limit=10,
            )
            subject = self._programming_subject(
                title=title,
                topic_candidates=combined_programming_topics,
                language=programming_language,
            )
            parameter_topic = self._find_topic(
                combined_programming_topics,
                ("parameter", "parameters"),
            )
            argument_topic = self._find_topic(
                combined_programming_topics,
                ("argument label", "argument labels", "external label", "external labels"),
            )
            return_topic = self._find_topic(
                combined_programming_topics,
                ("return keyword", "return value", "return values", "return", "returns"),
            )
            function_call_topic = self._find_topic(
                combined_programming_topics,
                ("function call", "function calls", "call syntax"),
            )
            if subject:
                display_subject = subject
                if display_subject.lower().endswith("functions"):
                    add_objective(
                        f"Explain how {display_subject} are defined and called."
                    )
                else:
                    add_objective(
                        f"Explain how {display_subject} works and why the syntax matters."
                    )
            if parameter_topic and argument_topic:
                add_objective(
                    "Differentiate parameters from argument labels in a function call."
                )
            elif parameter_topic:
                add_objective(
                    "Explain how parameters shape the inputs to a function."
                )
            if return_topic:
                add_objective(
                    "Use the return keyword to send a value back from a function."
                )
            if argument_topic:
                add_objective(
                    "Read and write function calls with clear argument labels."
                )
            elif function_call_topic:
                add_objective(
                    "Trace how a function call maps each input to the parameter it fills."
                )
            if subject and (parameter_topic or return_topic):
                example_subject = subject
                if example_subject.lower().endswith("functions"):
                    example_subject = example_subject[:-1]
                example_detail = "accepts inputs"
                if re.search(r"\btwo numbers\b", transcript_text.lower()):
                    example_detail = "adds two numbers"
                if return_topic:
                    example_detail = f"{example_detail} and returns a value"
                add_objective(
                    f"Build a small {example_subject} example that {example_detail}."
                )
            return objectives[:5]

        if content_format == "Meeting":
            for topic in topic_candidates[:3]:
                add_objective(f"Summarize the decision or alignment around {topic}.")
            if chapters:
                add_objective(
                    f"Track the next step tied to {chapters[0].title} and explain why it matters."
                )
            return objectives[:5]

        if content_format == "Lecture":
            if topic_candidates:
                add_objective(f"Define {topic_candidates[0]} and explain why it matters in this lesson.")
            if len(topic_candidates) >= 2:
                add_objective(f"Compare {topic_candidates[0]} with {topic_candidates[1]}.")
            for topic in topic_candidates[1:4]:
                add_objective(f"Apply {topic} to a new example or problem.")
            return objectives[:5]

        if content_format == "Workshop":
            for topic in topic_candidates[:2]:
                add_objective(f"Practice {topic} in a small hands-on example.")
            for topic in topic_candidates[2:4]:
                add_objective(f"Check the result of {topic} and explain what to verify.")
            return objectives[:5]

        if content_format in {"Podcast", "Talk", "Interview"}:
            if topic_candidates:
                add_objective(f"Explain the main point the speaker makes about {topic_candidates[0]}.")
            for topic in topic_candidates[1:4]:
                add_objective(f"Connect {topic} to the broader discussion or perspective in the video.")
            return objectives[:5]

        fallback_templates = (
            "Explain {topic} clearly and accurately.",
            "Analyze how {topic} connects to the rest of the video.",
            "Apply {topic} in a concrete example or use case.",
            "Evaluate why {topic} matters in the discussion.",
        )
        for topic, template in zip(topic_candidates[:4], fallback_templates, strict=False):
            add_objective(template.format(topic=topic))
        return objectives[:5]

    def _title_topic_candidates(self, title: str) -> list[str]:
        cleaned_title = re.sub(r"\(\d{4}\)", " ", title)
        cleaned_title = re.sub(
            r"\b(?:tutorial|tutorials|lesson|lessons|part|parts|episode|chapter|module)\s*\d*\b",
            " ",
            cleaned_title,
            flags=re.IGNORECASE,
        )
        cleaned_title = re.sub(r"\bfor beginners\b", " ", cleaned_title, flags=re.IGNORECASE)
        cleaned_title = re.sub(r"\bbeginners?\b", " ", cleaned_title, flags=re.IGNORECASE)
        extracted: list[str] = []
        for segment in re.split(r"[:|]", title):
            simplified_segment = re.sub(r"\(\d{4}\)", " ", segment)
            simplified_segment = re.sub(
                r"\b(?:tutorial|tutorials|lesson|lessons|part|parts|episode|chapter|module)\s*\d*\b",
                " ",
                simplified_segment,
                flags=re.IGNORECASE,
            )
            simplified_segment = re.sub(
                r"\bfor beginners\b|\bbeginners?\b",
                " ",
                simplified_segment,
                flags=re.IGNORECASE,
            )
            extracted.extend(extract_keywords(simplified_segment, limit=3))
        extracted.extend(extract_keywords(cleaned_title, limit=6))
        lowered = cleaned_title.lower()
        seeded: list[str] = []
        language = self._programming_language(title, [])
        if language:
            if "function" in lowered:
                seeded.append(f"{language} Functions")
            if "argument" in lowered or "label" in lowered:
                seeded.append("Argument Labels")
            if "parameter" in lowered:
                seeded.append("Parameters")
            if "return" in lowered:
                seeded.append("Return Values")
        return self._sanitize_topic_list([*seeded, *extracted], limit=4)

    def _is_programming_context(
        self,
        *,
        title: str,
        topic_candidates: list[str],
        chapters: list[ChapterItem],
    ) -> bool:
        evidence = " ".join(
            [
                title,
                *topic_candidates,
                *[chapter.title for chapter in chapters[:4]],
                *[
                    keyword
                    for chapter in chapters[:4]
                    for keyword in chapter.keywords[:4]
                ],
            ]
        ).lower()
        markers = (
            "swift",
            "python",
            "javascript",
            "typescript",
            "dart",
            "flutter",
            "react",
            "kotlin",
            "function",
            "functions",
            "parameter",
            "parameters",
            "argument",
            "arguments",
            "return value",
            "return",
            "variable",
            "variables",
            "method",
            "class",
            "array",
            "string",
            "syntax",
            "code",
            "coding",
        )
        return any(marker in evidence for marker in markers)

    def _programming_language(
        self,
        title: str,
        topic_candidates: list[str],
    ) -> str | None:
        evidence = " ".join([title, *topic_candidates]).lower()
        for marker, label in (
            ("swift", "Swift"),
            ("python", "Python"),
            ("javascript", "JavaScript"),
            ("typescript", "TypeScript"),
            ("dart", "Dart"),
            ("flutter", "Flutter"),
            ("react", "React"),
            ("kotlin", "Kotlin"),
        ):
            if marker in evidence:
                return label
        return None

    def _programming_subject(
        self,
        *,
        title: str,
        topic_candidates: list[str],
        language: str | None,
    ) -> str | None:
        function_topic = self._find_topic(topic_candidates, ("function", "functions"))
        if function_topic and language:
            return f"{language} functions"
        if function_topic:
            return function_topic
        if language:
            return f"{language} code"
        title_topics = self._title_topic_candidates(title)
        return title_topics[0] if title_topics else None

    def _programming_focus_candidates(
        self,
        *,
        title: str,
        transcript_text: str,
        topic_candidates: list[str],
    ) -> list[str]:
        evidence = " ".join([title, transcript_text, *topic_candidates]).lower()
        language = self._programming_language(title, topic_candidates)
        candidates: list[str] = []
        if language and "function" in evidence:
            candidates.append(f"{language} Functions")
        if re.search(r"\bparameter[s]?\b", evidence):
            candidates.append("Parameters")
        if re.search(r"\bargument label[s]?\b|\bexternal label[s]?\b", evidence):
            candidates.append("Argument Labels")
        if re.search(r"\bfunction call[s]?\b|\bcall the function\b", evidence):
            candidates.append("Function Calls")
        if re.search(r"\breturn keyword\b", evidence):
            candidates.append("Return Keyword")
        elif re.search(r"\breturn value\b|\breturns?\b", evidence):
            candidates.append("Return Values")
        if re.search(r"\bdata type[s]?\b", evidence):
            candidates.append("Data Types")
        return self._sanitize_topic_list(candidates, limit=6)

    def _find_topic(
        self,
        topics: list[str],
        markers: tuple[str, ...],
    ) -> str | None:
        for topic in topics:
            lowered = topic.lower()
            if any(marker in lowered for marker in markers):
                return topic
        return None

    def _best_objective_topic(
        self,
        topics: list[str],
        *,
        blocked: list[str],
    ) -> str | None:
        blocked_values = [value.lower() for value in blocked if value]
        for topic in topics:
            lowered = topic.lower()
            if any(blocked_value in lowered or lowered in blocked_value for blocked_value in blocked_values):
                continue
            if any(marker in lowered for marker in ("version", "people", "reading", "other people")):
                continue
            return topic
        return None

    def _topic_noise_penalty(self, label: str) -> float:
        lowered = strip_timecodes(normalize_whitespace(label)).lower()
        tokens = [token for token in tokenize(lowered, minimum_length=2) if len(token) > 1]
        if not tokens:
            return 2.0

        penalty = 0.0
        if lowered.startswith("other people") or "people reading" in lowered:
            penalty += 1.9
        if lowered in {
            "back value",
            "descriptive version",
            "help other people",
            "other people reading",
            "short call",
            "understand",
        }:
            penalty += 1.8
        if len(tokens) <= 3 and tokens[-1] in {"reading", "understand", "understanding"}:
            penalty += 1.4
        if len(tokens) <= 3 and tokens[0] in {"write", "look", "help"}:
            penalty += 0.9
        if len(tokens) <= 3 and tokens[0] in {"descriptive", "different", "other", "short"}:
            penalty += 0.9
        if len(tokens) <= 3 and tokens[-1] == "version" and tokens[0] in {"descriptive", "different", "short"}:
            penalty += 1.1
        if len(tokens) <= 3 and tokens[0] == "short" and tokens[-1] == "call":
            penalty += 1.1
        if len(tokens) <= 2 and tokens[0] == "back":
            penalty += 1.2
        if len(tokens) <= 2 and tokens[-1] in {"people", "version"}:
            penalty += 0.7
        return round(penalty, 4)

    def _sentence_contains_topic_noise(self, text: str) -> bool:
        lowered = strip_timecodes(normalize_whitespace(text)).lower()
        weak_phrases = (
            "descriptive version",
            "help other people",
            "other people reading",
            "short call",
        )
        return any(phrase in lowered for phrase in weak_phrases)

    def _build_glossary(
        self,
        sentence_records: list[dict[str, Any]],
        key_topics: list[str],
    ) -> list[GlossaryItem]:
        glossary: list[GlossaryItem] = []
        seen: set[str] = set()
        candidate_terms = key_topics[: self.settings.max_glossary_terms]
        for term in candidate_terms:
            evidence = self._evidence_for_term(sentence_records, term)
            normalized_term = normalize_whitespace(term).rstrip(".")
            if not normalized_term:
                continue
            key = normalized_term.lower()
            if key in seen:
                continue
            seen.add(key)
            glossary.append(
                GlossaryItem(
                    term=normalized_term,
                    definition=self._definition_from_evidence(normalized_term, evidence),
                    evidence=evidence,
                    relevance="high" if normalized_term.lower() in {topic.lower() for topic in key_topics[:3]} else "medium",
                )
            )
            if len(glossary) == self.settings.max_glossary_terms:
                break
        return glossary

    def _build_study_questions(
        self,
        chapters: list[ChapterItem],
        action_items: list[ActionItem],
        glossary: list[GlossaryItem],
        learning_objectives: list[str],
    ) -> list[StudyQuestion]:
        questions: list[StudyQuestion] = []
        seen: set[str] = set()

        def add_question(question: str, **kwargs: Any) -> None:
            normalized = normalize_whitespace(question)
            if not normalized:
                return
            key = normalized.lower()
            if key in seen:
                return
            seen.add(key)
            questions.append(StudyQuestion(question=normalized, **kwargs))

        for index, chapter in enumerate(chapters[:2]):
            add_question(
                f"What are the main takeaways from {chapter.title}?",
                answer=chapter.summary,
                question_type="concept",
                difficulty="introductory",
                related_topic=chapter.title,
                start_seconds=chapter.start_seconds,
                display_time=chapter.display_time,
            )
            connective_answer = truncate_text(
                (
                    f"{chapter.title} connects to the broader video through "
                    f"{', '.join(chapter.keywords[:3]).lower() or 'the surrounding discussion'}."
                ),
                180,
            )
            add_question(
                (
                    f"How does {chapter.title} connect to the rest of the video?"
                    if index == 0
                    else f"Why does {chapter.title} matter here?"
                ),
                answer=connective_answer,
                question_type="reflection",
                difficulty="intermediate",
                related_topic=chapter.title,
                start_seconds=chapter.start_seconds,
                display_time=chapter.display_time,
            )
            if len(questions) >= self.settings.max_study_questions:
                return questions[: self.settings.max_study_questions]

        for item in action_items[:1]:
            add_question(
                f"What actions or decisions are tied to {item.title}?",
                answer=item.detail,
                question_type="application",
                difficulty="intermediate",
                related_topic=item.title,
                start_seconds=item.start_seconds,
                display_time=item.display_time,
            )
            if len(questions) >= self.settings.max_study_questions:
                return questions[: self.settings.max_study_questions]

        for item in glossary[:2]:
            add_question(
                f"How would you explain {item.term} in your own words?",
                answer=item.definition,
                question_type="exam",
                difficulty="intermediate",
                related_topic=item.term,
            )
        for objective in learning_objectives[:1]:
            add_question(
                f"How could you apply this objective in practice: {objective}",
                answer="Use the transcript examples and chapter evidence to justify the application.",
                question_type="application",
                difficulty="advanced",
                related_topic=objective,
            )
        return questions[: self.settings.max_study_questions]

    def _build_analysis_metrics(
        self,
        *,
        transcript_text: str,
        sentence_records: list[dict[str, Any]],
        chapters: list[ChapterItem],
        action_items: list[ActionItem],
        glossary: list[GlossaryItem],
        study_questions: list[StudyQuestion],
    ) -> AnalysisMetrics:
        words = tokenize(transcript_text, minimum_length=1)
        transcript_word_count = len(words)
        unique_word_count = len(set(words))
        sentence_count = len(sentence_records)
        concept_density = round(len(glossary) / max(1, sentence_count), 4)
        return AnalysisMetrics(
            transcript_word_count=transcript_word_count,
            unique_word_count=unique_word_count,
            sentence_count=sentence_count,
            chapter_count=len(chapters),
            action_item_count=len(action_items),
            question_count=len(study_questions),
            estimated_reading_minutes=estimate_reading_minutes(transcript_word_count),
            lexical_diversity=lexical_diversity(transcript_text),
            concept_density=concept_density,
            academic_signal_score=academic_signal_score(transcript_text),
        )

    def _build_mind_map(
        self,
        *,
        title: str,
        transcript_text: str,
        key_topics: list[str],
        chapters: list[ChapterItem],
        glossary: list[GlossaryItem],
        action_items: list[ActionItem],
    ) -> MindMapNode:
        content_format = self._infer_content_format(
            title=title,
            transcript_text=transcript_text,
            chapters=chapters,
            action_items=action_items,
        )
        root_label = content_format or self._best_topic_label([title], fallback=title)
        structured_map = self._build_format_mind_map(
            root_label=root_label,
            content_format=content_format,
            key_topics=key_topics,
            chapters=chapters,
            glossary=glossary,
            action_items=action_items,
        )
        if structured_map is not None and not self._mind_map_is_low_quality(structured_map):
            return structured_map
        branches: list[MindMapNode] = []
        used_labels: list[str] = [root_label]

        def add_branch(branch_label: str, child_labels: list[str]) -> bool:
            normalized_branch = self._best_topic_label(
                [branch_label],
                fallback=branch_label,
                max_words=7,
            )
            if (
                not normalized_branch
                or any(self._texts_are_redundant(normalized_branch, existing) for existing in used_labels)
            ):
                return False
            unique_children = self._mind_map_unique_labels(
                child_labels,
                blocked_labels=[root_label, normalized_branch],
                global_labels=used_labels,
                limit=3,
            )
            minimum_children = 1 if not branches else 2
            if len(unique_children) < minimum_children:
                return False
            node = MindMapNode(
                label=normalized_branch,
                children=self._mind_map_leaves(unique_children),
            )
            branches.append(node)
            used_labels.append(normalized_branch)
            used_labels.extend(child.label for child in node.children)
            return True

        if content_format == "Meeting":
            action_branch = self._meeting_action_branch(action_items)
            if action_branch is not None:
                branches.append(action_branch)
                used_labels.append(action_branch.label)
                used_labels.extend(child.label for child in action_branch.children)
        for chapter in chapters[:4]:
            branch_label = self._best_topic_label(
                [chapter.title],
                fallback=chapter.title,
                max_words=7,
            )
            if (
                not branch_label
                or branch_label.lower() == root_label.lower()
                or any(self._texts_are_redundant(branch_label, node.label) for node in branches)
            ):
                continue
            chapter_children = self._chapter_mind_map_concepts(
                chapter,
                key_topics,
                glossary,
                root_label=root_label,
            )
            if not chapter_children:
                chapter_children = self._chapter_supporting_topics(chapter, limit=3)
            if not chapter_children:
                chapter_children = self._mind_map_branch_concepts(
                    branch_label,
                    key_topics,
                    chapters,
                    glossary,
            )
            if not chapter_children:
                chapter_children = chapter.keywords or extract_keywords(
                    chapter.summary,
                    limit=4,
                )
            if add_branch(branch_label, chapter_children) and len(branches) == 4:
                break

        for topic in key_topics:
            branch_label = self._best_topic_label([topic], fallback=topic)
            if (
                not branch_label
                or branch_label.lower() == root_label.lower()
                or any(self._texts_are_redundant(branch_label, node.label) for node in branches)
            ):
                continue
            topic_children = self._topic_mind_map_concepts(
                branch_label,
                chapters,
                glossary,
            )
            if add_branch(branch_label, topic_children) and len(branches) == 5:
                break

        if not branches:
            fallback_branches = chapters[:4] if chapters else []
            for chapter in fallback_branches:
                if not chapter.title:
                    continue
                add_branch(
                    chapter.title,
                    self._chapter_supporting_topics(chapter, limit=3)
                    or chapter.keywords
                    or extract_keywords(chapter.summary, limit=4),
                )
        return MindMapNode(label=root_label, children=branches)

    def _build_format_mind_map(
        self,
        *,
        root_label: str,
        content_format: str | None,
        key_topics: list[str],
        chapters: list[ChapterItem],
        glossary: list[GlossaryItem],
        action_items: list[ActionItem],
    ) -> MindMapNode | None:
        if not content_format:
            return None

        glossary_terms = [item.term for item in glossary]
        chapter_titles = [chapter.title for chapter in chapters]
        chapter_keywords = [
            keyword
            for chapter in chapters
            for keyword in chapter.keywords[:3]
        ]
        chapter_support = [
            topic
            for chapter in chapters
            for topic in self._chapter_supporting_topics(chapter, limit=2)
        ]
        branch_specs: list[tuple[str, list[str]]]

        if content_format == "Meeting":
            dependency_terms = [
                topic
                for topic in [*chapter_support, *chapter_keywords, *key_topics]
                if any(
                    marker in topic.lower()
                    for marker in ("timeline", "owner", "review", "risk", "launch", "dependency", "approval")
                )
            ]
            branch_specs = [
                ("Discussion Topics", [*chapter_titles, *key_topics]),
                ("Decisions", [*chapter_keywords, *chapter_support, *key_topics]),
                ("Action Items", [item.title for item in action_items]),
                ("Dependencies", dependency_terms or chapter_support),
            ]
        elif content_format == "Lecture":
            branch_specs = [
                ("Core Concepts", [*key_topics, *glossary_terms]),
                ("Section Flow", chapter_titles),
                ("Examples & Applications", [*chapter_support, *chapter_keywords]),
            ]
        elif content_format == "Workshop":
            branch_specs = [
                ("Workflow Steps", chapter_titles),
                ("Tools & Setup", [*key_topics, *glossary_terms, *chapter_keywords]),
                ("Practice Outcomes", [*chapter_support, *[item.title for item in action_items]]),
            ]
        elif content_format == "Interview":
            branch_specs = [
                ("Key Questions", chapter_titles),
                ("Perspectives", [*key_topics, *chapter_keywords]),
                ("Takeaways", [*chapter_support, *glossary_terms, *key_topics]),
            ]
        else:
            branch_specs = [
                ("Main Themes", [*key_topics, *chapter_titles]),
                ("Key Insights", [*chapter_support, *chapter_keywords, *glossary_terms]),
                ("Examples & Takeaways", [*glossary_terms, *chapter_keywords, *key_topics]),
            ]

        branches: list[MindMapNode] = []
        used_labels: list[str] = [root_label]
        for branch_label, candidates in branch_specs:
            unique_children = self._mind_map_unique_labels(
                candidates,
                blocked_labels=[root_label, branch_label],
                global_labels=used_labels,
                limit=3,
            )
            if not unique_children:
                continue
            node = MindMapNode(
                label=branch_label,
                children=self._mind_map_leaves(unique_children),
            )
            branches.append(node)
            used_labels.append(branch_label)
            used_labels.extend(child.label for child in node.children)
        if not branches:
            return None
        return MindMapNode(label=root_label, children=branches)

    def _infer_content_format(
        self,
        *,
        title: str,
        transcript_text: str,
        chapters: list[ChapterItem],
        action_items: list[ActionItem],
    ) -> str | None:
        title_text = strip_timecodes(normalize_whitespace(title)).lower()
        chapter_text = " ".join(chapter.title for chapter in chapters[:4]).lower()
        transcript_excerpt = strip_timecodes(transcript_text).lower()[:6000]
        evidence = " ".join([title_text, chapter_text, transcript_excerpt])
        scores: dict[str, float] = {
            "Meeting": 0.0,
            "Lecture": 0.0,
            "Podcast": 0.0,
            "Talk": 0.0,
            "Interview": 0.0,
            "Workshop": 0.0,
        }

        def add_score(
            label: str,
            phrases: tuple[str, ...],
            *,
            title_bonus: float,
            body_bonus: float,
        ) -> None:
            for phrase in phrases:
                if phrase in title_text:
                    scores[label] += title_bonus
                elif phrase in evidence:
                    scores[label] += body_bonus

        add_score(
            "Meeting",
            (
                "meeting",
                "standup",
                "sync",
                "agenda",
                "decision",
                "next steps",
                "follow up",
                "action item",
                "owner",
                "deadline",
            ),
            title_bonus=2.5,
            body_bonus=1.2,
        )
        add_score(
            "Lecture",
            (
                "lecture",
                "lesson",
                "course",
                "professor",
                "student",
                "class",
                "chapter",
                "module",
                "today we will",
                "in this lecture",
            ),
            title_bonus=2.6,
            body_bonus=1.15,
        )
        add_score(
            "Podcast",
            (
                "podcast",
                "episode",
                "host",
                "show notes",
                "subscribe",
                "guest",
                "listening",
            ),
            title_bonus=2.7,
            body_bonus=1.1,
        )
        add_score(
            "Talk",
            (
                "talk",
                "keynote",
                "presentation",
                "conference",
                "speaker",
                "audience",
                "today i want to talk",
            ),
            title_bonus=2.5,
            body_bonus=1.0,
        )
        add_score(
            "Interview",
            (
                "interview",
                "interviewer",
                "interviewee",
                "questions",
                "asked",
                "guest",
                "q and a",
            ),
            title_bonus=2.7,
            body_bonus=1.0,
        )
        add_score(
            "Workshop",
            (
                "workshop",
                "webinar",
                "training",
                "demo",
                "tutorial",
                "walkthrough",
                "hands on",
                "step by step",
                "for beginners",
            ),
            title_bonus=2.6,
            body_bonus=1.0,
        )

        if action_items:
            scores["Meeting"] += 1.2
        if re.search(r"\bwe need to\b|\bnext steps?\b|\bowner\b|\bagenda\b", transcript_excerpt):
            scores["Meeting"] += 1.0
        if re.search(r"\bin this lecture\b|\btoday we (?:will|are going to)\b|\bstudents?\b", transcript_excerpt):
            scores["Lecture"] += 1.0
        if re.search(r"\bhost\b|\bguest\b|\bsubscribe\b|\bepisode\b", transcript_excerpt):
            scores["Podcast"] += 0.9
        if re.search(r"\bquestion\b|\banswer\b|\basked\b", transcript_excerpt):
            scores["Interview"] += 0.8
        if re.search(r"\bdemo\b|\bwalkthrough\b|\bexercise\b|\bstep one\b", transcript_excerpt):
            scores["Workshop"] += 0.8
        if re.search(r"\btutorial\b|\bcoding along\b|\bbuild this\b", transcript_excerpt):
            scores["Workshop"] += 0.9

        best_label, best_score = max(scores.items(), key=lambda item: item[1])
        return best_label if best_score >= 2.1 else None

    def _meeting_action_branch(self, action_items: list[ActionItem]) -> MindMapNode | None:
        if not action_items:
            return None
        leaves = self._mind_map_leaves(
            [
                item.title
                for item in action_items[:4]
                if item.title and not is_low_signal_topic(item.title)
            ]
        )
        if not leaves:
            return None
        return MindMapNode(label="Action Items", children=leaves)

    def _mind_map_branch_concepts(
        self,
        branch_label: str,
        key_topics: list[str],
        chapters: list[ChapterItem],
        glossary: list[GlossaryItem],
    ) -> list[str]:
        candidates: list[str] = []
        for chapter in chapters:
            if not (
                self._labels_share_topic(branch_label, chapter.title)
                or branch_label.lower() in chapter.summary.lower()
                or any(self._labels_share_topic(branch_label, keyword) for keyword in chapter.keywords)
            ):
                continue
            if chapter.title.lower() != branch_label.lower():
                candidates.append(chapter.title)
            candidates.extend(chapter.keywords[:4])
            candidates.extend(extract_keywords(chapter.summary, limit=4))

        for topic in key_topics:
            if topic.lower() == branch_label.lower():
                continue
            if self._labels_share_topic(branch_label, topic):
                candidates.append(topic)

        for item in glossary:
            if item.term.lower() == branch_label.lower():
                continue
            if self._labels_share_topic(branch_label, item.term) or branch_label.lower() in item.definition.lower():
                candidates.append(item.term)

        return self._sanitize_topic_list(candidates, limit=6)

    def _labels_share_topic(self, left: str, right: str) -> bool:
        left_text = normalize_whitespace(left).lower()
        right_text = normalize_whitespace(right).lower()
        if not left_text or not right_text:
            return False
        if left_text in right_text or right_text in left_text:
            return True
        left_tokens = set(tokenize(left_text, minimum_length=2))
        right_tokens = set(tokenize(right_text, minimum_length=2))
        if not left_tokens or not right_tokens:
            return False
        overlap = left_tokens & right_tokens
        return len(overlap) >= max(1, min(len(left_tokens), len(right_tokens), 2))

    def _chapter_mind_map_concepts(
        self,
        chapter: ChapterItem,
        key_topics: list[str],
        glossary: list[GlossaryItem],
        *,
        root_label: str,
    ) -> list[str]:
        chapter_evidence = strip_timecodes(
            f"{chapter.title}. {chapter.summary}. {' '.join(chapter.keywords)}"
        ).lower()
        candidates = self._sanitize_topic_list(
            [
                *chapter.keywords,
                *extract_keywords(f"{chapter.title}. {chapter.summary}", limit=6),
                *[
                    topic
                    for topic in key_topics
                    if topic.lower() in chapter_evidence and topic.lower() != chapter.title.lower()
                ],
                *[
                    item.term
                    for item in glossary
                    if item.term.lower() in chapter_evidence and item.term.lower() != chapter.title.lower()
                ],
            ],
            limit=10,
        )
        ranked = sorted(
            candidates,
            key=lambda candidate: -self._mind_map_concept_score(
                candidate,
                branch_label=chapter.title,
                root_label=root_label,
                evidence_text=chapter_evidence,
            ),
        )
        refined = [
            candidate
            for candidate in ranked
            if not self._labels_share_topic(candidate, chapter.title)
            and not self._labels_share_topic(candidate, root_label)
        ]
        multiword = [candidate for candidate in refined if len(candidate.split()) >= 2]
        return (multiword or refined)[:3]

    def _topic_mind_map_concepts(
        self,
        topic: str,
        chapters: list[ChapterItem],
        glossary: list[GlossaryItem],
    ) -> list[str]:
        chapter_matches = [
            candidate
            for chapter in chapters
            if topic.lower() in chapter.title.lower()
            or topic.lower() in chapter.summary.lower()
            or any(topic.lower() in keyword.lower() for keyword in chapter.keywords)
            for candidate in (
                [
                    *chapter.keywords[:4],
                    *extract_keywords(chapter.summary, limit=4),
                    chapter.title,
                ]
            )
        ]
        glossary_matches = [
            item.term
            for item in glossary
            if topic.lower() in item.term.lower() or topic.lower() in item.definition.lower()
        ]
        return self._sanitize_topic_list(
            [*chapter_matches, *glossary_matches],
            limit=3,
        )

    def _mind_map_leaves(self, labels: list[str]) -> list[MindMapNode]:
        leaves: list[MindMapNode] = []
        seen: list[str] = []
        for label in labels:
            normalized = self._clean_topic_label(label)
            if not normalized:
                continue
            if any(self._texts_are_redundant(normalized, existing) for existing in seen):
                continue
            seen.append(normalized)
            leaves.append(
                MindMapNode(
                    label=truncate_text(normalized, 42),
                )
            )
            if len(leaves) == 3:
                break
        return leaves

    def _mind_map_unique_labels(
        self,
        labels: list[str],
        *,
        blocked_labels: list[str],
        global_labels: list[str],
        limit: int,
    ) -> list[str]:
        unique: list[str] = []
        blockers = [label for label in [*blocked_labels, *global_labels] if label.strip()]
        for label in labels:
            normalized = self._clean_topic_label(label)
            if not normalized:
                continue
            if any(
                self._texts_are_redundant(normalized, blocker)
                for blocker in [*blockers, *unique]
            ):
                continue
            unique.append(normalized)
            if len(unique) == limit:
                break
        return unique

    def _mind_map_concept_score(
        self,
        label: str,
        *,
        branch_label: str,
        root_label: str,
        evidence_text: str,
    ) -> float:
        cleaned = self._clean_topic_label(label)
        if not cleaned:
            return float("-inf")
        score = 0.0
        word_count = len(cleaned.split())
        if 2 <= word_count <= 4:
            score += 1.6
        elif word_count == 1:
            score += 0.2
        else:
            score += 0.8
        lowered = cleaned.lower()
        if lowered in evidence_text:
            score += 1.0
        if self._labels_share_topic(cleaned, branch_label):
            score -= 1.5
        if root_label and self._labels_share_topic(cleaned, root_label):
            score -= 0.6
        if is_low_signal_topic(cleaned):
            score -= 2.5
        return score

    def _mind_map_is_low_quality(self, node: MindMapNode | None) -> bool:
        if node is None or not node.children:
            return True
        branches = [child for child in node.children if self._clean_topic_label(child.label)]
        if len(branches) < 2:
            return True
        strong_branches = [
            branch
            for branch in branches
            if not is_low_signal_topic(branch.label)
            and len(branch.label.split()) <= 6
        ]
        if len(strong_branches) < 2:
            return True
        strong_leaves = sum(
            1
            for branch in branches
            for child in branch.children
            if not is_low_signal_topic(child.label)
        )
        if strong_leaves < 3:
            return True
        branch_keys = {
            re.sub(r"[^a-z0-9]+", " ", branch.label.lower()).strip()
            for branch in branches
        }
        leaf_keys = [
            re.sub(r"[^a-z0-9]+", " ", child.label.lower()).strip()
            for branch in branches
            for child in branch.children
            if child.label.strip()
        ]
        if len(branch_keys) < len(branches):
            return True
        if len(set(leaf_keys)) < max(3, len(leaf_keys) - 1):
            return True
        return False

    def _quote_candidate_score(
        self,
        sentence: str,
        *,
        record: dict[str, Any],
        context: str,
    ) -> float:
        lowered = sentence.lower()
        word_count = len(sentence.split())
        score = float(record.get("score", 0.0))
        if 10 <= word_count <= 22:
            score += 1.0
        elif word_count > 24:
            score -= 0.35
        if any(
            marker in lowered
            for marker in (
                "because",
                "which means",
                "this is why",
                "explains",
                "shows",
                "highlights",
                "reveals",
                "important",
                "difference",
                "tradeoff",
                "risk",
                "benefit",
            )
        ):
            score += 0.9
        if len(extract_keywords(sentence, limit=4)) >= 2:
            score += 0.45
        if context != "Transcript":
            score += 0.35
        if "," in sentence or ";" in sentence:
            score += 0.15
        return round(score, 4)

    def _best_topic_label(
        self,
        candidates: list[str],
        *,
        fallback: str,
        max_words: int = 6,
    ) -> str:
        for candidate in candidates:
            cleaned = self._clean_topic_label(candidate, max_words=max_words)
            if cleaned:
                return cleaned
        return fallback

    def _clean_topic_label(
        self,
        value: str | None,
        *,
        max_words: int = 6,
    ) -> str | None:
        if value is None:
            return None
        normalized_value = strip_timecodes(normalize_whitespace(value).replace("’", "'"))
        has_question = normalized_value.endswith("?")
        cleaned = normalized_value.strip(" .,:;!-")
        if not cleaned:
            return None
        if ":" in cleaned:
            cleaned = cleaned.split(":", 1)[0].strip()
        prefer_raw = len(cleaned.split()) <= (max_words + 1) and "." not in cleaned
        candidates = [cleaned, *extract_keywords(cleaned, limit=3)] if prefer_raw else [
            *extract_keywords(cleaned, limit=3),
            cleaned,
        ]
        for candidate in candidates:
            normalized = normalize_whitespace(candidate).strip(" .,:;!?-")
            if not normalized:
                continue
            tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]+", normalized)
            if not tokens:
                continue
            compact = " ".join(tokens[:max_words])
            if is_low_signal_topic(compact):
                continue
            if self._topic_noise_penalty(compact) >= 1.8:
                continue
            if (
                compact.lower() == compact
                or compact.upper() == compact
                or compact == compact.capitalize()
            ):
                compact = compact.title()
            if prefer_raw and has_question and not compact.endswith("?"):
                compact = f"{compact}?"
            return compact
        return None

    def _sanitize_topic_list(self, items: list[str], *, limit: int) -> list[str]:
        topics: list[str] = []
        seen: set[str] = set()
        token_sets: list[set[str]] = []
        for item in items:
            cleaned = self._clean_topic_label(item)
            if not cleaned:
                continue
            key = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
            if not key or key in seen:
                continue
            tokens = {
                token
                for token in tokenize(cleaned.lower(), minimum_length=2)
                if len(token) > 1
            }
            if not tokens:
                continue
            if any(key in existing or existing in key for existing in seen if len(key) >= 8 and len(existing) >= 8):
                continue
            if any(
                (
                    len(tokens) == 1
                    and next(iter(tokens)) in existing_tokens
                )
                or (
                    len(tokens) > 1
                    and len(tokens & existing_tokens) / max(1, min(len(tokens), len(existing_tokens))) >= 0.5
                )
                for existing_tokens in token_sets
            ):
                continue
            seen.add(key)
            token_sets.append(tokens)
            topics.append(cleaned)
            if len(topics) == limit:
                break
        return topics

    def _topic_support_score(
        self,
        label: str,
        *,
        evidence_text: str,
        sentence_records: list[dict[str, Any]],
        chunks: list[TranscriptChunk],
    ) -> float:
        cleaned_label = self._clean_topic_label(label)
        if not cleaned_label:
            return 0.0
        tokens = [token for token in tokenize(cleaned_label.lower(), minimum_length=2) if len(token) > 1]
        if not tokens:
            return 0.0
        lowered_label = cleaned_label.lower()
        lowered_evidence = strip_timecodes(evidence_text).lower()
        evidence_hit = 1.1 if lowered_label in lowered_evidence else 0.0
        chunk_hits = sum(
            1
            for chunk in chunks
            if lowered_label in strip_timecodes(chunk.text).lower()
            or all(token in tokenize(chunk.text.lower(), minimum_length=2) for token in tokens[:2])
        )
        sentence_hits = sum(
            1
            for record in sentence_records
            if lowered_label in strip_timecodes(str(record["text"])).lower()
            or all(token in tokenize(str(record["text"]).lower(), minimum_length=2) for token in tokens[:2])
        )
        return round(evidence_hit + (chunk_hits * 0.9) + min(1.8, sentence_hits * 0.25), 4)

    def _top_sentence_records(
        self,
        sentence_records: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not sentence_records or limit <= 0:
            return []
        ranked = sorted(
            sentence_records,
            key=lambda item: (-float(item["score"]), float(item["start_seconds"])),
        )
        chosen: list[dict[str, Any]] = []
        for candidate in ranked:
            if any(
                abs(float(candidate["start_seconds"]) - float(existing["start_seconds"])) < 20
                for existing in chosen
            ):
                continue
            if any(
                self._texts_are_redundant(str(candidate["text"]), str(existing["text"]))
                for existing in chosen
            ):
                continue
            chosen.append(candidate)
            if len(chosen) == limit:
                break
        if len(chosen) < limit:
            for record in sample_evenly(sentence_records, limit):
                if record in chosen:
                    continue
                if any(
                    self._texts_are_redundant(str(record["text"]), str(existing["text"]))
                    for existing in chosen
                ):
                    continue
                chosen.append(record)
                if len(chosen) == limit:
                    break
        return sorted(chosen, key=lambda item: float(item["start_seconds"]))

    def _in_span(
        self,
        point_seconds: float,
        *,
        start_seconds: float,
        end_seconds: float | None,
    ) -> bool:
        if end_seconds is None or end_seconds <= start_seconds:
            return point_seconds >= start_seconds
        return start_seconds <= point_seconds < end_seconds

    def _chapter_context(
        self,
        chapters: list[ChapterItem],
        start_seconds: float,
    ) -> str:
        for chapter in chapters:
            if chapter.start_seconds <= start_seconds <= (chapter.end_seconds or chapter.start_seconds):
                return chapter.title
        return "Transcript"

    def _evidence_for_term(
        self,
        sentence_records: list[dict[str, Any]],
        term: str,
    ) -> str:
        lowered_term = term.lower()
        for record in sentence_records:
            text = str(record["text"])
            if lowered_term in text.lower():
                return truncate_text(text, 180)
        for record in self._top_sentence_records(sentence_records, limit=2):
            return truncate_text(str(record["text"]), 180)
        return f"{term} is discussed throughout the transcript."

    def _definition_from_evidence(self, term: str, evidence: str) -> str:
        patterns = (
            rf"{re.escape(term)} is ([^.]+)",
            rf"{re.escape(term)} refers to ([^.]+)",
            rf"{re.escape(term)} means ([^.]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, evidence, flags=re.IGNORECASE)
            if match:
                return truncate_text(match.group(1).strip(), 160)
        cleaned_evidence = evidence.rstrip(".")
        if cleaned_evidence.lower().startswith(term.lower()):
            return truncate_text(cleaned_evidence, 160)
        return truncate_text(f"{term} relates to: {cleaned_evidence}.", 160)

    def _coerce_note_sections(self, value: Any) -> list[NoteSection]:
        items = value if isinstance(value, list) else []
        sections: list[NoteSection] = []
        for item in items:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    sections.append(
                        NoteSection(
                            heading=text,
                            bullet_points=[],
                            detail=text,
                        )
                    )
                continue
            if not isinstance(item, dict):
                continue

            bullets = self._coerce_string_list(
                item.get("bullet_points")
                or item.get("bullets")
                or item.get("key_points")
                or item.get("points"),
            )
            heading = self._coerce_text(
                item.get("heading")
                or item.get("title")
                or item.get("topic")
                or item.get("section"),
                fallback="",
            )
            detail = self._coerce_text(
                item.get("detail")
                or item.get("text")
                or item.get("summary")
                or item.get("description"),
                fallback="",
            )
            if not heading:
                heading = bullets[0] if bullets else "Section"
            if not detail:
                detail = " ".join(bullets) if bullets else heading
            heading = self._best_topic_label([heading, detail], fallback=heading or "Section")

            sections.append(
                NoteSection(
                    heading=heading,
                    bullet_points=bullets,
                    detail=detail,
                    start_seconds=self._coerce_seconds(item.get("start_seconds")),
                    display_time=self._coerce_optional_text(item.get("display_time")),
                    jump_url=self._coerce_optional_text(item.get("jump_url")),
                )
            )
        return sections

    def _coerce_chapters(
        self,
        source_url: str | None,
        value: Any,
    ) -> list[ChapterItem]:
        items = value if isinstance(value, list) else []
        chapters: list[ChapterItem] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            start_seconds = self._coerce_seconds(
                item["start_seconds"]
                if "start_seconds" in item
                else item["time_seconds"]
                if "time_seconds" in item
                else item.get("seconds"),
            )
            if start_seconds is None:
                continue
            title = self._coerce_text(
                item.get("title") or item.get("label") or item.get("heading"),
                fallback=f"Chapter {index + 1}",
            )
            summary = self._coerce_text(
                item.get("summary") or item.get("description") or item.get("text"),
                fallback=title,
            )
            keywords = self._chapter_keywords(
                summary,
                supporting_records=[],
                focus_candidates=[
                    title,
                    *self._coerce_string_list(item.get("keywords")),
                ],
            )
            cleaned_title = self._clean_topic_label(title, max_words=6)
            title = (
                cleaned_title
                if cleaned_title and not is_low_signal_topic(cleaned_title)
                else self._chapter_title(
                    [title, *keywords, summary],
                    chapter_text=summary,
                    supporting_records=[],
                    fallback=f"Chapter {index + 1}",
                )
            )
            chapters.append(
                ChapterItem(
                    title=title,
                    summary=summary,
                    start_seconds=start_seconds,
                    end_seconds=self._coerce_seconds(item.get("end_seconds")),
                    display_time=format_timestamp(start_seconds),
                    jump_url=build_youtube_jump_url(source_url, start_seconds),
                    keywords=keywords,
                    confidence=self._coerce_float(item.get("confidence"), fallback=0.72),
                )
            )
        return chapters

    def _coerce_timestamps(
        self,
        source_url: str | None,
        value: Any,
    ) -> list[TimestampItem]:
        items = value if isinstance(value, list) else []
        timestamps: list[TimestampItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            start_seconds = self._coerce_seconds(
                item["start_seconds"]
                if "start_seconds" in item
                else item["time_seconds"]
                if "time_seconds" in item
                else item["timestamp_seconds"]
                if "timestamp_seconds" in item
                else item.get("seconds"),
            )
            if start_seconds is None:
                continue
            label = self._coerce_text(
                item.get("label") or item.get("title") or item.get("heading"),
                fallback=f"Timestamp {len(timestamps) + 1}",
            )
            description = self._coerce_text(
                item.get("description")
                or item.get("detail")
                or item.get("summary")
                or item.get("text"),
                fallback=label,
            )
            label = self._chapter_title(
                [label, *extract_keywords(description, limit=4), description],
                chapter_text=description,
                supporting_records=[],
                fallback=label,
            )
            end_seconds = self._coerce_seconds(item.get("end_seconds"))
            timestamps.append(
                TimestampItem(
                    label=label,
                    description=description,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    display_time=format_timestamp(start_seconds),
                    jump_url=build_youtube_jump_url(source_url, start_seconds),
                )
            )
        return timestamps

    def _coerce_action_items(self, value: Any) -> list[ActionItem]:
        items = value if isinstance(value, list) else []
        actions: list[ActionItem] = []
        for item in items:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    actions.append(ActionItem(title=text, detail=text))
                continue
            if not isinstance(item, dict):
                continue
            title = self._coerce_text(
                item.get("title") or item.get("task") or item.get("action"),
                fallback="Follow up",
            )
            detail = self._coerce_text(
                item.get("detail")
                or item.get("description")
                or item.get("text")
                or item.get("summary"),
                fallback=title,
            )
            actions.append(
                ActionItem(
                    title=title,
                    detail=detail,
                    owner_hint=self._coerce_optional_text(
                        item.get("owner_hint") or item.get("owner") or item.get("assignee"),
                    ),
                    due_hint=self._coerce_optional_text(
                        item.get("due_hint") or item.get("due") or item.get("deadline"),
                    ),
                    completed=bool(item.get("completed", False)),
                    start_seconds=self._coerce_seconds(item.get("start_seconds")),
                    display_time=self._coerce_optional_text(item.get("display_time")),
                    jump_url=self._coerce_optional_text(item.get("jump_url")),
                )
            )
        return actions

    def _coerce_quotes(self, value: Any) -> list[QuoteItem]:
        items = value if isinstance(value, list) else []
        quotes: list[QuoteItem] = []
        for item in items:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    quotes.append(
                        QuoteItem(
                            quote=text,
                            context="",
                            start_seconds=0,
                            display_time=format_timestamp(0),
                        )
                    )
                continue
            if not isinstance(item, dict):
                continue
            start_seconds = self._coerce_seconds(
                item["start_seconds"]
                if "start_seconds" in item
                else item["time_seconds"]
                if "time_seconds" in item
                else item.get("timestamp_seconds"),
            )
            if start_seconds is None:
                start_seconds = 0
            quote = self._coerce_text(
                item.get("quote") or item.get("text") or item.get("statement"),
                fallback="",
            )
            if not quote:
                continue
            quotes.append(
                QuoteItem(
                    quote=quote,
                    context=self._coerce_text(
                        item.get("context") or item.get("description"),
                        fallback="",
                    ),
                    start_seconds=start_seconds,
                    display_time=format_timestamp(start_seconds),
                    jump_url=self._coerce_optional_text(item.get("jump_url")),
                )
            )
        return quotes

    def _coerce_glossary(self, value: Any) -> list[GlossaryItem]:
        items = value if isinstance(value, list) else []
        glossary: list[GlossaryItem] = []
        for item in items:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    glossary.append(
                        GlossaryItem(
                            term=text,
                            definition=text,
                            evidence=text,
                        )
                    )
                continue
            if not isinstance(item, dict):
                continue
            term = self._coerce_text(
                item.get("term") or item.get("title") or item.get("label"),
                fallback="Concept",
            )
            glossary.append(
                GlossaryItem(
                    term=term,
                    definition=self._coerce_text(
                        item.get("definition") or item.get("description") or item.get("text"),
                        fallback=term,
                    ),
                    evidence=self._coerce_text(
                        item.get("evidence") or item.get("context"),
                        fallback=term,
                    ),
                    relevance=self._coerce_text(
                        item.get("relevance"),
                        fallback="medium",
                    ),
                )
            )
        return glossary

    def _coerce_study_questions(self, value: Any) -> list[StudyQuestion]:
        items = value if isinstance(value, list) else []
        questions: list[StudyQuestion] = []
        for item in items:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    questions.append(
                        StudyQuestion(
                            question=text,
                            answer="Review the relevant transcript chapter for evidence.",
                        )
                    )
                continue
            if not isinstance(item, dict):
                continue
            start_seconds = self._coerce_seconds(
                item["start_seconds"]
                if "start_seconds" in item
                else item.get("time_seconds"),
            )
            questions.append(
                StudyQuestion(
                    question=self._coerce_text(
                        item.get("question") or item.get("prompt"),
                        fallback="What is the main idea?",
                    ),
                    answer=self._coerce_text(
                        item.get("answer") or item.get("response") or item.get("explanation"),
                        fallback="Review the transcript evidence.",
                    ),
                    question_type=self._coerce_question_type(item.get("question_type")),
                    difficulty=self._coerce_difficulty(item.get("difficulty")),
                    related_topic=self._coerce_optional_text(item.get("related_topic")),
                    start_seconds=start_seconds,
                    display_time=format_timestamp(start_seconds) if start_seconds is not None else None,
                )
            )
        return questions

    def _coerce_analysis_metrics(self, value: Any) -> AnalysisMetrics:
        if not isinstance(value, dict):
            return AnalysisMetrics()
        return AnalysisMetrics(
            transcript_word_count=self._coerce_int(value.get("transcript_word_count"), fallback=0),
            unique_word_count=self._coerce_int(value.get("unique_word_count"), fallback=0),
            sentence_count=self._coerce_int(value.get("sentence_count"), fallback=0),
            chapter_count=self._coerce_int(value.get("chapter_count"), fallback=0),
            action_item_count=self._coerce_int(value.get("action_item_count"), fallback=0),
            question_count=self._coerce_int(value.get("question_count"), fallback=0),
            estimated_reading_minutes=self._coerce_float(
                value.get("estimated_reading_minutes"),
                fallback=0.0,
            ),
            lexical_diversity=self._coerce_float(value.get("lexical_diversity"), fallback=0.0),
            concept_density=self._coerce_float(value.get("concept_density"), fallback=0.0),
            academic_signal_score=self._coerce_float(
                value.get("academic_signal_score"),
                fallback=0.0,
            ),
        )

    def _coerce_mind_map(self, value: Any, *, fallback_label: str) -> MindMapNode:
        if isinstance(value, dict):
            raw_label = self._coerce_text(
                value.get("label") or value.get("title") or value.get("name"),
                fallback=fallback_label,
            )
            label = (
                raw_label
                if raw_label in {"Meeting", "Lecture", "Podcast", "Talk", "Interview", "Workshop"}
                else self._best_topic_label([raw_label], fallback=fallback_label)
            )
            children = value.get("children")
            if not isinstance(children, list):
                children = value.get("nodes") if isinstance(value.get("nodes"), list) else []
            normalized_children: list[MindMapNode] = []
            for child in children:
                if not isinstance(child, dict):
                    continue
                child_node = self._coerce_mind_map(child, fallback_label=label)
                if not self._clean_topic_label(child_node.label):
                    continue
                normalized_children.append(child_node)
            return MindMapNode(
                label=label,
                children=normalized_children,
            )
        if isinstance(value, str) and value.strip():
            return MindMapNode(
                label=self._best_topic_label([value.strip()], fallback=fallback_label),
            )
        return MindMapNode(label=fallback_label)

    def _coerce_question_type(
        self,
        value: Any,
    ) -> str:
        normalized = self._coerce_text(value, fallback="concept").lower()
        if normalized not in {"concept", "application", "reflection", "exam"}:
            return "concept"
        return normalized

    def _coerce_difficulty(self, value: Any) -> str:
        normalized = self._coerce_text(value, fallback="intermediate").lower()
        if normalized not in {"introductory", "intermediate", "advanced"}:
            return "intermediate"
        return normalized

    def _coerce_string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [text for item in value if (text := self._coerce_optional_text(item))]
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return []

    def _coerce_text(self, value: Any, *, fallback: str) -> str:
        text = self._coerce_optional_text(value)
        return text if text is not None else fallback

    def _coerce_optional_text(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, (int, float)):
            return str(value)
        return None

    def _coerce_seconds(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                pass
            parts = text.split(":")
            try:
                seconds = 0.0
                for part in parts:
                    seconds = (seconds * 60) + float(part)
                return seconds
            except ValueError:
                return None
        return None

    def _coerce_int(self, value: Any, *, fallback: int) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except ValueError:
                return fallback
        return fallback

    def _coerce_float(self, value: Any, *, fallback: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return fallback
        return fallback

    def _action_title(self, sentence: str) -> str:
        cleaned = sentence.strip().rstrip(".")
        cleaned = re.sub(
            r"^(we|they|the team|speaker|speakers)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(should|need to|needs to|must|will|can)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        if not cleaned:
            return "Follow up"
        return build_headline(cleaned, fallback="Follow up")

    def _is_actionable_sentence(
        self,
        sentence: str,
        *,
        triggers: tuple[str, ...],
    ) -> bool:
        if any(trigger in sentence for trigger in triggers[:6]):
            return True
        if re.search(r"^(send|share|schedule|assign|confirm|review)\b", sentence):
            return True
        if re.search(r"^(we|the team|you)\s+(need to|should|must|will)\b", sentence):
            return True
        return False

    def _infer_due_hint(self, sentence: str) -> str | None:
        for hint in (
            "today",
            "tomorrow",
            "this week",
            "next week",
            "next meeting",
            "later",
        ):
            if hint in sentence:
                return hint.title()
        return None

    def _action_priority(self, sentence: str) -> float:
        lowered = sentence.lower()
        if not self._is_actionable_sentence(
            lowered,
            triggers=(
                "need to",
                "needs to",
                "should",
                "must",
                "follow up",
                "next step",
                "next steps",
                "review",
                "confirm",
                "send",
                "share",
                "schedule",
                "plan",
                "assign",
                "todo",
            ),
        ):
            return 0.0
        weights = {
            "need to": 3.0,
            "needs to": 3.0,
            "must": 2.8,
            "next": 2.4,
            "send": 2.2,
            "confirm": 2.0,
            "share": 1.8,
            "schedule": 1.8,
            "assign": 1.8,
            "review": 1.2,
            "should": 1.0,
        }
        score = sum(weight for marker, weight in weights.items() if marker in lowered)
        if self._infer_due_hint(lowered):
            score += 0.8
        return score

    def _merge_string_lists(
        self,
        primary: list[str],
        fallback: list[str],
        *,
        limit: int,
    ) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for value in [*primary, *fallback]:
            normalized = normalize_whitespace(value)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
            if len(merged) == limit:
                break
        return merged

    def _postprocess_artifact(
        self,
        artifact: VideoAnalysisArtifact,
    ) -> VideoAnalysisArtifact:
        artifact.five_minute_summary = self._dedupe_exact_strings(artifact.five_minute_summary)[:5]
        artifact.learning_objectives = self._dedupe_exact_strings(artifact.learning_objectives)[:5]
        artifact.key_topics = self._sanitize_topic_list(artifact.key_topics, limit=8)
        artifact.note_sections = self._dedupe_note_sections(artifact.note_sections)
        artifact.timestamps = self._dedupe_timestamps(artifact.timestamps)
        artifact.action_items = self._dedupe_action_items(artifact.action_items)
        artifact.key_quotes = self._dedupe_quotes(artifact.key_quotes)
        if artifact.mind_map is not None:
            artifact.mind_map = self._dedupe_mind_map(artifact.mind_map)
        return artifact

    def _dedupe_note_sections(
        self,
        sections: list[NoteSection],
    ) -> list[NoteSection]:
        deduped: list[NoteSection] = []
        for section in sections:
            if any(
                self._texts_are_redundant(section.heading, existing.heading)
                or self._texts_are_redundant(section.detail, existing.detail)
                for existing in deduped
            ):
                continue
            bullet_points = [
                bullet
                for bullet in self._dedupe_strings(section.bullet_points)
            ][:3]
            deduped.append(
                section.model_copy(
                    update={
                        "bullet_points": bullet_points,
                    }
                )
            )
            if len(deduped) == self.settings.max_chapters:
                break
        return deduped

    def _dedupe_timestamps(
        self,
        timestamps: list[TimestampItem],
    ) -> list[TimestampItem]:
        deduped: list[TimestampItem] = []
        for item in sorted(timestamps, key=lambda candidate: (candidate.start_seconds, candidate.label.lower())):
            if any(
                (
                    abs(item.start_seconds - existing.start_seconds) < 150
                    and self._texts_are_redundant(item.label, existing.label)
                )
                or (
                    self._texts_are_redundant(item.label, existing.label)
                    and self._texts_are_redundant(item.description, existing.description)
                )
                for existing in deduped
            ):
                continue
            deduped.append(item)
        return deduped[: max(self.settings.max_chapters * 2, 10)]

    def _dedupe_action_items(
        self,
        actions: list[ActionItem],
    ) -> list[ActionItem]:
        deduped: list[ActionItem] = []
        for item in actions:
            if any(
                self._texts_are_redundant(item.title, existing.title)
                or self._texts_are_redundant(item.detail, existing.detail)
                for existing in deduped
            ):
                continue
            deduped.append(item)
            if len(deduped) == 5:
                break
        return deduped

    def _dedupe_quotes(
        self,
        quotes: list[QuoteItem],
    ) -> list[QuoteItem]:
        deduped: list[QuoteItem] = []
        for item in quotes:
            if any(
                self._texts_are_redundant(item.quote, existing.quote)
                or (
                    item.context
                    and existing.context
                    and self._texts_are_redundant(item.context, existing.context)
                )
                for existing in deduped
            ):
                continue
            deduped.append(item)
            if len(deduped) == 4:
                break
        return deduped

    def _dedupe_mind_map(
        self,
        node: MindMapNode,
    ) -> MindMapNode:
        deduped_children: list[MindMapNode] = []
        for child in node.children:
            normalized_child = self._dedupe_mind_map(child)
            if any(
                self._texts_are_redundant(normalized_child.label, existing.label)
                for existing in deduped_children
            ):
                continue
            deduped_children.append(normalized_child)
        return node.model_copy(update={"children": deduped_children})

    def _dedupe_strings(self, items: list[str]) -> list[str]:
        deduped: list[str] = []
        for item in items:
            normalized = normalize_whitespace(item)
            if not normalized:
                continue
            if any(self._texts_are_redundant(normalized, existing) for existing in deduped):
                continue
            deduped.append(normalized)
        return deduped

    def _dedupe_exact_strings(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            normalized = normalize_whitespace(item)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    def _analysis_metrics_is_empty(self, metrics: AnalysisMetrics) -> bool:
        return (
            metrics.transcript_word_count == 0
            and metrics.sentence_count == 0
            and metrics.chapter_count == 0
            and metrics.question_count == 0
        )
