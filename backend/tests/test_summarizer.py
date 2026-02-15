import json
import re
from itertools import combinations
from pathlib import Path

from app.core.config import Settings
from app.models import (
    ChapterItem,
    MindMapNode,
    TranscriptChunk,
    TranscriptSegment,
    VideoAnalysisArtifact,
)
from app.summarization.summarizer import SummarizationService
from app.utils.text_intelligence import extract_keywords, is_low_signal_topic


def test_llm_summary_normalizes_note_section_variants() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=False))
    artifact = VideoAnalysisArtifact(
        job_id="job-1",
        title="Constitution 101",
        source_type="youtube",
        source_url="https://youtu.be/example12345",
    )

    updated = service._apply_llm_summary(
        artifact,
        {
            "quick_summary": "A lesson on the Constitution.",
            "five_minute_summary": ["Overview of the lecture"],
            "key_topics": ["Constitution", "Word chunks"],
            "note_sections": [
                {
                    "title": "What are Word Chunks?",
                    "text": "The lecturer explains how chunks simplify long transcripts.",
                }
            ],
            "timestamps": [
                {
                    "title": "Introduction",
                    "text": "The lecture opens with key framing.",
                    "start_seconds": 12,
                }
            ],
            "chapters": [
                {
                    "title": "Framing the Topic",
                    "summary": "The lecture introduces the governing question.",
                    "start_seconds": 12,
                    "end_seconds": 48,
                    "keywords": ["Constitution"],
                    "confidence": 0.88,
                }
            ],
            "action_items": [{"task": "Review notes", "description": "Go over the main points."}],
            "key_quotes": [{"text": "Words form patterns.", "timestamp_seconds": 24}],
            "learning_objectives": ["Explain the lecture framing."],
            "glossary": [
                {
                    "term": "Constitution",
                    "definition": "The governing framework under discussion.",
                    "evidence": "The lecture introduces the Constitution.",
                }
            ],
            "study_questions": [
                {
                    "question": "What is the lecture about?",
                    "answer": "It introduces the Constitution.",
                    "difficulty": "introductory",
                }
            ],
            "analysis_metrics": {"transcript_word_count": 120, "sentence_count": 8},
            "mind_map": {"title": "Lecture", "nodes": [{"title": "Definitions"}]},
        },
    )

    assert updated.note_sections[0].heading == "What are Word Chunks?"
    assert updated.note_sections[0].detail.startswith("The lecturer explains")
    assert updated.chapters[0].title == "Framing the Topic"
    assert updated.timestamps[0].label
    assert updated.timestamps[0].description.startswith("The lecture opens")
    assert updated.action_items[0].title == "Review notes"
    assert updated.key_quotes[0].quote == "Words form patterns."
    assert updated.learning_objectives == ["Explain the lecture framing."]
    assert updated.glossary[0].term == "Constitution"
    assert updated.study_questions[0].question == "What is the lecture about?"
    assert updated.analysis_metrics.transcript_word_count == 120
    assert updated.mind_map is not None
    assert updated.mind_map.label == "Lecture"


def test_heuristic_summary_uses_transcript_content() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-2",
        title="Product Review",
        source_type="upload",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=30,
                text=(
                    "We reviewed the onboarding drop-off and agreed the first fix should be a simpler "
                    "account creation flow for new users."
                ),
            ),
            TranscriptSegment(
                start_seconds=30,
                end_seconds=64,
                text=(
                    "Next we need to send the revised screens to design today and confirm engineering "
                    "ownership tomorrow morning."
                ),
            ),
        ],
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=64,
            semantic_focus="Onboarding drop-off review",
            text=(
                "We reviewed the onboarding drop-off and agreed the first fix should be a simpler "
                "account creation flow for new users. Next we need to send the revised screens "
                "to design today and confirm engineering ownership tomorrow morning."
            ),
        )
    ]

    updated = service.summarize(artifact, chunks)

    assert "onboarding" in updated.quick_summary.lower()
    assert updated.chapters
    assert updated.note_sections
    assert "Onboarding" in updated.note_sections[0].heading
    assert updated.timestamps
    assert updated.action_items
    assert updated.learning_objectives
    assert updated.glossary
    assert updated.study_questions
    assert updated.analysis_metrics.chapter_count >= 1
    assert "send the revised screens" in updated.action_items[0].detail.lower()


def test_heuristic_summary_filters_low_signal_topic_fragments() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-3",
        title="Remote Development Workflow",
        source_type="upload",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=40,
                text=(
                    "Hello hello testing. We are going to take this in three parts. "
                    "Remote SSH connections in Visual Studio Code let you open a development "
                    "environment on another machine without copying files around."
                ),
            ),
            TranscriptSegment(
                start_seconds=40,
                end_seconds=82,
                text=(
                    "The next section explains dev containers. Dev containers standardize runtime "
                    "dependencies, editor extensions, and terminal tools so each teammate starts "
                    "from the same environment."
                ),
            ),
            TranscriptSegment(
                start_seconds=82,
                end_seconds=126,
                text=(
                    "Finally, pull requests keep collaboration organized by attaching review comments "
                    "and decisions to each change before the main branch is updated."
                ),
            ),
        ],
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=40,
            semantic_focus="Hello Hello Testing",
            text=artifact.transcript_segments[0].text,
        ),
        TranscriptChunk(
            start_seconds=40,
            end_seconds=82,
            semantic_focus="Going Take",
            text=artifact.transcript_segments[1].text,
        ),
        TranscriptChunk(
            start_seconds=82,
            end_seconds=126,
            semantic_focus="Three Parts",
            text=artifact.transcript_segments[2].text,
        ),
    ]

    updated = service.summarize(artifact, chunks)

    assert updated.key_topics
    assert any("Remote" in topic or "Containers" in topic or "Pull Request" in topic for topic in updated.key_topics)
    assert all(not is_low_signal_topic(topic) for topic in updated.key_topics)
    assert updated.chapters
    assert all(not is_low_signal_topic(chapter.title) for chapter in updated.chapters)
    assert updated.timestamps
    assert all(not is_low_signal_topic(timestamp.label) for timestamp in updated.timestamps)
    assert updated.mind_map is not None
    assert updated.mind_map.children
    assert all(not is_low_signal_topic(node.label) for node in updated.mind_map.children)
    assert all(
        not is_low_signal_topic(child.label)
        for node in updated.mind_map.children
        for child in node.children
    )


def test_heuristic_summary_synthesizes_study_actions_from_chapters() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-4",
        title="Research Methods Lecture",
        source_type="upload",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=36,
                text=(
                    "The lecture begins by defining causal inference and explaining why observational "
                    "data makes causal claims difficult."
                ),
            ),
            TranscriptSegment(
                start_seconds=36,
                end_seconds=74,
                text=(
                    "The second section compares randomized experiments with observational studies and "
                    "highlights how confounding variables can distort interpretation."
                ),
            ),
        ],
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=36,
            semantic_focus="Causal Inference",
            text=artifact.transcript_segments[0].text,
        ),
        TranscriptChunk(
            start_seconds=36,
            end_seconds=74,
            semantic_focus="Confounding Variables",
            text=artifact.transcript_segments[1].text,
        ),
    ]

    updated = service.summarize(artifact, chunks)

    assert updated.action_items
    assert updated.action_items[0].title.startswith(("Review ", "Explain ", "Connect "))
    assert "main claim" in updated.action_items[0].detail.lower() or "short explanation" in updated.action_items[0].detail.lower()


def test_heuristic_summary_uses_source_chapters_when_available(tmp_path: Path) -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    media_dir = tmp_path / "source"
    media_dir.mkdir(parents=True)
    media_path = media_dir / "workflow.mp4"
    media_path.write_bytes(b"placeholder")
    (media_dir / "info.json").write_text(
        json.dumps(
            {
                "chapters": [
                    {"start_time": 0, "end_time": 44, "title": "<Untitled Chapter 1>"},
                    {
                        "start_time": 44,
                        "end_time": 90,
                        "title": "Pull request review workflow",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = VideoAnalysisArtifact(
        job_id="job-5",
        title="Remote Development Workflow",
        source_type="youtube",
        source_url="https://youtu.be/example12345",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=44,
                text=(
                    "Remote SSH connections in Visual Studio Code let you work on another "
                    "machine while keeping the editor, terminal, and file operations in sync."
                ),
            ),
            TranscriptSegment(
                start_seconds=44,
                end_seconds=90,
                text=(
                    "The pull request review workflow keeps comments, approvals, and changes "
                    "attached to each update before the main branch is merged."
                ),
            ),
        ],
        metadata={"local_media_path": str(media_path)},
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=44,
            semantic_focus="Remote SSH connections",
            text=artifact.transcript_segments[0].text,
        ),
        TranscriptChunk(
            start_seconds=44,
            end_seconds=90,
            semantic_focus="Pull request review workflow",
            text=artifact.transcript_segments[1].text,
        ),
    ]

    updated = service.summarize(artifact, chunks)

    assert len(updated.chapters) >= 2
    assert "Untitled" not in updated.chapters[0].title
    assert "Remote SSH" in updated.chapters[0].title
    assert "Pull Request Review" in updated.chapters[1].title
    assert len(updated.timestamps) == len(updated.chapters)
    assert [item.label for item in updated.timestamps] == [
        chapter.title for chapter in updated.chapters
    ]


def test_heuristic_summary_keeps_full_source_chapter_list(tmp_path: Path) -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    media_dir = tmp_path / "source-many"
    media_dir.mkdir(parents=True)
    media_path = media_dir / "lesson.mp4"
    media_path.write_bytes(b"placeholder")

    chapter_markers = []
    transcript_segments: list[TranscriptSegment] = []
    chunks: list[TranscriptChunk] = []
    for index in range(12):
        start_seconds = float(index * 30)
        end_seconds = float((index + 1) * 30)
        chapter_markers.append(
            {
                "start_time": start_seconds,
                "end_time": end_seconds,
                "title": f"Module {index + 1}: Swift Functions {index + 1}",
            }
        )
        segment_text = (
            f"Module {index + 1} explains Swift functions, parameter handling, and return values "
            f"through example {index + 1}."
        )
        transcript_segments.append(
            TranscriptSegment(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                text=segment_text,
            )
        )
        chunks.append(
            TranscriptChunk(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                semantic_focus=f"Swift Functions {index + 1}",
                text=segment_text,
            )
        )

    (media_dir / "info.json").write_text(
        json.dumps({"chapters": chapter_markers}),
        encoding="utf-8",
    )

    artifact = VideoAnalysisArtifact(
        job_id="job-5b",
        title="Swift Functions Deep Dive",
        source_type="youtube",
        source_url="https://youtu.be/example12345",
        transcript_segments=transcript_segments,
        metadata={"local_media_path": str(media_path)},
    )

    updated = service.summarize(artifact, chunks)

    assert len(updated.chapters) == 12


def test_heuristic_summary_infers_lecture_topic_map_root() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-6",
        title="Lecture 1: Research Methods",
        source_type="youtube",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=42,
                text=(
                    "In this lecture we define causal inference and explain why observational "
                    "data creates limits for strong claims."
                ),
            ),
            TranscriptSegment(
                start_seconds=42,
                end_seconds=88,
                text=(
                    "The next section compares randomized experiments with observational studies "
                    "and shows how confounding variables affect interpretation."
                ),
            ),
        ],
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=42,
            semantic_focus="Causal inference",
            text=artifact.transcript_segments[0].text,
        ),
        TranscriptChunk(
            start_seconds=42,
            end_seconds=88,
            semantic_focus="Randomized experiments",
            text=artifact.transcript_segments[1].text,
        ),
    ]

    updated = service.summarize(artifact, chunks)

    assert updated.mind_map is not None
    assert updated.mind_map.label == "Lecture"
    assert updated.mind_map.children
    assert all(not is_low_signal_topic(node.label) for node in updated.mind_map.children)
    assert any(
        len(child.label.split()) >= 2
        for node in updated.mind_map.children
        for child in node.children
    )


def test_merge_with_fallback_replaces_low_quality_mind_map() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-7",
        title="Planning Session",
        source_type="upload",
        mind_map=MindMapNode(
            label="Topic Map",
            children=[
                MindMapNode(label="Know", children=[MindMapNode(label="Think")]),
                MindMapNode(label="That's", children=[MindMapNode(label="Anything")]),
            ],
        ),
    )
    fallback = VideoAnalysisArtifact(
        job_id="job-7",
        title="Planning Session",
        source_type="upload",
        mind_map=MindMapNode(
            label="Meeting",
            children=[
                MindMapNode(
                    label="Action Items",
                    children=[MindMapNode(label="Review onboarding flow")],
                ),
                MindMapNode(
                    label="Launch Risks",
                    children=[MindMapNode(label="Timeline dependencies")],
                ),
            ],
        ),
    )

    merged = service._merge_with_fallback(artifact, fallback)

    assert merged.mind_map is not None
    assert merged.mind_map.label == "Meeting"
    assert merged.mind_map.children[0].label == "Action Items"


def test_heuristic_summary_generates_meeting_style_brief() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-8",
        title="Weekly Product Meeting",
        source_type="upload",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=32,
                text=(
                    "In today's meeting we review the onboarding drop-off and agree that the first fix "
                    "should simplify account creation for new users."
                ),
            ),
            TranscriptSegment(
                start_seconds=32,
                end_seconds=68,
                text=(
                    "The team also discusses revised screens, engineering ownership, and the launch "
                    "timeline for the onboarding update."
                ),
            ),
            TranscriptSegment(
                start_seconds=68,
                end_seconds=102,
                text=(
                    "Next steps are to send the revised screens to design today and confirm engineering "
                    "ownership tomorrow morning."
                ),
            ),
        ],
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=32,
            semantic_focus="Onboarding drop-off",
            text=artifact.transcript_segments[0].text,
        ),
        TranscriptChunk(
            start_seconds=32,
            end_seconds=68,
            semantic_focus="Revised screens and ownership",
            text=artifact.transcript_segments[1].text,
        ),
        TranscriptChunk(
            start_seconds=68,
            end_seconds=102,
            semantic_focus="Next steps",
            text=artifact.transcript_segments[2].text,
        ),
    ]

    updated = service.summarize(artifact, chunks)

    assert "onboarding drop-off" in updated.quick_summary.lower()
    assert "revised screens" in updated.quick_summary.lower() or "engineering ownership" in updated.quick_summary.lower()
    assert updated.five_minute_summary
    assert any("Close with the follow-ups" in item for item in updated.five_minute_summary)
    assert updated.action_items
    assert any("send the revised screens" in item.detail.lower() for item in updated.action_items)


def test_keyword_extraction_and_summary_resist_filler_fragments() -> None:
    keywords = extract_keywords(
        (
            "kind of good lot don't didn't good kind lot. "
            "GitHub workflow automation adds branch protection, pull request approvals, "
            "and release checklists for the team."
        ),
        limit=6,
    )

    assert any("GitHub" in keyword for keyword in keywords)
    assert all(keyword not in {"Kind", "Good", "Lot", "Don't", "Didn't"} for keyword in keywords)

    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-9",
        title="Engineering Workflow Review",
        source_type="upload",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=46,
                text=(
                    "Kind of good lot don't. GitHub workflow automation adds branch protection and "
                    "pull request approvals before code is merged."
                ),
            ),
            TranscriptSegment(
                start_seconds=46,
                end_seconds=94,
                text=(
                    "The next section compares automated checks with manual review and explains how "
                    "release checklists keep deployments consistent."
                ),
            ),
        ],
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=46,
            semantic_focus="Kind And Good",
            text=artifact.transcript_segments[0].text,
        ),
        TranscriptChunk(
            start_seconds=46,
            end_seconds=94,
            semantic_focus="Don't Lot",
            text=artifact.transcript_segments[1].text,
        ),
    ]

    updated = service.summarize(artifact, chunks)

    assert updated.key_topics
    assert any("GitHub" in topic or "Pull Request" in topic for topic in updated.key_topics)
    assert updated.chapters
    assert all(not re.search(r"\b(kind|good|lot|don't|didn't)\b", chapter.title.lower()) for chapter in updated.chapters)
    assert all(
        not re.search(r"\b(kind|good|lot|don't|didn't)\b", point.lower())
        for point in updated.five_minute_summary
    )


def test_programming_tutorial_learning_objectives_stay_grounded() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-10b",
        title="(2020) Swift Tutorial for Beginners: Lesson 8 Functions (Part 2)",
        source_type="upload",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=55,
                text=(
                    "Here we write a function that adds two numbers and use the return keyword "
                    "to send back a value."
                ),
            ),
            TranscriptSegment(
                start_seconds=55,
                end_seconds=130,
                text=(
                    "Next we look at parameters and argument labels so the function call is "
                    "easier to read and understand."
                ),
            ),
            TranscriptSegment(
                start_seconds=130,
                end_seconds=210,
                text=(
                    "Then we compare a short call with a more descriptive version and explain "
                    "why clear labels help other people reading the code."
                ),
            ),
        ],
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=55,
            semantic_focus="Write Function",
            text=artifact.transcript_segments[0].text,
        ),
        TranscriptChunk(
            start_seconds=55,
            end_seconds=130,
            semantic_focus="Parameters And Argument Labels",
            text=artifact.transcript_segments[1].text,
        ),
        TranscriptChunk(
            start_seconds=130,
            end_seconds=210,
            semantic_focus="Descriptive Version",
            text=artifact.transcript_segments[2].text,
        ),
    ]

    updated = service.summarize(artifact, chunks)
    joined_objectives = " ".join(updated.learning_objectives).lower()
    joined_topics = " ".join(updated.key_topics).lower()
    joined_summary = " ".join(updated.five_minute_summary).lower()

    assert "swift functions" in joined_objectives
    assert "parameters" in joined_objectives
    assert "argument labels" in joined_objectives
    assert "return keyword" in joined_objectives
    assert "descriptive version" not in joined_objectives
    assert "other people reading" not in joined_objectives
    assert "descriptive version" not in joined_topics
    assert "help other people" not in joined_topics
    assert "understand" not in joined_topics
    assert "descriptive version" not in joined_summary
    assert any("Return Keyword" == topic for topic in updated.key_topics)
    assert any("Argument Labels" in topic for topic in updated.key_topics)


def test_outputs_avoid_repeating_the_same_concept_across_artifacts() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-10c",
        title="Weekly Product Meeting",
        source_type="upload",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=34,
                text=(
                    "The team reviews the onboarding drop-off and agrees the account creation flow "
                    "needs to be simpler for new users."
                ),
            ),
            TranscriptSegment(
                start_seconds=34,
                end_seconds=74,
                text=(
                    "Next we assign engineering ownership for the redesign and confirm design review "
                    "with the revised onboarding screens."
                ),
            ),
            TranscriptSegment(
                start_seconds=74,
                end_seconds=118,
                text=(
                    "The group then discusses launch timing, release risks, and dependency tracking "
                    "before rollout."
                ),
            ),
        ],
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=34,
            semantic_focus="Onboarding Drop-Off",
            text=artifact.transcript_segments[0].text,
        ),
        TranscriptChunk(
            start_seconds=34,
            end_seconds=74,
            semantic_focus="Engineering Ownership",
            text=artifact.transcript_segments[1].text,
        ),
        TranscriptChunk(
            start_seconds=74,
            end_seconds=118,
            semantic_focus="Launch Risks",
            text=artifact.transcript_segments[2].text,
        ),
    ]

    updated = service.summarize(artifact, chunks)

    assert len(updated.five_minute_summary) >= 3
    summary_signatures = {
        service._concept_signature(item)
        for item in updated.five_minute_summary[:4]
        if service._concept_signature(item)
    }
    assert len(summary_signatures) >= 3
    assert updated.timestamps
    timestamp_signatures = {
        service._concept_signature(item.label)
        for item in updated.timestamps[:4]
        if service._concept_signature(item.label)
    }
    assert len(timestamp_signatures) >= min(3, len(updated.timestamps[:4]))
    assert updated.action_items
    action_signatures = {
        service._concept_signature(item.title)
        for item in updated.action_items[:3]
        if service._concept_signature(item.title)
    }
    assert len(action_signatures) >= min(2, len(updated.action_items[:3]))
    assert updated.note_sections
    for section in updated.note_sections[:3]:
        assert all(
            not service._texts_are_redundant(left, right)
            for left, right in combinations(section.bullet_points, 2)
        )
    assert updated.mind_map is not None
    branch_labels = [branch.label for branch in updated.mind_map.children]
    assert all(
        not service._texts_are_redundant(left, right)
        for left, right in combinations(branch_labels, 2)
    )


def test_mind_map_avoids_repeated_leaf_concepts_when_topics_overlap() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    chapters = [
        ChapterItem(
            title="Pull Request Workflow",
            summary="Pull request workflow covers branch protection, review approvals, and merge conditions.",
            start_seconds=0,
            end_seconds=40,
            display_time="00:00",
            keywords=["Branch Protection", "Review Approvals", "Merge Conditions"],
        ),
        ChapterItem(
            title="Release Operations",
            summary="Release operations focus on deployment checklists, release notes, and rollback planning.",
            start_seconds=40,
            end_seconds=88,
            display_time="00:40",
            keywords=["Deployment Checklists", "Release Notes", "Rollback Planning"],
        ),
        ChapterItem(
            title="Team Ownership",
            summary="Team ownership clarifies design review, engineering ownership, and decision tracking.",
            start_seconds=88,
            end_seconds=132,
            display_time="01:28",
            keywords=["Design Review", "Engineering Ownership", "Decision Tracking"],
        ),
    ]

    mind_map = service._build_mind_map(
        title="Engineering Planning Meeting",
        transcript_text=(
            "The team reviews pull request workflow, release operations, and team ownership. "
            "Branch protection, deployment checklists, and engineering ownership are discussed as distinct topics."
        ),
        key_topics=[
            "Pull Request Workflow",
            "Release Operations",
            "Team Ownership",
            "Engineering Ownership",
        ],
        chapters=chapters,
        glossary=[],
        action_items=[],
    )

    leaf_labels = [child.label.lower() for branch in mind_map.children for child in branch.children]
    assert mind_map.children
    assert len(leaf_labels) == len(set(leaf_labels))


def test_build_quotes_prefers_diverse_chapter_contexts() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    chapters = [
        ChapterItem(
            title="Onboarding Review",
            summary="The team reviews the onboarding drop-off and account creation changes.",
            start_seconds=0,
            end_seconds=40,
            display_time="00:00",
            keywords=["Onboarding Drop-Off", "Account Creation"],
        ),
        ChapterItem(
            title="Ownership Decisions",
            summary="The team clarifies engineering ownership and design review expectations.",
            start_seconds=40,
            end_seconds=80,
            display_time="00:40",
            keywords=["Engineering Ownership", "Design Review"],
        ),
        ChapterItem(
            title="Launch Timeline",
            summary="The group discusses launch timing, release risks, and rollout sequencing.",
            start_seconds=80,
            end_seconds=120,
            display_time="01:20",
            keywords=["Launch Timeline", "Release Risks"],
        ),
    ]
    sentence_records = [
        {
            "text": "The onboarding drop-off matters because account creation is the first point of friction for new users.",
            "start_seconds": 4.0,
            "end_seconds": 18.0,
            "score": 3.4,
            "keywords": ["Onboarding Drop-Off", "Account Creation"],
        },
        {
            "text": "Engineering ownership has to be explicit, otherwise design review comments stall and no one closes the loop.",
            "start_seconds": 48.0,
            "end_seconds": 63.0,
            "score": 3.2,
            "keywords": ["Engineering Ownership", "Design Review"],
        },
        {
            "text": "The launch timeline is risky because release dependencies stack up quickly when rollout sequencing is unclear.",
            "start_seconds": 92.0,
            "end_seconds": 108.0,
            "score": 3.1,
            "keywords": ["Launch Timeline", "Release Dependencies"],
        },
    ]

    quotes = service._build_quotes(
        "https://youtu.be/example12345",
        sentence_records,
        chapters,
    )

    assert len(quotes) >= 3
    assert len({quote.context for quote in quotes[:3]}) >= 3
    assert all("because" in quote.quote.lower() or "otherwise" in quote.quote.lower() for quote in quotes[:3])


def test_lecture_outputs_use_format_aware_notes_timeline_and_actions() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-10",
        title="Lecture 3: Causal Inference",
        source_type="upload",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=35,
                text=(
                    "In this lecture we define causal inference and explain why observational "
                    "data limits strong claims."
                ),
            ),
            TranscriptSegment(
                start_seconds=35,
                end_seconds=72,
                text=(
                    "The next section compares randomized experiments with observational studies "
                    "and gives an example of confounding variables."
                ),
            ),
            TranscriptSegment(
                start_seconds=72,
                end_seconds=108,
                text=(
                    "Finally the lecture explains how to evaluate evidence and design better "
                    "research questions."
                ),
            ),
        ],
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=35,
            semantic_focus="Causal inference",
            text=artifact.transcript_segments[0].text,
        ),
        TranscriptChunk(
            start_seconds=35,
            end_seconds=72,
            semantic_focus="Randomized experiments",
            text=artifact.transcript_segments[1].text,
        ),
        TranscriptChunk(
            start_seconds=72,
            end_seconds=108,
            semantic_focus="Evaluating evidence",
            text=artifact.transcript_segments[2].text,
        ),
    ]

    updated = service.summarize(artifact, chunks)

    assert any(point.startswith("Foundation:") for point in updated.five_minute_summary)
    assert updated.note_sections
    assert any(
        bullet.startswith("Core concept:") for bullet in updated.note_sections[0].bullet_points
    )
    assert any(
        item.label.startswith("Comparison:") or item.description.startswith("Comparison:")
        for item in updated.timestamps
    )
    assert updated.action_items
    assert updated.action_items[0].title.startswith(("Review ", "Explain ", "Compare "))
    assert updated.mind_map is not None
    assert updated.mind_map.label == "Lecture"


def test_workshop_outputs_use_step_focused_summary_and_practice_actions() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-11",
        title="Docker Workshop",
        source_type="upload",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=40,
                text=(
                    "First we install Docker and configure the project files for the local environment."
                ),
            ),
            TranscriptSegment(
                start_seconds=40,
                end_seconds=80,
                text=(
                    "Next we build the container image, run the service, and verify the output in the browser."
                ),
            ),
            TranscriptSegment(
                start_seconds=80,
                end_seconds=120,
                text=(
                    "Then we troubleshoot the database connection and test the final setup step by step."
                ),
            ),
        ],
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=40,
            semantic_focus="Install Docker",
            text=artifact.transcript_segments[0].text,
        ),
        TranscriptChunk(
            start_seconds=40,
            end_seconds=80,
            semantic_focus="Build the container",
            text=artifact.transcript_segments[1].text,
        ),
        TranscriptChunk(
            start_seconds=80,
            end_seconds=120,
            semantic_focus="Troubleshoot setup",
            text=artifact.transcript_segments[2].text,
        ),
    ]

    updated = service.summarize(artifact, chunks)

    assert any(point.startswith("Step 1:") for point in updated.five_minute_summary)
    assert updated.note_sections
    assert any(
        bullet.startswith("Step focus:") for bullet in updated.note_sections[0].bullet_points
    )
    assert updated.action_items
    assert updated.action_items[0].title.startswith(("Practice ", "Apply ", "Check "))
    assert updated.mind_map is not None
    assert updated.mind_map.label == "Workshop"


def test_interview_outputs_use_question_response_style_actions() -> None:
    service = SummarizationService(Settings(_env_file=None, demo_mode=True))
    artifact = VideoAnalysisArtifact(
        job_id="job-12",
        title="Founder Interview Episode 12",
        source_type="upload",
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=34,
                text=(
                    "In this interview I asked the founder why the product roadmap changed after launch."
                ),
            ),
            TranscriptSegment(
                start_seconds=34,
                end_seconds=76,
                text=(
                    "She explained that customer onboarding data changed the team's priorities because "
                    "activation was weaker than expected."
                ),
            ),
            TranscriptSegment(
                start_seconds=76,
                end_seconds=118,
                text=(
                    "The conversation ends with a practical takeaway on how to balance roadmap bets with "
                    "clearer evidence."
                ),
            ),
        ],
    )
    chunks = [
        TranscriptChunk(
            start_seconds=0,
            end_seconds=34,
            semantic_focus="Roadmap questions",
            text=artifact.transcript_segments[0].text,
        ),
        TranscriptChunk(
            start_seconds=34,
            end_seconds=76,
            semantic_focus="Customer onboarding data",
            text=artifact.transcript_segments[1].text,
        ),
        TranscriptChunk(
            start_seconds=76,
            end_seconds=118,
            semantic_focus="Practical takeaway",
            text=artifact.transcript_segments[2].text,
        ),
    ]

    updated = service.summarize(artifact, chunks)

    assert updated.mind_map is not None
    assert updated.mind_map.label == "Interview"
    assert updated.action_items
    assert updated.action_items[0].title.startswith(("Capture ", "Reflect On ", "Connect "))
    assert any(
        item.label.startswith("Question:") or item.description.startswith("Response:")
        for item in updated.timestamps
    )
