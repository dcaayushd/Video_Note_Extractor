from app.models import (
    ActionItem,
    AnalysisMetrics,
    MindMapNode,
    NoteSection,
    QuoteItem,
    TimestampItem,
    VideoAnalysisArtifact,
)
from app.schemas.video import ExportFormat
from app.services.export_service import ExportService


def test_markdown_export_includes_brief_quotes_and_topic_map() -> None:
    artifact = VideoAnalysisArtifact(
        job_id="job-export",
        title="Research Methods Lecture",
        source_type="youtube",
        quick_summary="This lecture focuses on causal inference and study design.",
        five_minute_summary=[
            "Begin with causal inference and why observational data limits strong claims.",
            "Then explain randomized experiments and confounding variables.",
        ],
        key_topics=["Causal Inference", "Randomized Experiments"],
        note_sections=[
            NoteSection(
                heading="Causal Inference",
                bullet_points=["Focus: study design and evidence."],
                detail="This section develops causal inference as a central idea in the discussion.",
                start_seconds=0,
                display_time="00:00",
            )
        ],
        timestamps=[
            TimestampItem(
                label="Causal Inference",
                description="This moment frames the core research question.",
                start_seconds=0,
                end_seconds=24,
                display_time="00:00",
            )
        ],
        chapters=[],
        action_items=[
            ActionItem(
                title="Explain Causal Inference",
                detail="Write a short explanation of causal inference using one transcript example.",
                start_seconds=0,
                display_time="00:00",
            )
        ],
        key_quotes=[
            QuoteItem(
                quote="Observational data makes causal claims difficult.",
                context="Causal Inference",
                start_seconds=6,
                display_time="00:06",
            )
        ],
        analysis_metrics=AnalysisMetrics(transcript_word_count=120, chapter_count=1),
        mind_map=MindMapNode(
            label="Lecture",
            children=[
                MindMapNode(
                    label="Causal Inference",
                    children=[MindMapNode(label="Observational Data")],
                )
            ],
        ),
    )

    service = ExportService()
    content, media_type, filename = service.render(artifact, ExportFormat.markdown)
    rendered = content.decode("utf-8")

    assert media_type == "text/markdown"
    assert filename.endswith(".md")
    assert "## Executive Brief" in rendered
    assert "## Timeline" in rendered
    assert "## Key Quotes" in rendered
    assert "## Topic Map" in rendered
    assert "Observational data makes causal claims difficult." in rendered
