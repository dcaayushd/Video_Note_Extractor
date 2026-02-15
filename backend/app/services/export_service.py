from __future__ import annotations

from io import BytesIO

from app.models import VideoAnalysisArtifact
from app.schemas.video import ExportFormat


class ExportService:
    def render(self, artifact: VideoAnalysisArtifact, export_format: ExportFormat) -> tuple[bytes, str, str]:
        if export_format == ExportFormat.markdown:
            content = self._to_markdown(artifact).encode("utf-8")
            return content, "text/markdown", f"{artifact.job_id}.md"
        if export_format == ExportFormat.notion:
            content = self._to_notion(artifact).encode("utf-8")
            return content, "text/plain", f"{artifact.job_id}-notion.txt"
        return self._to_pdf(artifact), "application/pdf", f"{artifact.job_id}.pdf"

    def _to_markdown(self, artifact: VideoAnalysisArtifact) -> str:
        lines = [
            f"# {artifact.title}",
            "",
            "## Quick Summary",
            artifact.quick_summary,
            "",
            "## Executive Brief",
        ]
        lines.extend(f"- {item}" for item in artifact.five_minute_summary)
        lines.extend(["", "## Key Topics"])
        lines.extend(f"- {item}" for item in artifact.key_topics)
        lines.extend(["", "## Learning Objectives"])
        lines.extend(f"- {item}" for item in artifact.learning_objectives)
        lines.extend(["", "## Chapters"])
        lines.extend(
            f"- {item.display_time} {item.title}: {item.summary}"
            for item in artifact.chapters
        )
        lines.extend(["", "## Timeline"])
        lines.extend(
            f"- {item.display_time} {item.label}: {item.description}" for item in artifact.timestamps
        )
        lines.extend(["", "## Notes"])
        for section in artifact.note_sections:
            lines.append(f"### {section.heading}")
            lines.extend(f"- {bullet}" for bullet in section.bullet_points)
            lines.append(section.detail)
            lines.append("")
        lines.extend(["## Action Items"])
        lines.extend(
            f"- {item.title}: {item.detail} ({item.owner_hint or 'Unassigned'})"
            for item in artifact.action_items
        )
        lines.extend(["", "## Key Quotes"])
        lines.extend(
            f"- {item.display_time} {item.context}: \"{item.quote}\""
            for item in artifact.key_quotes
        )
        lines.extend(["## Glossary"])
        lines.extend(
            f"- {item.term}: {item.definition}"
            for item in artifact.glossary
        )
        lines.extend(["", "## Study Questions"])
        lines.extend(
            f"- {item.question} Answer: {item.answer}"
            for item in artifact.study_questions
        )
        if artifact.mind_map is not None:
            lines.extend(["", "## Topic Map"])
            lines.extend(self._mind_map_lines(artifact.mind_map))
        lines.extend(
            [
                "",
                "## Analysis Metrics",
                f"- Transcript words: {artifact.analysis_metrics.transcript_word_count}",
                f"- Sentences: {artifact.analysis_metrics.sentence_count}",
                f"- Chapters: {artifact.analysis_metrics.chapter_count}",
                f"- Lexical diversity: {artifact.analysis_metrics.lexical_diversity}",
                f"- Academic signal score: {artifact.analysis_metrics.academic_signal_score}",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def _to_notion(self, artifact: VideoAnalysisArtifact) -> str:
        lines = [
            f"{artifact.title}",
            "=" * len(artifact.title),
            "",
            "Quick Summary",
            artifact.quick_summary,
            "",
            "Executive Brief",
        ]
        lines.extend(f"* {item}" for item in artifact.five_minute_summary)
        lines.extend(["", "Key Topics"])
        lines.extend(f"* {item}" for item in artifact.key_topics)
        lines.extend(["", "Learning Objectives"])
        lines.extend(f"* {item}" for item in artifact.learning_objectives)
        lines.extend(["", "Chapter Timeline"])
        lines.extend(
            f"* {item.display_time} | {item.title} | {item.summary}"
            for item in artifact.chapters
        )
        lines.extend(["", "Expanded Timeline"])
        lines.extend(
            f"* {item.display_time} | {item.label} | {item.description}"
            for item in artifact.timestamps
        )
        lines.extend(["", "Action Database"])
        lines.extend(
            f"[ ] {item.title} | Owner: {item.owner_hint or 'TBD'} | When: {item.due_hint or 'TBD'} | Detail: {item.detail}"
            for item in artifact.action_items
        )
        lines.extend(["", "Knowledge Blocks"])
        lines.extend(f"* {section.heading}: {section.detail}" for section in artifact.note_sections)
        lines.extend(["", "Key Quotes"])
        lines.extend(
            f"* {item.display_time} | {item.context} | \"{item.quote}\""
            for item in artifact.key_quotes
        )
        lines.extend(["", "Glossary"])
        lines.extend(f"* {item.term}: {item.definition}" for item in artifact.glossary)
        lines.extend(["", "Study Questions"])
        lines.extend(f"* {item.question} -> {item.answer}" for item in artifact.study_questions)
        if artifact.mind_map is not None:
            lines.extend(["", "Topic Map"])
            lines.extend(self._mind_map_lines(artifact.mind_map, bullet="*"))
        return "\n".join(lines)

    def _to_pdf(self, artifact: VideoAnalysisArtifact) -> bytes:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        except ImportError as exc:
            raise RuntimeError("Install reportlab to enable PDF exports.") from exc

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(artifact.title, styles["Title"]),
            Spacer(1, 12),
            Paragraph(artifact.quick_summary, styles["BodyText"]),
            Spacer(1, 12),
        ]
        if artifact.five_minute_summary:
            story.append(Paragraph("Executive Brief", styles["Heading2"]))
            for item in artifact.five_minute_summary:
                story.append(Paragraph(f"- {item}", styles["BodyText"]))
            story.append(Spacer(1, 8))
        if artifact.learning_objectives:
            story.append(Paragraph("Learning Objectives", styles["Heading2"]))
            for objective in artifact.learning_objectives:
                story.append(Paragraph(f"- {objective}", styles["BodyText"]))
            story.append(Spacer(1, 8))
        if artifact.chapters:
            story.append(Paragraph("Chapter Timeline", styles["Heading2"]))
            for chapter in artifact.chapters:
                story.append(
                    Paragraph(
                        f"{chapter.display_time} {chapter.title}: {chapter.summary}",
                        styles["BodyText"],
                    )
                )
            story.append(Spacer(1, 8))
        if artifact.timestamps:
            story.append(Paragraph("Timeline", styles["Heading2"]))
            for item in artifact.timestamps:
                story.append(
                    Paragraph(
                        f"{item.display_time} {item.label}: {item.description}",
                        styles["BodyText"],
                    )
                )
            story.append(Spacer(1, 8))
        for section in artifact.note_sections:
            story.append(Paragraph(section.heading, styles["Heading2"]))
            for bullet in section.bullet_points:
                story.append(Paragraph(f"- {bullet}", styles["BodyText"]))
            story.append(Paragraph(section.detail, styles["BodyText"]))
            story.append(Spacer(1, 8))
        if artifact.action_items:
            story.append(Paragraph("Action Items", styles["Heading2"]))
            for item in artifact.action_items:
                story.append(
                    Paragraph(
                        f"- {item.title}: {item.detail}",
                        styles["BodyText"],
                    )
                )
            story.append(Spacer(1, 8))
        if artifact.key_quotes:
            story.append(Paragraph("Key Quotes", styles["Heading2"]))
            for item in artifact.key_quotes:
                story.append(
                    Paragraph(
                        f"{item.display_time} {item.context}: \"{item.quote}\"",
                        styles["BodyText"],
                    )
                )
            story.append(Spacer(1, 8))
        if artifact.glossary:
            story.append(Paragraph("Glossary", styles["Heading2"]))
            for item in artifact.glossary:
                story.append(Paragraph(f"{item.term}: {item.definition}", styles["BodyText"]))
            story.append(Spacer(1, 8))
        if artifact.study_questions:
            story.append(Paragraph("Study Questions", styles["Heading2"]))
            for item in artifact.study_questions:
                story.append(Paragraph(f"{item.question} {item.answer}", styles["BodyText"]))
            story.append(Spacer(1, 8))
        if artifact.mind_map is not None:
            story.append(Paragraph("Topic Map", styles["Heading2"]))
            for line in self._mind_map_lines(artifact.mind_map):
                story.append(Paragraph(line, styles["BodyText"]))
            story.append(Spacer(1, 8))
        doc.build(story)
        return buffer.getvalue()

    def _mind_map_lines(
        self,
        node,
        *,
        depth: int = 0,
        bullet: str = "-",
    ) -> list[str]:
        prefix = "  " * depth
        lines = [f"{prefix}{bullet} {node.label}"]
        for child in node.children:
            lines.extend(self._mind_map_lines(child, depth=depth + 1, bullet=bullet))
        return lines
