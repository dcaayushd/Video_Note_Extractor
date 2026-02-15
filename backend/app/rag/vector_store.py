from __future__ import annotations

from collections import Counter
import re
import threading

from app.core.config import Settings
from app.models import ChatCitation, TranscriptChunk
from app.schemas.video import ChatResponse
from app.services.repository import ArtifactRepository
from app.utils.text_intelligence import (
    extract_keywords,
    normalize_whitespace,
    split_sentences,
    tokenize,
    truncate_text,
)
from app.utils.timecode import build_youtube_jump_url, format_timestamp


class TranscriptVectorStore:
    def __init__(self, settings: Settings, repository: ArtifactRepository) -> None:
        self.settings = settings
        self.repository = repository
        self._embeddings = None
        self._embeddings_lock = threading.Lock()

    def index(self, job_id: str, chunks: list[TranscriptChunk]) -> None:
        if self.settings.demo_mode or not chunks:
            return

        try:
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError("Install Chroma and embedding dependencies for RAG.") from exc

        persist_path = self.settings.chroma_persist_dir / job_id
        persist_path.mkdir(parents=True, exist_ok=True)
        if (persist_path / "chroma.sqlite3").exists():
            return
        embeddings = self._get_embeddings(HuggingFaceEmbeddings)
        store = Chroma(
            collection_name=f"video-note-extractor-{job_id}",
            persist_directory=str(persist_path),
            embedding_function=embeddings,
        )
        documents = [
            Document(
                page_content=self._document_text(chunk),
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "start_seconds": chunk.start_seconds,
                    "end_seconds": chunk.end_seconds,
                    "semantic_focus": chunk.semantic_focus,
                    "keywords": ",".join(extract_keywords(chunk.text, limit=5)),
                },
            )
            for chunk in chunks
        ]
        store.add_documents(documents)

    def ask(self, job_id: str, question: str) -> ChatResponse:
        artifact = self.repository.load_artifact(job_id)
        chunks = artifact.transcript_chunks
        if not chunks:
            return ChatResponse(
                job_id=job_id,
                answer="I could not find grounded transcript content for that question yet.",
                citations=[],
            )

        vector_scores = self._vector_retrieval(job_id, question)
        scored_chunks = self._hybrid_rank(chunks, question, vector_scores)
        best_chunks = [chunk for _, chunk in scored_chunks[:3]]

        if not self.settings.use_local_llm:
            answer = self._build_transcript_answer(best_chunks, question)
        else:
            answer = self._build_llm_answer(best_chunks, artifact.title, question)

        citations = [
            ChatCitation(
                label=chunk.semantic_focus,
                start_seconds=chunk.start_seconds,
                display_time=format_timestamp(chunk.start_seconds),
                jump_url=build_youtube_jump_url(artifact.source_url, chunk.start_seconds),
            )
            for chunk in best_chunks
        ]
        return ChatResponse(job_id=job_id, answer=answer, citations=citations)

    def _document_text(self, chunk: TranscriptChunk) -> str:
        keywords = ", ".join(extract_keywords(chunk.text, limit=5))
        return (
            f"Focus: {chunk.semantic_focus}\n"
            f"Keywords: {keywords}\n"
            f"Transcript: {chunk.text}"
        )

    def _vector_retrieval(self, job_id: str, question: str) -> dict[str, float]:
        if self.settings.demo_mode:
            return {}

        persist_path = self.settings.chroma_persist_dir / job_id
        if not persist_path.exists():
            return {}

        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            return {}

        try:
            embeddings = self._get_embeddings(HuggingFaceEmbeddings)
            store = Chroma(
                collection_name=f"video-note-extractor-{job_id}",
                persist_directory=str(persist_path),
                embedding_function=embeddings,
            )
            results = store.similarity_search_with_relevance_scores(
                question,
                k=self.settings.retrieval_k,
            )
        except Exception:
            return {}

        scores: dict[str, float] = {}
        for document, score in results:
            chunk_id = str(document.metadata.get("chunk_id") or "")
            if not chunk_id:
                continue
            scores[chunk_id] = float(score)
        return scores

    def _hybrid_rank(
        self,
        chunks: list[TranscriptChunk],
        question: str,
        vector_scores: dict[str, float],
    ) -> list[tuple[float, TranscriptChunk]]:
        query_tokens = [token for token in tokenize(question) if len(token) > 2]
        query_keywords = [keyword.lower() for keyword in extract_keywords(question, limit=5)]
        ranked: list[tuple[float, TranscriptChunk]] = []

        for chunk in chunks:
            lexical_score = self._keyword_score(chunk, query_tokens, query_keywords)
            vector_score = vector_scores.get(chunk.chunk_id, 0.0)
            focus_bonus = 0.0
            focus_text = normalize_whitespace(chunk.semantic_focus).lower()
            if any(keyword in focus_text for keyword in query_keywords):
                focus_bonus += 0.8
            score = (lexical_score * 0.65) + (vector_score * 0.3) + focus_bonus
            ranked.append((score, chunk))

        ranked.sort(
            key=lambda item: (item[0], item[1].start_seconds),
            reverse=True,
        )
        if ranked and ranked[0][0] > 0:
            return ranked

        return [(1.0 / (index + 1), chunk) for index, chunk in enumerate(chunks[:3])]

    def _keyword_score(
        self,
        chunk: TranscriptChunk,
        query_tokens: list[str],
        query_keywords: list[str],
    ) -> float:
        text = f"{chunk.semantic_focus} {chunk.text}".lower()
        if not query_tokens:
            return 0.0

        token_counts = Counter(tokenize(text))
        score = 0.0
        for token in query_tokens:
            score += token_counts.get(token, 0) * 0.45
            if token in chunk.semantic_focus.lower():
                score += 0.6
        for keyword in query_keywords:
            if keyword in text:
                score += 0.85
        return round(score, 4)

    def _build_transcript_answer(self, chunks: list[TranscriptChunk], question: str) -> str:
        if not chunks:
            return "I could not find grounded content for that question yet."

        points: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            sentences = split_sentences(chunk.text, minimum_words=6)
            candidate = sentences[0] if sentences else chunk.text
            compact = truncate_text(candidate, 180)
            key = compact.lower()
            if not compact or key in seen:
                continue
            seen.add(key)
            points.append(compact)
            if len(points) == 3:
                break

        if not points:
            points = [truncate_text(" ".join(chunk.text for chunk in chunks), 280)]

        if len(points) == 1:
            return points[0]
        return "\n".join(f"- {point}" for point in points)

    def _build_llm_answer(
        self,
        chunks: list[TranscriptChunk],
        title: str,
        question: str,
    ) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError("Install langchain-ollama to enable local chat answers.") from exc

        llm = ChatOllama(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
            temperature=0.1,
        )
        context = "\n\n".join(
            f"[{format_timestamp(chunk.start_seconds)}] {chunk.semantic_focus}\n{chunk.text}"
            for chunk in chunks
        )
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Answer the question using only the supplied transcript evidence. "
                            "Stay concise, do not invent details, and do not include timestamps, "
                            "timecodes, citation blocks, or section headers. Prefer 2-4 short bullets "
                            "or a short paragraph."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Video title: {title}\n"
                            f"Question: {question}\n"
                            f"Grounding context:\n{context}"
                        )
                    ),
                ]
            )
        except Exception:
            return self._build_transcript_answer(chunks, question)
        return self._clean_answer_text(self._stringify_response(response.content))

    def _get_embeddings(self, embeddings_cls):
        if self._embeddings is not None:
            return self._embeddings
        with self._embeddings_lock:
            if self._embeddings is None:
                self._embeddings = embeddings_cls(model_name=self.settings.embedding_model)
        return self._embeddings

    def _stringify_response(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(item) for item in content)
        return str(content)

    def _clean_answer_text(self, answer: str) -> str:
        cleaned_lines: list[str] = []
        seen: set[str] = set()
        for raw_line in answer.splitlines():
            line = re.sub(r"\b(?:\d{1,2}:)?\d{1,2}:\d{2}\b", "", raw_line)
            line = re.sub(r"^(answer|timeline|topics)\s*:?\s*", "", line.strip(), flags=re.IGNORECASE)
            line = normalize_whitespace(line.strip(" -*\t"))
            if not line:
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned_lines.append(line)

        if not cleaned_lines:
            return "I could not find grounded content for that question yet."
        if len(cleaned_lines) == 1:
            return cleaned_lines[0]
        return "\n".join(f"- {line}" for line in cleaned_lines[:4])
