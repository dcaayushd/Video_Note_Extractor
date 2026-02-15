from __future__ import annotations

import re
from collections import Counter
from typing import TypeVar

T = TypeVar("T")

_TIME_PATTERN = re.compile(r"\b(?:\d{1,2}:)?\d{1,2}:\d{2}\b")
_ALPHA_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'\-]+")

_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "another",
    "anything",
    "also",
    "an",
    "and",
    "any",
    "are",
    "around",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "bit",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cant",
    "could",
    "didn't",
    "didnt",
    "do",
    "does",
    "doesn't",
    "doesnt",
    "each",
    "even",
    "finally",
    "for",
    "found",
    "from",
    "get",
    "gonna",
    "had",
    "has",
    "have",
    "how",
    "here",
    "i",
    "i'm",
    "i've",
    "i'll",
    "id",
    "ill",
    "im",
    "in",
    "is",
    "it",
    "it's",
    "ive",
    "into",
    "if",
    "its",
    "just",
    "know",
    "like",
    "little",
    "let",
    "let's",
    "lets",
    "lot",
    "lots",
    "make",
    "maybe",
    "me",
    "more",
    "most",
    "need",
    "next",
    "not",
    "now",
    "of",
    "okay",
    "on",
    "other",
    "or",
    "our",
    "out",
    "over",
    "pretty",
    "really",
    "right",
    "same",
    "should",
    "some",
    "someone",
    "somebody",
    "so",
    "something",
    "still",
    "stuff",
    "such",
    "than",
    "that",
    "that's",
    "thats",
    "the",
    "their",
    "them",
    "then",
    "there",
    "there's",
    "theres",
    "these",
    "they",
    "think",
    "thing",
    "things",
    "this",
    "to",
    "through",
    "today",
    "uh",
    "um",
    "don't",
    "dont",
    "very",
    "want",
    "was",
    "wasn't",
    "wasnt",
    "way",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "well",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "won't",
    "wont",
    "without",
    "with",
    "would",
    "yeah",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
}

_NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "first",
    "second",
    "third",
}

_GENERIC_TOPIC_WORDS = {
    "beginner",
    "beginners",
    "chapter",
    "chapters",
    "clip",
    "example",
    "examples",
    "hello",
    "intro",
    "introduction",
    "lesson",
    "lessons",
    "part",
    "parts",
    "point",
    "points",
    "section",
    "sections",
    "segment",
    "segments",
    "test",
    "testing",
    "interview",
    "lecture",
    "meeting",
    "podcast",
    "topic",
    "topics",
    "talk",
    "tutorial",
    "tutorials",
    "video",
    "videos",
    "workshop",
}

_LOW_SIGNAL_LEADERS = {
    "going",
    "hello",
    "look",
    "only",
    "small",
    "take",
    "testing",
}

_LOW_SIGNAL_SINGLE_WORDS = {
    "anything",
    "brand",
    "dev",
    "didn't",
    "didnt",
    "easier",
    "don't",
    "dont",
    "drive",
    "files",
    "found",
    "focus",
    "focuses",
    "good",
    "going",
    "guide",
    "help",
    "helps",
    "hello",
    "home",
    "kind",
    "know",
    "lot",
    "lots",
    "new",
    "only",
    "part",
    "parts",
    "point",
    "points",
    "pretty",
    "quick",
    "right",
    "section",
    "sections",
    "specify",
    "understand",
    "understanding",
    "testing",
    "think",
    "there's",
    "theres",
    "machine",
}

_LOW_SIGNAL_FILLER_WORDS = {
    "actually",
    "basically",
    "didn't",
    "didnt",
    "don't",
    "dont",
    "good",
    "great",
    "kind",
    "kinda",
    "kindof",
    "lot",
    "lots",
    "maybe",
    "nice",
    "okay",
    "pretty",
    "quite",
    "really",
    "right",
    "sort",
    "still",
    "stuff",
    "thing",
    "things",
    "totally",
    "very",
}

_WEAK_TRAILING_TOPIC_WORDS = {
    "available",
    "better",
    "clear",
    "clearer",
    "consistent",
    "different",
    "easier",
    "good",
    "great",
    "help",
    "here",
    "important",
    "possible",
    "right",
    "simple",
    "simpler",
    "still",
}

_ACTION_LEADER_WORDS = {
    "assign",
    "confirm",
    "plan",
    "review",
    "schedule",
    "send",
    "share",
}

_TEMPORAL_TOPIC_WORDS = {
    "afternoon",
    "later",
    "meeting",
    "morning",
    "next",
    "today",
    "tomorrow",
    "week",
}

_VERBISH_WORDS = {
    "add",
    "adds",
    "align",
    "aligned",
    "aligning",
    "apply",
    "applied",
    "applying",
    "attach",
    "attaching",
    "build",
    "building",
    "compare",
    "compares",
    "copy",
    "copying",
    "cover",
    "covers",
    "define",
    "defining",
    "discuss",
    "discusses",
    "explain",
    "explains",
    "focus",
    "focused",
    "focusing",
    "focuses",
    "going",
    "help",
    "helps",
    "keep",
    "keeps",
    "look",
    "looks",
    "open",
    "organized",
    "organize",
    "merge",
    "merged",
    "review",
    "reviews",
    "show",
    "shows",
    "specify",
    "specified",
    "specifies",
    "standardize",
    "standardizes",
    "start",
    "starts",
    "take",
    "updated",
    "updating",
    "use",
    "using",
    "welcome",
    "work",
    "write",
    "writes",
    "run",
    "runs",
}

_UPPER_TOKENS = {"ai", "api", "cpu", "css", "ffmpeg", "gpu", "html", "llm", "ml", "pdf", "rag", "ssh", "ui", "ux"}

_TOKEN_NORMALIZATIONS = {
    "dart": "Dart",
    "flutter": "Flutter",
    "github": "GitHub",
    "javascript": "JavaScript",
    "kotlin": "Kotlin",
    "linkedin": "LinkedIn",
    "notion": "Notion",
    "openai": "OpenAI",
    "python": "Python",
    "react": "React",
    "slack": "Slack",
    "spotify": "Spotify",
    "swift": "Swift",
    "typescript": "TypeScript",
    "whatsapp": "WhatsApp",
    "youtube": "YouTube",
}

_TRANSITION_CUES = (
    "next",
    "moving on",
    "let's move",
    "turning to",
    "in contrast",
    "however",
    "on the other hand",
    "the next",
    "another",
    "finally",
    "to summarize",
    "in summary",
)

_ACADEMIC_MARKERS = (
    "framework",
    "hypothesis",
    "evidence",
    "analysis",
    "method",
    "concept",
    "model",
    "assumption",
    "theory",
    "therefore",
    "because",
    "data",
    "research",
    "principle",
    "derive",
    "compare",
    "evaluate",
)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def strip_timecodes(text: str) -> str:
    cleaned = _TIME_PATTERN.sub(" ", text)
    cleaned = re.sub(r"\[[^\]]*(?:\d{1,2}:)?\d{1,2}:\d{2}[^\]]*\]", " ", cleaned)
    return normalize_whitespace(cleaned)


def split_sentences(text: str, *, minimum_words: int = 5) -> list[str]:
    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        cleaned = strip_timecodes(normalize_whitespace(part).strip(" -•"))
        if len(cleaned.split()) < minimum_words:
            continue
        if cleaned and cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        if cleaned:
            sentences.append(cleaned)
    return sentences


def tokenize(text: str, *, minimum_length: int = 3) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]+", text.lower())
    return [token for token in tokens if len(token) >= minimum_length]


def extract_keywords(text: str, *, limit: int = 8) -> list[str]:
    cleaned = strip_timecodes(text)
    tokens = [token.lower() for token in _ALPHA_TOKEN_PATTERN.findall(cleaned)]
    filtered = [token for token in tokens if token not in _STOPWORDS]
    if not cleaned or not filtered:
        return []

    token_counts = Counter(filtered)
    phrase_scores: Counter[str] = Counter()

    for segment_index, segment in enumerate(_content_segments(cleaned)):
        max_words = min(3, len(segment))
        for size in range(max_words, 0, -1):
            for index in range(0, len(segment) - size + 1):
                phrase_tokens = segment[index : index + size]
                phrase = " ".join(phrase_tokens)
                if is_low_signal_topic(phrase):
                    continue
                if not _phrase_has_signal(phrase_tokens):
                    continue
                position_bonus = max(0.0, 0.45 - (segment_index * 0.08) - (index * 0.02))
                phrase_scores[phrase] += _phrase_score(phrase_tokens, token_counts) + position_bonus

    results: list[str] = []
    seen: set[str] = set()
    seen_signatures: set[str] = set()
    covered_tokens: set[str] = set()
    for phrase, _ in sorted(
        phrase_scores.items(),
        key=lambda item: (
            -item[1],
            0 if 2 <= len(item[0].split()) <= 3 else 1,
            -len(item[0]),
        ),
    ):
        normalized = _normalize_topic_phrase(phrase)
        key = normalized.lower()
        normalized_tokens = [token.lower() for token in normalized.split()]
        signature = " ".join(sorted(normalized_tokens))
        if key in seen:
            continue
        if len(normalized_tokens) == 1 and normalized_tokens[0] in covered_tokens:
            continue
        if len(normalized_tokens) == 2 and (
            signature in seen_signatures or all(token in covered_tokens for token in normalized_tokens)
        ):
            continue
        if any(key in existing or existing in key for existing in seen if len(key) >= 8):
            continue
        seen.add(key)
        if len(normalized_tokens) >= 2:
            seen_signatures.add(signature)
        covered_tokens.update(normalized_tokens)
        results.append(normalized)
        if len(results) == limit:
            break

    if results:
        return results

    fallback_results: list[str] = []
    for token, _ in token_counts.most_common(limit * 3):
        if is_low_signal_topic(token):
            continue
        normalized = _normalize_topic_phrase(token)
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        fallback_results.append(normalized)
        if len(fallback_results) == limit:
            break
    return fallback_results


def is_low_signal_topic(text: str) -> bool:
    normalized = strip_timecodes(normalize_whitespace(text)).strip(" .,:;!?-'\"")
    if not normalized:
        return True
    lowered = normalized.lower()
    if "untitled chapter" in lowered or lowered.startswith("chapter ") and len(lowered.split()) <= 3:
        return True
    tokens = [token.lower() for token in _ALPHA_TOKEN_PATTERN.findall(normalized)]
    if len(tokens) <= 3 and any(token.endswith("n't") for token in tokens):
        return True
    meaningful = [token for token in tokens if token not in _STOPWORDS]
    if not meaningful:
        return True
    low_signal_tokens = [
        token for token in meaningful if token in _LOW_SIGNAL_FILLER_WORDS
    ]
    if len(low_signal_tokens) == len(meaningful):
        return True
    if any(token.endswith("n't") for token in meaningful):
        return True
    if meaningful[0] in _ACTION_LEADER_WORDS and (
        len(meaningful) <= 4 or any(token in _TEMPORAL_TOPIC_WORDS for token in meaningful)
    ):
        return True
    if meaningful[0] in _GENERIC_TOPIC_WORDS and len(meaningful) <= 3:
        return True
    if meaningful[0] in _LOW_SIGNAL_LEADERS and len(meaningful) <= 3:
        return True
    if len(meaningful) == 1 and meaningful[0] in _LOW_SIGNAL_SINGLE_WORDS:
        return True
    if len(meaningful) <= 3 and len(low_signal_tokens) >= max(1, len(meaningful) - 1):
        return True
    if meaningful[-1] in _WEAK_TRAILING_TOPIC_WORDS and len(meaningful) <= 3:
        return True
    if meaningful[-1] in _TEMPORAL_TOPIC_WORDS and len(meaningful) <= 4:
        return True
    if len(meaningful) <= 2 and all(
        token in _NUMBER_WORDS or token in _GENERIC_TOPIC_WORDS for token in meaningful
    ):
        return True
    if len(set(meaningful)) == 1 and len(meaningful) > 1:
        return True
    if meaningful[0] in _NUMBER_WORDS and len(meaningful) <= 3:
        return True
    if lowered in _STOPWORDS:
        return True
    return False


def build_headline(text: str, *, fallback: str, max_words: int = 6) -> str:
    cleaned = strip_timecodes(normalize_whitespace(text)).strip(".")
    if not cleaned:
        return fallback

    for keyword in extract_keywords(cleaned, limit=4):
        if is_low_signal_topic(keyword):
            continue
        compact = " ".join(keyword.split()[:max_words]).strip()
        if compact:
            return _normalize_topic_phrase(compact)

    words = cleaned.split()
    for start in range(min(len(words), 5)):
        candidate = " ".join(words[start : start + max_words]).strip(" .,:;!?-")
        if not candidate or is_low_signal_topic(candidate):
            continue
        return candidate[:1].upper() + candidate[1:]

    headline = " ".join(words[:max_words])
    return headline[:1].upper() + headline[1:]


def truncate_text(text: str, limit: int) -> str:
    normalized = normalize_whitespace(text)
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rsplit(' ', 1)[0]}..."


def _content_segments(text: str) -> list[list[str]]:
    words = [token.lower() for token in _ALPHA_TOKEN_PATTERN.findall(text)]
    segments: list[list[str]] = []
    current: list[str] = []
    for word in words:
        if word in _STOPWORDS or word in _VERBISH_WORDS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(word)
    if current:
        segments.append(current)
    return [segment for segment in segments if segment]


def _phrase_has_signal(phrase_tokens: list[str]) -> bool:
    meaningful = [token for token in phrase_tokens if token not in _STOPWORDS]
    if not meaningful:
        return False
    strong_tokens = [token for token in meaningful if _is_strong_topic_token(token)]
    if not strong_tokens:
        return False
    if len(meaningful) <= 3 and sum(
        1 for token in meaningful if token in _LOW_SIGNAL_FILLER_WORDS
    ) >= max(1, len(meaningful) - 1):
        return False
    return True


def _is_strong_topic_token(token: str) -> bool:
    if token in _LOW_SIGNAL_FILLER_WORDS or token in _STOPWORDS:
        return False
    if token in _UPPER_TOKENS:
        return True
    return len(token) >= 5


def _phrase_score(phrase_tokens: list[str], token_counts: Counter[str]) -> float:
    if not phrase_tokens:
        return 0.0
    size = len(phrase_tokens)
    repetition_score = sum(token_counts[token] for token in set(phrase_tokens)) / max(1, size)
    length_bonus = {1: 0.15, 2: 0.8, 3: 0.95, 4: 0.45}.get(size, 0.2)
    specificity_bonus = sum(max(1, len(token) - 2) for token in phrase_tokens) / 18
    return round(repetition_score + length_bonus + specificity_bonus, 4)


def _normalize_topic_phrase(phrase: str) -> str:
    cleaned = normalize_whitespace(phrase.replace("-", " "))
    tokens = cleaned.split()
    normalized_tokens = [
        _TOKEN_NORMALIZATIONS.get(
            token.lower(),
            token.upper() if token.lower() in _UPPER_TOKENS else token.capitalize(),
        )
        for token in tokens
    ]
    return " ".join(normalized_tokens)


def sample_evenly(items: list[T], count: int) -> list[T]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]

    last_index = len(items) - 1
    indices = sorted(
        {
            min(last_index, max(0, round((last_index * step) / (count - 1))))
            for step in range(count)
        }
    )
    while len(indices) < count:
        for index in range(len(items)):
            if index in indices:
                continue
            indices.append(index)
            if len(indices) == count:
                break
    indices.sort()
    return [items[index] for index in indices[:count]]


def lexical_diversity(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def estimate_reading_minutes(word_count: int, *, words_per_minute: int = 170) -> float:
    if word_count <= 0:
        return 0.0
    return round(word_count / words_per_minute, 2)


def contains_transition_cue(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _TRANSITION_CUES)


def academic_signal_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    marker_hits = sum(1 for token in tokens if token in _ACADEMIC_MARKERS)
    return round(min(1.0, marker_hits / max(6, len(tokens) * 0.08)), 4)
