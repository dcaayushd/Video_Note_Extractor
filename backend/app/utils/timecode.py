from __future__ import annotations

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


def _is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "youtube.com" in host or "youtu.be" in host


def format_timestamp(total_seconds: float) -> str:
    seconds = int(total_seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_youtube_jump_url(url: str | None, start_seconds: float) -> str | None:
    if not url:
        return None
    if not _is_youtube_url(url):
        return None

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["t"] = [str(int(start_seconds))]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
