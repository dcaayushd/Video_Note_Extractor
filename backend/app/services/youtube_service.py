from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import certifi
import httpx

from app.core.config import Settings

logger = logging.getLogger("uvicorn.error")


class UnsupportedSourceError(RuntimeError):
    """Raised when a source URL points to unsupported or DRM-protected media."""


class _SilentYtdlpLogger:
    def debug(self, message: str) -> None:
        return None

    def info(self, message: str) -> None:
        return None

    def warning(self, message: str) -> None:
        return None

    def error(self, message: str) -> None:
        return None


class YouTubeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def download_media(
        self,
        source_url: str,
        output_dir: Path,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> tuple[Path, dict[str, str | int | float]]:
        unsupported_message = self.unsupported_source_message(source_url)
        if unsupported_message is not None:
            raise UnsupportedSourceError(unsupported_message)

        if self.settings.demo_mode:
            placeholder = output_dir / "sample-video.mp4"
            placeholder.parent.mkdir(parents=True, exist_ok=True)
            placeholder.write_bytes(b"sample-audio-placeholder")
            if progress_callback is not None:
                progress_callback(100, "100% | sample media prepared")
            return placeholder, {
                "title": "Weekly Planning Session",
                "source_url": source_url,
                "duration_seconds": 3120,
            }

        try:
            from yt_dlp import YoutubeDL
            from yt_dlp.utils import DownloadError
        except ImportError as exc:
            raise RuntimeError("Install yt-dlp to download video sources.") from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / "%(title)s.%(ext)s")
        info_path = output_dir / "info.json"
        has_ffmpeg = self._has_ffmpeg()

        base_options = self._base_options(output_template)
        try:
            with YoutubeDL(base_options) as ydl:
                info = ydl.extract_info(source_url, download=False)
        except DownloadError as exc:
            friendly_message = self._friendly_download_error_message(
                str(exc),
                source_url=source_url,
            )
            if friendly_message is not None:
                raise UnsupportedSourceError(friendly_message) from exc
            raise
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

        format_error: Exception | None = None
        for index, download_options in enumerate(
            self._download_attempts(
                output_template=output_template,
                has_ffmpeg=has_ffmpeg,
                progress_callback=progress_callback,
            ),
            start=1,
        ):
            selector = str(download_options.get("format", "<default>"))

            try:
                with YoutubeDL(download_options) as ydl:
                    ydl.download([source_url])
                format_error = None
                break
            except DownloadError as exc:
                message = str(exc)
                friendly_message = self._friendly_download_error_message(
                    message,
                    source_url=source_url,
                )
                if friendly_message is not None:
                    raise UnsupportedSourceError(friendly_message) from exc
                if not self._is_recoverable_download_error(message):
                    raise
                logger.warning(
                    "yt-dlp attempt failed | attempt=%s | selector=%s | reason=%s",
                    index,
                    selector,
                    message,
                )
                format_error = exc

        if format_error is not None:
            raise format_error

        media_path = self._resolve_downloaded_media(output_dir, preferred_suffix=".mp4")
        transcript_metadata = self._download_source_transcript(info, output_dir)

        metadata = {
            "title": info.get("title", "Untitled video"),
            "source_url": info.get("webpage_url", source_url),
            "duration_seconds": info.get("duration") or 0,
            "channel": info.get("channel"),
            "source_chapter_count": (
                len(info.get("chapters"))
                if isinstance(info.get("chapters"), list)
                else 0
            ),
            "local_media_path": str(media_path),
            **transcript_metadata,
        }
        return media_path, metadata

    def unsupported_source_message(self, source_url: str) -> str | None:
        host = self._normalized_host(source_url)
        if not host:
            return None

        if "spotify.com" in host:
            return (
                "Spotify links are not supported because Spotify streams are DRM-protected. "
                "Use a YouTube link, a direct downloadable media URL, or upload the audio/video file instead."
            )
        if any(
            blocked_host in host
            for blocked_host in (
                "netflix.com",
                "disneyplus.com",
                "hulu.com",
                "primevideo.com",
                "max.com",
            )
        ):
            return (
                "This source appears to come from a DRM-protected streaming service, so it cannot be processed here. "
                "Use a non-DRM media link or upload a local audio/video file instead."
            )
        return None

    def _base_options(
        self,
        output_template: str,
        *,
        use_mobile_clients: bool = True,
    ) -> dict[str, object]:
        options: dict[str, object] = {
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "logger": _SilentYtdlpLogger(),
            "skip_unavailable_fragments": True,
        }
        if use_mobile_clients:
            options["extractor_args"] = {
                "youtube": {
                    "player_client": ["android", "ios", "web"],
                }
            }
        return options

    def _has_ffmpeg(self) -> bool:
        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    def _media_format_selectors(self, has_ffmpeg: bool) -> list[str]:
        if has_ffmpeg:
            return [
                (
                    "bestvideo[ext=mp4][protocol!=m3u8]+"
                    "bestaudio[ext=m4a][protocol!=m3u8]/"
                    "best[ext=mp4][protocol!=m3u8]/"
                    "best[protocol!=m3u8]/best"
                ),
                "bestvideo[protocol!=m3u8]+bestaudio[protocol!=m3u8]/best[protocol!=m3u8]/best",
                "bestvideo+bestaudio/best",
                "best",
            ]
        return [
            "best[ext=mp4][protocol!=m3u8]/best[protocol!=m3u8]/best",
            "best[protocol!=m3u8]/best",
            "best",
        ]

    def _download_attempts(
        self,
        *,
        output_template: str,
        has_ffmpeg: bool,
        progress_callback: Callable[[int, str], None] | None,
    ) -> list[dict[str, object]]:
        attempts: list[dict[str, object]] = []
        progress_hooks = [self._build_progress_hook(progress_callback)]

        for selector in self._media_format_selectors(has_ffmpeg):
            options = self._base_options(output_template, use_mobile_clients=True)
            options["format"] = selector
            options["progress_hooks"] = progress_hooks
            if has_ffmpeg:
                options["merge_output_format"] = "mp4"
            attempts.append(options)

        relaxed_selectors = [
            "bv*+ba/b",
            "bestvideo+bestaudio/best",
            "best",
        ]
        for selector in relaxed_selectors:
            options = self._base_options(output_template, use_mobile_clients=False)
            options["format"] = selector
            options["progress_hooks"] = progress_hooks
            attempts.append(options)

        final_attempt = self._base_options(output_template, use_mobile_clients=False)
        final_attempt["progress_hooks"] = progress_hooks
        attempts.append(final_attempt)
        return attempts

    def _is_recoverable_download_error(self, message: str) -> bool:
        recoverable_markers = (
            "Requested format is not available",
            "No video formats found",
            "Requested format not available",
        )
        return any(marker in message for marker in recoverable_markers)

    def _friendly_download_error_message(
        self,
        message: str,
        *,
        source_url: str,
    ) -> str | None:
        lowered = message.lower()
        if "drm" in lowered and "not be supported" in lowered:
            return self.unsupported_source_message(source_url) or (
                "This link appears to use DRM protection, so it cannot be processed here. "
                "Use a non-DRM media link or upload the audio/video file instead."
            )
        return None

    def _normalized_host(self, source_url: str) -> str:
        parsed = urlparse(source_url)
        return parsed.netloc.lower().removeprefix("www.")

    def _build_progress_hook(
        self,
        progress_callback: Callable[[int, str], None] | None,
    ) -> Callable[[dict[str, object]], None]:
        def hook(status: dict[str, object]) -> None:
            if progress_callback is None:
                return
            state = str(status.get("status") or "")
            if state == "downloading":
                downloaded = int(status.get("downloaded_bytes") or 0)
                total = int(
                    status.get("total_bytes")
                    or status.get("total_bytes_estimate")
                    or 0
                )
                percent = int((downloaded / total) * 100) if total else 0
                speed = self._format_bytes(status.get("speed"))
                eta = status.get("eta")
                total_text = self._format_bytes(total) if total else "unknown"
                detail = (
                    f"{percent}% | {self._format_bytes(downloaded)}/{total_text}"
                    f" | {speed}/s | ETA {eta if eta is not None else '--'}s"
                )
                progress_callback(percent, detail)
            elif state == "finished":
                progress_callback(100, "100% | download finished")

        return hook

    def _format_bytes(self, value: object) -> str:
        if value in {None, 0, 0.0}:
            return "0 B"
        size = float(value)
        units = ["B", "KB", "MB", "GB"]
        unit_index = 0
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        return f"{size:.1f} {units[unit_index]}"

    def _resolve_downloaded_media(
        self,
        output_dir: Path,
        *,
        preferred_suffix: str | None = None,
    ) -> Path:
        candidates = [
            path
            for path in output_dir.iterdir()
            if path.is_file()
            and path.name != "info.json"
            and not path.name.endswith((".part", ".ytdl"))
        ]
        if preferred_suffix:
            preferred = [
                path for path in candidates if path.suffix.lower() == preferred_suffix.lower()
            ]
            if preferred:
                preferred.sort(key=lambda path: path.stat().st_mtime, reverse=True)
                return preferred[0]
        if not candidates:
            raise RuntimeError("YouTube download completed but no media file was created.")
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0]

    def _download_source_transcript(
        self,
        info: dict[str, object],
        output_dir: Path,
    ) -> dict[str, str]:
        asset = self._select_source_transcript_asset(info)
        if not asset:
            return {}

        url = asset.get("url")
        extension = asset.get("ext")
        language = asset.get("language")
        kind = asset.get("kind")
        if not url or not extension or not language or not kind:
            return {}

        transcript_path = output_dir / f"source-transcript.{extension}"
        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=20,
                verify=certifi.where(),
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                    )
                },
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                transcript_path.write_bytes(response.content)
        except (OSError, ValueError, httpx.HTTPError) as exc:
            logger.warning("Unable to download source transcript: %s", exc)
            return {}

        return {
            "source_transcript_path": str(transcript_path),
            "source_transcript_language": language,
            "source_transcript_kind": kind,
            "source_transcript_format": extension,
        }

    def _select_source_transcript_asset(
        self,
        info: dict[str, object],
    ) -> dict[str, str] | None:
        preferred_language = str(
            info.get("language") or self.settings.transcription_language or ""
        ).strip().lower()
        candidates = [
            *self._caption_candidates(
                info.get("subtitles"),
                preferred_language=preferred_language,
                kind="manual_subtitles",
            ),
            *self._caption_candidates(
                info.get("automatic_captions"),
                preferred_language=preferred_language,
                kind="automatic_captions",
            ),
        ]
        if not candidates:
            return None
        candidates.sort(key=self._caption_candidate_priority)
        return candidates[0]

    def _caption_candidates(
        self,
        caption_map: object,
        *,
        preferred_language: str,
        kind: str,
    ) -> list[dict[str, str]]:
        if not isinstance(caption_map, dict):
            return []

        preferred_languages = [
            preferred_language,
            "en",
        ]
        candidates: list[dict[str, str]] = []
        for language, entries in caption_map.items():
            if not isinstance(language, str) or language == "live_chat":
                continue
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                extension = str(entry.get("ext") or "").strip().lower()
                url = str(entry.get("url") or "").strip()
                protocol = str(entry.get("protocol") or "").strip().lower()
                if not extension or not url:
                    continue
                if extension not in {"json3", "srv3", "srv2", "srv1", "ttml", "xml", "vtt", "srt"}:
                    continue
                if protocol == "m3u8_native":
                    continue
                candidates.append(
                    {
                        "language": language.lower(),
                        "ext": extension,
                        "url": url,
                        "kind": kind,
                        "preferred": "1" if language.lower() in preferred_languages else "0",
                    }
                )
        return candidates

    def _caption_candidate_priority(self, candidate: dict[str, str]) -> tuple[int, int, int]:
        kind_rank = 0 if candidate.get("kind") == "manual_subtitles" else 1
        preferred_rank = 0 if candidate.get("preferred") == "1" else 1
        extension_rank = {
            "json3": 0,
            "srv3": 1,
            "ttml": 2,
            "xml": 3,
            "vtt": 4,
            "srt": 5,
            "srv2": 6,
            "srv1": 7,
        }.get(candidate.get("ext"), 9)
        return (kind_rank, preferred_rank, extension_rank)
