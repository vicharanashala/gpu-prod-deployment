"""
OpenAI-compatible audio routes backed by Sarvam AI streaming TTS/STT.

Environment is loaded from `.env` (see `Settings`). The Sarvam SDK also
reads `SARVAM_API_KEY` from the environment if not passed explicitly.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sarvamai import AsyncSarvamAI
from sarvamai.core.api_error import ApiError

from tts_language_detection import detect_tts_language_code, pick_tts_speaker

logger = logging.getLogger("voice_services")

# ~50ms silent MP3 returned when sanitization leaves nothing to speak (avoids client errors).
_SILENT_MP3_BYTES = base64.standard_b64decode(
    "SUQzBAAAAAAAIlRTU0UAAAAOAAADTGF2ZjYxLjcuMTAyAAAAAAAAAAAAAAD/+1DEAAPAAAGkAAAAIAAANIAAAARMQU1FMy4xMDBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//tSxF2DwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/+1LEoYPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQ=="
)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


class PrepTrace:
    """Collect prep/fallback steps for one request; echoed to stdout and file logs."""

    __slots__ = ("steps",)

    def __init__(self) -> None:
        self.steps: list[str] = []

    def add(self, step: str) -> None:
        self.steps.append(step)
        logger.info("prep: %s", step)

    def absorb(self, other: PrepTrace) -> None:
        self.steps.extend(other.steps)

    def format(self) -> str:
        return " → ".join(self.steps) if self.steps else "none"


def _install_voice_service_log_handlers() -> None:
    """
    Uvicorn's default logging often leaves the root logger above INFO or without propagation
    for app loggers. Attach explicit stderr handlers so voice_services / tts_language_detection
    INFO lines always show next to access logs.
    """
    fmt = logging.Formatter("%(levelname)s [%(name)s] %(message)s")
    for name in ("voice_services", "tts_language_detection"):
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        if log.handlers:
            continue
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(fmt)
        handler.setLevel(logging.INFO)
        log.addHandler(handler)
        log.propagate = False


_install_voice_service_log_handlers()

# --- Configuration -----------------------------------------------------------------


# Always load `.env` next to this file (cwd-independent; fixes missing flags when uvicorn
# is started from another directory).
_VOICE_ENV_FILE = Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_VOICE_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sarvam_api_key: str = Field(validation_alias="SARVAM_API_KEY")
    # Sarvam file STT (plain HTTP multipart — same as ``requests.post(..., files=...)``).
    sarvam_stt_rest_url: str = Field(
        default="https://api.sarvam.ai/speech-to-text",
        validation_alias="SARVAM_STT_URL",
    )

    default_tts_model: str = Field(default="bulbul:v3", validation_alias="DEFAULT_TTS_MODEL")
    default_stt_model: str = Field(default="saaras:v3", validation_alias="DEFAULT_STT_MODEL")
    default_tts_speaker: str = Field(default="shubh", validation_alias="DEFAULT_TTS_SPEAKER")
    default_tts_language: str = Field(default="hi-IN", validation_alias="DEFAULT_TTS_LANGUAGE")
    default_stt_language: str = Field(
        default="unknown",
        validation_alias="DEFAULT_STT_LANGUAGE",
        description="Sarvam STT BCP-47 code, or 'unknown' for auto language detection.",
    )
    stt_mode: str = Field(default="transcribe", validation_alias="STT_MODE")
    stt_high_vad_sensitivity: bool = Field(default=True, validation_alias="STT_HIGH_VAD_SENSITIVITY")
    stt_sample_rate: int = Field(default=16000, validation_alias="STT_SAMPLE_RATE")
    # REST file transcribe is usually faster than WebSocket for one-shot uploads (LibreChat).
    stt_use_rest: bool = Field(
        default=True,
        validation_alias="STT_USE_REST",
        description="true: Sarvam REST speech_to_text (multipart HTTP). false: WebSocket streaming STT.",
    )
    stt_max_audio_seconds: float = Field(
        default=60.0,
        validation_alias="STT_MAX_AUDIO_SECONDS",
        description="Maximum upload duration accepted by this service (seconds).",
    )
    stt_chunk_seconds: float = Field(
        default=30.0,
        validation_alias="STT_CHUNK_SECONDS",
        description="Max segment length per Sarvam REST call (seconds).",
    )
    stt_chunk_delay_seconds: float = Field(
        default=2.0,
        validation_alias="STT_CHUNK_DELAY_SECONDS",
        description="Pause between sequential Sarvam STT chunk requests (seconds).",
    )
    # When true, log the complete STT transcript string at INFO (may contain PII; use only in dev).
    log_full_stt_transcript: bool = Field(
        default=False,
        validation_alias="LOG_FULL_STT_TRANSCRIPT",
    )
    # When true, log the complete TTS input string (body.input) at INFO for /v1/audio/speech.
    log_full_tts_text: bool = Field(
        default=False,
        validation_alias="LOG_FULL_TTS_TEXT",
    )
    # Master switch for TTS file logging (default off). Requires TTS_LOG_DIR when enabled.
    tts_log_enabled: bool = Field(
        default=False,
        validation_alias="TTS_LOG_ENABLED",
    )
    # Directory for TTS log files when TTS_LOG_ENABLED=true (e.g. /app/logs mounted from host).
    tts_log_dir: str = Field(
        default="/app/logs",
        validation_alias="TTS_LOG_DIR",
    )
    # When true (and TTS_LOG_ENABLED=true), also write one .txt file per request with full text.
    tts_log_full_text_files: bool = Field(
        default=False,
        validation_alias="TTS_LOG_FULL_TEXT_FILES",
    )

    # TTS: optional LLM language detection (OpenAI-compatible chat). If unset, use DEFAULT_TTS_LANGUAGE.
    tts_lang_detect_base_url: Optional[str] = Field(
        default=None,
        validation_alias="TTS_LANG_DETECT_BASE_URL",
    )
    tts_lang_detect_model: str = Field(
        default="google/gemma-4-E4B-it",
        validation_alias="TTS_LANG_DETECT_MODEL",
    )
    tts_lang_detect_timeout: float = Field(default=60.0, validation_alias="TTS_LANG_DETECT_TIMEOUT")
    default_tts_gender: Literal["male", "female"] = Field(
        default="male",
        validation_alias="DEFAULT_TTS_GENDER",
    )
    # Sarvam REST TTS per-request char cap. Unset → model default (2500 bulbul:v3, 1500 bulbul:v2).
    tts_max_chars_per_request: Optional[int] = Field(
        default=None,
        validation_alias="TTS_MAX_CHARS_PER_REQUEST",
    )
    # Notice block starts at this marker (AjraSakha testing disclaimer).
    tts_notice_marker: str = Field(
        default="⚠️",
        validation_alias="TTS_NOTICE_MARKER",
        description="Start of the notice block spoken after the main answer.",
    )
    # Optional extra regex patterns (|| separated) removed anywhere in the text before TTS.
    tts_strip_patterns: str = Field(
        default="",
        validation_alias="TTS_STRIP_PATTERNS",
        description="Additional regex patterns merged with built-in WorkDrive / link patterns.",
    )
    # AjraSakha footer: speak main body until this line, then the notice block until the next.
    tts_footnote_separator: str = Field(
        default="_____________________________",
        validation_alias="TTS_FOOTNOTE_SEPARATOR",
        description="Line dividing main answer from footer metadata.",
    )
    tts_speak_notice_block: bool = Field(
        default=True,
        validation_alias="TTS_SPEAK_NOTICE_BLOCK",
        description=(
            "When true and the footnote separator appears twice, speak part 1 (before first "
            "separator), skip answered-by/sources, then speak the notice block until the "
            "second separator."
        ),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _pace_from_openai_speed(speed: float) -> float:
    """Map OpenAI speech speed (0.25–4.0, default 1.0) to Sarvam pace (roughly 0.5–2.0)."""
    return max(0.5, min(2.0, float(speed)))


def _map_openai_voice_to_speaker(voice: str, default_speaker: str) -> str:
    """OpenAI voice names → Sarvam defaults; otherwise treat `voice` as a Sarvam speaker id."""
    aliases = {
        "alloy": default_speaker,
        "echo": default_speaker,
        "fable": default_speaker,
        "onyx": default_speaker,
        "nova": default_speaker,
        "shimmer": default_speaker,
    }
    return aliases.get(voice.lower().strip(), voice.strip() or default_speaker)


class SpeechRequest(BaseModel):
    """Subset of OpenAI `POST /v1/audio/speech` JSON body."""

    model: Optional[str] = None
    input: str
    voice: str = "alloy"
    response_format: str = Field(default="mp3", description="mp3 supported")
    speed: float = 1.0
    stream: bool = False
    # Sarvam-specific optional override (not in OpenAI spec). When omitted and
    # TTS_LANG_DETECT_BASE_URL is set, language is inferred via chat completions.
    target_language_code: Optional[str] = None


def _parse_tts_strip_markers(raw: str) -> list[str]:
    """Parse ``||`` or newline separated lists (used for ``TTS_STRIP_PATTERNS``)."""
    if not raw.strip():
        return []
    if "||" in raw:
        parts = raw.split("||")
    else:
        parts = raw.splitlines()
    return [part.strip() for part in parts if part.strip()]


# Zoho WorkDrive refs and source-link lines that LibreChat sometimes sends as TTS input.
_BUILTIN_TTS_STRIP_REGEXES: tuple[tuple[str, str], ...] = (
    ("workdrive_url", r"https?://workdrive\.zoho(?:external)?\.in/file/[A-Za-z0-9]+"),
    ("in_file_fragment", r"\bin/file/[A-Za-z0-9]+\b"),
    ("markdown_link", r"\[[^\]]+\]\([^)]*\)"),
    ("orphan_paren_line", r"^\s*\)\s*$"),
    ("link_emoji_line", r"^\s*🔗\s*.+$"),
)


def _parse_tts_strip_patterns(raw: str) -> list[str]:
    """Parse ``TTS_STRIP_PATTERNS`` (``||`` or newline separated regexes)."""
    return _parse_tts_strip_markers(raw)


def _apply_tts_regex_strips(
    text: str,
    patterns: list[tuple[str, str]],
    prep: PrepTrace,
) -> str:
    out = text
    for name, pattern in patterns:
        try:
            new = re.sub(pattern, "", out, flags=re.MULTILINE | re.IGNORECASE)
        except re.error as exc:
            logger.warning("TTS strip regex invalid %r: %s", pattern, exc)
            prep.add(f"strip:regex:{name}:invalid({exc})")
            continue
        if len(new) != len(out):
            prep.add(f"strip:regex:{name}({len(out)}→{len(new)} chars)")
            out = new
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()



def _apply_tts_footnote_segments(
    text: str,
    settings: Settings,
    prep: PrepTrace,
) -> Optional[str]:
    """
    AjraSakha layout: [main answer] ___ [answered-by + sources] [notice] ___ [IMD footer…]

    Speak main answer and notice block only; omit the middle and trailing footer.
    """
    sep = settings.tts_footnote_separator.strip()
    if not settings.tts_speak_notice_block or not sep:
        return None

    first_sep = text.find(sep)
    if first_sep == -1:
        return None

    notice_marker = settings.tts_notice_marker.strip() or "⚠️"
    notice_idx = text.find(notice_marker)

    # LibreChat often sends a footer-only chunk: [source tail][notice]___[IMD footer]
    if notice_idx != -1 and notice_idx < first_sep:
        part2 = text[notice_idx:first_sep].strip()
        prep.add(
            f"segment:notice_only_footer_chunk({len(part2)} chars)"
            " skip_sources_tail_before_notice"
        )
        return part2

    part1 = text[:first_sep].rstrip()
    after_first = text[first_sep + len(sep) :]
    notice_rel = after_first.find(notice_marker)
    if notice_rel == -1:
        prep.add(f"segment:main_only({len(part1)} chars, no_notice_after_sep)")
        return part1

    after_notice = after_first[notice_rel:]
    second_rel = after_notice.find(sep, len(notice_marker))
    if second_rel == -1:
        part2 = after_notice.strip()
        prep.add(
            f"segment:main({len(part1)} chars)+notice({len(part2)} chars, no_second_sep)"
        )
    else:
        part2 = after_notice[:second_rel].strip()
        prep.add(
            f"segment:main({len(part1)} chars)+notice({len(part2)} chars)"
            " skip_answered_by_sources_and_imd_footer"
        )

    if part1 and part2:
        return f"{part1}\n\n{part2}"
    return part1 or part2


def _prepare_tts_text(text: str, settings: Settings, *, prep: PrepTrace) -> str:
    """
    Sanitize TTS input: regex cleanup, then AjraSakha footnote / notice segments.
    """
    patterns: list[tuple[str, str]] = list(_BUILTIN_TTS_STRIP_REGEXES)
    for i, pattern in enumerate(_parse_tts_strip_patterns(settings.tts_strip_patterns)):
        patterns.append((f"custom:{i}", pattern))

    out = _apply_tts_regex_strips(text, patterns, prep) if patterns else text

    segmented = _apply_tts_footnote_segments(out, settings, prep)
    if segmented is not None:
        return segmented

    sep = settings.tts_footnote_separator.strip()
    if sep:
        idx = out.find(sep)
        if idx != -1:
            stripped = out[:idx].rstrip()
            prep.add(f"strip:footnote_sep({len(out)}→{len(stripped)} chars)")
            return stripped

    notice_marker = settings.tts_notice_marker.strip() or "⚠️"
    idx = out.find(notice_marker)
    if idx != -1:
        stripped = out[:idx].rstrip()
        prep.add(f"strip:before_notice({len(out)}→{len(stripped)} chars)")
        return stripped

    if not any(step.startswith(("strip:", "segment:")) for step in prep.steps):
        prep.add("strip:none")
    return out


def _write_tts_log_file(
    log_dir: str,
    *,
    model: str,
    lang: str,
    speaker: str,
    stream: bool,
    raw_input: str,
    tts_text: str,
    prep_trace: str,
    write_full_text_files: bool,
    total_ms: Optional[float] = None,
    resolve_ms: Optional[float] = None,
    sarv_ms: Optional[float] = None,
    out_bytes: Optional[int] = None,
) -> None:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / "tts.log"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    ts_slug = ts.replace(":", "-")
    lines = [
        f"--- {ts} ---",
        f"model={model!r} lang={lang!r} speaker={speaker!r} stream={stream}",
        f"full_input_chars={len(raw_input)} text_spoken_chars={len(tts_text)}",
        f"prep={prep_trace}",
    ]
    if total_ms is not None:
        lines.append(
            "timing_ms "
            f"total={total_ms:.1f} resolve={resolve_ms:.1f} sarvam={sarv_ms:.1f} out_bytes={out_bytes}"
        )
    lines.extend(
        [
            "--- full_input ---",
            raw_input,
            "--- text_spoken ---",
            tts_text,
            "",
        ]
    )
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    if write_full_text_files:
        full_text_dir = path / "full-text"
        full_text_dir.mkdir(parents=True, exist_ok=True)
        full_text_file = full_text_dir / f"{ts_slug}.txt"
        full_text_body = "\n".join(
            [
                f"timestamp: {ts}",
                f"model: {model}",
                f"lang: {lang}",
                f"speaker: {speaker}",
                f"stream: {stream}",
                f"prep: {prep_trace}",
                "",
                "=== full_input ===",
                raw_input,
                "",
                "=== text_spoken ===",
                tts_text,
                "",
            ]
        )
        full_text_file.write_text(full_text_body, encoding="utf-8")


async def _append_tts_log(
    settings: Settings,
    *,
    model: str,
    lang: str,
    speaker: str,
    stream: bool,
    raw_input: str,
    tts_text: str,
    prep_trace: str,
    total_ms: Optional[float] = None,
    resolve_ms: Optional[float] = None,
    sarv_ms: Optional[float] = None,
    out_bytes: Optional[int] = None,
) -> None:
    if not settings.tts_log_enabled:
        return
    log_dir = (settings.tts_log_dir or "").strip() or "/app/logs"
    try:
        await asyncio.to_thread(
            _write_tts_log_file,
            log_dir,
            model=model,
            lang=lang,
            speaker=speaker,
            stream=stream,
            raw_input=raw_input,
            tts_text=tts_text,
            prep_trace=prep_trace,
            write_full_text_files=settings.tts_log_full_text_files,
            total_ms=total_ms,
            resolve_ms=resolve_ms,
            sarv_ms=sarv_ms,
            out_bytes=out_bytes,
        )
    except OSError as exc:
        logger.warning("TTS file log failed dir=%r: %s", log_dir, exc)


def _write_stt_log_file(
    log_dir: str,
    *,
    model: str,
    language_code: str,
    backend: str,
    filename: Optional[str],
    content_type: Optional[str],
    upload_bytes: int,
    duration_sec: Optional[float],
    n_chunks: int,
    chunked: bool,
    transcript: str,
    prep_trace: str,
    total_ms: float,
    read_ms: float,
    prep_ms: float,
    stt_ms: float,
    write_full_text_files: bool,
) -> None:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / "stt.log"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    ts_slug = ts.replace(":", "-")
    duration_log = f"{duration_sec:.2f}" if duration_sec is not None else "unknown"
    lines = [
        f"--- {ts} ---",
        f"model={model!r} language_code={language_code!r} backend={backend!r}",
        f"filename={filename!r} content_type={content_type!r} upload_bytes={upload_bytes}",
        f"duration_sec={duration_log} n_chunks={n_chunks} chunked={chunked}",
        f"transcript_len={len(transcript)}",
        f"prep={prep_trace}",
        f"timing_ms total={total_ms:.1f} read={read_ms:.1f} prep={prep_ms:.1f} sarvam={stt_ms:.1f}",
        "--- transcript ---",
        transcript,
        "",
    ]
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    if write_full_text_files:
        full_text_dir = path / "full-text-stt"
        full_text_dir.mkdir(parents=True, exist_ok=True)
        full_text_file = full_text_dir / f"{ts_slug}.txt"
        full_text_file.write_text(
            "\n".join(
                [
                    f"timestamp: {ts}",
                    f"model: {model}",
                    f"language_code: {language_code}",
                    f"backend: {backend}",
                    f"filename: {filename}",
                    f"content_type: {content_type}",
                    f"upload_bytes: {upload_bytes}",
                    f"duration_sec: {duration_log}",
                    f"n_chunks: {n_chunks}",
                    f"chunked: {chunked}",
                    f"prep: {prep_trace}",
                    "",
                    "=== transcript ===",
                    transcript,
                    "",
                ]
            ),
            encoding="utf-8",
        )


async def _append_stt_log(
    settings: Settings,
    *,
    model: str,
    language_code: str,
    backend: str,
    filename: Optional[str],
    content_type: Optional[str],
    upload_bytes: int,
    duration_sec: Optional[float],
    n_chunks: int,
    chunked: bool,
    transcript: str,
    prep_trace: str,
    total_ms: float,
    read_ms: float,
    prep_ms: float,
    stt_ms: float,
) -> None:
    if not settings.tts_log_enabled:
        return
    log_dir = (settings.tts_log_dir or "").strip() or "/app/logs"
    try:
        await asyncio.to_thread(
            _write_stt_log_file,
            log_dir,
            model=model,
            language_code=language_code,
            backend=backend,
            filename=filename,
            content_type=content_type,
            upload_bytes=upload_bytes,
            duration_sec=duration_sec,
            n_chunks=n_chunks,
            chunked=chunked,
            transcript=transcript,
            prep_trace=prep_trace,
            total_ms=total_ms,
            read_ms=read_ms,
            prep_ms=prep_ms,
            stt_ms=stt_ms,
            write_full_text_files=settings.tts_log_full_text_files,
        )
    except OSError as exc:
        logger.warning("STT file log failed dir=%r: %s", log_dir, exc)


async def resolve_tts_language_and_speaker(
    body: SpeechRequest,
    settings: Settings,
    *,
    tts_text: str,
    prep: PrepTrace,
) -> tuple[str, str]:
    """
    If `target_language_code` is set, use it with standard OpenAI→Sarvam voice mapping.
    Otherwise, when TTS_LANG_DETECT_BASE_URL is set, detect language via chat completions
    and pick a recommended speaker; else fall back to DEFAULT_TTS_LANGUAGE / DEFAULT_TTS_SPEAKER.
    """
    t0 = time.perf_counter()
    if body.target_language_code:
        lang = body.target_language_code.strip()
        speaker = _map_openai_voice_to_speaker(body.voice, settings.default_tts_speaker)
        prep.add(
            f"lang:manual(target={lang!r} voice={body.voice!r}→speaker={speaker!r} "
            f"{_elapsed_ms(t0):.1f}ms)"
        )
        return lang, speaker

    base = (settings.tts_lang_detect_base_url or "").strip()
    if base:
        t_detect = time.perf_counter()
        detected = await detect_tts_language_code(
            tts_text,
            base_url=base,
            model=settings.tts_lang_detect_model,
            timeout=settings.tts_lang_detect_timeout,
        )
        detect_ms = _elapsed_ms(t_detect)
        lang = detected or settings.default_tts_language
        speaker = pick_tts_speaker(lang, body.voice, settings.default_tts_gender)
        if detected:
            prep.add(
                f"lang:llm_detect(detected={detected!r}→{lang!r} speaker={speaker!r} "
                f"detect={detect_ms:.1f}ms total={_elapsed_ms(t0):.1f}ms)"
            )
        else:
            prep.add(
                f"lang:llm_detect_fail→default({settings.default_tts_language!r} "
                f"speaker={speaker!r} detect={detect_ms:.1f}ms total={_elapsed_ms(t0):.1f}ms)"
            )
        return lang, speaker

    lang = settings.default_tts_language
    speaker = _map_openai_voice_to_speaker(body.voice, settings.default_tts_speaker)
    prep.add(
        f"lang:no_detector_url(default={lang!r} speaker={speaker!r} {_elapsed_ms(t0):.1f}ms)"
    )
    return lang, speaker


def _normalize_stt_language(language: Optional[str], default_bcp47: str) -> str:
    if not language:
        return default_bcp47
    lang = language.strip()
    if lang.lower() == "unknown":
        return "unknown"
    two = {
        "en": "en-IN",
        "hi": "hi-IN",
        "bn": "bn-IN",
        "gu": "gu-IN",
        "kn": "kn-IN",
        "ml": "ml-IN",
        "mr": "mr-IN",
        "ta": "ta-IN",
        "te": "te-IN",
        "pa": "pa-IN",
    }
    if len(lang) == 2 and lang.lower() in two:
        return two[lang.lower()]
    return lang


# LibreChat / OpenAI clients often send model=whisper-1; Sarvam only accepts its own IDs.
_OPENAI_STT_MODEL_ALIASES = frozenset(
    {
        "whisper-1",
        "whisper",
    }
)


def _resolve_stt_model_for_sarvam(
    request_model: Optional[str],
    default_model: str,
    *,
    prep: Optional[PrepTrace] = None,
) -> str:
    if not request_model or not str(request_model).strip():
        return default_model
    m = str(request_model).strip()
    if m.lower() in _OPENAI_STT_MODEL_ALIASES:
        if prep is not None:
            prep.add(f"model:openai_alias({m!r}→{default_model!r})")
        else:
            logger.info("STT: mapping OpenAI model %r -> Sarvam default %r", m, default_model)
        return default_model
    return m


def _looks_like_riff_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _ffmpeg_to_wav_16k_mono(raw: bytes, *, input_suffix: str = ".audio") -> bytes:
    """Decode/transcode arbitrary audio (webm, mp3, …) to 16 kHz mono WAV via ffmpeg.

    Writes to a temp file — piping WAV to stdout leaves data chunk size 0xFFFFFFFF,
    which Sarvam reads as ~134k seconds and rejects.
    """
    with tempfile.NamedTemporaryFile(suffix=input_suffix) as inp:
        inp.write(raw)
        inp.flush()
        with tempfile.NamedTemporaryFile(suffix=".wav") as out:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    inp.name,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    out.name,
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
            if proc.returncode != 0:
                err = (proc.stderr or b"").decode("utf-8", errors="replace")[:1200]
                raise RuntimeError(err or f"ffmpeg failed with exit code {proc.returncode}")
            out.seek(0)
            data = out.read()
    if not data:
        raise RuntimeError("ffmpeg produced empty WAV output")
    return data


def _parse_ffprobe_duration_line(line: str) -> float:
    line = line.strip()
    if not line or line.upper() == "N/A":
        raise ValueError(f"invalid duration: {line!r}")
    duration = float(line)
    if duration <= 0:
        raise ValueError(f"non-positive duration: {duration}")
    return duration


def _ffprobe_duration_from_path(path: str) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:800]
        raise RuntimeError(err or f"ffprobe failed with exit code {proc.returncode}")
    line = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
    return _parse_ffprobe_duration_line(line)


def _wav_duration_from_header(raw: bytes) -> Optional[float]:
    """Best-effort WAV duration from RIFF header when ffprobe on stdin fails."""
    if not _looks_like_riff_wav(raw) or len(raw) < 44:
        return None
    try:
        byte_rate = struct.unpack_from("<I", raw, 28)[0]
        if byte_rate < 8000 or byte_rate > 384_000:
            return None
        offset = 12
        while offset + 8 <= len(raw):
            chunk_id = raw[offset : offset + 4]
            chunk_size = struct.unpack_from("<I", raw, offset + 4)[0]
            if chunk_id == b"data":
                if chunk_size <= 0 or chunk_size > len(raw):
                    return None
                duration = chunk_size / byte_rate
                if duration <= 0:
                    return None
                return duration
            if chunk_size > len(raw):
                return None
            offset += 8 + chunk_size
    except (struct.error, ZeroDivisionError):
        return None
    return None


def _max_plausible_duration_seconds(byte_len: int, *, configured_max: float) -> float:
    """Upper bound from file size (assumes ≥2 kbit/s effective bitrate for compressed voice)."""
    if byte_len <= 0:
        return 0.0
    from_size = (byte_len * 8) / 2000.0
    return max(configured_max * 1.5, from_size)


def _is_plausible_stt_duration(
    duration_sec: float,
    byte_len: int,
    *,
    configured_max: float,
) -> bool:
    if duration_sec <= 0:
        return False
    if duration_sec > _max_plausible_duration_seconds(byte_len, configured_max=configured_max):
        return False
    return True


def _guess_probe_suffix(filename: Optional[str], content_type: Optional[str]) -> str:
    ct = (content_type or "").split(";")[0].strip().lower()
    by_type = {
        "video/webm": ".webm",
        "audio/webm": ".webm",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".m4a",
        "video/mp4": ".mp4",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
    }
    if ct in by_type:
        return by_type[ct]
    if filename:
        suf = Path(filename).suffix.lower()
        if suf in {".webm", ".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".flac"}:
            return suf
    return ".audio"


def _ffmpeg_decode_duration_seconds(raw: bytes) -> float:
    """Decode via ffmpeg to WAV and measure duration (works when container metadata is N/A)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not available")
    wav = _ffmpeg_to_wav_16k_mono(raw)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(wav)
        tmp.flush()
        return _ffprobe_duration_from_path(tmp.name)


def _ffprobe_duration_seconds(
    raw: bytes,
    *,
    upload_filename: Optional[str] = None,
    content_type: Optional[str] = None,
    prep: Optional[PrepTrace] = None,
) -> float:
    """Return audio duration in seconds; tries ffprobe, then ffmpeg decode."""
    errors: list[str] = []
    suffix = _guess_probe_suffix(upload_filename, content_type)

    if shutil.which("ffprobe"):
        for label, args in (
            ("stdin", ["pipe:0"]),
            ("stdin_dash", ["-i", "-"]),
        ):
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    *args,
                ],
                input=raw,
                capture_output=True,
                timeout=30,
                check=False,
            )
            line = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
            if proc.returncode == 0 and line:
                try:
                    duration = _parse_ffprobe_duration_line(line)
                    if prep is not None:
                        prep.add(f"duration:ffprobe_{label}_ok({duration:.2f}s)")
                    return duration
                except ValueError as exc:
                    msg = f"ffprobe {label}: {exc}"
                    errors.append(msg)
                    if prep is not None:
                        prep.add(f"duration:ffprobe_{label}_fail({exc})")
            else:
                err = (proc.stderr or b"").decode("utf-8", errors="replace")[:400]
                msg = f"ffprobe {label}: exit={proc.returncode} line={line!r} stderr={err!r}"
                errors.append(msg)
                if prep is not None:
                    prep.add(f"duration:ffprobe_{label}_fail(exit={proc.returncode})")

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(raw)
                tmp.flush()
                duration = _ffprobe_duration_from_path(tmp.name)
                if prep is not None:
                    prep.add(f"duration:ffprobe_tempfile{suffix}_ok({duration:.2f}s)")
                return duration
        except (RuntimeError, ValueError) as exc:
            msg = f"ffprobe tempfile({suffix}): {exc}"
            errors.append(msg)
            if prep is not None:
                prep.add(f"duration:ffprobe_tempfile{suffix}_fail({exc})")

    wav_duration = _wav_duration_from_header(raw)
    if wav_duration is not None:
        if prep is not None:
            prep.add(f"duration:wav_header_ok({wav_duration:.2f}s)")
        return wav_duration

    if shutil.which("ffmpeg"):
        try:
            duration = _ffmpeg_decode_duration_seconds(raw)
            if prep is not None:
                prep.add(f"duration:ffmpeg_decode_ok({duration:.2f}s)")
            return duration
        except (RuntimeError, ValueError) as exc:
            msg = f"ffmpeg decode duration: {exc}"
            errors.append(msg)
            if prep is not None:
                prep.add(f"duration:ffmpeg_decode_fail({exc})")

    raise RuntimeError("; ".join(errors) or "ffprobe/ffmpeg not available")


async def _ffmpeg_decode_duration_seconds_async(
    raw: bytes,
    *,
    max_audio_seconds: float,
    prep: Optional[PrepTrace] = None,
) -> Optional[float]:
    try:
        duration = await asyncio.to_thread(_ffmpeg_decode_duration_seconds, raw)
    except RuntimeError as exc:
        if prep is not None:
            prep.add(f"duration:ffmpeg_decode_retry_fail({exc})")
        else:
            logger.warning("STT ffmpeg decode duration failed: %s", exc)
        return None
    if not _is_plausible_stt_duration(
        duration,
        len(raw),
        configured_max=max_audio_seconds,
    ):
        if prep is not None:
            prep.add(
                f"duration:ffmpeg_decode_retry_implausible({duration:.2f}s "
                f"bytes={len(raw)})"
            )
        else:
            logger.warning(
                "STT ffmpeg decode duration implausible duration_sec=%.2f upload_bytes=%d",
                duration,
                len(raw),
            )
        return None
    if prep is not None:
        prep.add(f"duration:ffmpeg_decode_retry_ok({duration:.2f}s)")
    return duration


def _is_sarvam_stt_duration_limit_error(exc: HTTPException) -> bool:
    if exc.status_code != 400:
        return False
    detail = str(exc.detail).lower()
    return "30 second" in detail or (
        "maximum limit" in detail and "duration" in detail
    )


async def _probe_stt_duration_seconds(
    raw: bytes,
    *,
    upload_filename: Optional[str],
    content_type: Optional[str],
    max_audio_seconds: float,
    prep: PrepTrace,
) -> Optional[float]:
    """Probe upload duration; log and return None on failure or implausible values."""
    try:
        duration_sec = await asyncio.to_thread(
            _ffprobe_duration_seconds,
            raw,
            upload_filename=upload_filename,
            content_type=content_type,
            prep=prep,
        )
    except RuntimeError as exc:
        prep.add(f"duration:probe_failed({exc})")
        logger.warning(
            "STT duration probe failed upload_bytes=%d filename=%r content_type=%r: %s",
            len(raw),
            upload_filename,
            content_type,
            exc,
        )
        return None

    if not _is_plausible_stt_duration(
        duration_sec,
        len(raw),
        configured_max=max_audio_seconds,
    ):
        prep.add(
            f"duration:implausible({duration_sec:.2f}s)→unknown "
            f"(max_plausible={_max_plausible_duration_seconds(len(raw), configured_max=max_audio_seconds):.1f}s)"
        )
        logger.warning(
            "STT duration probe implausible duration_sec=%.2f upload_bytes=%d "
            "max_plausible=%.1f filename=%r — treating as unknown",
            duration_sec,
            len(raw),
            _max_plausible_duration_seconds(len(raw), configured_max=max_audio_seconds),
            upload_filename,
        )
        return None

    prep.add(f"duration:probe_ok({duration_sec:.2f}s)")
    logger.info(
        "STT duration probe: duration_sec=%.2f upload_bytes=%d filename=%r content_type=%r",
        duration_sec,
        len(raw),
        upload_filename,
        content_type,
    )
    return duration_sec


def _sarvam_safe_chunk_seconds(chunk_seconds: float) -> float:
    """Sarvam REST rejects segments at the 30s boundary; stay slightly under."""
    return min(chunk_seconds, 29.0)


def _wav_duration_via_tempfile(wav_bytes: bytes) -> float:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        return _ffprobe_duration_from_path(tmp.name)


def _ensure_wav_max_duration(
    wav_bytes: bytes,
    max_sec: float,
    *,
    prep: Optional[PrepTrace] = None,
    chunk_index: Optional[int] = None,
) -> bytes:
    """Re-trim WAV when ffmpeg produced a segment Sarvam would reject."""
    try:
        duration = _wav_duration_via_tempfile(wav_bytes)
    except (RuntimeError, ValueError):
        return wav_bytes
    if duration <= max_sec + 0.05:
        return wav_bytes
    chunk_label = f"chunk={chunk_index} " if chunk_index is not None else ""
    if prep is not None:
        prep.add(f"chunk_trim:{chunk_label}{duration:.2f}s→{max_sec:.1f}s")
    logger.warning(
        "STT chunk trim: segment duration_sec=%.2f exceeds max=%.1f — re-encoding",
        duration,
        max_sec,
    )
    with tempfile.NamedTemporaryFile(suffix=".wav") as inp:
        inp.write(wav_bytes)
        inp.flush()
        with tempfile.NamedTemporaryFile(suffix=".wav") as out:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    inp.name,
                    "-t",
                    str(max_sec),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    out.name,
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
            if proc.returncode != 0:
                return wav_bytes
            out.seek(0)
            data = out.read()
    return data or wav_bytes


def _ffmpeg_extract_wav_segment(
    raw: bytes,
    start_sec: float,
    duration_sec: float,
    *,
    input_suffix: str = ".audio",
) -> bytes:
    """Extract a segment as 16 kHz mono WAV via ffmpeg (file output for valid RIFF headers)."""
    with tempfile.NamedTemporaryFile(suffix=input_suffix) as inp:
        inp.write(raw)
        inp.flush()
        with tempfile.NamedTemporaryFile(suffix=".wav") as out:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(start_sec),
                    "-t",
                    str(duration_sec),
                    "-i",
                    inp.name,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    out.name,
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
            if proc.returncode != 0:
                err = (proc.stderr or b"").decode("utf-8", errors="replace")[:1200]
                raise RuntimeError(err or f"ffmpeg failed with exit code {proc.returncode}")
            out.seek(0)
            data = out.read()
    if not data:
        raise RuntimeError("ffmpeg produced empty WAV segment")
    return data


def _split_audio_into_chunks(
    raw: bytes,
    chunk_seconds: float,
    *,
    duration_sec: float,
    input_suffix: str = ".audio",
    prep: Optional[PrepTrace] = None,
) -> list[bytes]:
    """Split audio into WAV segments safely under Sarvam's 30s REST limit."""
    safe_chunk = _sarvam_safe_chunk_seconds(chunk_seconds)
    if prep is not None:
        prep.add(
            f"split:start(duration={duration_sec:.2f}s safe_chunk={safe_chunk:.1f}s "
            f"suffix={input_suffix!r})"
        )
    chunks: list[bytes] = []
    start = 0.0
    chunk_index = 0
    while start < duration_sec:
        seg_len = min(safe_chunk, duration_sec - start)
        if seg_len <= 0:
            break
        chunk_index += 1
        wav = _ffmpeg_extract_wav_segment(raw, start, seg_len, input_suffix=input_suffix)
        wav = _ensure_wav_max_duration(
            wav,
            safe_chunk,
            prep=prep,
            chunk_index=chunk_index,
        )
        chunks.append(wav)
        start += safe_chunk
    if not chunks:
        raise RuntimeError("ffmpeg produced no audio segments")
    if prep is not None:
        prep.add(f"split:done(n_chunks={len(chunks)})")
    return chunks


def _require_ffmpeg_for_stt_chunking() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    raise HTTPException(
        status_code=503,
        detail=(
            "Audio longer than the Sarvam REST limit requires `ffmpeg` and `ffprobe` on the "
            "voice-services host to split uploads. Install them or upload shorter audio."
        ),
    )


async def _prepare_audio_for_sarvam_stt(
    raw: bytes,
    *,
    sample_rate: Optional[int],
    settings: Settings,
) -> tuple[bytes, int]:
    """
    Sarvam streaming STT `AudioData` only allows encoding ``audio/wav`` (SDK literal).

    Browsers often send ``video/webm`` from MediaRecorder; normalize with ffmpeg when
    available, otherwise require a real WAV file.
    """
    if shutil.which("ffmpeg"):
        try:
            t0 = time.perf_counter()
            out = await asyncio.to_thread(_ffmpeg_to_wav_16k_mono, raw)
            ms = _elapsed_ms(t0)
            logger.info(
                "STT prep: ffmpeg -> wav16k mono in_bytes=%d out_bytes=%d (%.1f ms)",
                len(raw),
                len(out),
                ms,
            )
            return out, 16000
        except RuntimeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode audio to WAV (ffmpeg): {exc}",
            ) from exc

    if _looks_like_riff_wav(raw):
        sr = int(sample_rate) if sample_rate is not None else settings.stt_sample_rate
        logger.info(
            "STT prep: passthrough RIFF WAV bytes=%d sample_rate=%d (no ffmpeg)",
            len(raw),
            sr,
        )
        return raw, sr

    raise HTTPException(
        status_code=503,
        detail=(
            "Sarvam STT expects WAV; this upload is not RIFF WAV (e.g. webm from the browser). "
            "Install `ffmpeg` on the voice-services host to transcode automatically, or upload WAV."
        ),
    )


# --- Sarvam helpers ----------------------------------------------------------------


def _sarvam_tts_max_chars(model: str) -> int:
    """Sarvam REST limits: bulbul:v3 → 2500, bulbul:v2 → 1500."""
    if "v2" in model.lower():
        return 1500
    return 2500


def _effective_tts_max_chars(model: str, settings: Settings) -> int:
    if settings.tts_max_chars_per_request is not None:
        return settings.tts_max_chars_per_request
    return _sarvam_tts_max_chars(model)


def _chunk_text_for_tts(text: str, max_len: int) -> list[str]:
    """Split long text at sentence/space boundaries (Sarvam REST char limit)."""
    chunks: list[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        break_at = remaining.rfind(".", 0, max_len)
        if break_at == -1:
            break_at = remaining.rfind(" ", 0, max_len)
        if break_at == -1:
            chunks.append(remaining[:max_len].strip())
            remaining = remaining[max_len:].strip()
            continue
        piece = remaining[: break_at + 1].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[break_at + 1 :].strip()
    return chunks


async def _sarvam_tts_rest_mp3_single(
    client: AsyncSarvamAI,
    *,
    model: str,
    text: str,
    speaker: str,
    target_language_code: str,
    pace: float,
) -> bytes:
    """Single POST ``/text-to-speech`` — one Sarvam REST call (no chunking)."""
    t0 = time.perf_counter()
    try:
        resp = await client.text_to_speech.convert(
            text=text,
            target_language_code=target_language_code,
            speaker=speaker,
            pace=pace,
            model=model,
            output_audio_codec="mp3",
        )
    except ApiError as exc:
        logger.warning(
            "TTS Sarvam REST error: status=%s body=%r",
            exc.status_code,
            exc.body,
        )
        code = exc.status_code
        try:
            c_int = int(code) if code is not None else 502
        except (TypeError, ValueError):
            c_int = 502
        raise HTTPException(
            status_code=c_int if 400 <= c_int < 600 else 502,
            detail=str(exc.body or exc),
        ) from exc

    chunks: list[bytes] = []
    for b64 in resp.audios or []:
        if b64:
            chunks.append(base64.standard_b64decode(b64))
    out = b"".join(chunks)
    if not out:
        raise HTTPException(status_code=502, detail="Sarvam TTS returned no audio (empty audios list).")
    logger.info(
        "TTS Sarvam REST: model=%r lang=%r speaker=%r out_bytes=%d text_chars=%d (%.1f ms)",
        model,
        target_language_code,
        speaker,
        len(out),
        len(text),
        _elapsed_ms(t0),
    )
    return out


async def _sarvam_tts_rest_mp3_bytes(
    client: AsyncSarvamAI,
    *,
    model: str,
    text: str,
    speaker: str,
    target_language_code: str,
    pace: float,
    max_chars: int,
) -> bytes:
    """Sarvam REST TTS; splits text when it exceeds ``max_chars`` and concatenates MP3."""
    text_chunks = _chunk_text_for_tts(text, max_chars)
    if len(text_chunks) == 1:
        return await _sarvam_tts_rest_mp3_single(
            client,
            model=model,
            text=text_chunks[0],
            speaker=speaker,
            target_language_code=target_language_code,
            pace=pace,
        )

    t0 = time.perf_counter()
    parts: list[bytes] = []
    for i, chunk in enumerate(text_chunks, start=1):
        part = await _sarvam_tts_rest_mp3_single(
            client,
            model=model,
            text=chunk,
            speaker=speaker,
            target_language_code=target_language_code,
            pace=pace,
        )
        parts.append(part)
    out = b"".join(parts)
    logger.info(
        "TTS Sarvam REST chunked: n_chunks=%d total_chars=%d out_bytes=%d model=%r (%.1f ms)",
        len(text_chunks),
        len(text),
        len(out),
        model,
        _elapsed_ms(t0),
    )
    return out


async def sarvam_tts_bytes(
    client: AsyncSarvamAI,
    *,
    model: str,
    text: str,
    speaker: str,
    target_language_code: str,
    pace: float,
    response_format: str,
    max_chars: int,
) -> tuple[bytes, str]:
    if response_format not in ("mp3", "mpeg"):
        raise HTTPException(
            status_code=400,
            detail="Sarvam TTS currently returns MP3; use response_format=mp3.",
        )
    audio = await _sarvam_tts_rest_mp3_bytes(
        client,
        model=model,
        text=text,
        speaker=speaker,
        target_language_code=target_language_code,
        pace=pace,
        max_chars=max_chars,
    )
    return audio, "audio/mpeg"


async def sarvam_tts_stream(
    client: AsyncSarvamAI,
    *,
    model: str,
    text: str,
    speaker: str,
    target_language_code: str,
    pace: float,
    response_format: str,
    max_chars: int,
) -> AsyncIterator[bytes]:
    if response_format not in ("mp3", "mpeg"):
        raise HTTPException(
            status_code=400,
            detail="Sarvam TTS currently returns MP3; use response_format=mp3.",
        )
    text_chunks = _chunk_text_for_tts(text, max_chars)
    if len(text_chunks) > 1:
        logger.info(
            "TTS Sarvam stream chunked: n_chunks=%d total_chars=%d model=%r",
            len(text_chunks),
            len(text),
            model,
        )
    for chunk in text_chunks:
        audio = await _sarvam_tts_rest_mp3_single(
            client,
            model=model,
            text=chunk,
            speaker=speaker,
            target_language_code=target_language_code,
            pace=pace,
        )
        yield audio


async def sarvam_stt_transcript_simple_http(
    *,
    settings: Settings,
    raw: bytes,
    upload_filename: Optional[str],
    content_type: Optional[str],
    model: str,
    language_code: str,
    mode: str,
) -> str:
    """
    Minimal Sarvam file STT: multipart ``file`` + ``api-subscription-key`` header
    (same idea as ``requests.post(SARVAM_URL, data=..., files=...)``).
    Sends the **original** upload bytes and content type (no ffmpeg re-wrap for REST).
    """
    t0 = time.perf_counter()
    url = settings.sarvam_stt_rest_url.rstrip("/")
    name = upload_filename or "audio"
    ct = (content_type or "application/octet-stream").split(";")[0].strip()
    data: dict[str, str] = {"model": model, "language_code": language_code}
    if model.lower().startswith("saaras"):
        data["mode"] = mode
    headers = {"api-subscription-key": settings.sarvam_api_key}
    files = {"file": (name, raw, ct)}
    timeout = httpx.Timeout(120.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as http:
        r = await http.post(url, headers=headers, data=data, files=files)
    try:
        payload: Any = r.json()
    except Exception:
        payload = None
    if r.status_code < 200 or r.status_code >= 300:
        logger.warning(
            "STT Sarvam REST HTTP error: status=%s body=%r",
            r.status_code,
            (r.text or "")[:2000],
        )
        detail = payload if isinstance(payload, dict) else (r.text or "")[:2000]
        code = r.status_code if 400 <= r.status_code < 600 else 502
        raise HTTPException(status_code=code, detail=str(detail)) from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Sarvam STT returned non-JSON body")
    err = payload.get("error")
    if err:
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise HTTPException(status_code=400, detail=msg or str(payload))
    text = (payload.get("transcript") or "").strip()
    logger.info(
        "STT Sarvam REST: model=%r language_code=%r transcript_len=%d upload_bytes=%d "
        "filename=%r content_type=%r (%.1f ms)",
        model,
        language_code,
        len(text),
        len(raw),
        name,
        ct,
        _elapsed_ms(t0),
    )
    return text


def _merge_stt_transcripts(parts: list[str]) -> str:
    return " ".join(t.strip() for t in parts if t.strip()).strip()


async def sarvam_stt_transcript_chunked(
    *,
    settings: Settings,
    raw: bytes,
    model: str,
    language_code: str,
    mode: str,
    duration_sec: float,
    input_suffix: str = ".audio",
    prep: Optional[PrepTrace] = None,
) -> tuple[str, int]:
    """
    Split audio longer than ``stt_chunk_seconds`` into WAV segments, transcribe each
    via Sarvam REST sequentially, and merge transcripts.
    """
    _require_ffmpeg_for_stt_chunking()
    safe_chunk = _sarvam_safe_chunk_seconds(settings.stt_chunk_seconds)
    t0 = time.perf_counter()
    chunks = await asyncio.to_thread(
        _split_audio_into_chunks,
        raw,
        settings.stt_chunk_seconds,
        duration_sec=duration_sec,
        input_suffix=input_suffix,
        prep=prep,
    )
    split_ms = _elapsed_ms(t0)
    logger.info(
        "STT chunk split: duration_sec=%.2f n_chunks=%d chunk_limit=%.1f safe_chunk=%.1f "
        "delay_sec=%.1f (%.1f ms)",
        duration_sec,
        len(chunks),
        settings.stt_chunk_seconds,
        safe_chunk,
        settings.stt_chunk_delay_seconds,
        split_ms,
    )

    parts: list[str] = []
    t_stt = time.perf_counter()
    for index, chunk in enumerate(chunks, start=1):
        if index > 1 and settings.stt_chunk_delay_seconds > 0:
            if prep is not None:
                prep.add(
                    f"chunk_delay:{settings.stt_chunk_delay_seconds}s "
                    f"before_chunk={index}/{len(chunks)}"
                )
            await asyncio.sleep(settings.stt_chunk_delay_seconds)
        try:
            chunk_duration = _wav_duration_via_tempfile(chunk)
        except (RuntimeError, ValueError):
            chunk_duration = -1.0
        if prep is not None:
            prep.add(
                f"chunk_stt:{index}/{len(chunks)} "
                f"wav_bytes={len(chunk)} duration_sec={chunk_duration:.2f}"
            )
        logger.info(
            "STT chunk %d/%d: wav_bytes=%d duration_sec=%.2f",
            index,
            len(chunks),
            len(chunk),
            chunk_duration,
        )
        parts.append(
            await sarvam_stt_transcript_simple_http(
                settings=settings,
                raw=chunk,
                upload_filename=f"chunk_{index:03d}.wav",
                content_type="audio/wav",
                model=model,
                language_code=language_code,
                mode=mode,
            )
        )
    text = _merge_stt_transcripts(parts)
    logger.info(
        "STT Sarvam REST chunked: duration_sec=%.2f n_chunks=%d transcript_len=%d (%.1f ms)",
        duration_sec,
        len(chunks),
        len(text),
        _elapsed_ms(t_stt),
    )
    return text, len(chunks)


async def sarvam_stt_transcript_ws_chunked(
    client: AsyncSarvamAI,
    *,
    settings: Settings,
    raw: bytes,
    model: str,
    language_code: str,
    mode: str,
    high_vad: bool,
    duration_sec: float,
    input_suffix: str = ".audio",
    prep: Optional[PrepTrace] = None,
) -> tuple[str, int]:
    """Split long audio and transcribe each segment via Sarvam WebSocket."""
    _require_ffmpeg_for_stt_chunking()
    safe_chunk = _sarvam_safe_chunk_seconds(settings.stt_chunk_seconds)
    t0 = time.perf_counter()
    chunks = await asyncio.to_thread(
        _split_audio_into_chunks,
        raw,
        settings.stt_chunk_seconds,
        duration_sec=duration_sec,
        input_suffix=input_suffix,
        prep=prep,
    )
    split_ms = _elapsed_ms(t0)
    logger.info(
        "STT WS chunk split: duration_sec=%.2f n_chunks=%d chunk_limit=%.1f safe_chunk=%.1f "
        "delay_sec=%.1f (%.1f ms)",
        duration_sec,
        len(chunks),
        settings.stt_chunk_seconds,
        safe_chunk,
        settings.stt_chunk_delay_seconds,
        split_ms,
    )

    parts: list[str] = []
    t_stt = time.perf_counter()
    for index, chunk in enumerate(chunks, start=1):
        if index > 1 and settings.stt_chunk_delay_seconds > 0:
            await asyncio.sleep(settings.stt_chunk_delay_seconds)
        audio_b64 = base64.b64encode(chunk).decode("utf-8")
        parts.append(
            await sarvam_stt_transcript(
                client,
                model=model,
                language_code=language_code,
                mode=mode,
                high_vad=high_vad,
                audio_b64=audio_b64,
                encoding="audio/wav",
                sample_rate=16000,
            )
        )
    text = _merge_stt_transcripts(parts)
    logger.info(
        "STT Sarvam WS chunked: duration_sec=%.2f n_chunks=%d transcript_len=%d (%.1f ms)",
        duration_sec,
        len(chunks),
        len(text),
        _elapsed_ms(t_stt),
    )
    return text, len(chunks)


async def sarvam_stt_transcript(
    client: AsyncSarvamAI,
    *,
    model: str,
    language_code: str,
    mode: str,
    high_vad: bool,
    audio_b64: str,
    encoding: str,
    sample_rate: int,
) -> str:
    transcripts: list[str] = []
    t0 = time.perf_counter()
    recv_count = 0
    async with client.speech_to_text_streaming.connect(
        model=model,
        mode=mode,
        language_code=language_code,
        high_vad_sensitivity=high_vad,
        sample_rate=str(sample_rate),
    ) as ws:
        await ws.transcribe(audio=audio_b64, encoding=encoding, sample_rate=sample_rate)
        await ws.flush()
        idle_rounds = 0
        while idle_rounds < 3:
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=8.0)
                recv_count += 1
            except asyncio.TimeoutError:
                idle_rounds += 1
                continue
            if response.type == "error":
                data: Any = response.data
                msg = getattr(data, "error", None) or str(data)
                raise HTTPException(status_code=502, detail=msg)
            if getattr(response.data, "transcript", None):
                transcripts.append(response.data.transcript)
                idle_rounds = 0
            else:
                idle_rounds += 1
    text = _merge_stt_transcripts(transcripts)
    logger.info(
        "STT Sarvam WS: model=%r language_code=%r recv_messages=%d transcript_len=%d "
        "audio_b64_len=%d (%.1f ms)",
        model,
        language_code,
        recv_count,
        len(text),
        len(audio_b64),
        _elapsed_ms(t0),
    )
    return text


# --- FastAPI -----------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    _install_voice_service_log_handlers()

    settings = get_settings()
    if settings.log_full_stt_transcript:
        logger.info(
            "LOG_FULL_STT_TRANSCRIPT enabled — full STT text will be logged per request "
            "(env file %s)",
            _VOICE_ENV_FILE,
        )
    if settings.log_full_tts_text:
        logger.info(
            "LOG_FULL_TTS_TEXT enabled — full TTS input text will be logged per /v1/audio/speech "
            "(env file %s)",
            _VOICE_ENV_FILE,
        )
    if settings.tts_log_enabled:
        log_dir = (settings.tts_log_dir or "").strip() or "/app/logs"
        logger.info(
            "TTS_LOG_ENABLED — file logs under %r/tts.log and %r/stt.log "
            "(per-request full text in %r/full-text/ and %r/full-text-stt/ when "
            "TTS_LOG_FULL_TEXT_FILES=true)",
            log_dir,
            log_dir,
            log_dir,
            log_dir,
        )
    try:
        app.state.sarvam = AsyncSarvamAI(api_subscription_key=settings.sarvam_api_key)
    except Exception as exc:
        raise RuntimeError(
            "Failed to create AsyncSarvamAI. Set SARVAM_API_KEY in `.env`."
        ) from exc
    yield
    # httpx client is owned by the SDK; no explicit close in public API.


app = FastAPI(title="Sarvam Voice (OpenAI-compatible)", lifespan=lifespan)


def get_client(request: Request) -> AsyncSarvamAI:
    return request.app.state.sarvam


SettingsDep = Annotated[Settings, Depends(get_settings)]
ClientDep = Annotated[AsyncSarvamAI, Depends(get_client)]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(settings: SettingsDep) -> dict[str, Any]:
    """Minimal model list for clients that probe `/v1/models`."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.default_tts_model,
                "object": "model",
                "created": 0,
                "owned_by": "sarvam-ai",
            },
            {
                "id": settings.default_stt_model,
                "object": "model",
                "created": 0,
                "owned_by": "sarvam-ai",
            },
        ],
    }


@app.post("/v1/audio/speech")
async def create_speech(
    body: SpeechRequest,
    settings: SettingsDep,
    client: ClientDep,
):
    req_t0 = time.perf_counter()
    model = body.model or settings.default_tts_model
    prep = PrepTrace()
    tts_text = _prepare_tts_text(body.input, settings, prep=prep)
    if not tts_text.strip():
        prep.add(f"output:silent_mp3(empty_after_sanitization in_chars={len(body.input)})")
        logger.info(
            "TTS skipped: no speakable text after sanitization (in_chars=%d) prep=%s",
            len(body.input),
            prep.format(),
        )
        await _append_tts_log(
            settings,
            model=model,
            lang="",
            speaker="",
            stream=body.stream,
            raw_input=body.input,
            tts_text="",
            prep_trace=prep.format(),
            total_ms=_elapsed_ms(req_t0),
            resolve_ms=_elapsed_ms(req_t0),
            sarv_ms=0.0,
            out_bytes=len(_SILENT_MP3_BYTES),
        )
        return Response(content=_SILENT_MP3_BYTES, media_type="audio/mpeg")
    lang, speaker = await resolve_tts_language_and_speaker(
        body, settings, tts_text=tts_text, prep=prep
    )
    resolve_ms = _elapsed_ms(req_t0)
    pace = _pace_from_openai_speed(body.speed)
    max_chars = _effective_tts_max_chars(model, settings)
    text_chunks = _chunk_text_for_tts(tts_text, max_chars)
    if len(text_chunks) > 1:
        prep.add(f"tts:text_split(n={len(text_chunks)} max_chars={max_chars})")

    if settings.log_full_tts_text:
        logger.info(
            "%s",
            "TTS full text stream=%s model=%r lang=%r speaker=%r prep=%s "
            "full_input_chars=%d text_spoken_chars=%d\n"
            "=== full_input ===\n%s\n"
            "=== text_spoken ===\n%s"
            % (
                body.stream,
                model,
                lang,
                speaker,
                prep.format(),
                len(body.input),
                len(tts_text),
                body.input,
                tts_text,
            ),
        )

    if body.stream:
        prep.add("output:stream_mp3")
        logger.info(
            "TTS request: stream=true model=%r prep=%s resolve=%.1f ms "
            "(Sarvam timing follows on generator close)",
            model,
            prep.format(),
            resolve_ms,
        )
        await _append_tts_log(
            settings,
            model=model,
            lang=lang,
            speaker=speaker,
            stream=True,
            raw_input=body.input,
            tts_text=tts_text,
            prep_trace=prep.format(),
            resolve_ms=resolve_ms,
        )
        return StreamingResponse(
            sarvam_tts_stream(
                client,
                model=model,
                text=tts_text,
                speaker=speaker,
                target_language_code=lang,
                pace=pace,
                response_format=body.response_format,
                max_chars=max_chars,
            ),
            media_type="audio/mpeg",
        )

    t_sarv = time.perf_counter()
    audio, media_type = await sarvam_tts_bytes(
        client,
        model=model,
        text=tts_text,
        speaker=speaker,
        target_language_code=lang,
        pace=pace,
        response_format=body.response_format,
        max_chars=max_chars,
    )
    sarv_ms = _elapsed_ms(t_sarv)
    total_ms = _elapsed_ms(req_t0)
    prep.add(f"output:mp3(bytes={len(audio)})")
    logger.info(
        "TTS request: stream=false total=%.1f ms (resolve+lang %.1f ms, Sarvam %.1f ms) "
        "out_bytes=%d prep=%s",
        total_ms,
        resolve_ms,
        sarv_ms,
        len(audio),
        prep.format(),
    )
    await _append_tts_log(
        settings,
        model=model,
        lang=lang,
        speaker=speaker,
        stream=False,
        raw_input=body.input,
        tts_text=tts_text,
        prep_trace=prep.format(),
        total_ms=total_ms,
        resolve_ms=resolve_ms,
        sarv_ms=sarv_ms,
        out_bytes=len(audio),
    )
    return Response(content=audio, media_type=media_type)


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    settings: SettingsDep,
    client: ClientDep,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None),
    sample_rate: Optional[int] = Form(None),
):
    del prompt, temperature  # OpenAI fields; not passed to Sarvam STT here.

    req_t0 = time.perf_counter()
    t_read = time.perf_counter()
    raw = await file.read()
    read_ms = _elapsed_ms(t_read)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    prep = PrepTrace()
    lang = _normalize_stt_language(language, settings.default_stt_language)
    m = _resolve_stt_model_for_sarvam(model, settings.default_stt_model, prep=prep)
    input_suffix = _guess_probe_suffix(file.filename, file.content_type)
    prep.add(f"input:filename={file.filename!r} suffix={input_suffix!r} bytes={len(raw)}")

    t_probe = time.perf_counter()
    duration_sec = await _probe_stt_duration_seconds(
        raw,
        upload_filename=file.filename,
        content_type=file.content_type,
        max_audio_seconds=settings.stt_max_audio_seconds,
        prep=prep,
    )
    probe_ms = _elapsed_ms(t_probe)

    if duration_sec is not None and duration_sec > settings.stt_max_audio_seconds:
        prep.add(
            f"reject:duration_exceeds_max({duration_sec:.2f}s>{settings.stt_max_audio_seconds}s)"
        )
        logger.warning(
            "STT rejected: duration_sec=%.2f exceeds max=%.1f upload_bytes=%d filename=%r prep=%s",
            duration_sec,
            settings.stt_max_audio_seconds,
            len(raw),
            file.filename,
            prep.format(),
        )
        raise HTTPException(
            status_code=400,
            detail=f"Audio exceeds maximum length of {settings.stt_max_audio_seconds}s.",
        )

    chunked = duration_sec is not None and duration_sec > settings.stt_chunk_seconds
    if chunked:
        prep.add(f"path:chunked(duration={duration_sec:.2f}s>{settings.stt_chunk_seconds}s)")
    elif duration_sec is None:
        prep.add("path:single_call(duration_unknown)")
        logger.info(
            "STT duration unknown — using single-call path (no chunking) upload_bytes=%d filename=%r",
            len(raw),
            file.filename,
        )
    else:
        prep.add(f"path:single_call(duration={duration_sec:.2f}s)")
    n_chunks = 1

    t_stt = time.perf_counter()
    if settings.stt_use_rest:
        prep.add("backend:rest")
        prep_ms = probe_ms
        if chunked:
            text, n_chunks = await sarvam_stt_transcript_chunked(
                settings=settings,
                raw=raw,
                model=m,
                language_code=lang,
                mode=settings.stt_mode,
                duration_sec=duration_sec,
                input_suffix=input_suffix,
                prep=prep,
            )
        else:
            try:
                text = await sarvam_stt_transcript_simple_http(
                    settings=settings,
                    raw=raw,
                    upload_filename=file.filename,
                    content_type=file.content_type,
                    model=m,
                    language_code=lang,
                    mode=settings.stt_mode,
                )
            except HTTPException as exc:
                if not _is_sarvam_stt_duration_limit_error(exc):
                    raise
                prep.add("sarvam:30s_limit_error→chunked_retry")
                retry_duration = duration_sec or await _ffmpeg_decode_duration_seconds_async(
                    raw,
                    max_audio_seconds=settings.stt_max_audio_seconds,
                    prep=prep,
                )
                if retry_duration is None or retry_duration <= settings.stt_chunk_seconds:
                    prep.add("sarvam:chunked_retry_aborted(duration_unavailable_or_short)")
                    raise
                if retry_duration > settings.stt_max_audio_seconds:
                    prep.add(
                        f"reject:retry_duration_exceeds_max({retry_duration:.2f}s)"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"Audio exceeds maximum length of {settings.stt_max_audio_seconds}s.",
                    )
                logger.info(
                    "STT Sarvam 30s limit — retrying chunked duration_sec=%.2f upload_bytes=%d prep=%s",
                    retry_duration,
                    len(raw),
                    prep.format(),
                )
                text, n_chunks = await sarvam_stt_transcript_chunked(
                    settings=settings,
                    raw=raw,
                    model=m,
                    language_code=lang,
                    mode=settings.stt_mode,
                    duration_sec=retry_duration,
                    input_suffix=input_suffix,
                    prep=prep,
                )
                chunked = True
                duration_sec = retry_duration
        backend = "rest"
        stt_body_len = len(raw)
    else:
        prep.add("backend:websocket")
        if not chunked:
            prep.add("prep:ffmpeg_wav16k")
            t_prep = time.perf_counter()
            wav_bytes, sr = await _prepare_audio_for_sarvam_stt(
                raw, sample_rate=sample_rate, settings=settings
            )
            prep_ms = probe_ms + _elapsed_ms(t_prep)
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            text = await sarvam_stt_transcript(
                client,
                model=m,
                language_code=lang,
                mode=settings.stt_mode,
                high_vad=settings.stt_high_vad_sensitivity,
                audio_b64=audio_b64,
                encoding="audio/wav",
                sample_rate=sr,
            )
            backend = "websocket"
            stt_body_len = len(wav_bytes)
        else:
            text, n_chunks = await sarvam_stt_transcript_ws_chunked(
                client,
                settings=settings,
                raw=raw,
                model=m,
                language_code=lang,
                mode=settings.stt_mode,
                high_vad=settings.stt_high_vad_sensitivity,
                duration_sec=duration_sec,
                input_suffix=input_suffix,
                prep=prep,
            )
            prep_ms = probe_ms
            backend = "websocket"
            stt_body_len = len(raw)
    stt_ms = _elapsed_ms(t_stt)
    prep.add(f"done:transcript_len={len(text)} n_chunks={n_chunks}")
    if settings.log_full_stt_transcript:
        # Single %s so '%' characters in the transcript cannot break logging formatting.
        logger.info(
            "%s",
            "STT full transcript (%d chars): %s" % (len(text), text),
        )
    total_ms = _elapsed_ms(req_t0)
    duration_log = f"{duration_sec:.2f}" if duration_sec is not None else "unknown"
    prep_trace = prep.format()
    logger.info(
        "STT request: total=%.1f ms (read_upload %.1f ms, prep %.1f ms, Sarvam %.1f ms) "
        "backend=%s upload_bytes=%d stt_body_bytes=%d language_code=%r "
        "duration_sec=%s n_chunks=%d chunked=%s prep=%s",
        total_ms,
        read_ms,
        prep_ms,
        stt_ms,
        backend,
        len(raw),
        stt_body_len,
        lang,
        duration_log,
        n_chunks,
        chunked,
        prep_trace,
    )
    await _append_stt_log(
        settings,
        model=m,
        language_code=lang,
        backend=backend,
        filename=file.filename,
        content_type=file.content_type,
        upload_bytes=len(raw),
        duration_sec=duration_sec,
        n_chunks=n_chunks,
        chunked=chunked,
        transcript=text,
        prep_trace=prep_trace,
        total_ms=total_ms,
        read_ms=read_ms,
        prep_ms=prep_ms,
        stt_ms=stt_ms,
    )

    rf = (response_format or "json").lower().strip()
    if rf == "json":
        return JSONResponse({"text": text})
    if rf == "text":
        return Response(content=text, media_type="text/plain")
    raise HTTPException(
        status_code=400,
        detail="Supported response_format values: json, text.",
    )
