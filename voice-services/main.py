"""
OpenAI-compatible audio routes backed by Sarvam AI streaming TTS/STT.

Environment is loaded from `.env` (see `Settings`). The Sarvam SDK also
reads `SARVAM_API_KEY` from the environment if not passed explicitly.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import subprocess
import sys
import time
from contextlib import asynccontextmanager
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


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


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


async def resolve_tts_language_and_speaker(body: SpeechRequest, settings: Settings) -> tuple[str, str]:
    """
    If `target_language_code` is set, use it with standard OpenAI→Sarvam voice mapping.
    Otherwise, when TTS_LANG_DETECT_BASE_URL is set, detect language via chat completions
    and pick a recommended speaker; else fall back to DEFAULT_TTS_LANGUAGE / DEFAULT_TTS_SPEAKER.
    """
    t0 = time.perf_counter()
    if body.target_language_code:
        lang = body.target_language_code.strip()
        speaker = _map_openai_voice_to_speaker(body.voice, settings.default_tts_speaker)
        logger.info(
            "TTS lang: manual target_language_code=%r voice=%r -> speaker=%r (%.1f ms)",
            lang,
            body.voice,
            speaker,
            _elapsed_ms(t0),
        )
        return lang, speaker

    base = (settings.tts_lang_detect_base_url or "").strip()
    if base:
        t_detect = time.perf_counter()
        detected = await detect_tts_language_code(
            body.input,
            base_url=base,
            model=settings.tts_lang_detect_model,
            timeout=settings.tts_lang_detect_timeout,
        )
        detect_ms = _elapsed_ms(t_detect)
        lang = detected or settings.default_tts_language
        speaker = pick_tts_speaker(lang, body.voice, settings.default_tts_gender)
        logger.info(
            "TTS lang: LLM detected=%r resolved=%r speaker=%r gender=%s "
            "(detect %.1f ms, resolve_total %.1f ms)%s",
            detected,
            lang,
            speaker,
            settings.default_tts_gender,
            detect_ms,
            _elapsed_ms(t0),
            "" if detected else " [fallback DEFAULT_TTS_LANGUAGE]",
        )
        return lang, speaker

    lang = settings.default_tts_language
    speaker = _map_openai_voice_to_speaker(body.voice, settings.default_tts_speaker)
    logger.info(
        "TTS lang: no detector URL; default_language=%r speaker=%r (%.1f ms)",
        lang,
        speaker,
        _elapsed_ms(t0),
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


def _resolve_stt_model_for_sarvam(request_model: Optional[str], default_model: str) -> str:
    if not request_model or not str(request_model).strip():
        return default_model
    m = str(request_model).strip()
    if m.lower() in _OPENAI_STT_MODEL_ALIASES:
        logger.info("STT: mapping OpenAI model %r -> Sarvam default %r", m, default_model)
        return default_model
    return m


def _looks_like_riff_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _ffmpeg_to_wav_16k_mono(raw: bytes) -> bytes:
    """Decode/transcode arbitrary audio (webm, mp3, …) to 16 kHz mono WAV via ffmpeg."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            "pipe:1",
        ],
        input=raw,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(err or f"ffmpeg failed with exit code {proc.returncode}")
    if not proc.stdout:
        raise RuntimeError("ffmpeg produced empty WAV output")
    return proc.stdout


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


async def _sarvam_tts_rest_mp3_bytes(
    client: AsyncSarvamAI,
    *,
    model: str,
    text: str,
    speaker: str,
    target_language_code: str,
    pace: float,
) -> bytes:
    """Single POST ``/text-to-speech`` — same contract as Sarvam docs (no WebSocket)."""
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


async def sarvam_tts_bytes(
    client: AsyncSarvamAI,
    *,
    model: str,
    text: str,
    speaker: str,
    target_language_code: str,
    pace: float,
    response_format: str,
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
) -> AsyncIterator[bytes]:
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
    text = " ".join(t.strip() for t in transcripts if t.strip()).strip()
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
    lang, speaker = await resolve_tts_language_and_speaker(body, settings)
    resolve_ms = _elapsed_ms(req_t0)
    pace = _pace_from_openai_speed(body.speed)

    if settings.log_full_tts_text:
        logger.info(
            "%s",
            "TTS full input (%d chars) stream=%s model=%r lang=%r speaker=%r:\n%s"
            % (
                len(body.input),
                body.stream,
                model,
                lang,
                speaker,
                body.input,
            ),
        )

    if body.stream:
        logger.info(
            "TTS request: stream=true model=%r resolve=%.1f ms (Sarvam timing follows on generator close)",
            model,
            resolve_ms,
        )
        return StreamingResponse(
            sarvam_tts_stream(
                client,
                model=model,
                text=body.input,
                speaker=speaker,
                target_language_code=lang,
                pace=pace,
                response_format=body.response_format,
            ),
            media_type="audio/mpeg",
        )

    t_sarv = time.perf_counter()
    audio, media_type = await sarvam_tts_bytes(
        client,
        model=model,
        text=body.input,
        speaker=speaker,
        target_language_code=lang,
        pace=pace,
        response_format=body.response_format,
    )
    sarv_ms = _elapsed_ms(t_sarv)
    total_ms = _elapsed_ms(req_t0)
    logger.info(
        "TTS request: stream=false total=%.1f ms (resolve+lang %.1f ms, Sarvam %.1f ms) out_bytes=%d",
        total_ms,
        resolve_ms,
        sarv_ms,
        len(audio),
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

    lang = _normalize_stt_language(language, settings.default_stt_language)
    m = _resolve_stt_model_for_sarvam(model, settings.default_stt_model)

    t_stt = time.perf_counter()
    if settings.stt_use_rest:
        prep_ms = 0.0
        text = await sarvam_stt_transcript_simple_http(
            settings=settings,
            raw=raw,
            upload_filename=file.filename,
            content_type=file.content_type,
            model=m,
            language_code=lang,
            mode=settings.stt_mode,
        )
        backend = "rest"
        stt_body_len = len(raw)
    else:
        t_prep = time.perf_counter()
        wav_bytes, sr = await _prepare_audio_for_sarvam_stt(
            raw, sample_rate=sample_rate, settings=settings
        )
        prep_ms = _elapsed_ms(t_prep)
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
    stt_ms = _elapsed_ms(t_stt)
    if settings.log_full_stt_transcript:
        # Single %s so '%' characters in the transcript cannot break logging formatting.
        logger.info(
            "%s",
            "STT full transcript (%d chars): %s" % (len(text), text),
        )
    total_ms = _elapsed_ms(req_t0)
    logger.info(
        "STT request: total=%.1f ms (read_upload %.1f ms, prep %.1f ms, Sarvam %.1f ms) "
        "backend=%s upload_bytes=%d stt_body_bytes=%d language_code=%r",
        total_ms,
        read_ms,
        prep_ms,
        stt_ms,
        backend,
        len(raw),
        stt_body_len,
        lang,
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
