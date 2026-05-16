"""
TTS text language detection (OpenAI-compatible chat completions) and
language-code → recommended Sarvam speaker mapping.

**What the model should return (parsable)**  
A single BCP-47 tag from the supported set, with nothing else required. Whitespace around
it is fine. Examples that parse successfully: ``hi-IN``, ``kn-IN\n``.

We scan the assistant message for the first substring matching ``xx-IN`` that is in the
allowlist (e.g. embedded in one sentence like ``Answer: kn-IN`` still works).

**Typical wrong formats (first attempt → no allowlisted tag found)**  
- Prose only, no tag: ``This is Kannada.``  
- JSON / structured: ``{"language": "hi-IN"}``  
- Out-of-list codes: ``de-IN``, ``en-US``, or bare ``hi``.  
- Multiple allowed tags in one reply (we take the **first** match only; ambiguous cases
  should still include one clear tag).

If the first reply cannot be parsed, we **retry once** with a corrective user message
that quotes the previous output and re-states the strict one-token rule.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Literal, Optional

import httpx

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0

# BCP-47 → recommended Sarvam speakers (male / female), first entry is default for that gender.
TTS_LANG_SPEAKERS: dict[str, dict[str, list[str]]] = {
    "en-IN": {"male": ["ratan"], "female": ["ishita"]},
    "hi-IN": {"male": ["shubh", "ashutosh"], "female": ["priya", "suhani"]},
    "te-IN": {"male": ["shubh", "ratan"], "female": ["neha", "priya"]},
    "kn-IN": {"male": ["shubh", "ratan"], "female": ["neha", "ishita"]},
    "bn-IN": {"male": ["rehan"], "female": ["roopa", "suhani"]},
    "ta-IN": {"male": ["ratan", "rohan"], "female": ["ishita", "ritu"]},
    "od-IN": {"male": ["shubh"], "female": ["ritu", "pooja"]},
    "ml-IN": {"male": ["shubh"], "female": ["pooja"]},
    "mr-IN": {"male": ["ratan"], "female": ["priya", "ritu"]},
    "pa-IN": {"male": ["mani"], "female": ["roopa", "suhani"]},
    "gu-IN": {"male": ["ratan"], "female": ["priya", "ritu"]},
}

ALLOWED_TTS_LANG_CODES: frozenset[str] = frozenset(TTS_LANG_SPEAKERS.keys())

_OPENAI_VOICE_ALIASES = frozenset({"alloy", "echo", "fable", "onyx", "nova", "shimmer"})


def _canonical_speaker(lang_code: str, voice: str) -> Optional[str]:
    """If `voice` matches a recommended speaker for `lang_code`, return canonical casing."""
    entry = TTS_LANG_SPEAKERS.get(lang_code)
    if not entry:
        return None
    v = voice.strip().lower()
    for group in ("male", "female"):
        for s in entry[group]:
            if s.lower() == v:
                return s
    return None


def pick_tts_speaker(
    lang_code: str,
    voice: str,
    gender: Literal["male", "female"],
) -> str:
    """
    Choose Sarvam speaker: honor explicit recommended names when valid for `lang_code`;
    map OpenAI built-in voice names to gender-based defaults; otherwise first default for gender.
    """
    entry = TTS_LANG_SPEAKERS.get(lang_code)
    if not entry:
        return voice.strip() or "shubh"

    v = voice.strip()
    if not v:
        return entry["male" if gender == "male" else "female"][0]

    if v.lower() in _OPENAI_VOICE_ALIASES:
        return entry["male" if gender == "male" else "female"][0]

    canon = _canonical_speaker(lang_code, v)
    if canon:
        return canon

    return entry["male" if gender == "male" else "female"][0]


_CODE_IN_TEXT = re.compile(r"\b([a-z]{2}-IN)\b", re.IGNORECASE)


def normalize_lang_code_from_llm(content: str) -> Optional[str]:
    """Extract first allowed xx-IN tag from model output."""
    if not content or not content.strip():
        return None
    for m in _CODE_IN_TEXT.finditer(content):
        cand = m.group(1)
        for code in ALLOWED_TTS_LANG_CODES:
            if cand.lower() == code.lower():
                return code
    return None


def build_detection_prompt(text: str) -> str:
    allowed = " ".join(sorted(ALLOWED_TTS_LANG_CODES))
    # Keep prompt short; model returns a single code.
    snippet = text.strip()
    if len(snippet) > 6000:
        snippet = snippet[:6000] + "\n[truncated]"
    return (
        "Identify the primary language of the following text. "
        "Reply with exactly one tag from this list and nothing else "
        f"(no quotes, no punctuation, no explanation): {allowed}\n\n"
        f"Text:\n{snippet}"
    )


def build_retry_prompt_after_bad_format(text: str, previous_reply: str, *, max_prev: int = 800) -> str:
    """Second user message: show what the model returned and demand a single allowlisted token."""
    allowed = " ".join(sorted(ALLOWED_TTS_LANG_CODES))
    prev = previous_reply.strip().replace("\n", " ")
    if len(prev) > max_prev:
        prev = prev[:max_prev] + "…"
    snippet = text.strip()
    if len(snippet) > 4000:
        snippet = snippet[:4000] + "\n[truncated]"
    return (
        "Your previous reply was not in the required machine-readable format.\n"
        f"Previous reply (truncated): {prev!r}\n\n"
        "Reply with EXACTLY ONE token from this list — no other characters, no JSON, "
        f"no natural language, no quotes: {allowed}\n"
        "Valid: hi-IN\n"
        'Invalid: The language is Hindi (hi-IN).   {"lang":"hi-IN"}   Kannada\n\n'
        "Classify the same text again. Output only the tag:\n\n"
        f"Text:\n{snippet}"
    )


def _message_content_from_completion(data: dict) -> Optional[str]:
    choices = data.get("choices") or []
    if not choices:
        return None
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    return None


async def detect_tts_language_code(
    text: str,
    *,
    base_url: str,
    model: str,
    timeout: float,
) -> Optional[str]:
    """
    Call an OpenAI-compatible `/v1/chat/completions` endpoint and parse a BCP-47 tag.

    On successful HTTP: if the assistant message does not contain a parsable allowlisted
    tag, sends **one** follow-up request with a corrective prompt, then parses again.

    Returns None on HTTP errors or if both attempts are unparsable (caller should fall
    back to configured defaults).
    """
    t_all = time.perf_counter()
    url = base_url.rstrip("/") + "/v1/chat/completions"

    async def _one_shot(client: httpx.AsyncClient, user_content: str) -> Optional[str]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.0,
            "max_tokens": 32,
        }
        t0 = time.perf_counter()
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        raw = _message_content_from_completion(data)
        logger.info(
            "TTS lang detect LLM round: http_ms=%.1f raw_preview=%r",
            _elapsed_ms(t0),
            ((raw[:120] + "…") if len(raw) > 120 else raw) if raw else None,
        )
        return raw

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            t1 = time.perf_counter()
            first_raw = await _one_shot(client, build_detection_prompt(text))
            first_http_ms = _elapsed_ms(t1)
            if first_raw is None:
                logger.warning(
                    "TTS lang detect: empty assistant content (attempt 1) (%.1f ms total)",
                    _elapsed_ms(t_all),
                )
                return None

            first_code = normalize_lang_code_from_llm(first_raw)
            if first_code:
                logger.info(
                    "TTS lang detect: parsed=%r from raw in one shot (http %.1f ms, total %.1f ms)",
                    first_code,
                    first_http_ms,
                    _elapsed_ms(t_all),
                )
                return first_code

            logger.info(
                "TTS lang detect: unparsable first raw=%r — retry (http %.1f ms so far)",
                first_raw[:500] + ("…" if len(first_raw) > 500 else ""),
                first_http_ms,
            )

            t2 = time.perf_counter()
            second_raw = await _one_shot(client, build_retry_prompt_after_bad_format(text, first_raw))
            second_http_ms = _elapsed_ms(t2)
            if second_raw is None:
                logger.warning("TTS lang detect: missing assistant content (attempt 2)")
                return None

            second_code = normalize_lang_code_from_llm(second_raw)
            if second_code:
                logger.info(
                    "TTS lang detect: parsed=%r after retry (2nd http %.1f ms, total %.1f ms)",
                    second_code,
                    second_http_ms,
                    _elapsed_ms(t_all),
                )
                return second_code

            logger.warning(
                "TTS lang detect: failed after retry total=%.1f ms attempt1=%r attempt2=%r",
                _elapsed_ms(t_all),
                first_raw[:200],
                second_raw[:200],
            )
            return None

    except httpx.HTTPError as exc:
        logger.warning("TTS lang detect HTTP error after %.1f ms: %s", _elapsed_ms(t_all), exc)
        return None
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("TTS lang detect parse error after %.1f ms: %s", _elapsed_ms(t_all), exc)
        return None
