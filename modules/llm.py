from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import httpx

from config import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_PARAMS, OLLAMA_API_URL
from modules.ai_helper import (
    Timeline,
    call_llm_with_retry,
    _extract_json,
    _repair_json_deterministic,
    LogFn
)

SYSTEM_PROMPT = """
You are a video planning assistant. Convert the user prompt into strict JSON only.
Return JSON with this schema:
{
  "title": "short title",
  "total_duration": number (seconds),
  "music_mood": "string",
  "scenes": [
    {
      "id": integer,
      "duration": number (seconds),
      "voiceover": "string",
      "visual_query": "string",
      "overlay_text": "string"
    }
  ]
}
Rules:
- JSON only. No markdown.
- Use double quotes for all strings.
- No trailing commas.
- Do not wrap in code fences.
- Ensure durations sum to total_duration.
- Make 3-8 scenes.
- The voiceover field should be a brief 1-sentence outline, not the final narration.
""".strip()


class LLMError(Exception):
    pass


async def _list_models(api_url: str) -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{api_url}/api/tags")
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError:
        return []
    models = [
        model.get("name") for model in data.get("models", []) if model.get("name")
    ]
    return models


def _build_payload(
    prompt: str,
    model: str,
    options: Dict[str, Any],
    force_json: bool = True,
    think: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\n\nUser prompt: {prompt}",
        "stream": False,
        "think": think,
        "options": options,
    }
    if force_json:
        payload["format"] = "json"
    return payload


def _schema_hint() -> str:
    return (
        "{"
        '"title": string, '
        '"total_duration": number, '
        '"music_mood": string, '
        '"scenes": ['
        "{"
        '"id": integer, '
        '"duration": number, '
        '"voiceover": string, '
        '"visual_query": string, '
        '"overlay_text": string'
        "}"
        "]"
        "}"
    )


def _fallback_timeline(prompt: str, target_duration: Optional[float]) -> Dict[str, Any]:
    total = float(target_duration or 30)
    per = round(total / 3, 2)
    base = prompt.strip().split("\n")[0]
    base = base[:140] if base else "A short thematic story."
    scenes = []
    for idx in range(1, 4):
        scenes.append(
            {
                "id": idx,
                "duration": per,
                "voiceover": f"Scene {idx}: {base}",
                "visual_query": base.split(".")[0][:60] or "cinematic background",
                "overlay_text": base.split(".")[0][:40] or "Story",
            }
        )
    return {
        "title": base[:60] or "Generated Story",
        "total_duration": total,
        "music_mood": "cinematic",
        "scenes": scenes,
    }


async def plan_timeline(
    prompt: str,
    model: str | None = None,
    options: Dict[str, Any] | None = None,
    api_url: str = OLLAMA_API_URL,
    timeout: float = 180.0,
    think: bool = False,
    helper_settings: Optional[Dict[str, Any]] = None,
    target_duration: Optional[float] = None,
    request_delay: float = 0.8,
    log: Optional[LogFn] = None,
) -> Dict[str, Any]:
    model = model or DEFAULT_OLLAMA_MODEL
    merged_options = {**DEFAULT_OLLAMA_PARAMS, **(options or {})}

    payload = _build_payload(
        prompt,
        model,
        merged_options,
        force_json=True,
        think=think,
    )

    try:
        if request_delay > 0:
            await asyncio.sleep(request_delay)
            
        result = await call_llm_with_retry(
            api_url,
            payload,
            timeout,
            max_retries=3,
            log=log,
            validation_model=Timeline
        )
        
        if isinstance(result, Timeline):
            return result.model_dump()
        return result

    except Exception as exc:
        if log:
            await log("error", f"Timeline planning failed after retries: {exc}")
        return _fallback_timeline(prompt, target_duration)


def _clean_voiceover_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("{") and "voiceover" in cleaned:
        try:
            data = _extract_json(cleaned)
            if isinstance(data, dict) and data.get("voiceover"):
                cleaned = str(data.get("voiceover", "")).strip()
        except Exception:
            pass
    cleaned = re.sub(
        r"^(narrator|voiceover)\s*[:\-]+\s*", "", cleaned, flags=re.IGNORECASE
    )
    cleaned = cleaned.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


async def generate_voiceover(
    scene: Dict[str, Any],
    language: str,
    target_words: int,
    model: str,
    options: Dict[str, Any],
    api_url: str,
    timeout: float,
    think: bool = False,
    add_greeting: bool = False,
    add_closing: bool = False,
    is_first: bool = False,
    is_last: bool = False,
    request_delay: float = 0.8,
    log: Optional[LogFn] = None,
) -> str:
    if target_words <= 0:
        target_words = 30
    min_words = max(10, int(target_words * 0.7))
    max_words = max(min_words + 4, int(target_words * 1.3))

    base_outline = (
        scene.get("voiceover")
        or scene.get("overlay_text")
        or scene.get("visual_query")
        or ""
    )
    visual_query = scene.get("visual_query") or ""
    overlay_text = scene.get("overlay_text") or ""

    local_options = dict(options or {})
    
    greeting_hint = ""
    closing_hint = ""
    if add_greeting and is_first:
        greeting_hint = "Include a short, friendly greeting to the audience at the beginning.\n"
    if add_closing and is_last:
        closing_hint = "Include a short, friendly closing line at the end.\n"

    prompt = (
        "You are a professional video narrator. Write the narration for ONE scene.\n"
        f"Language: {language}.\n"
        f"Target length: about {target_words} words (min {min_words}, max {max_words}).\n"
        "Do not use labels, quotes, or stage directions. Do not say 'narrator'.\n"
        "Return ONLY the narration text.\n\n"
        f"{greeting_hint}{closing_hint}"
        f"Scene outline: {base_outline}\n"
        f"Visuals: {visual_query}\n"
        f"Overlay text: {overlay_text}\n"
    )

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": bool(think),
        "options": local_options,
    }

    try:
        if request_delay > 0:
            await asyncio.sleep(request_delay)
            
        # For voiceover, we don't need JSON validation, just the raw text response
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{api_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            
        raw_text = data.get("response", "") if isinstance(data, dict) else ""
        cleaned = _clean_voiceover_text(raw_text)
        return cleaned or base_outline
        
    except Exception as exc:
        if log:
            await log("error", f"Voiceover generation failed: {exc}")
        return base_outline or "Brief scene."
