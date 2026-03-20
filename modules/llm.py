from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional

import httpx

from config import (
    OLLAMA_SETTINGS,
    LLM_DEFAULT_TIMEOUT,
    LLM_MAX_RETRIES,
    OLLAMA_API_URL,
)
from modules.ai_helper import (
    Timeline,
    call_llm_with_retry,
    _extract_json,
    repair_json,
    LLMResponseError,
    LogFn
)
from modules.utils import clean_and_tokenize

def get_planner_prompt(
    target_duration: float,
    language: str = "English",
    scene_count: Optional[int] = None,
    add_greeting: bool = False,
    add_closing: bool = False,
    words_per_sec: float = 2.0,
) -> str:
    scene_rule = f"Make exactly {scene_count} scenes." if scene_count else "Make 3-8 scenes."
    greeting_rule = (
        "Include a short, friendly greeting in the first scene outline."
        if add_greeting
        else ""
    )
    closing_rule = (
        "Include a short, friendly closing line in the last scene outline."
        if add_closing
        else ""
    )

    return f"""
You are a video planning assistant. Convert the user prompt into strict JSON only.
Target Language: {language}
Target Duration: {target_duration} seconds
Voiceover Density: ~{words_per_sec} words per second

Return JSON with this schema:
{{
  "title": "short title",
  "total_duration": {target_duration},
  "music_mood": "string",
  "scenes": [
    {{
      "id": integer,
      "duration": number (seconds),
      "voiceover": "string",
      "visual_query": "string",
      "overlay_text": "string"
    }}
  ]
}}
Rules:
- JSON only. No markdown.
- Use double quotes for all strings.
- No trailing commas.
- Do not wrap in code fences.
- {scene_rule}
- The sum of all scene durations MUST be exactly {target_duration} seconds.
- The voiceover field should be a brief 1-sentence outline, not the final narration.
- {greeting_rule}
- {closing_rule}
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
    target_duration: float,
    language: str = "English",
    scene_count: Optional[int] = None,
    add_greeting: bool = False,
    add_closing: bool = False,
    words_per_sec: float = 2.0,
    force_json: bool = True,
    think: bool = False,
) -> Dict[str, Any]:
    system_prompt = get_planner_prompt(
        target_duration=target_duration,
        language=language,
        scene_count=scene_count,
        add_greeting=add_greeting,
        add_closing=add_closing,
        words_per_sec=words_per_sec,
    )
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": f"{system_prompt}\n\nUser prompt: {prompt}",
        "stream": False,
        "think": think,
        "options": options,
    }
    
    # Certain models (Qwen, Llama 3) struggle with Ollama's 'format: json'
    # which can lead to empty responses or infinite loops.
    skip_json_format = any(m in model.lower() for m in ["qwen", "llama3", "llama-3"])
    
    if force_json and not skip_json_format:
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

    # Extract keywords for better visual queries
    keywords = clean_and_tokenize(base, max_words=5) or "cinematic background"
    overlay = clean_and_tokenize(base, max_words=3).title() or "Story"

    scenes = []
    for idx in range(1, 4):
        scenes.append(
            {
                "id": idx,
                "duration": per,
                "voiceover": f"Scene {idx}: {base}",
                "visual_query": keywords,
                "overlay_text": overlay,
            }
        )
    return {
        "title": base[:60].strip() or "Generated Story",
        "total_duration": total,
        "music_mood": "cinematic",
        "scenes": scenes,
    }


async def plan_timeline(
    prompt: str,
    model: str | None = None,
    options: Dict[str, Any] | None = None,
    api_url: str = OLLAMA_API_URL,
    timeout: float = LLM_DEFAULT_TIMEOUT,
    think: bool = False,
    helper_settings: Optional[Dict[str, Any]] = None,
    target_duration: Optional[float] = None,
    language: str = "English",
    scene_count: Optional[int] = None,
    add_greeting: bool = False,
    add_closing: bool = False,
    words_per_sec: float = 2.0,
    request_delay: float = 0.8,
    log: Optional[LogFn] = None,
) -> Dict[str, Any]:
    model = model or OLLAMA_SETTINGS["default"]["model"]
    merged_options = {**OLLAMA_SETTINGS["default"]["params"], **(options or {})}

    payload = _build_payload(
        prompt,
        model,
        merged_options,
        target_duration=float(target_duration or 30),
        language=language,
        scene_count=scene_count,
        add_greeting=add_greeting,
        add_closing=add_closing,
        words_per_sec=words_per_sec,
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
            max_retries=LLM_MAX_RETRIES,
            log=log,
            validation_model=Timeline
        )
        
        if isinstance(result, Timeline):
            return result.model_dump()
        return result

    except LLMResponseError as e:
        if log:
            await log("warn", f"LLM parsing failed. Attempting AI repair: {e}")
        try:
            repaired = await repair_json(
                raw_text=e.raw_text,
                schema_hint=_schema_hint(),
                api_url=api_url,
                model=e.model or model,
                options=e.options or merged_options,
                timeout=timeout,
                think=think,
                log=log
            )
            # Re-validate with Timeline model
            from modules.ai_helper import validate_json
            validated = validate_json(repaired, Timeline)
            return validated.model_dump()
        except Exception as repair_exc:
            if log:
                await log("error", f"JSON repair failed: {repair_exc}")
            return _fallback_timeline(prompt, target_duration)

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
    full_timeline: Optional[List[Dict[str, Any]]] = None,
    previous_voiceovers: Optional[List[str]] = None,
) -> str:
    if target_words <= 0:
        target_words = 30
    min_words = max(2, int(target_words * 0.7))
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

    timeline_context = ""
    if full_timeline:
        outlines = [
            f"Scene {s.get('id', i+1)}: {s.get('voiceover', '').strip()}"
            for i, s in enumerate(full_timeline)
        ]
        timeline_context = "Full Narrative Arc:\n" + "\n".join(outlines) + "\n\n"

    history_context = ""
    if previous_voiceovers:
        history_context = (
            "Previous Voiceovers (for continuity):\n"
            + "\n".join(previous_voiceovers[-2:])
            + "\n\n"
        )

    prompt = (
        "You are a professional video narrator. Write the narration for ONE scene while maintaining narrative flow with the rest of the video.\n"
        f"Language: {language}.\n"
        f"Target length: about {target_words} words (min {min_words}, max {max_words}).\n"
        "Do not use labels, quotes, or stage directions. Do not say 'narrator'.\n"
        "Return ONLY the narration text.\n\n"
        f"{timeline_context}"
        f"{history_context}"
        f"{greeting_hint}{closing_hint}"
        "CURRENT SCENE TO NARRATE:\n"
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
            
        if log:
            await log("raw", f"LLM Voiceover prompt:\n{payload.get('prompt', '')}")

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

async def generate_search_queries(
    scene: Dict[str, Any],
    asset_type: str,
    model: str,
    options: Dict[str, Any],
    api_url: str,
    timeout: float = 900.0,
    request_delay: float = 0.5,
    log: Optional[LogFn] = None,
) -> List[str]:
    """
    Refine a descriptive visual query into 3-5 specific keywords for search.
    This helps bypass search engine confusion with long sentences.
    """
    base_query = (
        scene.get("visual_query")
        or scene.get("voiceover")
        or scene.get("overlay_text")
        or ""
    )
    
    prompt = (
        "You are an expert in finding stock media. "
        f"Convert this scene description into 3-5 specific, searchable keywords for {asset_type} search.\n"
    )
    if asset_type == "music":
        prompt += "Focus on mood, genre, and instruments (e.g., 'uplifting, acoustic guitar, folk').\n"
    else:
        prompt += "Focus on concrete nouns and objects (e.g., 'modern office, businessman, typing').\n"
    
    prompt += (
        "Return ONLY the keywords, separated by commas. No other text.\n\n"
        f"Description: {base_query}"
    )

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    max_retries = 3
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            if request_delay > 0:
                await asyncio.sleep(request_delay)
                
            if log:
                await log("raw", f"LLM Search queries prompt (attempt {attempt}):\n{payload.get('prompt', '')}")

            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{api_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                
            raw_text = data.get("response", "") if isinstance(data, dict) else ""
            
            # Check for empty response BEFORE processing
            if not raw_text or not raw_text.strip():
                if log:
                    await log("warn", f"AI returned empty response for {asset_type} search queries (attempt {attempt})")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt * 0.5)
                    continue
                return []
            
            # Strip thinking/markdown
            raw_text = re.sub(r"", "", raw_text, flags=re.DOTALL).strip()
            raw_text = re.sub(r"```.*?```", "", raw_text, flags=re.DOTALL).strip()
            
            # Split and clean
            lines = [l.strip() for l in re.split(r"[,|\n]", raw_text) if l.strip()]
            cleaned = [re.sub(r"^[\-\*\d\.\s]+", "", l).strip() for l in lines]
            keywords = [k for k in cleaned if k][:5]
            
            if not keywords:
                if log:
                    await log("warn", f"AI returned unparseable response for {asset_type} (attempt {attempt}): '{raw_text[:100]}'")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt * 0.5)
                    continue
                # Fallback to simple tokenization
                if base_query:
                    from modules.utils import clean_and_tokenize
                    keywords = clean_and_tokenize(base_query, max_words=4).split()
                return keywords

            if log:
                await log("info", f"Generated {asset_type} keywords: {', '.join(keywords)}")
                
            return keywords
            
        except Exception as exc:
            last_error = exc
            error_type = type(exc).__name__
            error_msg = str(exc) if str(exc) else "(empty exception)"
            if log:
                await log("warn", f"AI search query generation failed (attempt {attempt}/{max_retries}): {error_type}: {error_msg}")
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt * 0.5)
                continue

    # All retries exhausted
    if log and last_error:
        error_type = type(last_error).__name__
        error_msg = str(last_error) if str(last_error) else "(empty exception)"
        await log("error", f"AI search query generation failed after {max_retries} attempts: {error_type}: {error_msg}")
    return []


async def evaluate_asset_match(
    visual_query: str,
    asset_description: str,
    model: str,
    options: Dict[str, Any],
    api_url: str,
    timeout: float,
    log: Optional[LogFn] = None,
) -> bool:
    """
    Evaluate if an existing video asset matches a scene's visual requirement using an LLM.
    Returns True if it's a good match, False otherwise.
    """
    prompt = (
        "You are an AI assistant evaluating if a video asset matches a scene's visual requirement.\n"
        f"Scene Requirement: {visual_query}\n"
        f"Video Description: {asset_description}\n\n"
        "Does the video description reasonably fulfill the scene requirement? "
        "Respond with 'YES' or 'NO' followed by a brief reason.\n"
        "If it's a good match, start with 'YES'. If not, start with 'NO'."
    )

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    if log:
        await log("raw", f"Asset match prompt:\n{prompt}")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{api_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            
        raw_text = data.get("response", "") if isinstance(data, dict) else ""
        # Strip thinking/markdown
        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        raw_text = raw_text.strip().upper()
        
        if log:
            await log("info", f"Asset match evaluation for '{visual_query}' vs '{asset_description}': {raw_text[:60]}...")
            
        return raw_text.startswith("YES")
    except Exception as exc:
        if log:
            await log("warn", f"Asset match evaluation failed: {exc}")
        return False
