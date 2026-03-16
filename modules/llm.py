from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

try:
    import json5  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    json5 = None

from config import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_PARAMS, OLLAMA_API_URL

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


LogFn = Callable[[str, str], Awaitable[None]]


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


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise LLMError("Empty response from model")
    cleaned = text.strip()
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for chunk in parts:
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{") and chunk.endswith("}"):
                cleaned = chunk
                break
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return _loads_relaxed(cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return _loads_relaxed(cleaned[start : end + 1])
    raise LLMError("No JSON object found in model response")


def _loads_relaxed(payload: str) -> Dict[str, Any]:
    try:
        if json5 is not None:
            return json5.loads(payload)
        return json.loads(payload)
    except Exception as exc:
        raise LLMError(str(exc))


def _looks_truncated_json(text: str) -> bool:
    if not text:
        return True
    stripped = text.strip()
    if stripped.endswith((",", "[", "{", ":")):
        return True
    if not stripped.endswith("}"):
        return True
    opens = stripped.count("{")
    closes = stripped.count("}")
    brackets_open = stripped.count("[")
    brackets_close = stripped.count("]")
    return opens > closes or brackets_open > brackets_close


def _sanitize_json_text(text: str) -> str:
    output = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
                output.append(ch)
                continue
            if ch == "\\":
                escape = True
                output.append(ch)
                continue
            if ch in ("\r", "\n"):
                output.append(" ")
                continue
            if ch == '"':
                in_string = False
            output.append(ch)
        else:
            if ch == '"':
                in_string = True
            output.append(ch)
    return "".join(output)


def _repair_json_deterministic(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    cleaned = text[start : end + 1]
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = _sanitize_json_text(cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return _loads_relaxed(cleaned)
    except json.JSONDecodeError:
        pass
    except LLMError:
        pass

    # Attempt to balance truncated JSON by closing brackets/braces.
    balanced = cleaned
    bracket_open = balanced.count("[")
    bracket_close = balanced.count("]")
    if bracket_open > bracket_close:
        balanced += "]" * (bracket_open - bracket_close)
    brace_open = balanced.count("{")
    brace_close = balanced.count("}")
    if brace_open > brace_close:
        balanced += "}" * (brace_open - brace_close)
    balanced = re.sub(r",\s*([}\]])", r"\1", balanced)
    try:
        return _loads_relaxed(balanced)
    except Exception:
        return None


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


def _parse_scene_lines(
    text: str, target_duration: Optional[float]
) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("scene") and "|" not in line:
            continue
        lines.append(line)

    scenes = []
    for idx, line in enumerate(lines, start=1):
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        scene_id = parts[0]
        if scene_id.lower().startswith("scene"):
            scene_id = scene_id.split()[-1].strip(":")
        try:
            scene_idx = int(re.sub(r"\D", "", scene_id)) if scene_id else idx
        except ValueError:
            scene_idx = idx
        duration = 0.0
        try:
            duration = float(parts[1])
        except (TypeError, ValueError):
            duration = 0.0
        visual_query = parts[2] if len(parts) > 2 else ""
        overlay_text = parts[3] if len(parts) > 3 else ""
        voiceover = parts[4] if len(parts) > 4 else ""
        scenes.append(
            {
                "id": scene_idx,
                "duration": duration,
                "voiceover": voiceover.strip(),
                "visual_query": visual_query.strip(),
                "overlay_text": overlay_text.strip(),
            }
        )

    if len(scenes) < 3:
        return None
    if len(scenes) > 8:
        scenes = scenes[:8]

    total = float(target_duration or 0)
    if total <= 0:
        total = sum(max(0.0, float(s.get("duration", 0) or 0)) for s in scenes)
    if total <= 0:
        total = float(target_duration or 60)

    if any(float(s.get("duration", 0) or 0) <= 0 for s in scenes):
        per = round(total / len(scenes), 2)
        for scene in scenes:
            scene["duration"] = per

    return {
        "title": "Generated Story",
        "total_duration": total,
        "music_mood": "cinematic",
        "scenes": scenes,
    }


async def _request_scene_lines(
    prompt: str,
    model: str,
    options: Dict[str, Any],
    api_url: str,
    timeout: float,
    log: Optional[LogFn],
    target_duration: Optional[float],
) -> Optional[Dict[str, Any]]:
    line_prompt = (
        "Return 3-8 lines. Each line must follow this exact format:\n"
        "SCENE|duration_seconds|visual_query|overlay_text|voiceover_outline\n"
        "Use '|' exactly as separators. No extra text or numbering outside the format.\n"
        "Keep visual_query and overlay_text short.\n\n"
        f"User prompt: {prompt}\n"
    )

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": line_prompt,
        "stream": False,
        "think": False,
        "options": options,
    }

    try:
        if timeout and timeout > 0:
            await asyncio.sleep(0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{api_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = data.get("response", "") if isinstance(data, dict) else ""
        if log and text:
            await log("raw", f"Line-based planner raw response:\n{text}")
            await log(
                "info", f"Line-based planner raw response length: {len(text)} chars"
            )
        parsed = _parse_scene_lines(text, target_duration)
        if not parsed and log:
            await log("error", "Line-based planner failed to parse response.")
        return parsed
    except Exception as exc:
        if log:
            await log("error", f"Line-based planner failed: {exc}")
        return None


def _validate_timeline(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise LLMError("Timeline is not a JSON object")
    if (
        "scenes" not in data
        or not isinstance(data["scenes"], list)
        or not data["scenes"]
    ):
        raise LLMError("Timeline missing scenes list")
    if not (3 <= len(data["scenes"]) <= 8):
        raise LLMError(f"Timeline must have 3-8 scenes, got {len(data['scenes'])}")
    if "total_duration" not in data:
        raise LLMError("Timeline missing total_duration")
    for idx, scene in enumerate(data["scenes"]):
        if not isinstance(scene, dict):
            raise LLMError(f"Scene {idx} is not an object")
        for key in ("duration", "voiceover", "visual_query"):
            if key not in scene:
                raise LLMError(f"Scene {idx} missing {key}")


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

    last_error = None
    base_prompt = prompt
    force_think = think
    for attempt in range(1, 4):
        raw_text: str = ""
        raw_thinking: str = ""
        text: str = ""
        attempt_options = dict(merged_options)
        attempt_prompt = base_prompt
        if log:
            await log("info", f"Planner attempt {attempt}/3 (think={force_think})")
        if attempt >= 2:
            attempt_options["temperature"] = min(
                float(attempt_options.get("temperature", 0.7)), 0.3
            )
            attempt_options["num_predict"] = max(
                int(attempt_options.get("num_predict", 512)), 512
            )
            attempt_prompt = (
                f"{base_prompt}\n\n"
                "Additional constraints:\n"
                "- Keep each voiceover <= 20 words.\n"
                "- Keep visual_query <= 8 words.\n"
                "- Keep overlay_text <= 8 words.\n"
                "- Use 3-6 scenes.\n"
                "- Avoid double quotes inside strings.\n"
                "- Do not use line breaks inside strings.\n"
                "Return ONLY valid JSON."
            )
        try:
            if request_delay > 0:
                await asyncio.sleep(request_delay)
            payload = _build_payload(
                attempt_prompt,
                model,
                attempt_options,
                force_json=True,
                think=force_think,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{api_url}/api/generate", json=payload)
                if (
                    response.status_code in {400, 422}
                    and "format" in response.text.lower()
                ):
                    payload = _build_payload(
                        attempt_prompt,
                        model,
                        attempt_options,
                        force_json=False,
                        think=force_think,
                    )
                    response = await client.post(
                        f"{api_url}/api/generate", json=payload
                    )
                if response.status_code == 404:
                    models = await _list_models(api_url)
                    if models and model not in models:
                        model = models[0]
                        continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    data = response.json()
                    if isinstance(data, dict) and data.get("error"):
                        raise LLMError(str(data.get("error")))
                    if isinstance(data, dict):
                        raw_text = data.get("response", "") or ""
                        raw_thinking = data.get("thinking", "") or ""
                        text = raw_text
                    else:
                        text = ""
                    if log and raw_text:
                        await log("raw", f"LLM raw response:\n{raw_text}")
                        await log(
                            "info", f"LLM raw response length: {len(raw_text)} chars"
                        )
                        await log("info", f"LLM raw response preview: {raw_text[:300]}")
                    if log and raw_thinking:
                        await log("raw", f"LLM thinking:\n{raw_thinking}")
                    if not text and isinstance(data, dict) and data.get("thinking"):
                        if log:
                            await log(
                                "error",
                                "Ollama returned empty response with reasoning output; retrying without thinking.",
                            )
                        if force_think:
                            force_think = False
                            raise LLMError("Empty response with reasoning output")
                        text = data.get("response", "")
                else:
                    raw_text = response.text
                    text = raw_text
            if raw_text and _looks_truncated_json(raw_text):
                if log:
                    await log(
                        "error",
                        "Detected truncated JSON response; retrying with higher num_predict.",
                    )
                merged_options["num_predict"] = max(
                    int(merged_options.get("num_predict", 512)), 1536
                )
                raise LLMError("Truncated JSON response")
            timeline = _extract_json(text)
            if not timeline:
                raise LLMError("Empty timeline")
            _validate_timeline(timeline)
            return timeline
        except (json.JSONDecodeError, LLMError) as exc:
            repaired = _repair_json_deterministic(raw_text or text)
            if repaired:
                if log:
                    await log("info", "Deterministic JSON repair succeeded.")
                _validate_timeline(repaired)
                return repaired
            helper_error = None
            if helper_settings and attempt >= 2:
                if log:
                    await log(
                        "info", "Deterministic repair failed; invoking AI helper."
                    )
                try:
                    from modules import ai_helper

                    if request_delay > 0:
                        await asyncio.sleep(max(0.2, request_delay))
                    helper_json = await ai_helper.repair_json(
                        raw_text or text,
                        _schema_hint(),
                        api_url=helper_settings.get("api_url", api_url),
                        model=helper_settings.get("model", model),
                        options=helper_settings.get("options", {}),
                        timeout=helper_settings.get("timeout", timeout),
                        think=helper_settings.get("think", False),
                        log=log,
                    )
                    if log:
                        await log("info", "AI helper repair succeeded.")
                    _validate_timeline(helper_json)
                    return helper_json
                except Exception as helper_exc:
                    helper_error = helper_exc
                    if log:
                        await log("error", f"AI helper failed: {helper_exc}")
            elif log:
                await log(
                    "info", "Deterministic repair failed; retrying without helper."
                )
            err_text = str(exc).strip() if str(exc).strip() else repr(exc)
            last_error = err_text
            if helper_error:
                last_error = f"{err_text}; helper: {helper_error}"
            if log:
                await log("error", f"LLM parse error: {type(exc).__name__}: {err_text}")
                if raw_text:
                    await log(
                        "error", f"LLM raw response (truncated): {raw_text[:400]}"
                    )
                if raw_thinking:
                    await log(
                        "error", f"LLM thinking (truncated): {raw_thinking[:400]}"
                    )
            base_prompt = (
                f"{base_prompt}\n\n"
                f"The previous JSON was invalid or incomplete. Error: {last_error}. "
                "Please return ONLY valid JSON using the exact schema."
            )
            await asyncio.sleep(max(0.2, request_delay) * attempt)
            continue
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            detail = detail.strip() if isinstance(detail, str) else str(detail)
            if not detail:
                detail = repr(exc)
            last_error = f"HTTP {exc.response.status_code if exc.response is not None else 'error'}: {detail}".strip()
            if log:
                await log("error", f"Ollama API error: {last_error}")
            base_prompt = (
                f"{base_prompt}\n\n"
                f"The previous attempt failed due to an API error: {detail}. "
                "Please return ONLY valid JSON using the exact schema."
            )
            await asyncio.sleep(max(0.2, request_delay) * attempt)
        except httpx.ReadTimeout as exc:
            err_text = f"ReadTimeout: {exc}".strip()
            last_error = err_text
            if log:
                await log("error", f"LLM parse error: {err_text}")
                await log(
                    "error",
                    f"Hint: increase Ollama timeout (current {timeout}s) or lower num_predict.",
                )
            base_prompt = (
                f"{base_prompt}\n\n"
                f"The previous JSON was invalid or incomplete. Error: {last_error}. "
                "Please return ONLY valid JSON using the exact schema."
            )
            await asyncio.sleep(max(0.2, request_delay) * attempt)
        except (json.JSONDecodeError, LLMError, httpx.HTTPError) as exc:
            err_text = str(exc).strip()
            if not err_text:
                err_text = repr(exc)
            last_error = err_text
            if log:
                await log("error", f"LLM parse error: {type(exc).__name__}: {err_text}")
                if raw_text:
                    await log(
                        "error", f"LLM raw response (truncated): {raw_text[:400]}"
                    )
                if raw_thinking:
                    await log(
                        "error", f"LLM thinking (truncated): {raw_thinking[:400]}"
                    )
            base_prompt = (
                f"{base_prompt}\n\n"
                f"The previous JSON was invalid or incomplete. Error: {last_error}. "
                "Please return ONLY valid JSON using the exact schema."
            )
            await asyncio.sleep(max(0.2, request_delay) * attempt)
        except Exception as exc:
            err_text = str(exc).strip()
            if not err_text:
                err_text = repr(exc)
            last_error = err_text
            if log:
                await log(
                    "error",
                    f"Unexpected planner error: {type(exc).__name__}: {err_text}",
                )
            base_prompt = (
                f"{base_prompt}\n\n"
                f"The previous JSON was invalid or incomplete. Error: {last_error}. "
                "Please return ONLY valid JSON using the exact schema."
            )
            await asyncio.sleep(max(0.2, request_delay) * attempt)

    line_based = await _request_scene_lines(
        base_prompt,
        model=model,
        options=merged_options,
        api_url=api_url,
        timeout=timeout,
        log=log,
        target_duration=target_duration,
    )
    if line_based:
        return line_based

    if log:
        await log("error", "Falling back to default timeline")
    return _fallback_timeline(base_prompt, target_duration)


async def generate_search_queries(
    scene: Dict[str, Any],
    asset_type: str,  # "video", "music", "image"
    model: str,
    options: Dict[str, Any],
    api_url: str,
    timeout: float,
    request_delay: float = 0.8,
    log: Optional[LogFn] = None,
) -> List[str]:
    """
    Generate specialized search queries using AI (Ollama).
    This replaces raw prompts with optimized queries for:
    - video: optimized for stock video search
    - music: optimized for background music search
    - image: optimized for image search
    """
    visual_query = scene.get("visual_query", "")
    voiceover = scene.get("voiceover", "")
    overlay_text = scene.get("overlay_text", "")

    # Build context from scene data
    context = f"Visual: {visual_query}" if visual_query else ""
    if voiceover:
        context += f" | Voiceover: {voiceover}"
    if overlay_text:
        context += f" | Overlay: {overlay_text}"

    # Create specialized prompts based on asset type
    prompts = {
        "video": (
            "Generate 3 short search queries (3-6 words each) for finding stock video footage. "
            "Return ONLY a JSON array of strings, nothing else.\n"
            "Focus on: action, movement, background, atmosphere.\n"
            f"Scene context: {context}"
        ),
        "music": (
            "Generate 3 short search queries (2-4 words each) for finding background music. "
            "Return ONLY a JSON array of strings, nothing else.\n"
            "Focus on: mood, genre, tempo, atmosphere.\n"
            f"Scene context: {context}"
        ),
        "image": (
            "Generate 3 short search queries (2-5 words each) for finding images. "
            "Return ONLY a JSON array of strings, nothing else.\n"
            "Focus on: subject, style, composition.\n"
            f"Scene context: {context}"
        ),
    }

    prompt = prompts.get(asset_type, prompts["video"])
    merged_options = {**options}
    merged_options["num_predict"] = min(merged_options.get("num_predict", 256), 256)

    try:
        if request_delay > 0:
            await asyncio.sleep(request_delay)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": merged_options,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{api_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

        text = data.get("response", "") if isinstance(data, dict) else ""

        if log:
            await log("raw", f"Search query generation ({asset_type}): {text[:200]}")

        # Try to parse as JSON array
        import json as json_module

        try:
            # Find JSON array in response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                queries = json_module.loads(text[start:end])
                if isinstance(queries, list) and queries:
                    return [str(q).strip() for q in queries if q]
        except (json_module.JSONDecodeError, ValueError, AttributeError):
            pass

        # Fallback: extract queries from text
        queries = []
        for line in text.split("\n"):
            line = line.strip().strip("- ").strip().strip('"').strip("'")
            if line and len(line) > 2:
                queries.append(line)
        if queries:
            return queries[:3]

    except Exception as exc:
        if log:
            await log("error", f"Search query generation failed ({asset_type}): {exc}")

    # Ultimate fallback - return simplified version of visual_query
    if visual_query:
        words = visual_query.split()[:5]
        return [" ".join(words)]
    return []


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
    num_predict = local_options.get("num_predict")
    if isinstance(num_predict, (int, float)):
        suggested = max(64, int(target_words * 4))
        local_options["num_predict"] = int(min(num_predict, suggested))

    last_text = ""
    force_think = think
    for attempt in range(1, 3):
        greeting_hint = ""
        closing_hint = ""
        if add_greeting and is_first:
            greeting_hint = (
                "Include a short, friendly greeting to the audience at the beginning.\n"
            )
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
        if attempt == 2 and last_text:
            word_count = _word_count(last_text)
            prompt = (
                f"The previous narration had {word_count} words. "
                f"Please rewrite to about {target_words} words.\n\n" + prompt
            )

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": bool(force_think),
            "options": local_options,
        }

        try:
            if log:
                await log("info", f"Writer attempt {attempt}/2 (think={force_think})")
            if request_delay > 0:
                await asyncio.sleep(request_delay)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{api_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
            if isinstance(data, dict) and data.get("error"):
                raise LLMError(str(data.get("error")))
            raw_text = data.get("response", "") if isinstance(data, dict) else ""
            raw_thinking = data.get("thinking", "") if isinstance(data, dict) else ""
            if log and raw_text:
                await log("raw", f"Writer raw response:\n{raw_text}")
                await log("info", f"Writer raw response length: {len(raw_text)} chars")
                await log("info", f"Writer raw response preview: {raw_text[:300]}")
            if log and raw_thinking:
                await log("raw", f"Writer thinking:\n{raw_thinking}")
            if not raw_text and raw_thinking and force_think:
                if log:
                    await log(
                        "error",
                        "Writer model returned empty response with reasoning output; retrying without thinking.",
                    )
                force_think = False
                continue
            cleaned = _clean_voiceover_text(raw_text)
            if cleaned:
                last_text = cleaned
                word_count = _word_count(cleaned)
                if min_words <= word_count <= max_words:
                    return cleaned
            if raw_text:
                last_text = _clean_voiceover_text(raw_text)
        except httpx.HTTPError as exc:
            if log:
                await log("error", f"Writer LLM error: {exc}")
            last_text = last_text or base_outline
            continue
        if attempt < 2:
            await asyncio.sleep(max(0.2, request_delay))

    return last_text or base_outline or "Brief scene."
