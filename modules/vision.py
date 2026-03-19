from __future__ import annotations

import base64
import json5 as json
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

LogFn = Callable[[str, str], Awaitable[None]]


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty response from model")
    cleaned = text.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            return json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"Invalid JSON in vision response: {e}. Raw: {cleaned[:200]}...") from e
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        content = cleaned[start : end + 1]
        try:
            return json.loads(content)
        except Exception as e:
            raise ValueError(f"Invalid JSON found in markers (vision): {e}. Raw: {content[:200]}...") from e
    raise ValueError(f"No JSON object found in vision response. Raw: {cleaned[:200]}...")


def _extract_json_array(text: str) -> List[Any]:
    """Helper to extract a JSON array from vision text."""
    data = _extract_json(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array from vision, but got {type(data).__name__}")
    return data


async def analyze_image(
    image_path: str,
    api_url: str,
    model: str,
    options: Dict[str, Any],
    timeout: float,
    think: bool = False,
    prompt: Optional[str] = None,
    log: Optional[LogFn] = None,
) -> Dict[str, Any]:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    base_prompt = (
        "Describe the image in one short sentence and provide 3-6 tags.\n"
        "Return JSON only with keys: caption (string), tags (array of strings).\n"
    )
    if prompt:
        base_prompt += f"\nContext: {prompt}\n"

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": base_prompt,
        "stream": False,
        "think": think,
        "format": "json",
        "images": [b64],
        "options": options,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{api_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise ValueError(str(data.get("error")))
        text = data.get("response", "") if isinstance(data, dict) else ""

    if not text and log:
        await log("error", "Vision model returned empty response")

    try:
        return _extract_json(text)
    except Exception as e:
        if log:
            await log("raw", f"Vision parsing failed. Raw: {text}")
            await log("error", f"Vision parsing failed: {e}")
        raise ValueError(f"Failed to parse vision response: {e}") from e
