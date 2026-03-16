from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx

LogFn = Callable[[str, str], Awaitable[None]]


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty response from model")
    cleaned = text.strip()
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
        return json.loads(cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("No JSON object found in model response")


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
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


async def repair_json(
    raw_text: str,
    schema_hint: str,
    api_url: str,
    model: str,
    options: Dict[str, Any],
    timeout: float,
    think: bool = False,
    log: Optional[LogFn] = None,
) -> Dict[str, Any]:
    prompt = (
        "You are a JSON repair assistant. Fix the input into valid JSON that matches the schema.\n"
        "Return ONLY valid JSON. Do not add commentary.\n\n"
        f"Schema:\n{schema_hint}\n\n"
        "Broken JSON:\n"
        f"{raw_text}\n"
    )

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "format": "json",
        "options": options,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{api_url}/api/generate", json=payload)
        if resp.status_code in {400, 422} and "format" in resp.text.lower():
            payload.pop("format", None)
            resp = await client.post(f"{api_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise ValueError(str(data.get("error")))
        text = data.get("response", "") if isinstance(data, dict) else ""
        thinking = data.get("thinking", "") if isinstance(data, dict) else ""
        if log and text:
            await log("raw", f"AI helper raw response:\n{text}")
            await log("info", f"AI helper raw response length: {len(text)} chars")
        if log and thinking:
            await log("raw", f"AI helper thinking:\n{thinking}")
        if not text and thinking and think:
            if log:
                await log("error", "AI helper returned empty response with thinking; retrying without think")
            payload["think"] = False
            resp = await client.post(f"{api_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "") if isinstance(data, dict) else ""

    if not text and log:
        await log("error", "AI helper returned empty response")

    try:
        return _extract_json(text)
    except Exception:
        repaired = _repair_json_deterministic(text)
        if repaired:
            return repaired
        raise


async def summarize_error(
    raw_text: str,
    api_url: str,
    model: str,
    options: Dict[str, Any],
    timeout: float,
    think: bool = False,
) -> str:
    prompt = (
        "Summarize the following error in one short sentence. "
        "Return plain text only.\n\n"
        f"{raw_text}"
    )
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": options,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{api_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data.get("response", "") if isinstance(data, dict) else ""
