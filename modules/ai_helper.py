from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, TypeVar, Union

import httpx
from pydantic import BaseModel, Field, ValidationError

LogFn = Callable[[str, str], Awaitable[None]]

T = TypeVar("T", bound=BaseModel)


class Scene(BaseModel):
    id: int
    duration: float
    voiceover: str
    visual_query: str
    overlay_text: str


class Timeline(BaseModel):
    title: str
    total_duration: float
    music_mood: str
    scenes: List[Scene]


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty response from model")
    cleaned = text.strip()
    
    # Handle thinking tags
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
    if start == -1:
        return None
    
    # For truncated JSON, 'end' might not exist. We take the rest of the string.
    end = text.rfind("}")
    if end == -1 or end <= start:
        cleaned = text[start:]
    else:
        cleaned = text[start : end + 1]
    
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = _sanitize_json_text(cleaned)
    
    # Remove trailing commas before closing brackets
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt to balance truncated JSON by closing brackets/braces.
    balanced = cleaned
    # Remove a trailing comma if it exists after the last balancing
    balanced = balanced.strip().rstrip(",")
    
    bracket_open = balanced.count("[")
    bracket_close = balanced.count("]")
    if bracket_open > bracket_close:
        balanced += "]" * (bracket_open - bracket_close)
        
    brace_open = balanced.count("{")
    brace_close = balanced.count("}")
    if brace_open > brace_close:
        balanced += "}" * (brace_open - brace_close)
        
    # Final cleanup of trailing commas that might have been exposed
    balanced = re.sub(r",\s*([}\]])", r"\1", balanced)
    
    try:
        return json.loads(balanced)
    except Exception:
        return None


def validate_json(data: Dict[str, Any], model: Type[T]) -> T:
    """Validate a dictionary against a Pydantic model."""
    return model.model_validate(data)


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
        if log and thinking:
            await log("raw", f"AI helper thinking:\n{thinking}")

    if not text:
        raise ValueError("AI helper returned empty response")

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


async def call_llm_with_retry(
    api_url: str,
    payload: Dict[str, Any],
    timeout: float,
    max_retries: int = 3,
    log: Optional[LogFn] = None,
    validation_model: Optional[Type[T]] = None,
) -> Union[Dict[str, Any], T]:
    """Call the LLM with a retry loop that handles failures and validation errors."""
    last_error = None
    current_payload = payload.copy()

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{api_url}/api/generate", json=current_payload)
                if resp.status_code in {400, 422} and "format" in resp.text.lower():
                    current_payload.pop("format", None)
                    resp = await client.post(f"{api_url}/api/generate", json=current_payload)
                resp.raise_for_status()
                data = resp.json()
                
            text = data.get("response", "") if isinstance(data, dict) else ""
            if not text:
                raise ValueError("Empty response from model")
            
            extracted = _extract_json(text)
            
            if validation_model:
                try:
                    return validate_json(extracted, validation_model)
                except ValidationError as e:
                    if log:
                        await log("warn", f"Validation failed on attempt {attempt}: {e}")
                    # On validation failure, try to repair or just retry with different params
                    if attempt < max_retries:
                        # Maybe try without 'format: json' or with 'think: False'
                        current_payload["think"] = False
                        continue
                    raise
            return extracted

        except Exception as e:
            last_error = e
            if log:
                await log("warn", f"LLM call failed on attempt {attempt}: {e}")
            
            if attempt < max_retries:
                # Exponential backoff
                await asyncio.sleep(2 ** attempt * 0.5)
                # Adjust payload for next attempt
                if "format" in current_payload and attempt == 2:
                    current_payload.pop("format")
                continue
            
    raise last_error
