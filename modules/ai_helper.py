from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, TypeVar, Union

import httpx
from pydantic import BaseModel, Field, ValidationError

LogFn = Callable[[str, str], Awaitable[None]]

T = TypeVar("T", bound=BaseModel)


class LLMResponseError(Exception):
    """Exception raised when LLM response extraction or validation fails."""
    def __init__(self, message: str, raw_text: str, model: str = "", options: Dict[str, Any] = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.model = model
        self.options = options or {}


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


def _extract_json(text: str) -> Any:
    if not text:
        raise ValueError("Empty response from model")
    cleaned = text.strip()
    
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    
    if "```" in cleaned:
        parts = cleaned.split("```")
        for chunk in parts:
            chunk = chunk.strip()
            if not chunk or not (chunk.startswith("{") or chunk.startswith("[") or chunk.lower().startswith("json")):
                continue
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{") or chunk.startswith("["):
                cleaned = chunk
                break
    
    if (cleaned.startswith("{") and cleaned.endswith("}")) or (cleaned.startswith("[") and cleaned.endswith("]")):
        return json.loads(cleaned)
    
    start_obj, start_arr = cleaned.find("{"), cleaned.find("[")
    if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
        start, end = start_obj, cleaned.rfind("}")
    elif start_arr != -1:
        start, end = start_arr, cleaned.rfind("]")
    else:
        raise ValueError("No JSON object or array found")

    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("No JSON object or array found")


def _sanitize_json_text(text: str) -> str:
    """Replaces unescaped newlines in strings with spaces."""
    output, in_string, escape = [], False, False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch in ("\r", "\n"):
                output.append(" ")
                continue
            elif ch == '"':
                in_string = False
            output.append(ch)
        else:
            if ch == '"':
                in_string = True
            output.append(ch)
    return "".join(output)


def _safe_replace(pattern: str, replacement: Any, target: str) -> str:
    """Matches double-quoted strings OR the pattern, protecting string contents."""
    full_pattern = r'("(?:\\.|[^"\\])*")|' + pattern
    def subst(match):
        if match.group(1): return match.group(1)
        return match.expand(replacement) if isinstance(replacement, str) else replacement(match)
    return re.sub(full_pattern, subst, target, flags=re.DOTALL)


def _repair_json_deterministic(text: str) -> Optional[Any]:
    start_obj, start_arr = text.find("{"), text.find("[")
    if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
        start, end = start_obj, text.rfind("}")
    elif start_arr != -1:
        start, end = start_arr, text.rfind("]")
    else:
        return None
    
    cleaned = text[start : (end + 1 if end != -1 and end > start else None)]
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = _safe_replace(r"'", '"', cleaned)
    cleaned = _sanitize_json_text(cleaned)
    
    # Structural repairs
    cleaned = _safe_replace(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\2"\3"\4', cleaned)
    cleaned = _safe_replace(r'([}\]])\s*([{\[])', r'\2, \3', cleaned)
    
    # Value-to-key repairs (e.g., missing comma between value and next key)
    val_pat = r'("(?:\\.|[^"\\])*"|true|false|null|(?<!\w)-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    key_pat = r'("(?:\\.|[^"\\])*"\s*:)'
    cleaned = re.sub(val_pat + r'\s+' + key_pat, r'\1, \2', cleaned, flags=re.DOTALL)
    cleaned = _safe_replace(r",\s*([}\]])", r"\2", cleaned)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Balance truncated JSON
    balanced = cleaned.strip().rstrip(",")
    def count_outside(char: str, target: str) -> int:
        count, in_str, esc = 0, False, False
        for ch in target:
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            elif ch == '"': in_str = True
            elif ch == char: count += 1
        return count

    b_open, b_close = count_outside("[", balanced), count_outside("]", balanced)
    if b_open > b_close: balanced += "]" * (b_open - b_close)
    br_open, br_close = count_outside("{", balanced), count_outside("}", balanced)
    if br_open > br_close: balanced += "}" * (br_open - br_close)
    
    balanced = _safe_replace(r",\s*([}\]])", r"\2", balanced)
    
    try:
        return json.loads(balanced)
    except Exception:
        if "," in balanced and not balanced.startswith("["):
            try: return json.loads("[" + balanced + "]")
            except Exception: pass
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
            
            try:
                extracted = _extract_json(text)
            except Exception as e:
                if attempt == max_retries:
                    raise LLMResponseError(
                        f"Failed to extract JSON after all retries: {e}",
                        raw_text=text,
                        model=current_payload.get("model", ""),
                        options=current_payload.get("options", {})
                    )
                raise

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
                    raise LLMResponseError(
                        f"Pydantic validation failed: {e}",
                        raw_text=text,
                        model=current_payload.get("model", ""),
                        options=current_payload.get("options", {})
                    )
            return extracted

        except LLMResponseError:
            # Re-raise our custom response error
            raise
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
