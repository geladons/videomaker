from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable, List, Optional, Tuple

from config import (
    DEFAULT_COQUI_MODEL,
    DEFAULT_VCTK_SPEAKERS,
    LANGUAGE_SPEAKERS,
    LANGUAGE_TO_PIPER,
    MODELS_DIR,
    PIPER_MAX_THREADS,
    PIPER_VOICE_CONFIG,
    PIPER_VOICE_PATH,
)

LogFn = Callable[[str, str], Awaitable[None]]


def _get_default_speaker(language: Optional[str] = None) -> str:
    """Get a default speaker ID based on language."""
    if language:
        return LANGUAGE_SPEAKERS.get(language, DEFAULT_VCTK_SPEAKERS[0])
    return DEFAULT_VCTK_SPEAKERS[0]


async def _run_piper(
    text: str, model_path: str, config_path: Optional[str], out_path: str, log: LogFn
) -> int:
    cmd = ["piper", "--model", model_path, "--output_file", out_path]
    if config_path and os.path.exists(config_path):
        cmd.extend(["--config", config_path])

    env = os.environ.copy()
    threads = str(PIPER_MAX_THREADS)
    env["OMP_NUM_THREADS"] = threads
    env["ONNXRUNTIME_INTRA_OP_NUM_THREADS"] = threads
    env["ONNXRUNTIME_INTER_OP_NUM_THREADS"] = threads
    env["ORT_DISABLE_CPU_AFFINITY"] = "1"
    env["ORT_DISABLE_CPU_EXT"] = "1"

    await log("info", f"$ {' '.join(cmd)}")
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    if process.stdin:
        process.stdin.write(text.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

    stdout_lines = []
    stderr_lines = []

    while True:
        if process.stdout:
            line = await process.stdout.readline()
            if line:
                msg = line.decode(errors="ignore").rstrip()
                if msg:
                    stdout_lines.append(msg)
            else:
                break
        else:
            break

    while True:
        if process.stderr:
            line = await process.stderr.readline()
            if line:
                msg = line.decode(errors="ignore").rstrip()
                if msg:
                    stderr_lines.append(msg)
            else:
                break
        else:
            break

    for msg in stdout_lines:
        await log("info", msg)
    for msg in stderr_lines:
        lowered = msg.lower()
        if "error" in lowered or "failed" in lowered:
            await log("error", msg)
        elif "warn" in lowered:
            await log("warn", msg)
        else:
            await log("info", msg)

    return await process.wait()


async def generate_voiceovers(
    texts: List[str],
    out_dir: str,
    log: LogFn,
    voice_path: Optional[str] = None,
    voice_config: Optional[str] = None,
    language: Optional[str] = None,
    engine: str = "piper",
    coqui_model: Optional[str] = None,
    coqui_speaker: Optional[str] = None,
    progress_fn: Optional[Callable[[float], Awaitable[None]]] = None,
) -> List[Optional[str]]:
    """
    Generate voiceovers with fallback mechanism.
    Validates input text and handles TTS engine failures gracefully.
    Maintains 1:1 mapping with input texts (None for failures).
    """
    os.makedirs(out_dir, exist_ok=True)

    if not texts:
        await log("warn", "No texts provided for voiceover generation")
        return []

    # Try primary engine (Coqui or Piper)
    primary_engine = engine.lower() if engine else "piper"

    if primary_engine == "coqui":
        await log(
            "info",
            f"TTS engine set to Coqui (model={coqui_model}, speaker={coqui_speaker or 'auto'})",
        )
        try:
            return await _generate_coqui(
                texts,
                out_dir,
                log,
                model_name=coqui_model,
                speaker=coqui_speaker,
                language=language,
                progress_fn=progress_fn,
            )
        except Exception as exc:
            await log("error", f"Coqui TTS failed: {exc}. Falling back to Piper.")
            primary_engine = "piper"

    if primary_engine == "piper" or primary_engine not in ("coqui", "piper"):
        await log("info", "TTS engine set to Piper.")
        resolved_voice, resolved_config, note = _resolve_voice(
            language, voice_path, voice_config
        )
        if note:
            await log("info", note)

        if not resolved_voice:
            await log("error", f"No valid Piper voice model found for language: {language}")
            return [None] * len(texts)

        outputs: List[Optional[str]] = [None] * len(texts)
        num_texts = len(texts)
        for idx, text in enumerate(texts, start=1):
            if progress_fn:
                await progress_fn((idx - 1) / num_texts)
            
            if not text or not text.strip() or len(text.strip()) < 3:
                await log("warn", f"Skipping invalid text at index {idx}")
                continue

            out_path = os.path.join(out_dir, f"voice_{idx}.wav")
            await log("info", f"Generating voiceover {idx}/{num_texts}")
            try:
                code = await _run_piper(
                    text.strip(), resolved_voice, resolved_config, out_path, log
                )
                if code != 0:
                    raise RuntimeError(
                        f"Piper failed for segment {idx} with exit code {code}"
                    )
                outputs[idx-1] = out_path
            except Exception as exc:
                await log("error", f"Piper failed for segment {idx}: {exc}")
                continue

        return outputs

    # Ultimate fallback
    await log("error", "All TTS engines failed")
    return [None] * len(texts)


_coqui_cache = {}


async def _generate_coqui(
    texts: List[str],
    out_dir: str,
    log: LogFn,
    model_name: Optional[str],
    speaker: Optional[str],
    language: Optional[str],
    progress_fn: Optional[Callable[[float], Awaitable[None]]] = None,
) -> List[Optional[str]]:
    """
    Generate voiceovers using Coqui TTS.
    Returns List[Optional[str]] to maintain alignment.
    """
    try:
        from TTS.api import TTS  # type: ignore
    except Exception as exc:
        await log("error", f"Coqui TTS not available ({exc}); falling back to Piper.")
        return await generate_voiceovers(
            texts,
            out_dir,
            log,
            language=language,
            engine="piper",
            progress_fn=progress_fn,
        )

    # Use default model if none specified
    model_name = model_name or DEFAULT_COQUI_MODEL

    if not speaker:
        speaker = _get_default_speaker(language)

    await log("info", f"Using Coqui with speaker: {speaker}")

    # Check cache
    tts = _coqui_cache.get(model_name)
    if tts is None:
        await log("info", f"Loading Coqui TTS model: {model_name}")
        try:
            tts = await asyncio.to_thread(
                TTS, model_name=model_name, progress_bar=False, gpu=False
            )
            _coqui_cache[model_name] = tts
        except Exception as exc:
            error_msg = str(exc).lower()
            if "eof" in error_msg or "download" in error_msg or "connection" in error_msg:
                await log("warn", f"Coqui model download/load failed ({exc}). Falling back to Piper.")
            else:
                await log("error", f"Failed to load Coqui model {model_name}: {exc}")
            return await generate_voiceovers(
                texts,
                out_dir,
                log,
                language=language,
                engine="piper",
                progress_fn=progress_fn,
            )

    outputs: List[Optional[str]] = [None] * len(texts)
    num_texts = len(texts)
    for idx, text in enumerate(texts, start=1):
        if progress_fn:
            await progress_fn((idx - 1) / num_texts)
        
        if not text or not text.strip() or len(text.strip()) < 3:
            await log("warn", f"Skipping invalid text at index {idx}")
            continue

        out_path = os.path.join(out_dir, f"voice_{idx}.wav")
        await log(
            "info",
            f"Generating voiceover {idx}/{num_texts} with Coqui (speaker={speaker})",
        )
        try:
            # Prepare arguments for tts_to_file
            tts_args = {
                "text": text.strip(),
                "file_path": out_path,
                "speaker": speaker,
            }
            # Only add language if the model is XTTS (or multilingual)
            if model_name and "xtts" in model_name.lower():
                tts_args["language"] = language

            await asyncio.to_thread(
                tts.tts_to_file, **tts_args
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                outputs[idx-1] = out_path
                await log("info", f"Coqui generated: {out_path}")
            else:
                await log("warn", f"Coqui did not create output file for segment {idx}")
        except Exception as exc:
            await log("error", f"Coqui failed for segment {idx}: {exc}")
            continue

    return outputs



def _resolve_voice(
    language: Optional[str],
    voice_path: Optional[str],
    voice_config: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if voice_path and os.path.exists(voice_path):
        config = _pick_config(voice_path, voice_config)
        return voice_path, config, f"Using custom voice model: {voice_path}"

    if voice_path and not os.path.exists(voice_path):
        note = f"Custom voice model not found at {voice_path}, falling back."
    else:
        note = None

    if language:
        voice_name = LANGUAGE_TO_PIPER.get(language)
        if voice_name:
            candidate = os.path.join(MODELS_DIR, "piper", f"{voice_name}.onnx")
            if os.path.exists(candidate):
                config = _pick_config(candidate, None)
                return candidate, config, f"Using {language} voice: {voice_name}"
            if note is None:
                note = f"No local voice for {language}. Falling back to default."

    if os.path.exists(PIPER_VOICE_PATH):
        config = _pick_config(PIPER_VOICE_PATH, PIPER_VOICE_CONFIG)
        return (
            PIPER_VOICE_PATH,
            config,
            note or f"Using default voice: {PIPER_VOICE_PATH}",
        )

    return None, None, note or "No valid Piper voice models found"


def _pick_config(model_path: str, config_path: Optional[str]) -> Optional[str]:
    if config_path and os.path.exists(config_path):
        return config_path
    guessed = f"{model_path}.json"
    if os.path.exists(guessed):
        return guessed
    return None
