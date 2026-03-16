from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable, List, Optional, Tuple

from config import LANGUAGE_TO_PIPER, MODELS_DIR, PIPER_VOICE_CONFIG, PIPER_VOICE_PATH

LogFn = Callable[[str, str], Awaitable[None]]


async def _run_piper(text: str, model_path: str, config_path: Optional[str], out_path: str, log: LogFn) -> int:
    cmd = ["piper", "--model", model_path, "--output_file", out_path]
    if config_path and os.path.exists(config_path):
        cmd.extend(["--config", config_path])

    env = os.environ.copy()
    cpu_count = os.cpu_count() or 4
    env.setdefault("OMP_NUM_THREADS", str(cpu_count))
    env.setdefault("ORT_DISABLE_CPU_AFFINITY", "1")

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

    async def _stream(reader: asyncio.StreamReader, level: str) -> None:
        while True:
            line = await reader.readline()
            if not line:
                break
            msg = line.decode(errors="ignore").rstrip()
            if msg:
                lowered = msg.lower()
                if "error" in lowered or "failed" in lowered:
                    await log("error", msg)
                elif "warn" in lowered:
                    await log("warn", msg)
                else:
                    await log(level, msg)

    await asyncio.gather(
        _stream(process.stdout, "info"),
        _stream(process.stderr, "error"),
    )
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
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    if engine == "coqui":
        await log("info", "TTS engine set to Coqui.")
        return await _generate_coqui(
            texts,
            out_dir,
            log,
            model_name=coqui_model,
            speaker=coqui_speaker,
            fallback_voice=(voice_path, voice_config, language),
        )
    await log("info", "TTS engine set to Piper.")
    resolved_voice, resolved_config, note = _resolve_voice(language, voice_path, voice_config)
    if note:
        await log("info", note)

    outputs: List[str] = []
    for idx, text in enumerate(texts, start=1):
        out_path = os.path.join(out_dir, f"voice_{idx}.wav")
        await log("info", f"Generating voiceover {idx}/{len(texts)}")
        code = await _run_piper(text, resolved_voice, resolved_config, out_path, log)
        if code != 0:
            raise RuntimeError(f"Piper failed for segment {idx} with exit code {code}")
        outputs.append(out_path)
    return outputs


_coqui_cache = {}


async def _generate_coqui(
    texts: List[str],
    out_dir: str,
    log: LogFn,
    model_name: Optional[str],
    speaker: Optional[str],
    fallback_voice: Tuple[Optional[str], Optional[str], Optional[str]],
) -> List[str]:
    try:
        from TTS.api import TTS  # type: ignore
    except Exception as exc:
        await log("error", f"Coqui TTS not available ({exc}); falling back to Piper.")
        voice_path, voice_config, language = fallback_voice
        return await generate_voiceovers(
            texts,
            out_dir,
            log,
            voice_path=voice_path,
            voice_config=voice_config,
            language=language,
            engine="piper",
        )

    model_name = model_name or "tts_models/en/vctk/vits"
    tts = _coqui_cache.get(model_name)
    if tts is None:
        await log("info", f"Loading Coqui TTS model: {model_name}")
        tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
        _coqui_cache[model_name] = tts

    outputs: List[str] = []
    for idx, text in enumerate(texts, start=1):
        out_path = os.path.join(out_dir, f"voice_{idx}.wav")
        await log("info", f"Generating voiceover {idx}/{len(texts)} with Coqui")
        await asyncio.to_thread(tts.tts_to_file, text=text, file_path=out_path, speaker=speaker or None)
        outputs.append(out_path)
    return outputs


def _resolve_voice(
    language: Optional[str],
    voice_path: Optional[str],
    voice_config: Optional[str],
) -> Tuple[str, Optional[str], Optional[str]]:
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
        return PIPER_VOICE_PATH, config, note or f"Using default voice: {PIPER_VOICE_PATH}"

    return voice_path or PIPER_VOICE_PATH, voice_config or PIPER_VOICE_CONFIG, note


def _pick_config(model_path: str, config_path: Optional[str]) -> Optional[str]:
    if config_path and os.path.exists(config_path):
        return config_path
    guessed = f"{model_path}.json"
    if os.path.exists(guessed):
        return guessed
    return None
