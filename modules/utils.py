from __future__ import annotations

import asyncio
import os
import re
import shlex
from typing import Awaitable, Callable, Dict, Iterable, List, Optional

LogFn = Callable[[str, str], Awaitable[None]]

# Shared stopwords for query cleaning across modules
STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "to", "in", "on", "with", "from",
    "about", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "but", "if", "or", "because", "as", "until",
    "while", "at", "by", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "up", "down", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}


def clean_and_tokenize(text: str, max_words: int = 5) -> str:
    """Shared utility to clean text and extract keywords for search queries."""
    if not text:
        return ""
    # Clean: lower, strip non-alphanumeric (keep spaces)
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    # Tokenize and filter using common stopwords
    tokens = [t for t in cleaned.split() if t and t not in STOPWORDS]
    if not tokens:
        # Fallback to raw tokens if everything was filtered
        tokens = [t for t in cleaned.split() if t][:max_words]
    return " ".join(tokens[:max_words])


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_wav_duration(file_path: str) -> float:
    """
    Get duration of a WAV file in seconds.
    Uses wave module for standard PCM and provides a fallback/safeguard.
    """
    import contextlib
    import wave

    if not os.path.exists(file_path):
        return 0.0
    try:
        with contextlib.closing(wave.open(file_path, "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception:
        # Fallback for non-standard WAVs if needed or just return 0
        return 0.0


async def run_command(
    cmd: List[str],
    log: LogFn,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    await log("info", f"$ {shlex.join(cmd)}")
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    async def _stream(reader: asyncio.StreamReader, level: str) -> None:
        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode(errors="ignore").rstrip()
            if text:
                lowered = text.lower()
                if "error" in lowered or "failed" in lowered:
                    await log("error", text)
                elif "warn" in lowered:
                    await log("warn", text)
                else:
                    await log(level, text)

    try:
        await asyncio.gather(
            _stream(process.stdout, "info"),
            _stream(process.stderr, "error"),
        )
        return await process.wait()
    except asyncio.CancelledError:
        try:
            process.kill()
        except OSError:
            pass
        raise
    finally:
        try:
            if process.returncode is None:
                process.kill()
        except OSError:
            pass


async def run_shell(
    command: str,
    log: LogFn,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    return await run_command(["bash", "-lc", command], log=log, cwd=cwd, env=env)
