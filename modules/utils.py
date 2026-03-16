from __future__ import annotations

import asyncio
import os
import shlex
from typing import Awaitable, Callable, Dict, Iterable, List, Optional

LogFn = Callable[[str, str], Awaitable[None]]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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

    await asyncio.gather(
        _stream(process.stdout, "info"),
        _stream(process.stderr, "error"),
    )

    return await process.wait()


async def run_shell(
    command: str,
    log: LogFn,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    return await run_command(["bash", "-lc", command], log=log, cwd=cwd, env=env)
