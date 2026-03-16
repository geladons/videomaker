from __future__ import annotations

import json
import os
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite

from config import DATA_DIR, DB_PATH

_db: Optional[aiosqlite.Connection] = None
_write_queue: Optional[asyncio.Queue] = None
_writer_task: Optional[asyncio.Task] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    options TEXT NOT NULL,
    output_path TEXT,
    error TEXT,
    progress INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS task_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _utcnow() -> str:
    return datetime.utcnow().isoformat()


async def init_db() -> None:
    global _db, _write_queue, _writer_task
    os.makedirs(DATA_DIR, exist_ok=True)
    if _db is None:
        _db = await aiosqlite.connect(DB_PATH)
        await _db.execute("PRAGMA journal_mode=WAL;")
        await _db.execute("PRAGMA synchronous=NORMAL;")
        await _db.execute("PRAGMA busy_timeout=5000;")
        await _db.executescript(SCHEMA)
        await _db.commit()

    if _write_queue is None:
        _write_queue = asyncio.Queue()
    if _writer_task is None:
        _writer_task = asyncio.create_task(_writer_loop())


async def _writer_loop() -> None:
    assert _db is not None
    assert _write_queue is not None
    while True:
        func, future = await _write_queue.get()
        try:
            result = await func(_db)
            await _db.commit()
            future.set_result(result)
        except Exception as exc:  # pragma: no cover - defensive
            try:
                await _db.rollback()
            except Exception:
                pass
            future.set_exception(exc)
        finally:
            _write_queue.task_done()


async def _enqueue_write(func) -> Any:
    if _write_queue is None:
        await init_db()
    assert _write_queue is not None
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    await _write_queue.put((func, future))
    return await future


def _read_connection() -> aiosqlite.Connection:
    return aiosqlite.connect(DB_PATH, timeout=30)


async def create_task(task_id: str, prompt: str, options: Dict[str, Any]) -> None:
    now = _utcnow()
    async def _write(db: aiosqlite.Connection) -> None:
        await db.execute(
            "INSERT INTO tasks (id, prompt, status, created_at, updated_at, options, progress) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, prompt, "Pending", now, now, json.dumps(options), 0),
        )
    await _enqueue_write(_write)


async def update_task_status(
    task_id: str,
    status: str,
    output_path: Optional[str] = None,
    error: Optional[str] = None,
    progress: Optional[int] = None,
) -> None:
    now = _utcnow()
    fields = ["status = ?", "updated_at = ?"]
    params: List[Any] = [status, now]

    if output_path is not None:
        fields.append("output_path = ?")
        params.append(output_path)
    if error is not None:
        fields.append("error = ?")
        params.append(error)
    if progress is not None:
        fields.append("progress = ?")
        params.append(progress)

    params.append(task_id)

    async def _write(db: aiosqlite.Connection) -> None:
        await db.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?", params)
    await _enqueue_write(_write)


async def add_log(task_id: str, level: str, message: str) -> None:
    async def _write(db: aiosqlite.Connection) -> None:
        await db.execute(
            "INSERT INTO task_logs (task_id, timestamp, level, message) VALUES (?, ?, ?, ?)",
            (task_id, _utcnow(), level, message),
        )
    await _enqueue_write(_write)


async def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    async with _read_connection() as db:
        await db.execute("PRAGMA busy_timeout=5000;")
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["options"] = json.loads(data["options"])
            return data


async def list_tasks(limit: int = 50) -> List[Dict[str, Any]]:
    async with _read_connection() as db:
        await db.execute("PRAGMA busy_timeout=5000;")
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)) as cursor:
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                data["options"] = json.loads(data["options"])
                results.append(data)
            return results


async def list_logs(task_id: str, limit: int = 5000) -> List[Dict[str, Any]]:
    async with _read_connection() as db:
        await db.execute("PRAGMA busy_timeout=5000;")
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM task_logs WHERE task_id = ? ORDER BY id ASC LIMIT ?",
            (task_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def set_setting(key: str, value: Any) -> None:
    payload = json.dumps(value)
    async def _write(db: aiosqlite.Connection) -> None:
        await db.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, payload),
        )
    await _enqueue_write(_write)


async def get_setting(key: str, default: Any = None) -> Any:
    async with _read_connection() as db:
        await db.execute("PRAGMA busy_timeout=5000;")
        async with db.execute("SELECT value FROM settings WHERE key = ?", (key,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return default
            return json.loads(row[0])


async def get_all_settings() -> Dict[str, Any]:
    async with _read_connection() as db:
        await db.execute("PRAGMA busy_timeout=5000;")
        async with db.execute("SELECT key, value FROM settings") as cursor:
            rows = await cursor.fetchall()
            return {row[0]: json.loads(row[1]) for row in rows}
