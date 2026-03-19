from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    DEFAULT_DIRS,
    OLLAMA_SETTINGS,
    DEFAULT_PIPELINE,
    DEFAULT_SCRAPER,
    DEFAULT_VIDEO,
    DEFAULT_TTS_ENGINE,
    DEFAULT_COQUI_MODEL,
    DEFAULT_COQUI_SPEAKER,
    DEFAULT_VOICEOVER_WPS,
    OLLAMA_API_URL,
    OUTPUT_DIR,
    SUPPORTED_LANGUAGES,
)
from database import (
    get_all_settings,
    get_setting,
    init_db,
    list_logs,
    list_tasks,
    set_setting,
    fail_orphaned_tasks,
)
from orchestrator import Orchestrator

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")


templates = Jinja2Templates(directory="app/templates")


class LogHub:
    def __init__(self) -> None:
        self._connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, task_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.setdefault(task_id, []).append(websocket)

    async def disconnect(self, task_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            if task_id in self._connections:
                self._connections[task_id] = [ws for ws in self._connections[task_id] if ws != websocket]

    async def broadcast(self, task_id: str, payload: Dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._connections.get(task_id, [])) + list(self._connections.get("all", []))
        for websocket in targets:
            try:
                await websocket.send_text(json.dumps(payload))
            except WebSocketDisconnect:
                await self.disconnect(task_id, websocket)


log_hub = LogHub()


async def log_callback(task_id: str, level: str, message: str) -> None:
    payload = {"task_id": task_id, "level": level, "message": message}
    await log_hub.broadcast(task_id, payload)


orchestrator = Orchestrator(log_callback)


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()
    await fail_orphaned_tasks()

    # Consolidate Ollama settings setup
    for key, settings in OLLAMA_SETTINGS.items():
        prefix = f"ollama_{key}_" if key != "default" else "ollama_"
        if await get_setting(f"{prefix}model") is None:
            await set_setting(f"{prefix}model", settings["model"])
        if await get_setting(f"{prefix}params") is None:
            await set_setting(f"{prefix}params", settings["params"])
        if await get_setting(f"{prefix}timeout") is None:
            await set_setting(f"{prefix}timeout", settings["timeout"])
        if await get_setting(f"{prefix}think") is None:
            await set_setting(f"{prefix}think", settings["think"])
        # Special case for vision 'enabled' flag
        if key == "vision" and await get_setting(f"{prefix}enabled") is None:
            await set_setting(f"{prefix}enabled", settings["enabled"])
        # API URL can be set per-service or fallback to global
        if await get_setting(f"{prefix}api_url") is None:
             await set_setting(f"{prefix}api_url", OLLAMA_API_URL)

    # General settings
    if await get_setting("video_settings") is None:
        await set_setting("video_settings", DEFAULT_VIDEO)
    if await get_setting("supported_languages") is None:
        await set_setting("supported_languages", SUPPORTED_LANGUAGES)
    if await get_setting("pipeline_defaults") is None:
        await set_setting("pipeline_defaults", DEFAULT_PIPELINE)
    if await get_setting("voiceover_words_per_sec") is None:
        await set_setting("voiceover_words_per_sec", DEFAULT_VOICEOVER_WPS)
    if await get_setting("tts_engine") is None:
        await set_setting("tts_engine", DEFAULT_TTS_ENGINE)
    if await get_setting("coqui_model") is None:
        await set_setting("coqui_model", DEFAULT_COQUI_MODEL)
    if await get_setting("coqui_speaker") is None:
        await set_setting("coqui_speaker", DEFAULT_COQUI_SPEAKER)
    if await get_setting("scraper_settings") is None:
        await set_setting("scraper_settings", DEFAULT_SCRAPER)

    await _ensure_valid_ollama_model()
    await orchestrator.start()


async def _ensure_valid_ollama_model() -> None:
    async def _fetch_models(api_url: str) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{api_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [model.get("name") for model in data.get("models", []) if model.get("name")]
        except httpx.HTTPError:
            return []

    for key, settings in OLLAMA_SETTINGS.items():
        prefix = f"ollama_{key}_" if key != "default" else "ollama_"
        api_url = await get_setting(f"{prefix}api_url", OLLAMA_API_URL)
        selected_model = await get_setting(f"{prefix}model", settings["model"])
        
        available_models = await _fetch_models(api_url)
        
        if available_models and selected_model not in available_models:
            await set_setting(f"{prefix}model", available_models[0])
            if key == "vision":
                await set_setting(f"{prefix}enabled", False)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("settings.html", {"request": request})


@app.get("/settings/helper")
async def settings_helper_redirect() -> RedirectResponse:
    return RedirectResponse(url="/settings#helper")


@app.get("/settings/vision")
async def settings_vision_redirect() -> RedirectResponse:
    return RedirectResponse(url="/settings#vision")


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/api/tasks")
async def api_tasks() -> JSONResponse:
    tasks = await list_tasks(limit=100)
    return JSONResponse({"tasks": tasks})


@app.get("/api/tasks/{task_id}")
async def api_task(task_id: str) -> JSONResponse:
    from database import get_task

    task = await get_task(task_id)
    return JSONResponse({"task": task})


@app.post("/api/tasks/{task_id}/cancel")
async def api_cancel_task(task_id: str) -> JSONResponse:
    success = await orchestrator.cancel_task(task_id)
    if success:
        return JSONResponse({"status": "cancelled"})
    return JSONResponse({"error": "Task not active or not found"}, status_code=400)

@app.get("/api/tasks/{task_id}/logs")
async def api_logs(task_id: str, limit: int = 5000) -> JSONResponse:
    logs = await list_logs(task_id, limit=limit)
    return JSONResponse({"logs": logs})


@app.post("/api/generate")
async def api_generate(payload: Dict[str, Any]) -> JSONResponse:
    prompt = payload.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "Prompt is required"}, status_code=400)

    defaults = await get_setting("pipeline_defaults", DEFAULT_PIPELINE)
    format_choice = payload.get("format", "16:9")
    if format_choice not in {"16:9", "9:16"}:
        format_choice = "16:9"
    language = payload.get("language", "English")
    try:
        duration = int(payload.get("duration", 30))
    except (TypeError, ValueError):
        duration = 30
    duration = max(5, duration)

    options = {
        "format": format_choice,
        "language": language,
        "duration": duration,
        "add_music": bool(payload.get("add_music", defaults.get("add_music", True))),
        "add_greeting": bool(payload.get("add_greeting", defaults.get("add_greeting", False))),
        "add_closing": bool(payload.get("add_closing", defaults.get("add_closing", False))),
        "use_stock_video": bool(payload.get("use_stock_video", defaults.get("use_stock_video", True))),
        "use_images": bool(payload.get("use_images", defaults.get("use_images", True))),
        "burn_subtitles": bool(payload.get("burn_subtitles", defaults.get("burn_subtitles", True))),
    }

    full_prompt = (
        f"{prompt}\n\n"
        f"Target duration: {duration} seconds."
        f" Output language: {language}."
        f" Format: {format_choice}."
    )

    task_id = await orchestrator.enqueue(full_prompt, options)
    return JSONResponse({"task_id": task_id})


@app.get("/api/settings")
async def api_settings() -> JSONResponse:
    settings = await get_all_settings()

    async def _fetch_models(api_url: str) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{api_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [model.get("name") for model in data.get("models", []) if model.get("name")]
        except httpx.HTTPError:
            return []

    api_url = settings.get("ollama_api_url", OLLAMA_API_URL)
    planner_api_url = settings.get("ollama_planner_api_url", OLLAMA_API_URL)
    helper_api_url = settings.get("ollama_helper_api_url", OLLAMA_API_URL)
    vision_api_url = settings.get("ollama_vision_api_url", OLLAMA_API_URL)

    models = await _fetch_models(api_url)
    planner_models = await _fetch_models(planner_api_url)
    helper_models = await _fetch_models(helper_api_url)
    vision_models = await _fetch_models(vision_api_url)

    return JSONResponse(
        {
            "settings": settings,
            "models": models,
            "planner_models": planner_models,
            "helper_models": helper_models,
            "vision_models": vision_models,
        }
    )


@app.post("/api/settings")
async def api_settings_update(payload: Dict[str, Any]) -> JSONResponse:
    # Loop through consolidated Ollama settings
    for key, settings in OLLAMA_SETTINGS.items():
        prefix = f"ollama_{key}_" if key != "default" else "ollama_"
        await set_setting(f"{prefix}api_url", payload.get(f"{prefix}api_url", OLLAMA_API_URL))
        await set_setting(f"{prefix}model", payload.get(f"{prefix}model", settings["model"]))
        await set_setting(f"{prefix}params", payload.get(f"{prefix}params", settings["params"]))
        await set_setting(f"{prefix}timeout", payload.get(f"{prefix}timeout", settings["timeout"]))
        await set_setting(f"{prefix}think", payload.get(f"{prefix}think", settings["think"]))
        if key == "vision":
            await set_setting(f"{prefix}enabled", payload.get(f"{prefix}enabled", settings["enabled"]))

    # General settings
    await set_setting("video_settings", payload.get("video_settings", DEFAULT_VIDEO))
    await set_setting("voiceover_words_per_sec", payload.get("voiceover_words_per_sec", DEFAULT_VOICEOVER_WPS))
    await set_setting("tts_engine", payload.get("tts_engine", DEFAULT_TTS_ENGINE))
    await set_setting("coqui_model", payload.get("coqui_model", DEFAULT_COQUI_MODEL))
    await set_setting("coqui_speaker", payload.get("coqui_speaker", DEFAULT_COQUI_SPEAKER))
    await set_setting("tts_voice_path", payload.get("tts_voice_path", None))
    await set_setting("tts_voice_config", payload.get("tts_voice_config", None))
    await set_setting("pipeline_defaults", payload.get("pipeline_defaults", DEFAULT_PIPELINE))
    await set_setting("scraper_settings", payload.get("scraper_settings", DEFAULT_SCRAPER))
    return JSONResponse({"status": "ok"})


@app.get("/api/ollama/tags")
async def api_ollama_tags(url: str) -> JSONResponse:
    models: list[str] = []
    if not url:
        return JSONResponse({"models": []}, status_code=400)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [model.get("name") for model in data.get("models", []) if model.get("name")]
    except httpx.HTTPError:
        models = []
    return JSONResponse({"models": models})


@app.get("/api/download/{task_id}")
async def api_download(task_id: str) -> FileResponse:
    from database import get_task

    task = await get_task(task_id)
    if not task or not task.get("output_path"):
        raise HTTPException(status_code=404, detail="Output not found")
    output_path = task["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output missing on disk")
    return FileResponse(output_path, filename=f"{task_id}.mp4")


@app.get("/api/logs/{task_id}")
async def api_download_logs(task_id: str) -> FileResponse:
    log_path = os.path.join(OUTPUT_DIR, DEFAULT_DIRS["logs"], f"{task_id}.log")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")
    return FileResponse(log_path, filename=f"{task_id}.log")


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket) -> None:
    task_id = websocket.query_params.get("task_id", "all")
    await log_hub.connect(task_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await log_hub.disconnect(task_id, websocket)
