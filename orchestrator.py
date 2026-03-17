from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from config import (
    DEFAULT_COQUI_MODEL,
    DEFAULT_COQUI_SPEAKER,
    DEFAULT_DIRS,
    DEFAULT_OLLAMA_HELPER_MODEL,
    DEFAULT_OLLAMA_HELPER_PARAMS,
    DEFAULT_OLLAMA_HELPER_THINK,
    DEFAULT_OLLAMA_HELPER_TIMEOUT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_PARAMS,
    DEFAULT_OLLAMA_PLANNER_MODEL,
    DEFAULT_OLLAMA_PLANNER_PARAMS,
    DEFAULT_OLLAMA_PLANNER_THINK,
    DEFAULT_OLLAMA_PLANNER_TIMEOUT,
    DEFAULT_OLLAMA_REQUEST_DELAY,
    DEFAULT_OLLAMA_VISION_ENABLED,
    DEFAULT_OLLAMA_VISION_MODEL,
    DEFAULT_OLLAMA_VISION_PARAMS,
    DEFAULT_OLLAMA_VISION_THINK,
    DEFAULT_OLLAMA_VISION_TIMEOUT,
    DEFAULT_SCRAPER,
    DEFAULT_TTS_ENGINE,
    DEFAULT_VIDEO,
    DEFAULT_VOICEOVER_WPS,
    LLM_DEFAULT_TIMEOUT,
    OLLAMA_API_URL,
    OUTPUT_DIR,
    WORKSPACE_DIR,
)
from database import add_log, create_task, get_setting, update_task_status
from modules import compositor, llm, scraper, subtitles, tts_engine, vision

LogFn = Callable[[str, str], Awaitable[None]]


class Orchestrator:
    def __init__(self, log_callback: Callable[[str, str, str], Awaitable[None]]):
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.active_task_id: Optional[str] = None
        self.log_callback = log_callback
        self.worker_task: Optional[asyncio.Task] = None
        self.log_dir = os.path.join(OUTPUT_DIR, DEFAULT_DIRS["logs"])

    async def _get_effective_settings(self) -> Dict[str, Any]:
        """Fetch all configuration with database overrides and config.py fallbacks."""
        return {
            "ollama_api_url": await get_setting("ollama_api_url", OLLAMA_API_URL),
            "ollama_model": await get_setting("ollama_model", DEFAULT_OLLAMA_MODEL),
            "ollama_params": await get_setting("ollama_params", DEFAULT_OLLAMA_PARAMS),
            "ollama_timeout": await get_setting("ollama_timeout", LLM_DEFAULT_TIMEOUT),
            "ollama_think": await get_setting("ollama_think", False),
            "ollama_request_delay": await get_setting(
                "ollama_request_delay", DEFAULT_OLLAMA_REQUEST_DELAY
            ),
            "planner": {
                "model": await get_setting(
                    "ollama_planner_model", DEFAULT_OLLAMA_PLANNER_MODEL
                ),
                "params": await get_setting(
                    "ollama_planner_params", DEFAULT_OLLAMA_PLANNER_PARAMS
                ),
                "api_url": await get_setting("ollama_planner_api_url", OLLAMA_API_URL),
                "timeout": await get_setting(
                    "ollama_planner_timeout", DEFAULT_OLLAMA_PLANNER_TIMEOUT
                ),
                "think": await get_setting(
                    "ollama_planner_think", DEFAULT_OLLAMA_PLANNER_THINK
                ),
            },
            "helper": {
                "api_url": await get_setting("ollama_helper_api_url", OLLAMA_API_URL),
                "model": await get_setting(
                    "ollama_helper_model", DEFAULT_OLLAMA_HELPER_MODEL
                ),
                "params": await get_setting(
                    "ollama_helper_params", DEFAULT_OLLAMA_HELPER_PARAMS
                ),
                "timeout": await get_setting(
                    "ollama_helper_timeout", DEFAULT_OLLAMA_HELPER_TIMEOUT
                ),
                "think": await get_setting(
                    "ollama_helper_think", DEFAULT_OLLAMA_HELPER_THINK
                ),
            },
            "vision": {
                "api_url": await get_setting("ollama_vision_api_url", OLLAMA_API_URL),
                "model": await get_setting(
                    "ollama_vision_model", DEFAULT_OLLAMA_VISION_MODEL
                ),
                "params": await get_setting(
                    "ollama_vision_params", DEFAULT_OLLAMA_VISION_PARAMS
                ),
                "timeout": await get_setting(
                    "ollama_vision_timeout", DEFAULT_OLLAMA_VISION_TIMEOUT
                ),
                "think": await get_setting(
                    "ollama_vision_think", DEFAULT_OLLAMA_VISION_THINK
                ),
                "enabled": await get_setting(
                    "ollama_vision_enabled", DEFAULT_OLLAMA_VISION_ENABLED
                ),
            },
            "scraper": await get_setting("scraper_settings", DEFAULT_SCRAPER),
            "video": await get_setting("video_settings", DEFAULT_VIDEO),
            "tts": {
                "engine": await get_setting("tts_engine", DEFAULT_TTS_ENGINE),
                "voice_path": await get_setting("tts_voice_path", None),
                "voice_config": await get_setting("tts_voice_config", None),
                "coqui_model": await get_setting("coqui_model", DEFAULT_COQUI_MODEL),
                "coqui_speaker": await get_setting(
                    "coqui_speaker", DEFAULT_COQUI_SPEAKER
                ),
                "voiceover_wps": float(
                    await get_setting("voiceover_words_per_sec", DEFAULT_VOICEOVER_WPS)
                ),
            },
            "ai_query": {
                "model": await get_setting("ai_query_model", None),
                "api_url": await get_setting("ai_query_api_url", OLLAMA_API_URL),
            },
        }

    async def start(self) -> None:
        if self.worker_task is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.worker_task = asyncio.create_task(self._worker())

    async def enqueue(self, prompt: str, options: Dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())
        await create_task(task_id, prompt, options)
        await self.queue.put(task_id)
        return task_id

    async def _log(self, task_id: str, level: str, message: str) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.utcnow().isoformat()
        log_line = f"[{timestamp}] {level.upper()}: {message}\n"
        log_path = os.path.join(self.log_dir, f"{task_id}.log")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line)
        except OSError:
            pass

        if level.lower() in {"raw", "debug"}:
            return

        db_message = (
            message if len(message) <= 2000 else message[:2000] + " ...[truncated]"
        )
        await add_log(task_id, level, db_message)
        await self.log_callback(task_id, level, db_message)

    async def _set_progress(self, task_id: str, progress: int) -> None:
        await update_task_status(task_id, "Running", progress=progress)
        await self.log_callback(task_id, "progress", json.dumps({"progress": progress}))

    async def _worker(self) -> None:
        while True:
            task_id = await self.queue.get()
            self.active_task_id = task_id
            await self._run_task(task_id)
            self.active_task_id = None
            self.queue.task_done()

    async def _run_task(self, task_id: str) -> None:
        from database import get_task

        task = await get_task(task_id)
        if not task:
            return

        options = task["options"]
        prompt = task["prompt"]
        workspace = os.path.join(WORKSPACE_DIR, task_id)
        os.makedirs(workspace, exist_ok=True)

        await update_task_status(task_id, "Running", progress=0)
        await self._log(task_id, "info", "Task started")

        output_path = None
        try:
            cfg = await self._get_effective_settings()

            await self._set_progress(task_id, 5)
            await self._log(
                task_id, "info", f"Ollama request delay: {cfg['ollama_request_delay']}s"
            )
            await self._log(task_id, "info", f"Scraper settings: {cfg['scraper']}")
            await self._log(task_id, "info", "Requesting plan from Planner LLM")
            timeline = await llm.plan_timeline(
                prompt,
                model=cfg["planner"]["model"],
                options=cfg["planner"]["params"],
                api_url=cfg["planner"]["api_url"],
                timeout=float(cfg["planner"]["timeout"]),
                think=bool(cfg["planner"]["think"]),
                helper_settings=cfg["helper"],
                target_duration=float(options.get("duration", 0) or 0),
                request_delay=float(cfg["ollama_request_delay"]),
                log=lambda lvl, msg: self._log(task_id, lvl, msg),
            )

            scenes = timeline.get("scenes", [])
            await self._normalize_scene_durations(
                task_id,
                scenes,
                float(options.get("duration", timeline.get("total_duration", 0)) or 0),
            )
            await self._set_progress(task_id, 15)
            await self._log(task_id, "info", f"Generated {len(scenes)} scenes")

            timeline_preview = json.dumps(timeline, ensure_ascii=False)
            await self._log(task_id, "info", f"LLM timeline: {timeline_preview[:500]}")
            await self._log(
                task_id,
                "raw",
                f"LLM timeline (full): {json.dumps(timeline, ensure_ascii=False, indent=2)}",
            )

            for scene in scenes:
                if not scene.get("overlay_text"):
                    scene["overlay_text"] = (
                        scene.get("visual_query") or scene.get("voiceover") or ""
                    ).strip()

            await self._log(
                task_id, "info", "Generating scene voiceovers with Writer LLM"
            )
            num_scenes = len(scenes)
            for idx, scene in enumerate(scenes, start=1):
                # Voiceover script generation: 15% -> 30%
                p = 15 + int((idx / num_scenes) * 15)
                await self._set_progress(task_id, p)
                
                duration = float(scene.get("duration", 0) or 0)
                target_words = max(3, int(duration * cfg["tts"]["voiceover_wps"]))
                try:
                    voice_text = await llm.generate_voiceover(
                        scene,
                        language=options.get("language", "English"),
                        target_words=target_words,
                        model=cfg["ollama_model"],
                        options=cfg["ollama_params"],
                        api_url=cfg["ollama_api_url"],
                        timeout=float(cfg["ollama_timeout"]),
                        think=bool(cfg["ollama_think"]),
                        add_greeting=bool(options.get("add_greeting", False)),
                        add_closing=bool(options.get("add_closing", False)),
                        is_first=idx == 1,
                        is_last=idx == len(scenes),
                        request_delay=float(cfg["ollama_request_delay"]),
                        log=lambda lvl, msg: self._log(task_id, lvl, msg),
                    )
                    scene["voiceover"] = voice_text
                    await self._log(
                        task_id,
                        "info",
                        f"Voiceover {idx}/{len(scenes)}: {len(voice_text.split())} words",
                    )
                except Exception as exc:
                    await self._log(
                        task_id,
                        "error",
                        f"Voiceover generation failed for scene {idx}: {exc}",
                    )
                    scene["voiceover"] = scene.get("voiceover", "")

            await self._set_progress(task_id, 30)
            use_stock_video = bool(options.get("use_stock_video", True))
            use_images = bool(options.get("use_images", True))
            
            # Limit fallback topic to prevent massive search queries
            fallback_topic = prompt.splitlines()[0].strip() if prompt else None
            if fallback_topic and len(fallback_topic) > 100:
                fallback_topic = fallback_topic[:100] + "..."

            # Ensure workspace and subdirs exist
            os.makedirs(workspace, exist_ok=True)
            music_dir = os.path.join(workspace, DEFAULT_DIRS["music"])
            os.makedirs(music_dir, exist_ok=True)

            # Asset gathering: 30% -> 60%
            async def asset_progress(p_module: float):
                # p_module is 0.0 to 1.0
                p_overall = 30 + int(p_module * 30)
                await self._set_progress(task_id, p_overall)

            try:
                assets = await scraper.gather_scene_assets(
                    scenes,
                    workspace,
                    use_stock_video,
                    use_images,
                    lambda lvl, msg: self._log(task_id, lvl, msg),
                    fallback_topic=fallback_topic,
                    scraper_settings=cfg["scraper"],
                    ai_query_model=cfg["ai_query"]["model"],
                    ai_query_api_url=cfg["ai_query"]["api_url"],
                    progress_fn=asset_progress,
                )
            except Exception as e:
                await self._log(task_id, "error", f"Asset gathering failed: {e}")
                # Don't re-raise yet, try to continue with empty assets if possible
                assets = [{"video": None, "image": None} for _ in scenes]

            await self._set_progress(task_id, 60)
            fallback_video = next(
                (a.get("video") for a in assets if a.get("video")), None
            )
            fallback_image = next(
                (a.get("image") for a in assets if a.get("image")), None
            )
            if fallback_video or fallback_image:
                for entry in assets:
                    if use_stock_video and not entry.get("video") and fallback_video:
                        entry["video"] = fallback_video
                    if use_images and not entry.get("image") and fallback_image:
                        entry["image"] = fallback_image
                await self._log(
                    task_id,
                    "info",
                    "Reused first available video/image for scenes missing assets.",
                )

            if cfg["vision"]["enabled"]:
                await self._log(
                    task_id, "info", "Running vision analysis on downloaded images"
                )
                for idx, scene in enumerate(scenes, start=1):
                    # Vision analysis: 60% -> 70%
                    p = 60 + int((idx / num_scenes) * 10)
                    await self._set_progress(task_id, p)
                    
                    image_path = (
                        assets[idx - 1].get("image") if idx - 1 < len(assets) else None
                    )
                    if not image_path:
                        continue
                    try:
                        analysis = await vision.analyze_image(
                            image_path,
                            api_url=cfg["vision"]["api_url"],
                            model=cfg["vision"]["model"],
                            options=cfg["vision"]["params"],
                            timeout=float(cfg["vision"]["timeout"]),
                            think=bool(cfg["vision"].get("think", False)),
                            prompt=scene.get("visual_query"),
                            log=lambda lvl, msg: self._log(task_id, lvl, msg),
                        )
                        caption = analysis.get("caption")
                        tags = analysis.get("tags") or []
                        if caption and not scene.get("overlay_text"):
                            scene["overlay_text"] = caption
                        if tags:
                            scene["visual_query"] = (
                                f"{scene.get('visual_query', '')} {' '.join(tags[:3])}".strip()
                            )
                    except Exception as exc:
                        await self._log(
                            task_id,
                            "error",
                            f"Vision analysis failed for scene {idx}: {exc}",
                        )

            await self._set_progress(task_id, 70)
            texts = [scene.get("voiceover", "") for scene in scenes]
            voice_dir = os.path.join(workspace, DEFAULT_DIRS["voice"])
            await self._log(
                task_id,
                "info",
                f"TTS engine: {cfg['tts']['engine']} (coqui_model={cfg['tts']['coqui_model']}, speaker={cfg['tts']['coqui_speaker']})",
            )
            
            # TTS generation: 70% -> 85%
            async def tts_progress(p_module: float):
                p_overall = 70 + int(p_module * 15)
                await self._set_progress(task_id, p_overall)

            voiceovers = await tts_engine.generate_voiceovers(
                texts,
                voice_dir,
                lambda lvl, msg: self._log(task_id, lvl, msg),
                voice_path=cfg["tts"]["voice_path"],
                voice_config=cfg["tts"]["voice_config"],
                language=options.get("language", "English"),
                engine=cfg["tts"]["engine"],
                coqui_model=cfg["tts"]["coqui_model"],
                coqui_speaker=cfg["tts"]["coqui_speaker"],
                progress_fn=tts_progress,
            )

            await self._set_progress(task_id, 85)
            subtitle_path = None
            if options.get("burn_subtitles", True):
                offsets = []
                total = 0.0
                for scene in scenes:
                    offsets.append(total)
                    total += float(scene.get("duration", 0))
                subtitle_path = os.path.join(workspace, "subtitles.ass")
                await subtitles.generate_ass(
                    voiceovers,
                    offsets,
                    subtitle_path,
                    lambda lvl, msg: self._log(task_id, lvl, msg),
                    video_settings=cfg["video"],
                    format_choice=options.get("format", "16:9"),
                )

            await self._set_progress(task_id, 90)
            music_path = None
            music_dir = os.path.join(workspace, DEFAULT_DIRS["music"])

            if options.get("add_music", True):
                # Ensure music directory exists before download
                os.makedirs(music_dir, exist_ok=True)
                
                base_query = (
                    f"{timeline.get('music_mood', 'cinematic')} background music"
                )
                queries = [
                    base_query,
                    f"{fallback_topic} background music" if fallback_topic else None,
                    "ambient background music",
                    "cinematic music",
                ]
                for query in [q for q in queries if q]:
                    # Defensive check: recreate dir if disappeared
                    if not os.path.exists(music_dir):
                        await self._log(task_id, "warn", f"Music directory missing, recreating: {music_dir}")
                        os.makedirs(music_dir, exist_ok=True)
                        
                    await self._log(
                        task_id, "info", f"Trying to download music: {query}"
                    )
                    try:
                        music_path = await scraper.download_cc_audio(
                            query,
                            music_dir,
                            lambda lvl, msg: self._log(task_id, lvl, msg),
                            scraper_settings=cfg["scraper"],
                            mood=timeline.get("music_mood", "cinematic"),
                        )
                        if music_path and os.path.exists(music_path):
                            await self._log(task_id, "info", f"Successfully obtained music: {os.path.basename(music_path)}")
                            break
                    except Exception as e:
                        await self._log(task_id, "error", f"Music download attempt failed for query '{query}': {e}")

                # If still no music, check if there's any existing audio file in music dir
                if not music_path:
                    import os as os_module

                    if os_module.path.exists(music_dir):
                        for f in os_module.listdir(music_dir):
                            if f.endswith((".mp3", ".wav", ".m4a", ".ogg")):
                                music_path = os_module.path.join(music_dir, f)
                                await self._log(
                                    task_id, "info", f"Using existing audio file: {f}"
                                )
                                break

                if not music_path:
                    await self._log(
                        task_id,
                        "warn",
                        "No music found, continuing without background music",
                    )

            await self._set_progress(task_id, 95)
            output_path = os.path.join(OUTPUT_DIR, f"{task_id}.mp4")
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            await compositor.compose_video(
                scenes,
                assets,
                voiceovers,
                output_path,
                workspace,
                lambda lvl, msg: self._log(task_id, lvl, msg),
                format_choice=options.get("format", "16:9"),
                fps=int(cfg["video"].get("fps", 30)),
                video_settings=cfg["video"],
                background_music=music_path,
                burn_subtitles=subtitle_path,
            )

            await self._set_progress(task_id, 100)
            await update_task_status(
                task_id, "Completed", output_path=output_path, progress=100
            )
            await self._log(task_id, "info", "Task completed successfully")
        except Exception as exc:
            await update_task_status(task_id, "Failed", error=str(exc))
            await self._log(task_id, "error", f"Task failed: {exc}")
        finally:
            self._cleanup_workspace(workspace, keep=[output_path])

    async def _normalize_scene_durations(
        self,
        task_id: str,
        scenes: list[dict[str, Any]],
        target_duration: float,
    ) -> None:
        if not scenes or target_duration <= 0:
            return

        durations: list[float] = []
        for scene in scenes:
            try:
                durations.append(max(0.0, float(scene.get("duration", 0))))
            except (TypeError, ValueError):
                durations.append(0.0)

        total = sum(durations)
        if total <= 0:
            per = round(target_duration / len(scenes), 2)
            for scene in scenes:
                scene["duration"] = per
            await self._log(
                task_id,
                "info",
                f"Assigned uniform scene durations for {target_duration:.2f}s target",
            )
            return

        scale = target_duration / total
        for idx, scene in enumerate(scenes):
            scene["duration"] = round(durations[idx] * scale, 2)

        adjusted_total = sum(float(scene.get("duration", 0)) for scene in scenes)
        diff = round(target_duration - adjusted_total, 2)
        if abs(diff) > 0 and scenes:
            scenes[-1]["duration"] = round(
                float(scenes[-1].get("duration", 0)) + diff, 2
            )

        await self._log(
            task_id, "info", f"Normalized durations to {target_duration:.2f}s target"
        )

    def _cleanup_workspace(
        self, workspace: str, keep: Optional[list[str]] = None
    ) -> None:
        keep = [os.path.abspath(path) for path in (keep or []) if path]
        if not os.path.exists(workspace):
            return
            
        workspace_abs = os.path.abspath(workspace)
        
        # NEVER delete the entire WORKSPACE_DIR
        if workspace_abs == os.path.abspath(WORKSPACE_DIR):
            return

        for root, dirs, files in os.walk(workspace, topdown=False):
            for name in files:
                path = os.path.abspath(os.path.join(root, name))
                if any(path == k or path.startswith(k + os.sep) for k in keep):
                    continue
                try:
                    os.remove(path)
                except OSError:
                    pass
            for name in dirs:
                path = os.path.abspath(os.path.join(root, name))
                if any(path == k or path.startswith(k + os.sep) for k in keep):
                    continue
                try:
                    shutil.rmtree(path)
                except OSError:
                    pass
        
        try:
            # Only remove the workspace dir itself if no 'keep' files are inside it
            if not any(k.startswith(workspace_abs + os.sep) for k in keep):
                shutil.rmtree(workspace)
        except OSError:
            pass
