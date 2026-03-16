from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_HELPER_MODEL,
    DEFAULT_OLLAMA_HELPER_PARAMS,
    DEFAULT_OLLAMA_HELPER_THINK,
    DEFAULT_OLLAMA_HELPER_TIMEOUT,
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
    DEFAULT_VIDEO,
    DEFAULT_TTS_ENGINE,
    DEFAULT_COQUI_MODEL,
    DEFAULT_COQUI_SPEAKER,
    DEFAULT_VOICEOVER_WPS,
    OLLAMA_API_URL,
    OUTPUT_DIR,
    WORKSPACE_DIR,
)
from database import add_log, create_task, update_task_status
from modules import compositor, llm, scraper, subtitles, tts_engine, vision

LogFn = Callable[[str, str], Awaitable[None]]


class Orchestrator:
    def __init__(self, log_callback: Callable[[str, str, str], Awaitable[None]]):
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.active_task_id: Optional[str] = None
        self.log_callback = log_callback
        self.worker_task: Optional[asyncio.Task] = None
        self.log_dir = os.path.join(OUTPUT_DIR, "logs")

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

        db_message = message if len(message) <= 2000 else message[:2000] + " ...[truncated]"
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
        from database import get_task, get_setting

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
            writer_model = await get_setting("ollama_model", DEFAULT_OLLAMA_MODEL)
            writer_params = await get_setting("ollama_params", DEFAULT_OLLAMA_PARAMS)
            writer_api_url = await get_setting("ollama_api_url", OLLAMA_API_URL)
            writer_timeout = await get_setting("ollama_timeout", 180)
            writer_think = await get_setting("ollama_think", False)
            ollama_delay = await get_setting("ollama_request_delay", DEFAULT_OLLAMA_REQUEST_DELAY)

            planner_model = await get_setting("ollama_planner_model", DEFAULT_OLLAMA_PLANNER_MODEL)
            planner_params = await get_setting("ollama_planner_params", DEFAULT_OLLAMA_PLANNER_PARAMS)
            planner_api_url = await get_setting("ollama_planner_api_url", OLLAMA_API_URL)
            planner_timeout = await get_setting("ollama_planner_timeout", DEFAULT_OLLAMA_PLANNER_TIMEOUT)
            planner_think = await get_setting("ollama_planner_think", DEFAULT_OLLAMA_PLANNER_THINK)
            scraper_settings = await get_setting("scraper_settings", DEFAULT_SCRAPER)
            helper_settings = {
                "api_url": await get_setting("ollama_helper_api_url", OLLAMA_API_URL),
                "model": await get_setting("ollama_helper_model", DEFAULT_OLLAMA_HELPER_MODEL),
                "options": await get_setting("ollama_helper_params", DEFAULT_OLLAMA_HELPER_PARAMS),
                "timeout": await get_setting("ollama_helper_timeout", DEFAULT_OLLAMA_HELPER_TIMEOUT),
                "think": await get_setting("ollama_helper_think", DEFAULT_OLLAMA_HELPER_THINK),
            }
            vision_settings = {
                "api_url": await get_setting("ollama_vision_api_url", OLLAMA_API_URL),
                "model": await get_setting("ollama_vision_model", DEFAULT_OLLAMA_VISION_MODEL),
                "options": await get_setting("ollama_vision_params", DEFAULT_OLLAMA_VISION_PARAMS),
                "timeout": await get_setting("ollama_vision_timeout", DEFAULT_OLLAMA_VISION_TIMEOUT),
                "think": await get_setting("ollama_vision_think", DEFAULT_OLLAMA_VISION_THINK),
                "enabled": await get_setting("ollama_vision_enabled", DEFAULT_OLLAMA_VISION_ENABLED),
            }
            video_settings = await get_setting("video_settings", DEFAULT_VIDEO)
            voice_path = await get_setting("tts_voice_path", None)
            voice_config = await get_setting("tts_voice_config", None)
            tts_engine_name = await get_setting("tts_engine", DEFAULT_TTS_ENGINE)
            coqui_model = await get_setting("coqui_model", DEFAULT_COQUI_MODEL)
            coqui_speaker = await get_setting("coqui_speaker", DEFAULT_COQUI_SPEAKER)
            voiceover_wps = await get_setting("voiceover_words_per_sec", DEFAULT_VOICEOVER_WPS)
            try:
                voiceover_wps = float(voiceover_wps)
            except (TypeError, ValueError):
                voiceover_wps = float(DEFAULT_VOICEOVER_WPS)

            await self._set_progress(task_id, 5)
            await self._log(task_id, "info", f"Ollama request delay: {ollama_delay}s")
            await self._log(task_id, "info", f"Scraper settings: {scraper_settings}")
            await self._log(task_id, "info", "Requesting plan from Planner LLM")
            timeline = await llm.plan_timeline(
                prompt,
                model=planner_model,
                options=planner_params,
                api_url=planner_api_url,
                timeout=float(planner_timeout),
                think=bool(planner_think),
                helper_settings=helper_settings,
                target_duration=float(options.get("duration", 0) or 0),
                request_delay=float(ollama_delay),
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
            await self._log(task_id, "raw", f"LLM timeline (full): {json.dumps(timeline, ensure_ascii=False, indent=2)}")

            for scene in scenes:
                if not scene.get("overlay_text"):
                    scene["overlay_text"] = (scene.get("visual_query") or scene.get("voiceover") or "").strip()

            await self._log(task_id, "info", "Generating scene voiceovers with Writer LLM")
            for idx, scene in enumerate(scenes, start=1):
                duration = float(scene.get("duration", 0) or 0)
                target_words = max(12, int(duration * voiceover_wps))
                try:
                    voice_text = await llm.generate_voiceover(
                        scene,
                        language=options.get("language", "English"),
                        target_words=target_words,
                        model=writer_model,
                        options=writer_params,
                        api_url=writer_api_url,
                        timeout=float(writer_timeout),
                        think=bool(writer_think),
                        add_greeting=bool(options.get("add_greeting", False)),
                        add_closing=bool(options.get("add_closing", False)),
                        is_first=idx == 1,
                        is_last=idx == len(scenes),
                        request_delay=float(ollama_delay),
                        log=lambda lvl, msg: self._log(task_id, lvl, msg),
                    )
                    scene["voiceover"] = voice_text
                    await self._log(
                        task_id,
                        "info",
                        f"Voiceover {idx}/{len(scenes)}: {len(voice_text.split())} words",
                    )
                    await self._log(task_id, "info", f"Voiceover text {idx}: {voice_text[:300]}")
                    await self._log(task_id, "raw", f"Voiceover text {idx} (full): {voice_text}")
                except Exception as exc:
                    await self._log(task_id, "error", f"Voiceover generation failed for scene {idx}: {exc}")
                    scene["voiceover"] = scene.get("voiceover", "")

            use_stock_video = bool(options.get("use_stock_video", True))
            use_images = bool(options.get("use_images", True))
            fallback_topic = prompt.splitlines()[0].strip() if prompt else None
            assets = await scraper.gather_scene_assets(
                scenes,
                workspace,
                use_stock_video,
                use_images,
                lambda lvl, msg: self._log(task_id, lvl, msg),
                fallback_topic=fallback_topic,
                scraper_settings=scraper_settings,
            )
            fallback_video = next((a.get("video") for a in assets if a.get("video")), None)
            fallback_image = next((a.get("image") for a in assets if a.get("image")), None)
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

            if vision_settings.get("enabled"):
                await self._log(task_id, "info", "Running vision analysis on downloaded images")
                for idx, scene in enumerate(scenes, start=1):
                    image_path = assets[idx - 1].get("image") if idx - 1 < len(assets) else None
                    if not image_path:
                        continue
                    try:
                        analysis = await vision.analyze_image(
                            image_path,
                            api_url=vision_settings["api_url"],
                            model=vision_settings["model"],
                            options=vision_settings["options"],
                            timeout=float(vision_settings["timeout"]),
                            think=bool(vision_settings.get("think", False)),
                            prompt=scene.get("visual_query"),
                            log=lambda lvl, msg: self._log(task_id, lvl, msg),
                        )
                        caption = analysis.get("caption")
                        tags = analysis.get("tags") or []
                        if caption and not scene.get("overlay_text"):
                            scene["overlay_text"] = caption
                        if tags:
                            scene["visual_query"] = f"{scene.get('visual_query', '')} {' '.join(tags[:3])}".strip()
                    except Exception as exc:
                        await self._log(task_id, "error", f"Vision analysis failed for scene {idx}: {exc}")

            await self._set_progress(task_id, 35)
            texts = [scene.get("voiceover", "") for scene in scenes]
            voice_dir = os.path.join(workspace, "voice")
            await self._log(
                task_id,
                "info",
                f"TTS engine: {tts_engine_name} (coqui_model={coqui_model}, speaker={coqui_speaker})",
            )
            voiceovers = await tts_engine.generate_voiceovers(
                texts,
                voice_dir,
                lambda lvl, msg: self._log(task_id, lvl, msg),
                voice_path=voice_path,
                voice_config=voice_config,
                language=options.get("language", "English"),
                engine=tts_engine_name,
                coqui_model=coqui_model,
                coqui_speaker=coqui_speaker,
            )

            await self._set_progress(task_id, 55)
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
                    video_settings=video_settings,
                    format_choice=options.get("format", "16:9"),
                )

            await self._set_progress(task_id, 70)
            music_path = None
            if options.get("add_music", True):
                base_query = f"{timeline.get('music_mood', 'cinematic')} background music"
                queries = [
                    base_query,
                    f"{fallback_topic} background music" if fallback_topic else None,
                    "ambient background music",
                    "cinematic background music",
                ]
                for query in [q for q in queries if q]:
                    music_path = await scraper.download_cc_audio(
                        query,
                        os.path.join(workspace, "music"),
                        lambda lvl, msg: self._log(task_id, lvl, msg),
                        scraper_settings=scraper_settings,
                    )
                    if music_path:
                        break

            await self._set_progress(task_id, 80)
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
                fps=int(video_settings.get("fps", 30)),
                video_settings=video_settings,
                background_music=music_path,
                burn_subtitles=subtitle_path,
            )

            await self._set_progress(task_id, 100)
            await update_task_status(task_id, "Completed", output_path=output_path, progress=100)
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
            await self._log(task_id, "info", f"Assigned uniform scene durations for {target_duration:.2f}s target")
            return

        scale = target_duration / total
        for idx, scene in enumerate(scenes):
            scene["duration"] = round(durations[idx] * scale, 2)

        adjusted_total = sum(float(scene.get("duration", 0)) for scene in scenes)
        diff = round(target_duration - adjusted_total, 2)
        if abs(diff) > 0 and scenes:
            scenes[-1]["duration"] = round(float(scenes[-1].get("duration", 0)) + diff, 2)

        await self._log(task_id, "info", f"Normalized durations to {target_duration:.2f}s target")

    def _cleanup_workspace(self, workspace: str, keep: Optional[list[str]] = None) -> None:
        keep = [path for path in (keep or []) if path]
        if not os.path.exists(workspace):
            return
        for root, dirs, files in os.walk(workspace, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                if path in keep:
                    continue
                try:
                    os.remove(path)
                except OSError:
                    pass
            for name in dirs:
                path = os.path.join(root, name)
                try:
                    shutil.rmtree(path)
                except OSError:
                    pass
        try:
            shutil.rmtree(workspace)
        except OSError:
            pass
