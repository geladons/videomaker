from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
from duckduckgo_search import DDGS

import database
from modules import llm
from config import (
    DEFAULT_DIRS,
    OLLAMA_SETTINGS,
    OLLAMA_API_URL,
    SCRAPER_URLS,
)
from modules.utils import ensure_dir, run_command, clean_and_tokenize

LogFn = Callable[[str, str], Awaitable[None]]


def _build_scraper_constraints(settings: Optional[Dict[str, Any]] = None) -> List[str]:
    """Build yt-dlp constraints for VIDEO downloads."""
    settings = settings or {}
    duration_filter = int(settings.get("yt_dlp_duration_filter", 300))
    download_section = int(settings.get("yt_dlp_download_section", 60))
    return [
        "--match-filter",
        f"duration < {duration_filter}",
        "--download-sections",
        f"*0-{download_section}",
    ]


def _build_music_constraints() -> List[str]:
    """Build yt-dlp constraints for MUSIC downloads.
    Music needs FULL audio (no sections) and should be LONG ENOUGH (no duration filter).
    """
    return []  # No constraints - accept any length, download full audio

async def _latest_file(
    directory: str, exts: List[str], min_mtime: Optional[float] = None
) -> Optional[str]:
    candidates = []
    for name in os.listdir(directory):
        if any(name.lower().endswith(ext) for ext in exts):
            path = os.path.join(directory, name)
            mtime = os.path.getmtime(path)
            if min_mtime is None or mtime >= min_mtime:
                candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _cleanup_temp_files(directory: str, exts: List[str], min_mtime: float) -> None:
    """Clean up temporary files created after min_mtime."""
    for name in os.listdir(directory):
        if any(name.lower().endswith(ext) for ext in exts):
            path = os.path.join(directory, name)
            try:
                mtime = os.path.getmtime(path)
                if mtime >= min_mtime:
                    os.remove(path)
            except (OSError, FileNotFoundError):
                pass


def _limit_query(query: str, max_words: int = 6) -> str:
    words = [w for w in query.split() if w]
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return query


async def download_cc_video(
    query: str,
    out_dir: str,
    log: LogFn,
    scraper_settings: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    ensure_dir(out_dir)
    await asyncio.sleep(0.1)

    settings = scraper_settings or {}
    search_count = int(settings.get("yt_dlp_search_count", 8))
    search_count = min(20, max(3, search_count))
    sleep_min = float(settings.get("yt_dlp_sleep_min", 1.0))
    sleep_max = float(settings.get("yt_dlp_sleep_max", 3.0))
    if sleep_max < sleep_min:
        sleep_max = sleep_min
    delay_sec = float(settings.get("request_delay_sec", 0.0))
    safe_query = _limit_query(query)
    constraints = _build_scraper_constraints(settings)
    base_cmd = [
        "yt-dlp",
        f"ytsearch{search_count}:{safe_query} creative commons",
        "--max-downloads",
        "1",
        "--no-progress",
        "--format",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bestvideo+bestaudio/best",
        *constraints,
        "-o",
        os.path.join(out_dir, "bg_%(id)s.%(ext)s"),
        "--no-check-certificate",
        "--prefer-insecure",
        "--force-ipv4",
        "--fragment-retries", "3",
        "--retries", "3",
    ]
    if sleep_min > 0:
        base_cmd += ["--sleep-interval", str(sleep_min)]
    if sleep_max > 0:
        base_cmd += ["--max-sleep-interval", str(sleep_max)]
    if delay_sec > 0:
        await asyncio.sleep(delay_sec)

    start_time = time.time()
    code = await run_command(base_cmd, log=log)
    # Check if a file was actually downloaded despite potential exit code 1 (warnings)
    latest = await _latest_file(out_dir, [".mp4", ".mkv", ".webm"], min_mtime=start_time)
    
    if code != 0:
        if latest:
            await log("warn", f"yt-dlp exited with code {code} but video file was produced.")
        else:
            await log("error", f"No Creative Commons video found for query: {query}")
            _cleanup_temp_files(out_dir, [".part", ".ytdl"], min_mtime=start_time)
            return None
    
    if not latest:
        await log(
            "error", f"yt-dlp completed but no video file found for query: {query}"
        )
    return latest


async def download_cc_audio(
    query: str,
    out_dir: str,
    log: LogFn,
    scraper_settings: Optional[Dict[str, Any]] = None,
    mood: str = "cinematic",
) -> Optional[str]:
    ensure_dir(out_dir)
    await asyncio.sleep(0.1)

    settings = scraper_settings or {}
    search_count = int(settings.get("yt_dlp_search_count", 4))  # Reduced for audio
    search_count = min(10, max(1, search_count))
    sleep_min = float(settings.get("yt_dlp_sleep_min", 1.0))
    sleep_max = float(settings.get("yt_dlp_sleep_max", 3.0))
    if sleep_max < sleep_min:
        sleep_max = sleep_min
    delay_sec = float(settings.get("request_delay_sec", 0.0))

    # Clean query - remove extra words
    clean_query = query
    for term in ["background music", "cinematic", "ambient", "royalty free"]:
        clean_query = clean_query.replace(term, "").strip()
    safe_query = _limit_query(clean_query, max_words=4)

    if not safe_query:
        safe_query = f"{mood} music"

    constraints = _build_music_constraints()
    base_cmd = [
        "yt-dlp",
        f"ytsearch{search_count}:{safe_query} creative commons",
        "--max-downloads",
        "1",
        "--no-progress",
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        *constraints,
        "-o",
        os.path.join(out_dir, "music_%(id)s.%(ext)s"),
        "--no-check-certificate",
        "--prefer-insecure",
        "--force-ipv4",
        "--fragment-retries", "3",
        "--retries", "3",
    ]
    if sleep_min > 0:
        base_cmd += ["--sleep-interval", str(sleep_min)]
    if sleep_max > 0:
        base_cmd += ["--max-sleep-interval", str(sleep_max)]
    if delay_sec > 0:
        await asyncio.sleep(delay_sec)

    await log("info", f"Downloading audio: yt-dlp {safe_query} creative commons")
    start_time = time.time()
    code = await run_command(base_cmd, log=log)
    
    # Check if a file was actually downloaded despite potential exit code 1 (warnings)
    latest = await _latest_file(out_dir, [".mp3", ".m4a", ".wav", ".ogg"], min_mtime=start_time)
    
    if code != 0:
        if latest:
            await log("warn", f"yt-dlp exited with code {code} but audio file was produced.")
        else:
            await log("warn", f"No Creative Commons audio found for query: {query}. Trying fallback...")
            _cleanup_temp_files(out_dir, [".part", ".ytdl"], min_mtime=start_time)
            return await _download_audio_fallback(query, out_dir, log, settings, mood=mood)
    
    if not latest:
        await log(
            "warn", f"yt-dlp completed but no audio file found for query: {query}. Trying fallback..."
        )
        return await _download_audio_fallback(query, out_dir, log, settings, mood=mood)

    return latest


async def _download_from_internet_archive(
    query: str, out_dir: str, log: LogFn
) -> Optional[str]:

    """Download audio from Internet Archive (CC0/CC-BY music)."""
    try:
        # Search Internet Archive for music
        safe_query = _limit_query(query, max_words=4)
        search_url = f"{SCRAPER_URLS['archive_search']}?q={safe_query}+music&fl[]=identifier&output=json&rows=10"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(search_url)
            response.raise_for_status()
            data = response.json()

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            await log("info", f"No results from Internet Archive for: {safe_query}")
            return None

        # Try to find a suitable audio file
        for doc in docs[:5]:
            identifier = doc.get("identifier")
            if not identifier:
                continue

            # Check if this item has audio files
            detail_url = f"{SCRAPER_URLS['archive_metadata']}{identifier}"
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.get(detail_url)
                    response.raise_for_status()
                    metadata = response.json()

                # Find audio file
                for file in metadata.get("files", []):
                    if file.get("format", "").lower() in [
                        "vbr mp3",
                        "mp3",
                        "ogg vorbis",
                    ] or file.get("name", "").lower().endswith((".mp3", ".ogg")):
                        if (
                            file.get("size", "0") and int(file.get("size", 0)) > 100000
                        ):  # >100KB
                            # Download this file
                            download_url = f"{SCRAPER_URLS['archive_download']}{identifier}/{file['name']}"
                            out_path = os.path.join(out_dir, f"music_{identifier}.mp3")

                            await log(
                                "info",
                                f"Downloading from Internet Archive: {download_url}",
                            )

                            async with httpx.AsyncClient(timeout=120) as client:
                                response = await client.get(download_url)
                                response.raise_for_status()
                                with open(out_path, "wb") as f:
                                    async for chunk in response.aiter_bytes(
                                        chunk_size=8192
                                    ):
                                        f.write(chunk)

                            if (
                                os.path.exists(out_path)
                                and os.path.getsize(out_path) > 0
                            ):
                                await log(
                                    "info",
                                    f"Downloaded from Internet Archive: {out_path}",
                                )
                                return out_path
            except Exception as e:
                await log("warn", f"Failed to get audio from {identifier}: {e}")
                continue

        await log("info", "No suitable audio found on Internet Archive")
        return None

    except Exception as e:
        await log("warn", f"Internet Archive search failed: {e}")
        return None


async def _download_audio_fallback(
    query: str, out_dir: str, log: LogFn, settings: Dict[str, Any], mood: str = "cinematic"
) -> Optional[str]:
    """Fallback audio download - tries simpler approach."""
    try:
        # Simplify query - remove "background music", "cinematic" etc
        simple_query = query
        for term in ["background music", "cinematic", "ambient", "royalty free"]:
            simple_query = simple_query.replace(term, "").strip()
        simple_query = _limit_query(simple_query, max_words=3)

        if not simple_query:
            simple_query = f"{mood} music"

        # Try yt-dlp with simpler settings - use music constraints (no duration filter, no sections)
        constraints = _build_music_constraints()
        cmd = [
            "yt-dlp",
            f"ytsearch3:{simple_query}",
            "--max-downloads",
            "1",
            "--no-progress",
            "-x",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "0",  # Best quality
            *constraints,
            "-o",
            os.path.join(out_dir, "music_%(id)s.%(ext)s"),
            "--no-check-certificate",
            "--prefer-insecure",
        ]

        await log("info", f"Trying audio download with query: {simple_query}")
        start_time = time.time()
        code = await run_command(cmd, log=log)

        # Check for success even with warnings (non-zero exit)
        latest = await _latest_file(out_dir, [".mp3", ".m4a", ".wav"], min_mtime=start_time)

        if code == 0 or latest:
            if latest:
                if code != 0:
                    await log("warn", f"Fallback audio download exited with code {code} but file was produced.")
                await log("info", f"Audio downloaded: {latest}")
                return latest
        
        if code != 0:
            _cleanup_temp_files(out_dir, [".part", ".ytdl"], min_mtime=start_time)

        await log("warn", f"Audio download failed for query: {simple_query}")
        return None

    except Exception as e:
        await log("warn", f"Audio fallback failed: {e}")
        return None


async def search_duckduckgo_images(query: str, limit: int = 3) -> List[str]:
    """Search for images using DuckDuckGo library with thread-safe execution."""
    def _search():
        with DDGS() as ddgs:
            return [r["image"] for r in ddgs.images(query, max_results=limit)]

    try:
        return await asyncio.to_thread(_search)
    except Exception:
        return []


async def search_wikimedia_images(query: str, limit: int = 3) -> List[str]:
    api_url = SCRAPER_URLS["wikimedia"]
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": str(limit),
        "gsrnamespace": "6",
        "prop": "imageinfo",
        "iiprop": "url",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(api_url, params=params)
        response.raise_for_status()
        payload = response.json()

    pages = payload.get("query", {}).get("pages", {})
    urls: List[str] = []
    for page in pages.values():
        info = page.get("imageinfo")
        if not info:
            continue
        url = info[0].get("url")
        if url:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


async def download_image(url: str, out_path: str) -> bool:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(response.content)
        return True
    except httpx.HTTPError:
        return False


async def generate_ai_search_queries(
    scene: Dict[str, Any],
    asset_type: str,
    log: LogFn,
    api_url: str = OLLAMA_API_URL,
    model: str = OLLAMA_SETTINGS["default"]["model"],
    options: Optional[Dict[str, Any]] = None,
    timeout: float = 900.0,
) -> List[str]:
    """
    Generate specialized search queries using AI for video, music, or images.
    Runs asynchronously to avoid blocking the main pipeline.
    """
    from modules import llm

    merged_options = {**OLLAMA_SETTINGS["default"]["params"], **(options or {})}
    merged_options["num_predict"] = min(merged_options.get("num_predict", 256), 256)

    if log:
        await log("info", f"Generating AI search queries for {asset_type} using {model} at {api_url}")

    try:
        queries = await llm.generate_search_queries(
            scene=scene,
            asset_type=asset_type,
            model=model,
            options=merged_options,
            api_url=api_url,
            timeout=timeout,
            request_delay=0.5,
            log=log,
        )
        if queries:
            if log:
                await log("info", f"AI generated {len(queries)} {asset_type} queries: {queries}")
            return queries
        else:
            if log:
                await log("warn", f"AI returned empty queries for {asset_type}, using fallback")
    except Exception as exc:
        if log:
            await log("error", f"AI query generation failed for {asset_type}: {type(exc).__name__}: {exc}")

    # Fallback to simple query processing
    fallback = _alternate_queries(scene.get("visual_query", ""), fallback_topic=None)
    if log:
        await log("info", f"Using fallback queries for {asset_type}: {fallback}")
    return fallback


def _simplify_query(query: str) -> str:
    return clean_and_tokenize(query)


def _alternate_queries(query: str, fallback_topic: Optional[str] = None) -> List[str]:
    if not query:
        query = ""
    cleaned = re.sub(r"[^\w\s]", " ", query).strip()
    simplified = _simplify_query(query)
    words = simplified.split()

    candidates = []
    if cleaned:
        candidates.append(cleaned)
    if simplified and simplified != cleaned:
        candidates.append(simplified)
    if words:
        if "logo" in words:
            candidates.append(f"{words[-1]} logo")
        if "screenshot" in cleaned.lower() and words:
            candidates.append(words[-1])
        if len(words) > 4:
            candidates.append(" ".join(words[:4]))
        if len(words) > 2:
            candidates.append(" ".join(words[:2]))

    if fallback_topic:
        candidates.append(fallback_topic)
        simplified_topic = _simplify_query(fallback_topic)
        if simplified_topic:
            candidates.append(simplified_topic)
            # Add topic-based specific fallbacks
            topic_words = simplified_topic.split()
            if len(topic_words) >= 2:
                candidates.append(f"{topic_words[0]} {topic_words[-1]}")
            candidates.append(f"{simplified_topic} screenshot")
            candidates.append(f"{simplified_topic} gameplay")
            candidates.append(f"{simplified_topic} tutorial")

    # Progressive fallbacks - more specific first, then generic
    candidates.append("minecraft gameplay")
    candidates.append("minecraft building")
    candidates.append("nature landscape")
    candidates.append("city skyline")
    candidates.append("abstract background")
    candidates.append("cinematic background")

    seen = set()
    ordered: List[str] = []
    for item in candidates:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


async def gather_scene_assets(
    scenes: List[Dict[str, Any]],
    workspace: str,
    use_stock_video: bool,
    use_images: bool,
    log: LogFn,
    fallback_topic: Optional[str] = None,
    scraper_settings: Optional[Dict[str, Any]] = None,
    ai_query_model: Optional[str] = None,
    ai_query_api_url: Optional[str] = None,
    ai_query_timeout: Optional[float] = None,
    progress_fn: Optional[Callable[[float], Awaitable[None]]] = None,
) -> List[Dict[str, Optional[str]]]:
    """
    Gather assets for all scenes with proper rate limiting.
    Processes scenes sequentially to avoid YouTube rate limits.
    """
    assets: List[Dict[str, Optional[str]]] = []
    video_dir = os.path.join(workspace, DEFAULT_DIRS["videos"])
    image_dir = os.path.join(workspace, DEFAULT_DIRS["images"])
    music_dir = os.path.join(workspace, DEFAULT_DIRS["music"])

    # CRITICAL: Create ALL directories BEFORE any downloads
    ensure_dir(video_dir)
    ensure_dir(image_dir)
    ensure_dir(music_dir)

    settings = scraper_settings or {}
    image_delay = float(settings.get("image_delay_sec", 0.6))
    request_delay = float(settings.get("request_delay_sec", 0.0))

    # Increase delay between scene processing to avoid rate limits
    scene_delay = max(5.0, request_delay)  # At least 5 seconds between scenes

    # Whether to use AI for query generation
    use_ai = ai_query_model is not None
    query_api_url = ai_query_api_url or OLLAMA_API_URL

    # Track first successful assets for fallback
    first_video = None
    first_image = None

    try:
        cached_assets = await database.get_all_cached_assets()
    except Exception as e:
        await log("warn", f"Failed to fetch cached assets: {e}")
        cached_assets = []

    num_scenes = len(scenes)
    for idx, scene in enumerate(scenes, start=1):
        if progress_fn:
            await progress_fn((idx - 1) / num_scenes)

        entry: Dict[str, Optional[str]] = {"video": None, "image": None}

        if idx > 1 and request_delay > 0:
            await log("info", f"Waiting {scene_delay}s before processing scene {idx}...")
            await asyncio.sleep(scene_delay)

        if use_stock_video:
            visual_query = scene.get("visual_query", "cinematic background")
            video_path = None
            
            # Cache lookup
            if cached_assets:
                valid_cache = [a for a in cached_assets if os.path.exists(a["path"])][:10]
                for asset in valid_cache:
                    if await llm.evaluate_asset_match(
                        visual_query, 
                        asset["description"], 
                        model=ai_query_model or OLLAMA_SETTINGS["default"]["model"],
                        options={**OLLAMA_SETTINGS["default"]["params"], **(settings or {})},
                        api_url=query_api_url,
                        timeout=60.0,
                        log=log
                    ):
                        video_path = asset["path"]
                        await log("info", f"Cache hit for scene {idx}: {video_path}")
                        break
            
            # Download fallback
            if not video_path:
                queries = []
                if use_ai:
                    queries = await generate_ai_search_queries(
                        scene, "video", log, api_url=query_api_url, model=ai_query_model,
                        timeout=ai_query_timeout or 900.0,
                    )
                if not queries:
                    queries = _alternate_queries(visual_query, fallback_topic=fallback_topic)

                for q_idx, query in enumerate(queries):
                    await log("info", f"Searching CC video for scene {idx} (attempt {q_idx+1}): {query}")
                    try:
                        video_path = await download_cc_video(query, video_dir, log, settings)
                        if video_path:
                            first_video = first_video or video_path
                            await database.add_to_cache(video_path, visual_query, query)
                            break
                    except Exception as e:
                        await log("warn", f"Video search attempt {q_idx+1} failed: {e}")
                    
                    if request_delay > 0:
                        await asyncio.sleep(request_delay)

            entry["video"] = video_path
            if not video_path:
                await log("warn", f"No CC video for scene {idx} after all attempts")

        # Image download
        if use_images:
            queries = []
            if use_ai:
                queries = await generate_ai_search_queries(
                    scene, "image", log, api_url=query_api_url, model=ai_query_model,
                    timeout=ai_query_timeout or 900.0,
                )
            if not queries:
                queries = _alternate_queries(scene.get("visual_query", "abstract background"), fallback_topic=fallback_topic)

            image_path = None
            for q_idx, query in enumerate(queries):
                await log("info", f"Searching images for scene {idx} (attempt {q_idx+1}): {query}")
                urls = []
                try:
                    urls = await search_duckduckgo_images(query, limit=2)
                except Exception:
                    pass

                # Fallback to Wikimedia
                if not urls:
                    try:
                        urls = await search_wikimedia_images(query, limit=2)
                    except httpx.HTTPError:
                        pass

                if urls:

                    target = os.path.join(image_dir, f"scene_{idx}_{q_idx}.jpg")
                    for url in urls:
                        try:
                            if await download_image(url, target):
                                image_path = target
                                if first_image is None:
                                    first_image = image_path
                                break
                        except Exception as e:
                            await log("warn", f"Image download failed from {url}: {e}")
                    if image_path:
                        break
                if image_delay > 0:
                    await asyncio.sleep(image_delay)

            entry["image"] = image_path
            if not image_path:
                await log("warn", f"No images for scene {idx} after all attempts")

        assets.append(entry)

    # Apply global fallback if still missing - use first available from ANY scene
    # or a generic nature/abstract asset if none were found at all
    if not first_video and use_stock_video:
        await log("warn", "NO videos found for ANY scene. Attempting ultimate fallback search.")
        first_video = await download_cc_video("nature cinematic CC", video_dir, log, scraper_settings=settings)
        
    if not first_image and use_images:
        await log("warn", "NO images found for ANY scene. Attempting ultimate fallback search.")
        urls = await search_duckduckgo_images("nature", limit=1)
        if not urls:
            urls = await search_wikimedia_images("nature", limit=1)
        
        if urls:
            target = os.path.join(image_dir, "ultimate_fallback.jpg")
            if await download_image(urls[0], target):
                first_image = target

    # Apply fallback for missing assets - use first available (including ultimate fallbacks)
    if first_video or first_image:
        for entry in assets:
            if use_stock_video and not entry.get("video") and first_video:
                entry["video"] = first_video
            if use_images and not entry.get("image") and first_image:
                entry["image"] = first_image
        await log(
            "info", "Reused available assets for scenes missing specific content"
        )

    return assets


