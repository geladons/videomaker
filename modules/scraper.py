from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

from config import (
    DEFAULT_DIRS,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_PARAMS,
    OLLAMA_API_URL,
    SCRAPER_URLS,
)
from modules.utils import ensure_dir, run_command, clean_and_tokenize

LogFn = Callable[[str, str], Awaitable[None]]


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
    # Ensure directory exists before download
    ensure_dir(out_dir)
    await asyncio.sleep(0.1)  # Small delay to ensure FS sync

    settings = scraper_settings or {}
    search_count = int(settings.get("yt_dlp_search_count", 8))
    search_count = min(20, max(3, search_count))
    sleep_min = float(settings.get("yt_dlp_sleep_min", 1.0))
    sleep_max = float(settings.get("yt_dlp_sleep_max", 3.0))
    if sleep_max < sleep_min:
        sleep_max = sleep_min
    delay_sec = float(settings.get("request_delay_sec", 0.0))
    match_filter = "license~='Creative Commons'"
    safe_query = _limit_query(query)
    base_cmd = [
        "yt-dlp",
        f"ytsearch{search_count}:{safe_query} creative commons",
        "--max-downloads",
        "1",
        "--format",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "-o",
        os.path.join(out_dir, "bg_%(id)s.%(ext)s"),
    ]
    if sleep_min > 0:
        base_cmd += ["--sleep-interval", str(sleep_min)]
    if sleep_max > 0:
        base_cmd += ["--max-sleep-interval", str(sleep_max)]
    if delay_sec > 0:
        await asyncio.sleep(delay_sec)
    
    start_time = time.time()
    code = await run_command(base_cmd + ["--match-filter", match_filter], log=log)
    
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
) -> Optional[str]:
    # Ensure directory exists before download
    ensure_dir(out_dir)
    await asyncio.sleep(0.1)  # Small delay to ensure FS sync

    settings = scraper_settings or {}
    search_count = int(settings.get("yt_dlp_search_count", 4))  # Reduced for audio
    search_count = min(10, max(1, search_count))
    sleep_min = float(settings.get("yt_dlp_sleep_min", 1.0))
    sleep_max = float(settings.get("yt_dlp_sleep_max", 3.0))
    if sleep_max < sleep_min:
        sleep_max = sleep_min
    delay_sec = float(settings.get("request_delay_sec", 0.0))
    match_filter = "license~='Creative Commons'"

    # Clean query - remove extra words
    clean_query = query
    for term in ["background music", "cinematic", "ambient", "royalty free"]:
        clean_query = clean_query.replace(term, "").strip()
    safe_query = _limit_query(clean_query, max_words=4)

    if not safe_query:
        safe_query = "upbeat music"

    base_cmd = [
        "yt-dlp",
        f"ytsearch{search_count}:{safe_query} creative commons",
        "--max-downloads",
        "1",
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "-o",
        os.path.join(out_dir, "music_%(id)s.%(ext)s"),
    ]
    if sleep_min > 0:
        base_cmd += ["--sleep-interval", str(sleep_min)]
    if sleep_max > 0:
        base_cmd += ["--max-sleep-interval", str(sleep_max)]
    if delay_sec > 0:
        await asyncio.sleep(delay_sec)

    await log("info", f"Downloading audio: yt-dlp {safe_query} creative commons")
    start_time = time.time()
    code = await run_command(base_cmd + ["--match-filter", match_filter], log=log)
    
    # Check if a file was actually downloaded despite potential exit code 1 (warnings)
    latest = await _latest_file(out_dir, [".mp3", ".m4a", ".wav", ".ogg"], min_mtime=start_time)
    
    if code != 0:
        if latest:
            await log("warn", f"yt-dlp exited with code {code} but audio file was produced.")
        else:
            await log("error", f"No Creative Commons audio found for query: {query}")
            _cleanup_temp_files(out_dir, [".part", ".ytdl"], min_mtime=start_time)
            return None
    
    if not latest:
        await log(
            "error", f"yt-dlp completed but no audio file found for query: {query}"
        )
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
    query: str, out_dir: str, log: LogFn, settings: Dict[str, Any]
) -> Optional[str]:
    """Fallback audio download - tries simpler approach."""
    try:
        # Simplify query - remove "background music", "cinematic" etc
        simple_query = query
        for term in ["background music", "cinematic", "ambient", "royalty free"]:
            simple_query = simple_query.replace(term, "").strip()
        simple_query = _limit_query(simple_query, max_words=3)

        if not simple_query:
            simple_query = "upbeat music"

        # Try yt-dlp with simpler settings
        cmd = [
            "yt-dlp",
            f"ytsearch3:{simple_query}",
            "--max-downloads",
            "1",
            "-x",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "0",  # Best quality
            "-o",
            os.path.join(out_dir, "music_%(id)s.%(ext)s"),
            "--nocheckcertificate",
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
    url = SCRAPER_URLS["duckduckgo"]
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        response = await client.post(url, data=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

    image_urls = []
    for img in soup.select("img"):
        src = img.get("data-src") or img.get("src")
        if not src:
            continue
        if src.startswith("data:"):
            continue
        if "duckduckgo.com" in src:
            continue
        image_urls.append(src)
        if len(image_urls) >= limit:
            break
    return image_urls


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
    model: str = DEFAULT_OLLAMA_MODEL,
    options: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> List[str]:
    """
    Generate specialized search queries using AI for video, music, or images.
    Runs asynchronously to avoid blocking the main pipeline.
    """
    from modules import llm

    merged_options = {**DEFAULT_OLLAMA_PARAMS, **(options or {})}
    merged_options["num_predict"] = min(merged_options.get("num_predict", 256), 256)

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
            return queries
    except Exception as exc:
        if log:
            await log("warn", f"AI query generation failed for {asset_type}: {exc}")

    # Fallback to simple query processing
    return _alternate_queries(scene.get("visual_query", ""), fallback_topic=None)


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
    use_ai = ai_query_model is not None and ai_query_api_url
    query_api_url = ai_query_api_url or OLLAMA_API_URL

    # Track first successful assets for fallback
    first_video = None
    first_image = None

    # Process scenes SEQUENTIALLY to avoid rate limiting
    num_scenes = len(scenes)
    for idx, scene in enumerate(scenes, start=1):
        if progress_fn:
            # Progress within gather_scene_assets is from 0.0 to 1.0
            await progress_fn((idx - 1) / num_scenes)

        entry: Dict[str, Optional[str]] = {"video": None, "image": None}

        # Add delay between scenes to avoid YouTube rate limits
        if idx > 1 and request_delay > 0:
            await log(
                "info", f"Waiting {scene_delay}s before processing scene {idx}..."
            )
            await asyncio.sleep(scene_delay)

        # Video download - try multiple queries for better hit rate
        if use_stock_video:
            base_query = scene.get("visual_query", "cinematic background")
            
            # 1. Primary query
            video_query = _clean_query_for_video(base_query)
            # 2. Broader query (fewer words)
            words = video_query.split()
            broad_query = " ".join(words[:2]) + " CC" if len(words) > 2 else video_query
            # 3. Fallback topic query
            fallback_q = _clean_query_for_video(fallback_topic) if fallback_topic else "nature CC"

            queries = [video_query, broad_query, fallback_q]
            # Remove duplicates while preserving order
            seen_q = set()
            unique_queries = []
            for q in queries:
                if q not in seen_q:
                    unique_queries.append(q)
                    seen_q.add(q)

            video_path = None
            for q_idx, query in enumerate(unique_queries):
                await log("info", f"Searching CC video for scene {idx} (attempt {q_idx+1}): {query}")
                try:
                    video_path = await download_cc_video(
                        query, video_dir, log, scraper_settings=settings
                    )
                except Exception as e:
                    await log("warn", f"Video search attempt {q_idx+1} failed: {e}")
                    
                if video_path:
                    if first_video is None:
                        first_video = video_path
                    break
                # Small delay between queries
                if request_delay > 0:
                    await asyncio.sleep(request_delay)

            entry["video"] = video_path
            if not video_path:
                await log("warn", f"No CC video for scene {idx} after all attempts")

        # Image download
        if use_images:
            base_query = scene.get("visual_query", "abstract background")

            # Clean up query for images
            image_query = _clean_query_for_image(base_query)
            # Broader query
            words = image_query.split()
            broad_query = " ".join(words[:1]) if len(words) > 1 else image_query
            # Fallback
            fallback_q = _clean_query_for_image(fallback_topic) if fallback_topic else "abstract"

            queries = [image_query, broad_query, fallback_q]
            seen_q = set()
            unique_queries = []
            for q in queries:
                if q not in seen_q:
                    unique_queries.append(q)
                    seen_q.add(q)

            image_path = None
            for q_idx, query in enumerate(unique_queries):
                await log("info", f"Searching images for scene {idx} (attempt {q_idx+1}): {query}")
                urls = []
                try:
                    urls = await search_wikimedia_images(query, limit=2)
                except httpx.HTTPError:
                    pass
                if not urls:
                    try:
                        urls = await search_duckduckgo_images(query, limit=2)
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


def _clean_query_for_video(query: str) -> str:
    """
    Clean up query for video search - short, video-focused, with CC tag.
    """
    if not query:
        return "minecraft gameplay"

    # Remove common prefixes
    for prefix in [
        "A shot of ",
        "A montage of ",
        "A comparison shot of ",
        "A news headline of ",
        "image of ",
        "screenshot of ",
        "photo of ",
        "picture of ",
    ]:
        if query.lower().startswith(prefix.lower()):
            query = query[len(prefix) :]
            break

    # Remove common suffixes
    for suffix in [" footage", " gameplay", " video", " clip"]:
        if query.lower().endswith(suffix.lower()):
            query = query[: -len(suffix)]
            break

    # Clean up and limit to 4 words
    words = query.split()
    if len(words) > 4:
        words = words[:4]

    # Add CC tag for better results
    cleaned = " ".join(words) + " CC"

    return cleaned


def _clean_query_for_image(query: str) -> str:
    """
    Clean up query for image search - short, image-focused.
    """
    if not query:
        return "abstract"

    # Remove video-related terms
    for suffix in [" footage", " gameplay", " video", " clip"]:
        if query.lower().endswith(suffix.lower()):
            query = query[: -len(suffix)]
            break

    # Clean up and limit to 3 words
    words = query.split()
    if len(words) > 3:
        words = words[:3]

    return " ".join(words)
