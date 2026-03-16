from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

from modules.utils import ensure_dir, run_command

LogFn = Callable[[str, str], Awaitable[None]]


async def _latest_file(directory: str, exts: List[str]) -> Optional[str]:
    candidates = []
    for name in os.listdir(directory):
        if any(name.lower().endswith(ext) for ext in exts):
            path = os.path.join(directory, name)
            candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


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
    code = await run_command(base_cmd + ["--match-filter", match_filter], log=log)
    if code != 0:
        await log("error", f"No Creative Commons video found for query: {query}")
        return None
    latest = await _latest_file(out_dir, [".mp4", ".mkv", ".webm"])
    if not latest:
        await log("error", f"yt-dlp completed but no video file found for query: {query}")
    return latest

async def download_cc_audio(
    query: str,
    out_dir: str,
    log: LogFn,
    scraper_settings: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    ensure_dir(out_dir)
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
        "-x",
        "--audio-format",
        "mp3",
        "-o",
        os.path.join(out_dir, "music_%(id)s.%(ext)s"),
    ]
    if sleep_min > 0:
        base_cmd += ["--sleep-interval", str(sleep_min)]
    if sleep_max > 0:
        base_cmd += ["--max-sleep-interval", str(sleep_max)]
    if delay_sec > 0:
        await asyncio.sleep(delay_sec)
    code = await run_command(base_cmd + ["--match-filter", match_filter], log=log)
    if code != 0:
        await log("error", f"No Creative Commons audio found for query: {query}")
        return None
    latest = await _latest_file(out_dir, [".mp3", ".m4a", ".wav"])
    if not latest:
        await log("error", f"yt-dlp completed but no audio file found for query: {query}")
    return latest


async def search_duckduckgo_images(query: str, limit: int = 3) -> List[str]:
    url = "https://html.duckduckgo.com/html/"
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
    api_url = "https://commons.wikimedia.org/w/api.php"
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


_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "and",
    "to",
    "in",
    "on",
    "with",
    "from",
    "about",
    "screenshot",
    "image",
    "photo",
    "picture",
}


def _simplify_query(query: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", query).lower().strip()
    tokens = [t for t in cleaned.split() if t and t not in _STOPWORDS]
    if not tokens:
        return cleaned.strip()
    return " ".join(tokens)


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
) -> List[Dict[str, Optional[str]]]:
    assets: List[Dict[str, Optional[str]]] = []
    video_dir = os.path.join(workspace, "videos")
    image_dir = os.path.join(workspace, "images")
    ensure_dir(video_dir)
    ensure_dir(image_dir)
    settings = scraper_settings or {}
    image_delay = float(settings.get("image_delay_sec", 0.6))
    request_delay = float(settings.get("request_delay_sec", 0.0))

    for idx, scene in enumerate(scenes, start=1):
        entry: Dict[str, Optional[str]] = {"video": None, "image": None}
        if use_stock_video:
            base_query = scene.get("visual_query", "ambient cinematic")
            video_path = None
            for query in _alternate_queries(base_query, fallback_topic=fallback_topic):
                await log("info", f"Searching CC video for scene {idx}: {query}")
                video_path = await download_cc_video(query, video_dir, log, scraper_settings=settings)
                if video_path:
                    break
                if request_delay > 0:
                    await asyncio.sleep(request_delay)
            entry["video"] = video_path

        if use_images:
            base_query = scene.get("visual_query", "abstract background")
            await log("info", f"Searching images for scene {idx}: {base_query}")
            image_found = False
            for query in _alternate_queries(base_query, fallback_topic=fallback_topic):
                if image_found:
                    break
                try:
                    urls = await search_wikimedia_images(query, limit=2)
                except httpx.HTTPError:
                    urls = []
                if not urls:
                    try:
                        urls = await search_duckduckgo_images(query, limit=2)
                    except httpx.HTTPError:
                        urls = []
                if urls:
                    target = os.path.join(image_dir, f"scene_{idx}.jpg")
                    for url in urls:
                        success = await download_image(url, target)
                        if success:
                            entry["image"] = target
                            image_found = True
                            break
                if image_delay > 0:
                    await asyncio.sleep(image_delay)
            if not image_found:
                await log("error", f"No images found for scene {idx}; continuing")

        assets.append(entry)
    return assets
