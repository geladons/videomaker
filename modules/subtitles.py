from __future__ import annotations

import os
from typing import Any, Awaitable, Callable, Dict, List, Tuple

from faster_whisper import WhisperModel

from config import DEFAULT_VIDEO, WHISPER_MODEL_PATH

LogFn = Callable[[str, str], Awaitable[None]]


def _format_ass_time(seconds: float) -> str:
    total_cs = int(seconds * 100)
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _parse_resolution(resolution: str) -> tuple[int, int]:
    try:
        width, height = resolution.lower().split("x")
        return int(width), int(height)
    except (ValueError, AttributeError):
        return 1280, 720


def _resolution_for(format_choice: str, settings: Dict[str, str]) -> str:
    if format_choice == "9:16":
        return settings.get("resolution_portrait", DEFAULT_VIDEO["resolution_portrait"])
    return settings.get("resolution_landscape", DEFAULT_VIDEO["resolution_landscape"])


def _build_ass_header(
    style_name: str = "Default",
    resolution: str = "1280x720",
    video_settings: Dict[str, Any] | None = None,
) -> str:
    settings = {**DEFAULT_VIDEO, **(video_settings or {})}
    font = settings["font_name"]
    size = settings["font_size"]
    primary = settings["font_color"]
    secondary = "&H0000E5FF"  # bright yellow
    outline = settings["outline_color"]
    outline_size = settings["outline"]
    shadow = settings["shadow"]
    position = settings.get("subtitle_position", "bottom")
    margin_x = int(settings.get("subtitle_margin_x", 60))
    margin_y = int(settings.get("subtitle_margin_y", 60))
    alignment = 2
    if position == "top":
        alignment = 8
    elif position == "center":
        alignment = 5
    width, height = _parse_resolution(resolution)
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: {style_name},{font},{size},{primary},{secondary},{outline},&H64000000,0,0,0,0,100,100,0,0,1,{outline_size},{shadow},{alignment},{margin_x},{margin_x},{margin_y},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _words_to_karaoke(words: List[Tuple[str, float, float]]) -> str:
    parts = []
    for word, start, end in words:
        dur_cs = max(1, int((end - start) * 100))
        safe_word = word.replace("{", "").replace("}", "")
        parts.append(f"{{\\k{dur_cs}}}{safe_word}")
    return " ".join(parts)


async def generate_ass(
    wav_files: List[str],
    offsets: List[float],
    out_path: str,
    log: LogFn,
    model_path: str = WHISPER_MODEL_PATH,
    video_settings: Dict[str, Any] | None = None,
    format_choice: str = "16:9",
) -> str:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Whisper model not found at {model_path}")

    cpu_threads = min(20, os.cpu_count() or 20)
    model = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
    all_segments = []

    for wav, offset in zip(wav_files, offsets):
        await log("info", f"Transcribing subtitles for {os.path.basename(wav)}")
        segments, _info = model.transcribe(
            wav,
            word_timestamps=True,
            vad_filter=True,
        )
        for segment in segments:
            words = []
            for word in segment.words or []:
                words.append((word.word.strip(), word.start + offset, word.end + offset))
            if words:
                all_segments.append(words)

    resolution = _resolution_for(format_choice, video_settings or DEFAULT_VIDEO)
    header = _build_ass_header(resolution=resolution, video_settings=video_settings)
    lines = [header]

    for words in all_segments:
        start_time = _format_ass_time(words[0][1])
        end_time = _format_ass_time(words[-1][2])
        text = _words_to_karaoke(words)
        lines.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return out_path
