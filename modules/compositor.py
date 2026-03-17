from __future__ import annotations

import os
import tempfile
from typing import Awaitable, Callable, Dict, List, Optional

from config import DEFAULT_VIDEO
from modules.utils import ensure_dir, get_wav_duration, run_command

LogFn = Callable[[str, str], Awaitable[None]]


def _resolution_for(format_choice: str, settings: Dict[str, str]) -> str:
    if format_choice == "9:16":
        return settings.get("resolution_portrait", DEFAULT_VIDEO["resolution_portrait"])
    return settings.get("resolution_landscape", DEFAULT_VIDEO["resolution_landscape"])


def _escape_ffmpeg_path(path: str) -> str:
    """
    Escapes a path for use in FFmpeg filters (e.g., drawtext textfile, ass).
    FFmpeg filtergraph parser requires escaping of colons and backslashes.
    """
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _build_scene_command(
    duration: float,
    bg_video: Optional[str],
    overlay_image: Optional[str],
    text_file: Optional[str],
    voiceover: str,
    out_path: str,
    resolution: str,
    fps: int,
    font_name: str,
    font_size: int,
) -> List[str]:
    width, height = resolution.split("x")
    inputs = []
    filters = []

    if bg_video:
        inputs.extend(["-stream_loop", "-1", "-i", bg_video])
        bg_filter = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},fps={fps},format=yuv420p[bg]"
        )
        filter_inputs = "[bg]"
        input_offset = 1
    else:
        inputs.extend(["-f", "lavfi", "-i", f"color=c=0x0f172a:s={resolution}:r={fps}"])
        bg_filter = "[0:v]format=yuv420p,noise=alls=8:allf=t+u[bg]"
        filter_inputs = "[bg]"
        input_offset = 1

    filters.append(bg_filter)
    current_label = "[bg]"

    if text_file and not overlay_image:
        safe_path = _escape_ffmpeg_path(text_file)
        draw = (
            f"{current_label}drawtext=textfile='{safe_path}':"
            f"font='{font_name}':fontsize={font_size}:fontcolor=white:"
            "box=1:boxcolor=black@0.4:boxborderw=20:"
            "x=(w-text_w)/2:y=(h-text_h)/2[txt]"
        )
        filters.append(draw)
        current_label = "[txt]"

    if overlay_image:
        fade_dur = min(0.5, max(0.1, duration * 0.2))
        fade_out_start = max(0.0, duration - fade_dur)
        inputs.extend(["-i", overlay_image])
        img_index = input_offset
        overlay = (
            f"[{img_index}:v]scale=iw*0.8:ih*0.8:force_original_aspect_ratio=decrease,"
            f"format=rgba,fade=t=in:st=0:d={fade_dur}:alpha=1,"
            f"fade=t=out:st={fade_out_start}:d={fade_dur}:alpha=1[img];"
            f"{current_label}[img]overlay=(W-w)/2:(H-h)/2:enable='between(t,0,{duration})'[vout]"
        )
        filters.append(overlay)
        video_map = "[vout]"
        input_offset += 1
    else:
        video_map = current_label

    # Audio source and padding
    if voiceover and os.path.exists(voiceover):
        inputs.extend(["-i", voiceover])
        audio_source = f"{input_offset}:a"
    else:
        # Fallback to silent audio if voiceover is missing
        inputs.extend(["-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:d={duration}"])
        audio_source = f"{input_offset}:a"
    
    input_offset += 1

    filter_complex = ";".join(filters)
    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        video_map,
        "-map",
        audio_source,
        "-af",
        "apad",  # Pad with silence if audio is shorter than -t
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        out_path,
    ]
    return cmd


async def compose_video(
    scenes: List[Dict[str, float]],
    assets: List[Dict[str, Optional[str]]],
    voiceovers: List[Optional[str]],
    output_path: str,
    workspace: str,
    log: LogFn,
    format_choice: str = "16:9",
    fps: int = 30,
    video_settings: Optional[Dict[str, str]] = None,
    background_music: Optional[str] = None,
    burn_subtitles: Optional[str] = None,
) -> str:
    ensure_dir(workspace)
    scene_dir = os.path.join(workspace, "scenes")
    ensure_dir(scene_dir)

    video_settings = video_settings or DEFAULT_VIDEO
    resolution = _resolution_for(format_choice, video_settings)

    scene_files: List[str] = []
    for idx, scene in enumerate(scenes, start=1):
        duration = float(scene.get("duration", 4))
        voice = voiceovers[idx - 1] if idx - 1 < len(voiceovers) else None

        # Double safety: ensure scene is at least as long as audio (with padding)
        if voice and os.path.exists(voice):
            audio_dur = get_wav_duration(voice)
            if audio_dur + 0.1 > duration:
                duration = round(audio_dur + 0.3, 2)
                
        bg_video = assets[idx - 1].get("video") if idx - 1 < len(assets) else None
        overlay = assets[idx - 1].get("image") if idx - 1 < len(assets) else None
        overlay_text = scene.get("overlay_text") if not overlay else None
        font_name = str(video_settings.get("font_name", DEFAULT_VIDEO["font_name"]))
        font_size = int(video_settings.get("font_size", DEFAULT_VIDEO["font_size"]))
        out_path = os.path.join(scene_dir, f"scene_{idx}.mp4")

        text_file = None
        if overlay_text:
            # Write overlay_text to a temp file to avoid escaping issues
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt", encoding="utf-8"
            ) as tf:
                tf.write(overlay_text)
                text_file = tf.name

        try:
            cmd = _build_scene_command(
                duration,
                bg_video,
                overlay,
                text_file,
                voice,
                out_path,
                resolution,
                fps,
                font_name,
                font_size,
            )
            await log("info", f"Rendering scene {idx}/{len(scenes)}")
            code = await run_command(cmd, log=log)
            if code != 0:
                raise RuntimeError(f"ffmpeg failed while rendering scene {idx}")
            scene_files.append(out_path)
        finally:
            if text_file and os.path.exists(text_file):
                os.remove(text_file)

    concat_list = os.path.join(scene_dir, "concat.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for path in scene_files:
            # Escape single quotes in path for FFmpeg concat file format
            safe_path = path.replace("'", "''")
            f.write(f"file '{safe_path}'\n")

    concat_out = os.path.join(workspace, "concat.mp4")
    concat_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        concat_out,
    ]
    await log("info", "Concatenating scenes")
    code = await run_command(concat_cmd, log=log)
    if code != 0:
        raise RuntimeError("ffmpeg concat failed")

    final_input = concat_out
    mix_out = os.path.join(workspace, "mix.mp4")

    if background_music:
        await log("info", "Adding background music with ducking")
        music_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            final_input,
            "-stream_loop",
            "-1",
            "-i",
            background_music,
            "-filter_complex",
            "[1:a]volume=0.3[bgm];"
            "[bgm][0:a]sidechaincompress=threshold=0.05:ratio=8:attack=20:release=250[bgmduck];"
            "[0:a][bgmduck]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            mix_out,
        ]
        code = await run_command(music_cmd, log=log)
        if code != 0:
            raise RuntimeError("ffmpeg music mix failed")
        final_input = mix_out

    if burn_subtitles:
        await log("info", "Burning subtitles")
        safe_ass_path = _escape_ffmpeg_path(burn_subtitles)
        burn_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            final_input,
            "-vf",
            f"ass='{safe_ass_path}'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            output_path,
        ]
        code = await run_command(burn_cmd, log=log)
        if code != 0:
            raise RuntimeError("ffmpeg subtitle burn failed")
    else:
        await log("info", "Finalizing video output")
        final_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            final_input,
            "-c",
            "copy",
            output_path,
        ]
        code = await run_command(final_cmd, log=log)
        if code != 0:
            raise RuntimeError("ffmpeg finalize failed")

    return output_path
