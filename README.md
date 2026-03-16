# Video Generator (Local, Autonomous)

A fully local, headless video-generation pipeline that turns a text prompt into a complete MP4 with narration, subtitles, stock visuals, and background music. Built for CPU-only LXC deployments with a strict FIFO queue and detailed live logs.

## Key Features
- **Local-only**: Uses a local Ollama server for all LLM tasks.
- **FIFO queue**: One task at a time to avoid CPU/RAM overload.
- **Modular pipeline**: Planner → Voiceover → Assets → Subtitles → Compose.
- **CC media**: Pulls Creative Commons stock video/music via `yt-dlp`.
- **Subtitles**: Word-level karaoke `.ass` subtitles using `faster-whisper`.
- **TTS**: Piper (default) and optional Coqui TTS for higher quality.
- **Realtime logs**: Live stdout/stderr via WebSockets.

## Quick Start
1. Ensure you have a running Ollama server (example: `http://192.168.1.137:11434`).
2. Run the installer:
   ```bash
   bash install.sh
   ```
3. Open the UI:
   ```
   http://<server-ip>:8065
   ```

## Ollama Models
Use Settings to select models per role:
- **Planner / Orchestrator** (fast + reliable JSON): recommended `llama3.2:3b`.
- **Writer / Narration** (better text): recommended `qwen3.5:9b`.
- **Helper** for JSON repair: small, fast model.
- **Vision** (optional): `qwen3-vl:2b`.

## TTS
- **Piper (default)**: fast and light, best for CPU.
- **Coqui TTS (optional)**: higher quality but heavier.
  - Select `TTS Engine = coqui` in Settings.
  - Choose a model (default: `tts_models/en/vctk/vits`).

> Note: The installer provisions Python 3.11 (via apt, deadsnakes, or source build) to ensure Coqui TTS installs reliably, even on Ubuntu 24.04.

## Settings Highlights
- **num_ctx**: Context length for all Ollama models.
- **Greeting / Closing**: Optional intro/outro toggles.
- **Subtitle position**: Top / Center / Bottom with margins.
- **Voiceover words/sec**: Controls narration length per scene.

## Project Structure
```
app/                # FastAPI templates & static UI
modules/            # LLM, scraper, TTS, subtitles, compositor
orchestrator.py     # FIFO queue + pipeline controller
main.py             # API + Web UI
config.py           # Defaults
```

## Troubleshooting
- **No stock media**: CC content is limited. Try simpler prompts or adjust queries.
- **JSON parse errors**: Increase `num_ctx`, reduce `num_predict`, or disable `think`.
- **Short video**: Ensure `-shortest` is not used (already handled in compositor).
- **Voice quality**: Switch to Coqui TTS or provide a better Piper voice file.

## License
This project is intended for internal use. Ensure you respect Creative Commons licenses for any media downloaded.
