from __future__ import annotations

import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspaces")

DB_PATH = os.path.join(DATA_DIR, "videogen.db")

DEFAULT_DIRS = {
    "videos": "videos",
    "images": "images",
    "music": "music",
    "voice": "voice",
    "logs": "logs",
}

SCRAPER_URLS = {
    "archive_search": "https://archive.org/advancedsearch.php",
    "archive_metadata": "https://archive.org/metadata/",
    "archive_download": "https://archive.org/download/",
    "wikimedia": "https://commons.wikimedia.org/w/api.php",
}

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
WHISPER_MODEL_PATH = os.environ.get(
    "WHISPER_MODEL_PATH", os.path.join(MODELS_DIR, "whisper", "small")
)
PIPER_VOICE_PATH = os.environ.get(
    "PIPER_VOICE_PATH", os.path.join(MODELS_DIR, "piper", "en_US-lessac-medium.onnx")
)
PIPER_VOICE_CONFIG = os.environ.get(
    "PIPER_VOICE_CONFIG",
    os.path.join(MODELS_DIR, "piper", "en_US-lessac-medium.onnx.json"),
)

# TTS Engine Defaults
DEFAULT_TTS_ENGINE = os.environ.get("TTS_ENGINE", "coqui")
DEFAULT_COQUI_MODEL = os.environ.get("COQUI_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
DEFAULT_COQUI_SPEAKER = os.environ.get("COQUI_TTS_SPEAKER", "")

# VCTK speaker IDs - p225-p335 are available
DEFAULT_VCTK_SPEAKERS = [
    "p225",
    "p226",
    "p227",
    "p228",  # Common English speakers
]

# Fallback speakers for different languages
LANGUAGE_SPEAKERS = {
    "English": "p226",  # Male voice
    "Spanish": "p225",
    "German": "p225",
    "French": "p225",
    "Italian": "p225",
    "Portuguese": "p225",
    "Russian": "p225",  # Use English speaker as fallback
}

# Ollama API Defaults
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
LLM_DEFAULT_TIMEOUT = 180.0
LLM_MAX_RETRIES = 3
DEFAULT_OLLAMA_REQUEST_DELAY = float(os.environ.get("OLLAMA_REQUEST_DELAY", "0.8"))

OLLAMA_SETTINGS = {
    "default": {
        "model": os.environ.get("OLLAMA_MODEL", "qwen3.5:9b"),
        "timeout": int(os.environ.get("OLLAMA_TIMEOUT", "180")),
        "think": os.environ.get("OLLAMA_THINK", "false").lower() in {"1", "true", "yes", "on"},
        "params": {
            "num_ctx": 4096,
            "num_thread": 20,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 1024,
        }
    },
    "planner": {
        "model": os.environ.get("OLLAMA_PLANNER_MODEL", "llama3.2:3b"),
        "timeout": int(os.environ.get("OLLAMA_PLANNER_TIMEOUT", "120")),
        "think": os.environ.get("OLLAMA_PLANNER_THINK", "false").lower() in {"1", "true", "yes", "on"},
        "params": {
            "num_ctx": 4096,
            "num_thread": 20,
            "temperature": 0.3,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_predict": 512,
        }
    },
    "helper": {
        "model": os.environ.get("OLLAMA_HELPER_MODEL", "qwen3.5:2b"),
        "timeout": int(os.environ.get("OLLAMA_HELPER_TIMEOUT", "120")),
        "think": os.environ.get("OLLAMA_HELPER_THINK", "false").lower() in {"1", "true", "yes", "on"},
        "params": {
            "num_ctx": 4096,
            "num_thread": 20,
            "temperature": 0.2,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_predict": 512,
        }
    },
    "vision": {
        "enabled": os.environ.get("OLLAMA_VISION_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
        "model": os.environ.get("OLLAMA_VISION_MODEL", "qwen3-vl:2b"),
        "timeout": int(os.environ.get("OLLAMA_VISION_TIMEOUT", "120")),
        "think": os.environ.get("OLLAMA_VISION_THINK", "false").lower() in {"1", "true", "yes", "on"},
        "params": {
            "num_ctx": 4096,
            "num_thread": 20,
            "temperature": 0.2,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_predict": 256,
        }
    }
}

DEFAULT_VOICEOVER_WPS = float(os.environ.get("VOICEOVER_WORDS_PER_SEC", "2.0"))

DEFAULT_SCRAPER = {
    "request_delay_sec": float(os.environ.get("SCRAPER_REQUEST_DELAY", "1.2")),
    "yt_dlp_sleep_min": float(os.environ.get("SCRAPER_YTDLP_SLEEP_MIN", "1.0")),
    "yt_dlp_sleep_max": float(os.environ.get("SCRAPER_YTDLP_SLEEP_MAX", "3.0")),
    "yt_dlp_search_count": int(os.environ.get("SCRAPER_YTDLP_SEARCH_COUNT", "8")),
    "image_delay_sec": float(os.environ.get("SCRAPER_IMAGE_DELAY", "0.6")),
}

DEFAULT_VIDEO = {
    "resolution_landscape": "1280x720",
    "resolution_portrait": "720x1280",
    "fps": 30,
    "font_name": "Inter",
    "font_size": 52,
    "font_color": "&H00FFFFFF",
    "secondary_color": "&H0000E5FF",
    "outline_color": "&H00000000",
    "shadow_color": "&H64000000",
    "outline": 2,
    "shadow": 1,
    "subtitle_position": "bottom",
    "subtitle_margin_x": 60,
    "subtitle_margin_y": 60,
}

DEFAULT_PIPELINE = {
    "add_music": True,
    "use_stock_video": True,
    "use_images": True,
    "burn_subtitles": True,
    "add_greeting": False,
    "add_closing": False,
}

SUPPORTED_LANGUAGES = [
    "English",
    "Spanish",
    "Russian",
    "German",
    "French",
    "Italian",
    "Portuguese",
]

LANGUAGE_TO_PIPER = {
    "English": "en_US-lessac-medium",
    "Spanish": "es_ES-sharvard-medium",
    "Russian": "ru_RU-dmitri-medium",
    "German": "de_DE-thorsten-medium",
    "French": "fr_FR-gilles-medium",
    "Italian": "it_IT-riccardo-medium",
    "Portuguese": "pt_BR-faber-medium",
}


@dataclass
class AppPaths:
    base_dir: str = BASE_DIR
    data_dir: str = DATA_DIR
    output_dir: str = OUTPUT_DIR
    workspace_dir: str = WORKSPACE_DIR
    models_dir: str = MODELS_DIR
    whisper_model_path: str = WHISPER_MODEL_PATH
    piper_voice_path: str = PIPER_VOICE_PATH
    piper_voice_config: str = PIPER_VOICE_CONFIG
