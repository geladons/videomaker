"""
Test configuration for Ollama service.
This module provides utilities to ensure Ollama is available for tests.
"""

import os
import subprocess
import time
from typing import Optional


def is_ollama_running() -> bool:
    """Check if Ollama service is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def ensure_ollama_model(model_name: str = "qwen3.5:9b") -> bool:
    """Ensure the specified Ollama model is available."""
    try:
        # Check if model exists
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if model_name in result.stdout:
            return True
            
        # Pull the model if it doesn't exist
        print(f"Pulling model {model_name}...")
        pull_result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for model download
        )
        
        if pull_result.returncode == 0:
            print(f"Successfully pulled model {model_name}")
            return True
        else:
            print(f"Failed to pull model {model_name}: {pull_result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout while ensuring model {model_name}")
        return False
    except Exception as e:
        print(f"Error ensuring model {model_name}: {e}")
        return False


def setup_test_environment():
    """Setup test environment with Ollama service."""
    if not is_ollama_running():
        print("Ollama service is not running. Tests may fail.")
        print("To run Ollama locally, install it from: https://ollama.com/download")
        return False
    
    # Ensure required models are available
    models = ["qwen3.5:9b", "llama3.2:3b", "qwen3.5:2b", "qwen3-vl:2b", "qwen3.5:4b"]
    success = True
    
    for model in models:
        if not ensure_ollama_model(model):
            success = False
    
    return success


if __name__ == "__main__":
    setup_test_environment()