"""ARPA configuration."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARPA_ROOT = Path(__file__).resolve().parents[1]

# Load .env file explicitly
load_dotenv(PROJECT_ROOT / ".env")


class ARPASettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARPA_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM backend selection: "ollama", "gemini", "openrouter", "nvidia", or "groq"
    llm_backend: str = "ollama"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_general_model: str = "mistral:7b-instruct"
    ollama_code_model: str = "qwen2.5-coder:7b"
    ollama_timeout_s: float = 600.0

    # Gemini (Google Generative Language API)
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_general_model: str = "gemini-2.0-flash-lite"  # Will be overridden by .env
    gemini_code_model: str = "gemini-2.0-flash-lite"  # Will be overridden by .env
    gemini_timeout_s: float = 120.0
    gemini_max_retries: int = 3

    # OpenRouter (unified LLM access)
    openrouter_api_key: str | None = None
    openrouter_general_model: str = "google/gemini-2.0-flash-exp:free"
    openrouter_code_model: str = "google/gemini-2.0-flash-exp:free"

    # NVIDIA NIM (free access to 100+ models)
    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.3-70b-instruct"

    # Groq (fast, free, 30 req/min)
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # Dataset agent
    dataset_verify_max_retries: int = 3
    dataset_resolution_timeout_s: int = 60
    huggingface_cache_dir: Path | None = None

    # Docker sandbox
    docker_sandbox_image: str = "arpa-sandbox:latest"
    docker_network: str | None = None
    docker_verify_timeout_s: int = 300

    # Paths
    sandbox_dockerfile: Path = Field(
        default=PROJECT_ROOT / "docker" / "sandbox" / "Dockerfile"
    )
    work_dir: Path = Field(default=PROJECT_ROOT / ".arpa_runs")

    def ensure_work_dir(self) -> Path:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return self.work_dir


def get_settings() -> ARPASettings:
    return ARPASettings()
