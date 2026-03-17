from __future__ import annotations

import os
from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
CONFIG_PATH = ROOT_DIR / "config.ini"

load_dotenv(ENV_PATH)

DEFAULT_CHAT_MODEL = "gpt-5.4"
DEFAULT_CHAT_MODEL_FALLBACK = "gpt-5-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_EMBED_DIM = 1536


def _env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _load_config() -> ConfigParser:
    config = ConfigParser()
    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)
    return config


@dataclass
class Settings:
    openai_api_key: str | None
    openai_base_url: str
    chat_model: str
    chat_model_fallback: str
    embedding_model: str
    embedding_dimensions: int
    duplicate_threshold: float
    nearby_radius_m: int
    cluster_radius_m: int
    max_results: int
    geocoding_user_agent: str
    geocoding_provider: str
    google_maps_api_key: str | None
    google_places_region: str
    google_places_language: str
    log_level: str


_config = _load_config()

settings = Settings(
    openai_api_key=_env_value("OPENAI_API_KEY"),
    openai_base_url=_env_value(
        "OPENAI_BASE_URL",
        _config.get("openai", "base_url", fallback="https://api.openai.com/v1"),
    ),
    chat_model=_config.get("openai", "chat_model", fallback=DEFAULT_CHAT_MODEL),
    chat_model_fallback=_config.get(
        "openai", "chat_model_fallback", fallback=DEFAULT_CHAT_MODEL_FALLBACK
    ),
    embedding_model=_config.get("openai", "embedding_model", fallback=DEFAULT_EMBED_MODEL),
    embedding_dimensions=int(
        _config.get("openai", "embedding_dimensions", fallback=str(DEFAULT_EMBED_DIM))
    ),
    duplicate_threshold=float(_config.get("app", "duplicate_threshold", fallback="0.85")),
    nearby_radius_m=int(_config.get("app", "nearby_radius_m", fallback="500")),
    cluster_radius_m=int(
        _env_value("CLUSTER_RADIUS_M", _config.get("app", "cluster_radius_m", fallback="100")) or "100"
    ),
    max_results=int(_config.get("app", "max_results", fallback="20")),
    geocoding_user_agent=_env_value(
        "GEOCODING_USER_AGENT",
        _config.get("geocoding", "user_agent", fallback="urban-agent"),
    ),
    geocoding_provider=_env_value(
        "GEOCODING_PROVIDER",
        _config.get("geocoding", "provider", fallback="nominatim"),
    ),
    google_maps_api_key=_env_value("GOOGLE_MAPS_API_KEY"),
    google_places_region=_env_value(
        "GOOGLE_PLACES_REGION",
        _config.get("geocoding", "google_region", fallback="pa"),
    ),
    google_places_language=_env_value(
        "GOOGLE_PLACES_LANGUAGE",
        _config.get("geocoding", "google_language", fallback="es"),
    ),
    log_level=_env_value("LOG_LEVEL", "info") or "info",
)

if settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

if settings.openai_base_url:
    os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)
    os.environ.setdefault("OPENAI_API_BASE", settings.openai_base_url)
