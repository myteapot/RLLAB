"""Shared support helpers for configuration, logging, and simple persistence."""

from __future__ import annotations

import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def default_config_path() -> Path:
    """Return the default configuration file path."""
    return resolve_project_path("configs/default.yaml")


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging with console output."""
    logger = logging.getLogger("evolve-grasp")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML configuration and normalize well-known repository paths."""
    config_path = resolve_project_path(path or default_config_path())
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    path_overrides = [
        ("checkpoint", "base_dir"),
        ("paths", "rewards_dir"),
        ("paths", "configs_dir"),
        ("paths", "reports_dir"),
        ("paths", "prompts_dir"),
        ("paths", "logs_dir"),
    ]
    for section, key in path_overrides:
        section_data = config.get(section)
        if section_data and section_data.get(key):
            section_data[key] = str(resolve_project_path(section_data[key]))

    return config


def save_yaml(data: Any, path: str | Path) -> None:
    """Save data to YAML, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_yaml(path: str | Path) -> Any:
    """Load YAML data from disk."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def now_iso() -> str:
    """Return the current timestamp as ISO 8601."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def hash_code(code: str) -> str:
    """Return a short stable hash for generated code."""
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories if they do not already exist."""
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
