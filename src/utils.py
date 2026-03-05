"""
Utility functions: logging, YAML helpers, safe code execution.
"""

import logging
import os
import sys
import yaml
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging with console output."""
    logger = logging.getLogger("evolve-grasp")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Any, path: str | Path) -> None:
    """Save data to YAML file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_yaml(path: str | Path) -> Any:
    """Load data from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_iso() -> str:
    """Current time as ISO 8601 string."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def hash_code(code: str) -> str:
    """SHA-256 hash of code string (for tracking)."""
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response (handles ```python blocks)."""
    # Try to find ```python ... ``` block
    markers = ["```python", "```Python", "```py"]
    for marker in markers:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start)
            return text[start:end].strip()

    # Fallback: if text starts with def, treat whole thing as code
    stripped = text.strip()
    if stripped.startswith("def ") or stripped.startswith("import "):
        return stripped

    # Last resort: return as-is
    return stripped


def safe_exec_reward(code: str) -> callable:
    """
    Safely execute LLM-generated reward function code.
    Only allows numpy and math in the namespace.
    Returns the reward_fn callable.
    """
    import numpy as np
    import math

    # Restricted namespace
    namespace = {
        "np": np,
        "numpy": np,
        "math": math,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "float": float,
        "int": int,
        "bool": bool,
        "__builtins__": {},  # Restrict builtins
    }

    try:
        exec(code, namespace)
    except Exception as e:
        raise RuntimeError(f"Failed to execute reward code: {e}\n\nCode:\n{code}")

    # Find the reward function
    if "reward_fn" in namespace:
        return namespace["reward_fn"]
    elif "compute_reward" in namespace:
        return namespace["compute_reward"]
    else:
        # Look for any function defined
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_") and name not in (
                "abs", "min", "max", "sum", "len", "range", "float", "int", "bool"
            ):
                return obj
        raise RuntimeError(
            f"No reward function found in generated code. "
            f"Expected 'reward_fn' or 'compute_reward'. "
            f"Found names: {[k for k in namespace if not k.startswith('_')]}"
        )
