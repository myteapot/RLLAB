"""Reward-code extraction and sandboxed execution helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Callable


class MissingNumpyProxy:
    """Small placeholder that raises a clear error when NumPy is actually needed."""

    def __getattr__(self, name):
        raise ModuleNotFoundError("The `numpy` package is required for reward code that uses NumPy APIs.")


try:
    import numpy as np
except ModuleNotFoundError:
    np = MissingNumpyProxy()


def extract_python_code(text: str) -> str:
    """Extract Python code from an LLM response."""
    markers = ["```python", "```Python", "```py"]
    for marker in markers:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start)
            return text[start:end].strip()

    stripped = text.strip()
    if stripped.startswith("def ") or stripped.startswith("import "):
        return stripped
    return stripped


def safe_exec_reward(code: str) -> Callable:
    """Execute generated reward code in a restricted namespace."""
    lines = code.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(import |from \S+ import )", stripped):
            continue
        cleaned_lines.append(line)
    code = "\n".join(cleaned_lines)

    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "float": float,
        "int": int,
        "bool": bool,
        "str": str,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "True": True,
        "False": False,
        "None": None,
        "print": print,
        "isinstance": isinstance,
        "round": round,
    }

    namespace = {
        "np": np,
        "numpy": np,
        "math": math,
        "__builtins__": safe_builtins,
    }

    try:
        exec(code, namespace)
    except Exception as exc:
        raise RuntimeError(f"Failed to execute reward code: {exc}\n\nCode:\n{code}") from exc

    if "reward_fn" in namespace:
        return namespace["reward_fn"]
    if "compute_reward" in namespace:
        return namespace["compute_reward"]

    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("_") and name not in {"abs", "min", "max", "sum", "len", "range", "float", "int", "bool"}:
            return obj

    raise RuntimeError(
        "No reward function found in generated code. "
        "Expected 'reward_fn' or 'compute_reward'. "
        f"Found names: {[key for key in namespace if not key.startswith('_')]}"
    )
