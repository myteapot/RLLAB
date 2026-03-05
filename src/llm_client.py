"""
LLM client: OpenAI-compatible API wrapper for reward generation and analysis.
"""

import os
import time
import json
import logging
from typing import Optional

from openai import OpenAI
from pathlib import Path
from src.utils import extract_python_code

logger = logging.getLogger("evolve-grasp")


class LLMClient:
    """OpenAI-compatible LLM client for reward function generation."""

    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 2.0)

        self.client = OpenAI(
            base_url=config.get("base_url", os.environ.get("OPENAI_BASE_URL")),
            api_key=config.get("api_key", os.environ.get("OPENAI_API_KEY", "sk-none")),
        )

        # Load prompt templates
        self.prompts_dir = Path(config.get("prompts_dir", "prompts"))

    def _call(self, system: str, user: str) -> str:
        """Call LLM with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM call failed after {self.retry_attempts} attempts: {e}")

    def _load_prompt(self, name: str) -> str:
        """Load a prompt template from file."""
        path = self.prompts_dir / f"{name}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Prompt template not found: {path}")

    def generate_reward(
        self,
        env_description: str,
        history_best: Optional[dict] = None,
        failure_analysis: Optional[str] = None,
        previous_rewards: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate a new reward function.
        Returns raw Python code string.
        """
        system = (
            "You are an expert robotics reward engineer. "
            "You write Python reward functions for MuJoCo robotic grasping tasks. "
            "Always output a complete Python function named `reward_fn(obs, action, info)` "
            "that returns a float. You may use numpy (imported as np) and math. "
            "Output ONLY the Python code in a ```python block, no explanations."
        )

        prompt = self._load_prompt("reward_init")
        prompt = prompt.replace("{ENV_DESCRIPTION}", env_description)
        prompt = prompt.replace(
            "{BEST_SCORE}",
            str(history_best.get("score", "N/A")) if history_best else "N/A"
        )
        prompt = prompt.replace(
            "{BEST_REWARD_CODE}",
            history_best.get("reward_code", "None yet") if history_best else "None yet"
        )
        prompt = prompt.replace(
            "{FAILURE_ANALYSIS}",
            failure_analysis or "No failure analysis yet (first generation)."
        )

        # Add previous rewards context if available
        if previous_rewards:
            prev_ctx = "\n".join([
                f"- Gen {r['gen']} (score={r['score']:.4f}): {r.get('summary', 'no summary')}"
                for r in previous_rewards[-5:]  # Last 5
            ])
            prompt = prompt.replace("{PREVIOUS_REWARDS}", prev_ctx)
        else:
            prompt = prompt.replace("{PREVIOUS_REWARDS}", "No previous rewards yet.")

        response = self._call(system, prompt)
        return extract_python_code(response)

    def mutate_reward(
        self,
        current_code: str,
        current_score: float,
        metrics: dict,
        failure_analysis: str,
    ) -> str:
        """
        Mutate an existing reward function based on its performance.
        Returns improved Python code string.
        """
        system = (
            "You are an expert robotics reward engineer. "
            "You are improving an existing reward function based on training results. "
            "Output ONLY the improved Python function named `reward_fn(obs, action, info)` "
            "in a ```python block. Make targeted improvements, don't rewrite from scratch."
        )

        prompt = self._load_prompt("reward_mutate")
        prompt = prompt.replace("{CURRENT_CODE}", current_code)
        prompt = prompt.replace("{CURRENT_SCORE}", f"{current_score:.4f}")
        prompt = prompt.replace("{METRICS}", json.dumps(metrics, indent=2))
        prompt = prompt.replace("{FAILURE_ANALYSIS}", failure_analysis)

        response = self._call(system, prompt)
        return extract_python_code(response)

    def analyze_results(
        self,
        generation: int,
        results: list[dict],
        best_ever: Optional[dict],
        failure_cases: list[str],
    ) -> dict:
        """
        Analyze a generation's results. Returns analysis dict.
        """
        system = (
            "You are a robotics training analyst. "
            "Analyze the training results and provide insights. "
            "Output a JSON object with keys: "
            "'summary' (str), 'key_findings' (list[str]), "
            "'improvement_direction' (str), 'failure_patterns' (list[str])."
        )

        prompt = self._load_prompt("failure_analysis")
        prompt = prompt.replace("{GENERATION}", str(generation))
        prompt = prompt.replace("{RESULTS}", json.dumps(results, indent=2, default=str))
        prompt = prompt.replace(
            "{BEST_EVER}",
            json.dumps(best_ever, indent=2, default=str) if best_ever else "None yet"
        )
        prompt = prompt.replace("{FAILURE_CASES}", "\n".join(failure_cases) if failure_cases else "None")

        response = self._call(system, prompt)

        # Parse JSON from response
        try:
            # Try to extract JSON from response
            text = response.strip()
            if "```json" in text:
                start = text.index("```json") + 7
                end = text.index("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.index("```") + 3
                end = text.index("```", start)
                text = text[start:end].strip()
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {
                "summary": response[:500],
                "key_findings": [],
                "improvement_direction": "Could not parse structured analysis",
                "failure_patterns": [],
            }
