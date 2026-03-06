"""
Evaluator: Multi-dimensional policy evaluation.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger("evolve-grasp")


@dataclass
class EvalMetrics:
    """Multi-dimensional evaluation metrics."""
    success_rate: float = 0.0      # Fraction of episodes where object was grasped
    stability: float = 0.0         # How long the grasp was maintained (normalized)
    efficiency: float = 0.0        # How quickly the task was completed (normalized)
    smoothness: float = 0.0        # Action smoothness (inverse of jerk)
    composite_score: float = 0.0   # Weighted combination

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_policy(
    policy,
    env_config: dict,
    reward_fn=None,
    n_episodes: int = 50,
    weights: Optional[dict] = None,
) -> tuple[EvalMetrics, list[str]]:
    """
    Evaluate a trained policy across multiple episodes.

    Args:
        policy: Trained SB3 policy (model)
        env_config: Environment configuration
        reward_fn: Reward function (for consistency with training)
        n_episodes: Number of evaluation episodes
        weights: Metric weights for composite score

    Returns:
        (EvalMetrics, failure_descriptions)
    """
    from src.interfaces.simulation.grasp_env import create_env

    if weights is None:
        weights = {
            "success_rate": 0.40,
            "stability": 0.25,
            "efficiency": 0.20,
            "smoothness": 0.15,
        }

    env = create_env(config=env_config, reward_fn=reward_fn)

    successes = []
    grasp_durations = []
    completion_steps = []
    action_jerks = []
    failure_descriptions = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        grasped_steps = 0
        prev_action = None
        prev_prev_action = None
        jerks = []
        was_successful = False

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            # Track success
            if info.get("is_grasped", False):
                grasped_steps += 1
                was_successful = True

            # Track action smoothness (jerk = second derivative of action)
            if prev_prev_action is not None:
                jerk = np.linalg.norm(action - 2 * prev_action + prev_prev_action)
                jerks.append(jerk)
            prev_prev_action = prev_action
            prev_action = action.copy() if isinstance(action, np.ndarray) else np.array(action)

        successes.append(float(was_successful))
        grasp_durations.append(grasped_steps)
        completion_steps.append(step)

        if jerks:
            action_jerks.append(float(np.mean(jerks)))

        # Record failure description
        if not was_successful:
            dist = info.get("distance", -1)
            height = info.get("object_height", -1)
            failure_descriptions.append(
                f"Ep {ep}: Failed. Final distance={dist:.3f}, "
                f"object_height={height:.3f}, steps={step}"
            )

    env.close()

    # Compute metrics
    horizon = env_config.get("horizon", 200)

    success_rate = float(np.mean(successes))

    # Stability: average fraction of episode spent grasping (among successful episodes)
    successful_durations = [d for d, s in zip(grasp_durations, successes) if s > 0]
    stability = float(np.mean(successful_durations)) / horizon if successful_durations else 0.0

    # Efficiency: how quickly you complete (inverse of steps, normalized)
    successful_steps = [s for s, succ in zip(completion_steps, successes) if succ > 0]
    if successful_steps:
        efficiency = 1.0 - float(np.mean(successful_steps)) / horizon
        efficiency = max(0.0, efficiency)
    else:
        efficiency = 0.0

    # Smoothness: inverse of mean jerk, normalized to [0, 1]
    if action_jerks:
        mean_jerk = float(np.mean(action_jerks))
        smoothness = 1.0 / (1.0 + mean_jerk)  # higher = smoother
    else:
        smoothness = 0.0

    # Composite score
    composite = (
        weights["success_rate"] * success_rate +
        weights["stability"] * stability +
        weights["efficiency"] * efficiency +
        weights["smoothness"] * smoothness
    )

    metrics = EvalMetrics(
        success_rate=round(success_rate, 4),
        stability=round(stability, 4),
        efficiency=round(efficiency, 4),
        smoothness=round(smoothness, 4),
        composite_score=round(composite, 4),
    )

    logger.info(
        f"Evaluation ({n_episodes} episodes): "
        f"success={metrics.success_rate:.2%}, "
        f"stability={metrics.stability:.3f}, "
        f"efficiency={metrics.efficiency:.3f}, "
        f"smoothness={metrics.smoothness:.3f}, "
        f"composite={metrics.composite_score:.4f}"
    )

    # Keep only a sample of failures for LLM analysis
    if len(failure_descriptions) > 10:
        failure_descriptions = failure_descriptions[:5] + failure_descriptions[-5:]

    return metrics, failure_descriptions
