"""
RL Trainer: SB3 PPO training with dynamic reward injection.
"""

import logging
import time
from typing import Callable, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger("evolve-grasp")


def train_policy(
    env_config: dict,
    reward_fn: Callable,
    training_config: dict,
    save_path: Optional[str] = None,
) -> dict:
    """
    Train a policy using PPO with the given reward function.

    Args:
        env_config: Environment configuration dict
        reward_fn: The reward function to use
        training_config: Training hyperparameters
        save_path: Optional path to save the trained model

    Returns:
        dict with keys: policy, training_stats, duration_seconds
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from src.interfaces.simulation.grasp_env import make_vec_env

    n_envs = training_config.get("n_envs", 4)
    total_timesteps = training_config.get("total_timesteps", 500_000)

    logger.info(f"Starting training: {total_timesteps} steps, {n_envs} parallel envs")
    start_time = time.time()

    # Create vectorized environment
    vec_env = make_vec_env(
        config=env_config,
        reward_fn=reward_fn,
        n_envs=n_envs,
    )

    # Training progress tracker
    class ProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_successes = []

        def _on_step(self) -> bool:
            # Collect episode info from vectorized envs
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                if "success" in info:
                    self.episode_successes.append(float(info["success"]))
            return True

    callback = ProgressCallback()

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=training_config.get("learning_rate", 3e-4),
        batch_size=training_config.get("batch_size", 2048),
        n_epochs=training_config.get("n_epochs", 10),
        gamma=training_config.get("gamma", 0.99),
        gae_lambda=training_config.get("gae_lambda", 0.95),
        clip_range=training_config.get("clip_range", 0.2),
        device=training_config.get("device", "auto"),
        verbose=0,
        seed=training_config.get("seed", 42),
    )

    # Train
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        vec_env.close()
        raise

    duration = time.time() - start_time

    # Save model if path specified
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    # Compile training stats
    training_stats = {
        "total_timesteps": total_timesteps,
        "duration_seconds": round(duration, 1),
        "duration_minutes": round(duration / 60, 1),
        "n_episodes_completed": len(callback.episode_rewards),
        "mean_episode_reward": float(np.mean(callback.episode_rewards)) if callback.episode_rewards else 0.0,
        "std_episode_reward": float(np.std(callback.episode_rewards)) if callback.episode_rewards else 0.0,
        "mean_episode_length": float(np.mean(callback.episode_lengths)) if callback.episode_lengths else 0.0,
        "mean_success_rate": float(np.mean(callback.episode_successes[-100:])) if callback.episode_successes else 0.0,
    }

    logger.info(
        f"Training complete in {training_stats['duration_minutes']:.1f} min | "
        f"Episodes: {training_stats['n_episodes_completed']} | "
        f"Mean reward: {training_stats['mean_episode_reward']:.2f} | "
        f"Success rate: {training_stats['mean_success_rate']:.2%}"
    )

    vec_env.close()

    return {
        "policy": model,
        "training_stats": training_stats,
    }
