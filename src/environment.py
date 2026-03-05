"""
Gymnasium wrapper around Robosuite's Lift environment
with pluggable LLM-generated reward functions.
"""

import logging
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger("evolve-grasp")


class GraspEnvWrapper(gym.Env):
    """
    Wraps Robosuite's Lift task as a Gymnasium environment.
    Accepts a custom reward_fn that overrides the built-in reward.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: dict,
        reward_fn: Optional[Callable] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        import robosuite as suite
        from robosuite.wrappers import GymWrapper

        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.config = config

        # Create Robosuite environment
        self._robosuite_env = suite.make(
            env_name=config.get("task", "Lift"),
            robots=config.get("robot", "Panda"),
            has_renderer=render_mode == "human",
            has_offscreen_renderer=config.get("has_offscreen_renderer", False),
            use_camera_obs=config.get("use_camera_obs", False),
            horizon=config.get("horizon", 200),
            reward_shaping=config.get("reward_shaping", False),
            control_freq=config.get("control_freq", 20),
        )

        # Wrap with GymWrapper for Gymnasium compatibility
        self._gym_env = GymWrapper(self._robosuite_env)

        # Define action and observation space from the wrapper
        self.action_space = self._gym_env.action_space
        self.observation_space = self._gym_env.observation_space

        # Track episode state
        self._step_count = 0
        self._episode_info = {}

    def _get_info(self) -> dict:
        """Extract useful info from the Robosuite env state for reward computation."""
        env = self._robosuite_env

        info = {}

        # Gripper position
        if hasattr(env, "sim"):
            try:
                gripper_site = env.robots[0].gripper.important_sites.get(
                    "grip_site", "grip_site"
                )
                info["gripper_pos"] = np.array(
                    env.sim.data.site_xpos[env.sim.model.site_name2id(gripper_site)]
                )
            except Exception:
                # Fallback: use eef_pos if available
                info["gripper_pos"] = np.array(getattr(env, "_eef_xpos", [0, 0, 0]))

            # Object position
            try:
                obj_body = "cube_main"
                info["object_pos"] = np.array(
                    env.sim.data.body_xpos[env.sim.model.body_name2id(obj_body)]
                )
            except Exception:
                info["object_pos"] = np.array([0, 0, 0])

            # Object height (z-coordinate, useful for lift detection)
            info["object_height"] = float(info["object_pos"][2])

            # Distance between gripper and object
            info["distance"] = float(np.linalg.norm(
                info["gripper_pos"] - info["object_pos"]
            ))

            # Check if object is grasped (lifted above table)
            table_height = 0.8  # approximate Robosuite table height
            info["is_grasped"] = info["object_height"] > table_height + 0.04

            # Gripper state (open/closed)
            try:
                info["gripper_open"] = float(env.robots[0].gripper.current_action[0])
            except Exception:
                info["gripper_open"] = 0.0

        info["step"] = self._step_count
        info["horizon"] = self.config.get("horizon", 200)

        return info

    def step(self, action):
        obs, original_reward, terminated, truncated, gym_info = self._gym_env.step(action)
        self._step_count += 1

        # Get enriched info
        info = self._get_info()
        info.update(gym_info)

        # Use custom reward if provided, else fall back to original
        if self.reward_fn is not None:
            try:
                reward = float(self.reward_fn(obs, action, info))
                # Clamp reward to prevent extreme values
                reward = max(-10.0, min(10.0, reward))
            except Exception as e:
                logger.warning(f"Custom reward_fn error: {e}, using 0.0")
                reward = 0.0
        else:
            reward = original_reward

        # Track success for evaluation
        info["success"] = info.get("is_grasped", False)

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self._gym_env.reset()
        self._step_count = 0
        return obs, info

    def render(self):
        if self.render_mode == "human":
            self._robosuite_env.render()

    def close(self):
        self._robosuite_env.close()


def create_env(
    config: Optional[dict] = None,
    reward_fn: Optional[Callable] = None,
) -> GraspEnvWrapper:
    """Factory function to create a grasp environment."""
    if config is None:
        config = {
            "robot": "Panda",
            "task": "Lift",
            "horizon": 200,
            "reward_shaping": False,
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "control_freq": 20,
        }
    return GraspEnvWrapper(config=config, reward_fn=reward_fn)


def make_vec_env(
    config: dict,
    reward_fn: Optional[Callable] = None,
    n_envs: int = 4,
) -> gym.vector.VectorEnv:
    """Create vectorized environment for parallel training."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    def make_env_fn(rank: int):
        def _init():
            env = create_env(config=config, reward_fn=reward_fn)
            return env
        return _init

    if n_envs > 1:
        try:
            return SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])
        except Exception as e:
            logger.warning(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
            return DummyVecEnv([make_env_fn(i) for i in range(n_envs)])
    else:
        return DummyVecEnv([make_env_fn(0)])
