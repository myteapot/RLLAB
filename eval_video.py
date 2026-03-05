"""
Visualization script: render the best trained policy and save as MP4.

Usage:
    python eval_video.py                    # Use best checkpoint
    python eval_video.py --episodes 5       # Record 5 episodes
    python eval_video.py --live             # Try live WSLg rendering
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.utils import load_config, setup_logging, load_yaml, safe_exec_reward
from src.environment import GraspEnvWrapper

logger = logging.getLogger("evolve-grasp")


def find_best_checkpoint(config: dict) -> dict:
    """Find the best checkpoint from registry."""
    registry_path = Path(config["checkpoint"]["base_dir"]) / "registry.yaml"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"No checkpoint registry found at {registry_path}. "
            "Run training first: python run.py --max-generations 1 --timesteps 10000"
        )

    registry = load_yaml(registry_path)
    # Find the task with the best score
    best_task = None
    best_score = -float("inf")
    for task, info in registry.get("tasks", {}).items():
        if info.get("score", 0) > best_score:
            best_score = info["score"]
            best_task = task

    if best_task is None:
        raise ValueError("No trained checkpoints found in registry")

    task_info = registry["tasks"][best_task]
    task_dir = Path(config["checkpoint"]["base_dir"]) / f"task_{best_task}"

    return {
        "task": best_task,
        "score": best_score,
        "generation": task_info.get("best_generation", -1),
        "policy_path": str(task_dir / "best.zip"),
        "reward_path": str(task_dir / "best_reward.py"),
    }


def record_video(config: dict, n_episodes: int = 3, output_path: str = "best_policy.mp4"):
    """Record the best policy as MP4 video."""
    try:
        import mujoco
        from stable_baselines3 import PPO
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return

    # Find best checkpoint
    best = find_best_checkpoint(config)
    logger.info(f"Best checkpoint: task={best['task']}, score={best['score']:.4f}, gen={best['generation']}")

    # Load reward function
    reward_fn = None
    if Path(best["reward_path"]).exists():
        code = Path(best["reward_path"]).read_text(encoding="utf-8")
        reward_fn = safe_exec_reward(code)

    # Load policy
    policy_path = best["policy_path"]
    if not Path(policy_path).exists():
        logger.error(f"Policy file not found: {policy_path}")
        return

    policy = PPO.load(policy_path)

    # Create environment with offscreen rendering
    env_config = dict(config["environment"])
    env_config["has_offscreen_renderer"] = True
    env_config["camera_names"] = "agentview"
    env_config["camera_heights"] = 480
    env_config["camera_widths"] = 640

    env = GraspEnvWrapper(config=env_config, reward_fn=reward_fn)

    # Record episodes
    frames = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Capture frame
            try:
                frame = env.env.sim.render(
                    camera_name="agentview",
                    width=640,
                    height=480,
                )
                frames.append(frame)
            except Exception:
                pass

        success = "✅" if info.get("is_grasped", False) else "❌"
        logger.info(f"  Episode {ep+1}/{n_episodes}: reward={total_reward:.2f}, steps={steps}, {success}")

    env.close()

    if not frames:
        logger.warning("No frames captured. Offscreen rendering may not be available.")
        logger.info("Try: pip install imageio[ffmpeg]")
        return

    # Save video
    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=30)
        for frame in frames:
            # MuJoCo returns upside-down frames
            writer.append_data(np.flipud(frame))
        writer.close()
        logger.info(f"\n🎬 Video saved: {output_path} ({len(frames)} frames, {len(frames)/30:.1f}s)")
    except ImportError:
        logger.error("Install imageio for video: pip install imageio[ffmpeg]")
        # Fallback: save as numpy
        np.save("frames.npy", np.array(frames))
        logger.info("Frames saved as frames.npy (install imageio for MP4)")


def live_render(config: dict, n_episodes: int = 3):
    """Try live rendering via WSLg/X11."""
    from stable_baselines3 import PPO

    best = find_best_checkpoint(config)
    logger.info(f"Live render: task={best['task']}, score={best['score']:.4f}")

    # Load reward function
    reward_fn = None
    if Path(best["reward_path"]).exists():
        code = Path(best["reward_path"]).read_text(encoding="utf-8")
        reward_fn = safe_exec_reward(code)

    policy = PPO.load(best["policy_path"])

    # Create environment with live renderer
    env_config = dict(config["environment"])
    env_config["has_renderer"] = True

    env = GraspEnvWrapper(config=env_config, reward_fn=reward_fn)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            try:
                env.env.render()
            except Exception as e:
                logger.error(f"Live render failed: {e}")
                logger.info("WSLg may not be available. Use --record mode instead.")
                env.close()
                return

    env.close()
    logger.info("Live render complete")


def main():
    parser = argparse.ArgumentParser(description="Visualize best trained policy")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="best_policy.mp4")
    parser.add_argument("--live", action="store_true", help="Try live WSLg rendering")
    args = parser.parse_args()

    setup_logging("INFO")
    config = load_config(args.config)

    if args.live:
        logger.info("🖥️ Attempting live render (requires WSLg or X11)...")
        live_render(config, args.episodes)
    else:
        logger.info("🎬 Recording best policy to video...")
        record_video(config, args.episodes, args.output)


if __name__ == "__main__":
    main()
