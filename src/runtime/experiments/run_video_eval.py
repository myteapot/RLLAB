"""
Visualization CLI for the best trained policy.

Usage:
    python -m src.runtime.experiments.run_video_eval
    python -m src.runtime.experiments.run_video_eval --episodes 5
    python -m src.runtime.experiments.run_video_eval --live
"""

import argparse
import logging
from pathlib import Path

from src.shared.support import default_config_path, load_config, load_yaml, setup_logging

logger = logging.getLogger("evolve-grasp")


def find_best_checkpoint(config: dict) -> dict:
    """Find the best checkpoint from registry."""
    registry_path = Path(config["checkpoint"]["base_dir"]) / "registry.yaml"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"No checkpoint registry found at {registry_path}. "
            "Run training first: python -m src.runtime.experiments.run_evolution --max-generations 1 --timesteps 10000"
        )

    registry = load_yaml(registry_path)
    global_best = registry.get("global_best")
    if global_best:
        best_task = global_best["task"]
        best_score = global_best["score"]
    else:
        best_task = None
        best_score = -float("inf")
        for task, info in registry.get("tasks", {}).items():
            candidate_score = info.get("best_score", 0.0)
            if candidate_score > best_score:
                best_score = candidate_score
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


def record_video(config: dict, n_episodes: int = 3, output_path: str = "scratch/videos/best_policy.mp4"):
    """Record the best policy as MP4 video."""
    try:
        import imageio
        import numpy as np
        from stable_baselines3 import PPO
    except ImportError as exc:
        logger.error(f"Missing dependency: {exc}")
        return

    from src.domain.rewards.execution import safe_exec_reward
    from src.interfaces.simulation.grasp_env import GraspEnvWrapper

    best = find_best_checkpoint(config)
    logger.info(f"Best checkpoint: task={best['task']}, score={best['score']:.4f}, gen={best['generation']}")

    reward_fn = None
    if Path(best["reward_path"]).exists():
        code = Path(best["reward_path"]).read_text(encoding="utf-8")
        reward_fn = safe_exec_reward(code)

    policy_path = best["policy_path"]
    if not Path(policy_path).exists():
        logger.error(f"Policy file not found: {policy_path}")
        return

    policy = PPO.load(policy_path)
    env_config = dict(config["environment"])
    env_config["has_offscreen_renderer"] = True
    env_config["camera_names"] = "agentview"
    env_config["camera_heights"] = 480
    env_config["camera_widths"] = 640
    env = GraspEnvWrapper(config=env_config, reward_fn=reward_fn)

    frames = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        info = {}

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            try:
                frame = env.render_frame(camera_name="agentview", width=640, height=480)
                frames.append(frame)
            except Exception:
                pass

        success = "✅" if info.get("is_grasped", False) else "❌"
        logger.info(f"  Episode {episode + 1}/{n_episodes}: reward={total_reward:.2f}, steps={steps}, {success}")

    env.close()

    if not frames:
        logger.warning("No frames captured. Offscreen rendering may not be available.")
        logger.info("Try: pip install imageio[ffmpeg]")
        return

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_file, fps=30)
    for frame in frames:
        writer.append_data(np.flipud(frame))
    writer.close()
    logger.info(f"\nVideo saved: {output_file} ({len(frames)} frames, {len(frames) / 30:.1f}s)")


def live_render(config: dict, n_episodes: int = 3):
    """Try live rendering via WSLg/X11."""
    from stable_baselines3 import PPO

    from src.domain.rewards.execution import safe_exec_reward
    from src.interfaces.simulation.grasp_env import GraspEnvWrapper

    best = find_best_checkpoint(config)
    logger.info(f"Live render: task={best['task']}, score={best['score']:.4f}")

    reward_fn = None
    if Path(best["reward_path"]).exists():
        code = Path(best["reward_path"]).read_text(encoding="utf-8")
        reward_fn = safe_exec_reward(code)

    policy = PPO.load(best["policy_path"])
    env_config = dict(config["environment"])
    env_config["has_renderer"] = True
    env = GraspEnvWrapper(config=env_config, reward_fn=reward_fn)

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            try:
                env.render()
            except Exception as exc:
                logger.error(f"Live render failed: {exc}")
                logger.info("WSLg may not be available. Use record mode instead.")
                env.close()
                return

    env.close()
    logger.info("Live render complete")


def main():
    parser = argparse.ArgumentParser(description="Visualize best trained policy")
    parser.add_argument("--config", type=str, default=str(default_config_path()))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="scratch/videos/best_policy.mp4")
    parser.add_argument("--live", action="store_true", help="Try live WSLg rendering")
    args = parser.parse_args()

    setup_logging("INFO")
    config = load_config(args.config)

    if args.live:
        logger.info("Attempting live render (requires WSLg or X11)...")
        live_render(config, args.episodes)
    else:
        logger.info("Recording best policy to video...")
        record_video(config, args.episodes, args.output)


if __name__ == "__main__":
    main()
