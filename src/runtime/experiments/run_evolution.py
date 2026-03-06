"""
CLI entry point for reward evolution experiments.

Usage:
    python -m src.runtime.experiments.run_evolution
    python -m src.runtime.experiments.run_evolution --max-generations 5
    python -m src.runtime.experiments.run_evolution --timesteps 10000
    python -m src.runtime.experiments.run_evolution --config configs/default.yaml
    python -m src.runtime.experiments.run_evolution --two-stage
"""

import argparse
import logging
import os
import signal
import sys
import warnings
from pathlib import Path

from src.shared.support import default_config_path, load_config, setup_logging


def suppress_noisy_loggers():
    """Suppress verbose warnings from robosuite, gymnasium, and SB3."""
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
    warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

    for name in ["robosuite", "mujoco", "gymnasium"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    os.environ.setdefault("ROBOSUITE_SUPPRESS_WARNINGS", "1")


def setup_file_logging(log_dir: str | Path):
    """Add a file handler for detailed logging."""
    from datetime import datetime

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"evolution_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)
    return log_file


def main():
    parser = argparse.ArgumentParser(description="Auto-Evolving Grasp System")
    parser.add_argument("--config", type=str, default=str(default_config_path()), help="Path to configuration YAML file")
    parser.add_argument("--max-generations", type=int, default=None, help="Override max generations (default from config)")
    parser.add_argument("--timesteps", type=int, default=None, help="Override training timesteps per candidate (for quick testing)")
    parser.add_argument("--two-stage", action="store_true", help="Enable two-stage training: quick screen -> full train")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Terminal logging level")
    parser.add_argument("--verbose", action="store_true", help="Show all warnings (don't suppress robosuite/gym logs)")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(args.log_level)
    log_file = setup_file_logging(config["paths"].get("logs_dir", "scratch/logs"))

    if not args.verbose:
        suppress_noisy_loggers()

    from src.runtime.experiments.evolver import RewardEvolver

    logger.info("=" * 60)
    logger.info("  Auto-Evolving Grasp System")
    logger.info("  LLM-driven reward evolution for robotic grasping")
    logger.info(f"  Detailed logs -> {log_file}")
    logger.info("=" * 60)
    logger.info(f"Config loaded from {args.config}")

    if args.two_stage:
        config.setdefault("evolution", {})
        config["evolution"]["two_stage"] = True
        logger.info("Two-stage training enabled")

    evolver = RewardEvolver(config)

    def signal_handler(sig, frame):
        logger.info("\n\nInterrupted! Saving current state...")
        logger.info("Checkpoint registry has been saved.")
        logger.info("You can resume from where you left off by running again.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    evolver.evolve(max_generations=args.max_generations, timesteps_override=args.timesteps)

    logger.info("\nDone! Check the results:")
    logger.info(f"  - Checkpoints: {config['checkpoint']['base_dir']}")
    logger.info(f"  - Rewards:     {config['paths']['rewards_dir']}")
    logger.info(f"  - Reports:     {config['paths']['reports_dir']}")
    logger.info(f"  - Detailed log: {log_file}")


if __name__ == "__main__":
    main()
