"""
Auto-Evolving Grasp — Entry Point

Usage:
    python run.py                                # Default config
    python run.py --max-generations 5            # Quick test
    python run.py --timesteps 10000              # Fast training
    python run.py --config my_config.yaml        # Custom config
    python run.py --two-stage                    # Two-stage training
"""

import argparse
import signal
import sys
import os
import logging
import warnings
from datetime import datetime
from pathlib import Path

from src.utils import load_config, setup_logging
from src.reward_evolver import RewardEvolver


def suppress_noisy_loggers():
    """Suppress verbose warnings from robosuite, gymnasium, and SB3."""
    # Suppress Python warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
    warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

    # Suppress robosuite logger
    for name in ["robosuite", "mujoco", "gymnasium"]:
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)

    # Suppress robosuite print-based warnings by redirecting stderr briefly
    os.environ.setdefault("ROBOSUITE_SUPPRESS_WARNINGS", "1")


def setup_file_logging(log_dir: str = "logs"):
    """Add a file handler for detailed logging."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"evolution_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    ))

    # Add to root logger so all libraries' logs go to file
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    return log_file


def main():
    parser = argparse.ArgumentParser(description="Auto-Evolving Grasp System")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--max-generations", type=int, default=None,
        help="Override max generations (default from config)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override training timesteps per candidate (for quick testing)"
    )
    parser.add_argument(
        "--two-stage", action="store_true",
        help="Enable two-stage training: quick screen → full train"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Terminal logging level"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show all warnings (don't suppress robosuite/gym logs)"
    )
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)
    log_file = setup_file_logging()

    # Suppress noisy loggers unless --verbose
    if not args.verbose:
        suppress_noisy_loggers()

    logger.info("=" * 60)
    logger.info("  🤖 Auto-Evolving Grasp System")
    logger.info("  LLM-driven reward evolution for robotic grasping")
    logger.info(f"  📋 Detailed logs → {log_file}")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)
    logger.info(f"Config loaded from {args.config}")

    # Apply two-stage if requested
    if args.two_stage:
        config.setdefault("evolution", {})
        config["evolution"]["two_stage"] = True
        logger.info("🔀 Two-stage training enabled")

    # Create evolver
    evolver = RewardEvolver(config)

    # Graceful shutdown handler
    def signal_handler(sig, frame):
        logger.info("\n\n⚠️ Interrupted! Saving current state...")
        logger.info("Checkpoint registry has been saved.")
        logger.info("You can resume from where you left off by running again.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run evolution
    evolver.evolve(
        max_generations=args.max_generations,
        timesteps_override=args.timesteps,
    )

    logger.info("\n✅ Done! Check the results:")
    logger.info(f"  - Checkpoints: checkpoints/")
    logger.info(f"  - Rewards:     rewards/")
    logger.info(f"  - Reports:     reports/")
    logger.info(f"  - Detailed log: {log_file}")


if __name__ == "__main__":
    main()

