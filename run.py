"""
Auto-Evolving Grasp — Entry Point

Usage:
    python run.py                                # Default config
    python run.py --max-generations 5            # Quick test
    python run.py --timesteps 10000              # Fast training
    python run.py --config my_config.yaml        # Custom config
"""

import argparse
import signal
import sys
import logging

from src.utils import load_config, setup_logging
from src.reward_evolver import RewardEvolver


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
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("  🤖 Auto-Evolving Grasp System")
    logger.info("  LLM-driven reward evolution for robotic grasping")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)
    logger.info(f"Config loaded from {args.config}")

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
    logger.info(f"  - Configs:     configs/snapshots/")


if __name__ == "__main__":
    main()
