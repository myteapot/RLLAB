"""
Checkpoint Registry: track best weights per task, with config snapshots.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from src.shared.support import load_yaml, now_iso, save_yaml

logger = logging.getLogger("evolve-grasp")


class CheckpointRegistry:
    """Manages best checkpoints per task with Top-K retention."""

    def __init__(self, config: dict):
        self.base_dir = Path(config.get("base_dir", "scratch/checkpoints"))
        self.top_k = config.get("top_k", 5)
        self.registry_path = self.base_dir / "registry.yaml"
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load or initialize the registry."""
        if self.registry_path.exists():
            return load_yaml(self.registry_path)
        return {
            "last_updated": now_iso(),
            "global_best": None,
            "tasks": {},
        }

    def _save_registry(self):
        """Persist registry to disk."""
        self.registry["last_updated"] = now_iso()
        save_yaml(self.registry, self.registry_path)

    def register(
        self,
        task: str,
        generation: int,
        score: float,
        policy,
        config_snapshot: dict,
        reward_code: str = "",
    ) -> bool:
        """
        Register a training result. Saves checkpoint if it's in Top-K.
        Returns True if this is a new best.
        """
        task_dir = self.base_dir / f"task_{task}"
        history_dir = task_dir / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        # Initialize task entry if needed
        if task not in self.registry["tasks"]:
            self.registry["tasks"][task] = {
                "best_score": 0.0,
                "best_generation": -1,
                "total_generations": 0,
                "improvement_history": [],
                "top_k_checkpoints": [],
            }

        task_entry = self.registry["tasks"][task]
        task_entry["total_generations"] = max(
            task_entry["total_generations"], generation + 1
        )

        # Save checkpoint
        ckpt_name = f"ckpt_gen{generation:04d}_score{score:.3f}"
        ckpt_path = history_dir / f"{ckpt_name}.zip"  # SB3 saves as .zip
        policy.save(str(ckpt_path.with_suffix("")))  # SB3 adds .zip automatically

        # Save config snapshot alongside
        save_yaml(config_snapshot, history_dir / f"{ckpt_name}_config.yaml")

        # Save reward code
        if reward_code:
            reward_path = history_dir / f"{ckpt_name}_reward.py"
            reward_path.write_text(reward_code, encoding="utf-8")

        # Update top-k list
        task_entry["top_k_checkpoints"].append({
            "gen": generation,
            "score": round(score, 4),
            "path": str(ckpt_path),
            "timestamp": now_iso(),
        })
        task_entry["top_k_checkpoints"].sort(key=lambda x: x["score"], reverse=True)

        # Prune to top-k
        if len(task_entry["top_k_checkpoints"]) > self.top_k:
            removed = task_entry["top_k_checkpoints"][self.top_k:]
            task_entry["top_k_checkpoints"] = task_entry["top_k_checkpoints"][:self.top_k]
            # Clean up old files
            for r in removed:
                try:
                    p = Path(r["path"])
                    if p.exists():
                        p.unlink()
                    # Also remove associated config and reward files
                    for suffix in ["_config.yaml", "_reward.py"]:
                        assoc = p.with_suffix("").with_suffix(suffix)
                        if assoc.exists():
                            assoc.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up old checkpoint: {e}")

        # Check if new best
        is_new_best = score > task_entry["best_score"]
        if is_new_best:
            old_best = task_entry["best_score"]
            task_entry["best_score"] = round(score, 4)
            task_entry["best_generation"] = generation
            task_entry["improvement_history"].append({
                "gen": generation,
                "score": round(score, 4),
                "delta": f"+{score - old_best:.4f}",
                "timestamp": now_iso(),
            })

            # Copy as "best" for easy access
            best_path = task_dir / "best"
            policy.save(str(best_path))
            save_yaml(config_snapshot, task_dir / "best_config.yaml")
            if reward_code:
                (task_dir / "best_reward.py").write_text(reward_code, encoding="utf-8")

            # Update global best
            if (
                self.registry["global_best"] is None
                or score > self.registry["global_best"]["score"]
            ):
                self.registry["global_best"] = {
                    "task": task,
                    "generation": generation,
                    "score": round(score, 4),
                    "checkpoint": str(task_dir / "best.zip"),
                }

            logger.info(
                f"🏆 New best for '{task}': gen={generation}, "
                f"score={score:.4f} (Δ={score - old_best:+.4f})"
            )
        else:
            logger.info(
                f"Gen {generation}: score={score:.4f} "
                f"(best={task_entry['best_score']:.4f})"
            )

        self._save_registry()
        return is_new_best

    def get_best(self, task: str) -> Optional[dict]:
        """Get info about the best checkpoint for a task."""
        if task in self.registry["tasks"]:
            entry = self.registry["tasks"][task]
            return {
                "score": entry["best_score"],
                "generation": entry["best_generation"],
                "total_generations": entry["total_generations"],
                "improvement_history": entry["improvement_history"],
            }
        return None

    def get_best_reward_code(self, task: str) -> Optional[str]:
        """Load the best reward code for a task."""
        path = self.base_dir / f"task_{task}" / "best_reward.py"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def get_improvement_rate(self, task: str, last_n: int = 10) -> float:
        """Check if we're still improving (for convergence detection)."""
        entry = self.registry["tasks"].get(task, {})
        history = entry.get("improvement_history", [])
        if len(history) < 2:
            return 1.0  # Still early, assume improving

        recent = history[-last_n:]
        if len(recent) < 2:
            return 0.0

        # Rate of improvement: total delta / number of generations
        total_delta = recent[-1]["score"] - recent[0]["score"]
        gen_span = recent[-1]["gen"] - recent[0]["gen"]
        if gen_span == 0:
            return 0.0
        return total_delta / gen_span
