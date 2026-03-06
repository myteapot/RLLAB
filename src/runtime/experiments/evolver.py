"""
Reward Evolver: main orchestration loop for LLM-driven reward evolution.
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Optional

from src.domain.rewards.execution import safe_exec_reward
from src.runtime.experiments.checkpoints import CheckpointRegistry
from src.shared.support import ensure_dirs, hash_code, now_iso, save_yaml

logger = logging.getLogger("evolve-grasp")

# Environment description for LLM prompts
ENV_DESCRIPTION = """
Robot: Panda 7-DOF arm with parallel-jaw gripper
Task: Lift a cube (2.5cm) from the table surface
Simulation: MuJoCo via Robosuite

Observation space: A flat numpy array containing:
  - Robot joint positions (7)
  - Robot joint velocities (7)
  - Gripper finger positions (2)
  - Object position (3)
  - Object orientation (4, quaternion)
  Total: ~32 dimensions, all floats

Action space: Continuous, 7 dimensions [-1, 1]
  - Joint velocity targets (6) + gripper open/close (1)
  - Negative gripper value = close, positive = open

Available info dict keys for reward_fn(obs, action, info):
  - info["gripper_pos"]: np.array shape (3,) — gripper center position (x, y, z)
  - info["object_pos"]: np.array shape (3,) — cube center position (x, y, z)
  - info["object_height"]: float — cube z-coordinate
  - info["distance"]: float — Euclidean distance gripper ↔ object
  - info["is_grasped"]: bool — True if object is lifted above table
  - info["gripper_open"]: float — gripper state, 0=closed, 1=fully open
  - info["step"]: int — current step in episode
  - info["horizon"]: int — max steps per episode (200)
"""


class RewardEvolver:
    """Orchestrates the LLM reward evolution loop."""

    def __init__(self, config: dict):
        self.config = config
        llm_config = dict(config["llm"])
        llm_config["prompts_dir"] = config["paths"]["prompts_dir"]
        self._llm_config = llm_config
        self.llm = None
        self.checkpoint = CheckpointRegistry(config["checkpoint"])
        self.task_name = config["environment"].get("task", "Lift").lower()

        # Directories
        self.rewards_dir = Path(config["paths"]["rewards_dir"])
        self.configs_dir = Path(config["paths"]["configs_dir"])
        self.reports_dir = Path(config["paths"]["reports_dir"])
        ensure_dirs(self.rewards_dir, self.configs_dir, self.reports_dir)

        # Evolution state
        self.generation = 0
        self.population = []  # List of {code, score, metrics, gen}
        self.best_ever = None

    def _get_llm(self):
        """Create the LLM client only when it is actually needed."""
        if self.llm is None:
            from src.interfaces.llm.client import LLMClient

            self.llm = LLMClient(self._llm_config)
        return self.llm

    def evolve(
        self,
        max_generations: Optional[int] = None,
        timesteps_override: Optional[int] = None,
    ):
        """
        Main evolution loop.

        Args:
            max_generations: Override max generations from config
            timesteps_override: Override training timesteps (for quick testing)
        """
        evo_config = self.config["evolution"]
        max_gen = evo_config.get("max_generations", 100) if max_generations is None else max_generations
        n_candidates = evo_config.get("candidates_per_generation", 4)
        patience = evo_config.get("convergence_patience", 10)

        training_config = dict(self.config["training"])
        if timesteps_override:
            training_config["total_timesteps"] = timesteps_override

        eval_config = self.config["evaluation"]
        env_config = self.config["environment"]

        # Resume from previous state if available
        existing_best = self.checkpoint.get_best(self.task_name)
        if existing_best and existing_best.get("generation", -1) >= 0:
            self.generation = existing_best.get("total_generations", 0)
            best_code = self.checkpoint.get_best_reward_code(self.task_name)
            self.best_ever = {
                "score": existing_best["score"],
                "reward_code": best_code or "",
                "gen": existing_best.get("generation", 0),
            }
            logger.info(
                f"Resuming from generation {self.generation}, "
                f"best score={self.best_ever['score']:.4f}"
            )

        gens_without_improvement = 0

        logger.info(f"\n{'='*60}")
        logger.info(f"  Starting Reward Evolution")
        logger.info(f"  Max generations: {max_gen}")
        logger.info(f"  Candidates per generation: {n_candidates}")
        logger.info(f"  Training steps per candidate: {training_config['total_timesteps']}")
        logger.info(f"{'='*60}\n")

        for gen_idx in range(max_gen):
            gen = self.generation + gen_idx
            logger.info(f"\n--- Generation {gen} ---")

            two_stage = evo_config.get("two_stage", False)

            if two_stage:
                # === TWO-STAGE TRAINING ===
                screen_n = evo_config.get("screen_candidates", 12)
                screen_steps = evo_config.get("screen_timesteps", 2000)
                screen_top_k = evo_config.get("screen_top_k", 3)

                # Stage 1: Generate many candidates, quick screen
                logger.info(f"📊 Stage 1: Screening {screen_n} candidates ({screen_steps} steps each)")
                candidates = self._generate_candidates(screen_n, gen)
                if not candidates:
                    logger.error("No valid candidates generated, retrying...")
                    continue

                screen_config = dict(training_config)
                screen_config["total_timesteps"] = screen_steps

                screen_results = []
                for i, code in enumerate(candidates):
                    logger.info(f"  ▶ Screen {i+1}/{len(candidates)}")
                    result = self._train_and_evaluate(
                        code=code, gen=gen, candidate_idx=i,
                        env_config=env_config,
                        training_config=screen_config,
                        eval_config=eval_config,
                    )
                    if result:
                        screen_results.append(result)
                        logger.info(f"    score={result['score']:.4f}")

                if not screen_results:
                    logger.warning(f"Gen {gen}: No successful candidates in screening")
                    continue

                # Select top-K
                screen_results.sort(key=lambda r: r["score"], reverse=True)
                survivors = screen_results[:screen_top_k]
                logger.info(
                    f"🏅 Stage 1 done | Top-{screen_top_k}: "
                    + ", ".join(f"{r['score']:.4f}" for r in survivors)
                )

                # Stage 2: Full training on survivors
                logger.info(f"🔥 Stage 2: Full training on {len(survivors)} survivors ({training_config['total_timesteps']} steps)")
                gen_results = []
                for i, survivor in enumerate(survivors):
                    logger.info(f"  ▶ Full train {i+1}/{len(survivors)}")
                    result = self._train_and_evaluate(
                        code=survivor["code"], gen=gen, candidate_idx=survivor["candidate_idx"],
                        env_config=env_config,
                        training_config=training_config,
                        eval_config=eval_config,
                    )
                    if result:
                        gen_results.append(result)
                        logger.info(f"    score={result['score']:.4f}")
            else:
                # === SINGLE-STAGE (original) ===
                candidates = self._generate_candidates(n_candidates, gen)
                if not candidates:
                    logger.error("No valid candidates generated, retrying...")
                    continue

                gen_results = []
                for i, code in enumerate(candidates):
                    logger.info(f"  ▶ Candidate {i+1}/{len(candidates)}")
                    result = self._train_and_evaluate(
                        code=code, gen=gen, candidate_idx=i,
                        env_config=env_config,
                        training_config=training_config,
                        eval_config=eval_config,
                    )
                    if result:
                        gen_results.append(result)

            if not gen_results:
                logger.warning(f"Gen {gen}: No successful candidates")
                continue

            # 3. Record results + update best
            best_in_gen = max(gen_results, key=lambda r: r["score"])
            improved = self._record_results(
                gen=gen,
                results=gen_results,
                best_in_gen=best_in_gen,
            )

            if improved:
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1

            # 4. LLM analysis of the generation
            self._analyze_generation(gen, gen_results)

            # 5. Save generation report
            self._save_generation_report(gen, gen_results, best_in_gen)

            # 6. Check convergence
            if gens_without_improvement >= patience:
                logger.info(
                    f"\n🎯 Converged! No improvement for {patience} generations. "
                    f"Best score: {self.best_ever['score']:.4f} (gen {self.best_ever['gen']})"
                )
                break

            best_ever_str = f"{self.best_ever['score']:.4f}" if self.best_ever else "N/A"
            logger.info(
                f"Gen {gen} complete | Best this gen: {best_in_gen['score']:.4f} | "
                f"Best ever: {best_ever_str} | "
                f"Patience: {gens_without_improvement}/{patience}"
            )

        logger.info(f"\n{'='*60}")
        logger.info(f"  Evolution Complete!")
        if self.best_ever:
            logger.info(f"  Best score: {self.best_ever['score']:.4f}")
            logger.info(f"  Best generation: {self.best_ever['gen']}")
        logger.info(f"{'='*60}")

    def _generate_candidates(self, n: int, gen: int) -> list[str]:
        """Generate N candidate reward function codes using LLM."""
        candidates = []
        previous_rewards = [
            {"gen": r["gen"], "score": r["score"], "summary": r.get("summary", "")}
            for r in sorted(self.population, key=lambda x: x["score"], reverse=True)[:5]
        ]

        for i in range(n):
            try:
                if gen == 0 or self.best_ever is None:
                    # First generation: generate from scratch
                    code = self._get_llm().generate_reward(
                        env_description=ENV_DESCRIPTION,
                        history_best=None,
                        failure_analysis=None,
                        previous_rewards=previous_rewards,
                    )
                elif i < n // 2:
                    # Mutate the best
                    best_code = self.best_ever.get("reward_code", "")
                    best_metrics = self.best_ever.get("metrics", {})
                    failure_str = self.best_ever.get("failure_analysis", "")
                    code = self._get_llm().mutate_reward(
                        current_code=best_code,
                        current_score=self.best_ever["score"],
                        metrics=best_metrics,
                        failure_analysis=failure_str,
                    )
                else:
                    # Generate novel reward
                    code = self._get_llm().generate_reward(
                        env_description=ENV_DESCRIPTION,
                        history_best=self.best_ever,
                        failure_analysis=self.best_ever.get("failure_analysis"),
                        previous_rewards=previous_rewards,
                    )

                # Validate: can we exec it?
                safe_exec_reward(code)
                candidates.append(code)
                logger.info(f"  Candidate {i+1}: generated OK (hash={hash_code(code)})")

            except Exception as e:
                logger.warning(f"  Candidate {i+1}: generation failed: {e}")

        return candidates

    def _train_and_evaluate(
        self,
        code: str,
        gen: int,
        candidate_idx: int,
        env_config: dict,
        training_config: dict,
        eval_config: dict,
    ) -> Optional[dict]:
        """Train a policy with the given reward code, then evaluate it."""
        try:
            # Parse reward function
            reward_fn = safe_exec_reward(code)

            from src.runtime.experiments.evaluator import evaluate_policy
            from src.runtime.experiments.trainer import train_policy

            # Train
            train_result = train_policy(
                env_config=env_config,
                reward_fn=reward_fn,
                training_config=training_config,
            )

            # Evaluate
            metrics, failures = evaluate_policy(
                policy=train_result["policy"],
                env_config=env_config,
                reward_fn=reward_fn,
                n_episodes=eval_config.get("n_episodes", 50),
                weights=eval_config.get("metrics_weights"),
            )

            # Save reward code
            gen_dir = self.rewards_dir / f"gen_{gen:04d}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            reward_path = gen_dir / f"candidate_{candidate_idx}.py"
            reward_path.write_text(code, encoding="utf-8")

            return {
                "gen": gen,
                "candidate_idx": candidate_idx,
                "code": code,
                "code_hash": hash_code(code),
                "score": metrics.composite_score,
                "metrics": metrics.to_dict(),
                "training_stats": train_result["training_stats"],
                "failure_descriptions": failures,
                "failure_analysis": "\n".join(failures[:5]) if failures else "No failures",
                "policy": train_result["policy"],
                "reward_path": str(reward_path),
            }

        except Exception as e:
            logger.error(f"  Train/Eval failed: {e}\n{traceback.format_exc()}")
            return None

    def _record_results(
        self,
        gen: int,
        results: list[dict],
        best_in_gen: dict,
    ) -> bool:
        """Record results to checkpoint registry. Returns True if new best."""
        # Add to population memory
        for r in results:
            self.population.append({
                "gen": r["gen"],
                "score": r["score"],
                "code": r["code"],
                "metrics": r["metrics"],
                "summary": f"score={r['score']:.4f}, success={r['metrics']['success_rate']:.2%}",
            })

        # Save config snapshot
        config_snapshot = {
            "generation": gen,
            "timestamp": now_iso(),
            "environment": self.config["environment"],
            "training": self.config["training"],
            "evolution": self.config["evolution"],
            "reward_code_hash": best_in_gen["code_hash"],
            "metrics": best_in_gen["metrics"],
        }
        save_yaml(config_snapshot, self.configs_dir / f"gen_{gen:04d}.yaml")

        # Register with checkpoint system
        is_new_best = self.checkpoint.register(
            task=self.task_name,
            generation=gen,
            score=best_in_gen["score"],
            policy=best_in_gen["policy"],
            config_snapshot=config_snapshot,
            reward_code=best_in_gen["code"],
        )

        if is_new_best:
            self.best_ever = {
                "score": best_in_gen["score"],
                "gen": gen,
                "reward_code": best_in_gen["code"],
                "metrics": best_in_gen["metrics"],
                "failure_analysis": best_in_gen.get("failure_analysis", ""),
            }

        return is_new_best

    def _analyze_generation(self, gen: int, results: list[dict]):
        """Have LLM analyze the generation results."""
        try:
            analysis = self._get_llm().analyze_results(
                generation=gen,
                results=[{
                    "candidate": r["candidate_idx"],
                    "score": r["score"],
                    "metrics": r["metrics"],
                    "code_hash": r["code_hash"],
                } for r in results],
                best_ever={
                    "score": self.best_ever["score"],
                    "gen": self.best_ever["gen"],
                } if self.best_ever else None,
                failure_cases=[
                    f for r in results for f in r.get("failure_descriptions", [])[:3]
                ],
            )
            logger.info(f"LLM Analysis: {analysis.get('summary', 'N/A')[:200]}")
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")

    def _save_generation_report(self, gen: int, results: list[dict], best: dict):
        """Write a markdown report for this generation."""
        report = f"""# Generation {gen} Report

**Timestamp**: {now_iso()}

## Results Summary

| Candidate | Score | Success Rate | Stability | Efficiency | Smoothness |
|-----------|-------|--------------|-----------|------------|------------|
"""
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            m = r["metrics"]
            marker = " 🏆" if r is best else ""
            report += (
                f"| {r['candidate_idx']}{marker} "
                f"| {r['score']:.4f} "
                f"| {m['success_rate']:.2%} "
                f"| {m['stability']:.3f} "
                f"| {m['efficiency']:.3f} "
                f"| {m['smoothness']:.3f} |\n"
            )

        if self.best_ever:
            report += f"\n## Best Ever: score={self.best_ever['score']:.4f} (gen {self.best_ever['gen']})\n"

        report += f"\n## Best Reward Code (this gen)\n\n```python\n{best['code']}\n```\n"

        if best.get("failure_descriptions"):
            report += "\n## Failure Cases\n\n"
            for f in best["failure_descriptions"][:5]:
                report += f"- {f}\n"

        report_path = self.reports_dir / f"generation_{gen:04d}.md"
        report_path.write_text(report, encoding="utf-8")
