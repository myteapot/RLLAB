# Project Rules — RLLAB

## Purpose
- Build and evolve robotic grasping reward functions with an LLM-guided PPO experiment loop.

## Source of Truth
- `spec/` is the only formal rule and coordination source.
- `src/` documents local implementation structure through `walk.md`, but never overrides `spec/`.

## Review Boundary
- Humans review `spec/` and `src/`.
- `scratch/` is outside the main review surface unless content is explicitly promoted.

## Runtime Entrypoints
- Evolution CLI: `python -m src.runtime.experiments.run_evolution --config configs/default.yaml`
- Video eval CLI: `python -m src.runtime.experiments.run_video_eval --config configs/default.yaml`

## Configuration Contract
- Default config lives at `configs/default.yaml`.
- Secrets must come from environment variables, especially `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
- Paths in config are repository-root relative and resolved by `src/shared/support.py`.

## Output Contract
- Active outputs go under `scratch/`.
- Historical pre-CKP artifacts stay under `scratch/legacy/pre-ckp/`.
- New checkpoints, reports, rewards, config snapshots, logs, and videos must not be written to the repository root.

## Module Boundaries
- `src/runtime/experiments`: orchestration, training, evaluation, checkpointing, CLI.
- `src/interfaces`: external boundaries such as LLM APIs and Robosuite/Gym wrappers.
- `src/domain/rewards`: reward-code extraction and execution rules.
- `src/shared`: generic support code without experiment-specific semantics.

## Agent Rules
- Read `spec/ownership.md` before editing implementation.
- Prefer the smallest owned directory possible.
- Update the nearest `walk.md` when a directory’s role changes materially.
- Keep `scratch/` for drafts, probes, and non-reviewed outputs.
