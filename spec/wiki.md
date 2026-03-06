# Auto-Evolving Grasp — Project Wiki

## Overview

这是一个 `LLM + Robosuite + PPO` 的实验平台：主循环会生成或变异 reward 函数，训练策略，执行评估，然后把结果反馈给 LLM 继续迭代。

## CKP Structure

```text
RLLAB/
├── spec/
│   ├── project-rules.md
│   ├── ownership.md
│   ├── wiki.md
│   ├── roadmap.md
│   └── details/
├── configs/
│   └── default.yaml
├── prompts/
├── src/
│   ├── runtime/experiments/
│   ├── interfaces/llm/
│   ├── interfaces/simulation/
│   ├── domain/rewards/
│   └── shared/
└── scratch/
    ├── checkpoints/
    ├── reports/
    ├── rewards/
    ├── config_snapshots/
    ├── logs/
    └── legacy/pre-ckp/
```

## Runtime Flow

```text
LLM prompt templates
  → src/interfaces/llm/client.py
  → reward Python code
  → src/domain/rewards/execution.py
  → src/interfaces/simulation/grasp_env.py
  → src/runtime/experiments/trainer.py
  → src/runtime/experiments/evaluator.py
  → src/runtime/experiments/checkpoints.py
  → reports / reward snapshots / next generation analysis
```

## Primary Commands

```bash
python -m src.runtime.experiments.run_evolution
python -m src.runtime.experiments.run_evolution --max-generations 1 --timesteps 10000
python -m src.runtime.experiments.run_evolution --config configs/default.yaml --two-stage

python -m src.runtime.experiments.run_video_eval
python -m src.runtime.experiments.run_video_eval --episodes 5
python -m src.runtime.experiments.run_video_eval --live
```

## Configuration Notes

- Default config: `configs/default.yaml`
- Secret inputs: `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- Active output roots:
  - `scratch/checkpoints/`
  - `scratch/rewards/`
  - `scratch/reports/`
  - `scratch/config_snapshots/`
  - `scratch/logs/`

## Historical Notes

- Pre-CKP outputs are preserved under `scratch/legacy/pre-ckp/`.
- Pre-CKP planning and validation notes were migrated from `_agent/` into `spec/`.
