> 🌟 愿代码无 bug，训练一帆风顺，万事如意！

# 🤖 Auto-Evolving Grasp

LLM 驱动的机器人抓取奖励进化系统：自动生成/变异奖励函数，训练 PPO 策略，并把实验产物统一写入 `scratch/`。

## 快速开始

### 前提条件

- Python 3.10+
- Git
- 建议在 WSL2 / Linux 上运行 `robosuite` + `mujoco`

### 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 配置 LLM

默认配置文件是 `configs/default.yaml`。敏感信息不要写入仓库，请用环境变量：

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-endpoint/v1"
```

### 运行实验

```bash
python -m src.runtime.experiments.run_evolution
python -m src.runtime.experiments.run_evolution --max-generations 1 --timesteps 10000
python -m src.runtime.experiments.run_evolution --config configs/default.yaml --two-stage
```

### 录制最佳策略

```bash
python -m src.runtime.experiments.run_video_eval
python -m src.runtime.experiments.run_video_eval --episodes 5
python -m src.runtime.experiments.run_video_eval --live
```

## 目录说明

- `spec/`：正式规则、路线图、Wiki、协作记录
- `src/runtime/experiments/`：实验编排、训练、评估、checkpoint、CLI
- `src/interfaces/`：LLM 与仿真环境边界
- `src/domain/rewards/`：reward 代码抽取与安全执行
- `src/shared/`：日志、配置、YAML 等共享支持
- `scratch/`：运行产物与非正式实验区
- `scratch/legacy/pre-ckp/`：CKP 重构前的历史产物备份

## 当前默认产物路径

- `scratch/checkpoints/`
- `scratch/rewards/`
- `scratch/reports/`
- `scratch/config_snapshots/`
- `scratch/logs/`

## 验证安装

```bash
python -m src.runtime.experiments.run_evolution --help
python -m src.runtime.experiments.run_video_eval --help
python -c "from src.interfaces.simulation import create_env; print(create_env)"
python -c "from src.domain.rewards import safe_exec_reward; safe_exec_reward('def reward_fn(obs, action, info): return 0.0')"
```
