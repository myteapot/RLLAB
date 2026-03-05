# 🤖 Auto-Evolving Grasp

LLM 驱动的持续进化机器人抓取系统 —— 利用无限 LLM 调用自动迭代奖励函数，训练最优抓取策略。

## 环境配置

### 前提条件

- Python 3.10+
- Git
- (推荐) NVIDIA GPU + CUDA（训练更快，但 CPU 也能跑）

### 1. 创建虚拟环境

```bash
# 使用 conda
conda create -n evolve-grasp python=3.10 -y
conda activate evolve-grasp

# 或使用 venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> **Windows 注意**：Robosuite 官方主要支持 macOS/Linux。如果在 Windows 上遇到问题：
> - 方案 A：使用 WSL2（推荐）
> - 方案 B：使用云环境训练
> - 方案 C：尝试 `pip install robosuite`，部分 Windows 用户报告可以正常工作

### 3. 配置 LLM API

编辑 `config.yaml`，修改 `llm` 部分：

```yaml
llm:
  base_url: "https://your-free-api-endpoint/v1"
  api_key: "your-api-key"
  model: "your-model-name"
```

或通过环境变量：

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-endpoint/v1"
```

### 4. 验证安装

```bash
# 测试 MuJoCo + Robosuite
python -c "import robosuite; print('Robosuite version:', robosuite.__version__)"

# 测试环境可用
python -c "from src.environment import create_env; env = create_env(); print('OK'); env.close()"
```

## 使用方法

### 启动进化

```bash
# 默认配置运行
python run.py

# 快速测试（1 代，少量训练步数）
python run.py --max-generations 1 --timesteps 10000

# 自定义配置
python run.py --config my_config.yaml
```

### 查看结果

- `checkpoints/registry.yaml` — 最优权重索引
- `rewards/gen_XXX/` — 每代的 reward 代码和结果
- `reports/` — LLM 分析报告
- `configs/snapshots/` — 每代配置快照

## 项目结构

```
auto-evolving-grasp/
├── config.yaml          # 主配置
├── run.py               # 入口
├── src/
│   ├── llm_client.py    # LLM API 客户端
│   ├── environment.py   # Robosuite 环境封装
│   ├── trainer.py       # SB3 PPO 训练
│   ├── evaluator.py     # 多维度评估
│   ├── reward_evolver.py # 奖励进化主循环
│   ├── checkpoint.py    # Checkpoint 管理
│   └── utils.py         # 工具函数
├── prompts/             # LLM Prompt 模板
├── checkpoints/         # 权重存档
├── rewards/             # 进化的 reward 代码
├── configs/snapshots/   # 配置快照
└── reports/             # LLM 分析报告
```
