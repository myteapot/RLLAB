# Auto-Evolving Grasp — 项目 Wiki

## 概述

用 LLM 自动进化机器人抓取策略。核心循环：**LLM 生成 reward 函数 → PPO 训练 → 评估 → LLM 分析反馈 → 迭代**。

## 架构

```
run.py                       CLI 入口
src/
├── reward_evolver.py        主循环编排器
├── llm_client.py            LLM API（OpenAI 兼容）
├── environment.py           Robosuite Lift 任务 Gymnasium 包装
├── trainer.py               SB3 PPO 训练
├── evaluator.py             策略评估 + 失败案例收集
├── checkpoint.py            Top-K checkpoint 管理
└── utils.py                 日志、YAML、代码沙箱
prompts/
├── reward_init.txt          初始 reward 生成 prompt
├── reward_mutate.txt        reward 变异 prompt
└── failure_analysis.txt     失败分析 prompt
config.yaml                  所有配置集中管理
eval_video.py                可视化：MP4 录制 / WSLg 实时渲染
```

## 数据流

```
LLM API → reward 代码 (Python str)
    ↓ safe_exec_reward() 沙箱执行
reward_fn(obs, action, info) → float
    ↓ 注入 GraspEnvWrapper.step()
PPO 训练 (SB3) → 策略模型
    ↓ evaluate_policy()
metrics (success_rate, stability, efficiency, smoothness) → composite_score
    ↓ CheckpointRegistry
最优策略 + reward 代码 + config 快照
    ↓ LLM 分析
下一代 reward 变异/生成
```

## 两阶段训练

启用方式：`python run.py --two-stage`

| 阶段 | 候选数 | 训练步数 | 目的 |
|------|--------|---------|------|
| Stage 1（快筛） | 12 | 2,000 | 淘汰明显差的 reward |
| Stage 2（精训） | top-3 | 500,000 | 真正训练出策略 |

配置项在 `config.yaml` → `evolution` 下：`screen_timesteps`, `screen_candidates`, `screen_top_k`

## safe_exec_reward 沙箱

LLM 生成的代码在受限命名空间中执行：

- ✅ 允许：`numpy`, `math`, 基础内建函数
- ❌ 禁止：`import`（自动 strip）、`os`、`sys`、`__builtins__` 中的危险函数
- 函数查找顺序：`reward_fn` → `compute_reward` → 任意 callable

## 环境说明

**机器人**: Panda 7-DOF + 平行夹爪
**任务**: Lift（从桌面抬起 2.5cm 方块）
**观测**: 32 维 flat array（关节位置/速度 + 夹爪 + 物体位姿）
**动作**: 7 维连续 [-1,1]（6 关节速度 + 1 夹爪开合）

reward_fn 签名：
```python
def reward_fn(obs: np.ndarray, action: np.ndarray, info: dict) -> float:
    # info keys:
    # gripper_pos, object_pos, object_height, distance,
    # is_grasped, gripper_open, step, horizon
```

## ⚠️ 注意事项

### 依赖
- **opencv**: 必须用 `opencv-python-headless`，不要装 `opencv-python`（WSL 无 GUI）
- **h5py**: robosuite 隐式依赖，必须手动装
- **numpy**: opencv-headless 会拉 numpy 2.x，但 mink 要 <2.0。我们不用 mink，暂可忽略；如出问题 `pip install numpy==1.26.4`
- **imageio[ffmpeg]**: eval_video.py 录制 MP4 需要，尚未加入 requirements.txt

### 训练
- **device: cpu** — PPO + MLP 策略用 CPU 更快（矩阵太小，GPU 数据传输开销大）
- **10K steps** 不够学会抓取（success=0% 是预期），至少需要 500K+
- **SubprocVecEnv** 可能在 WSL + /mnt/ 下 segfault，代码会自动 fallback 到 DummyVecEnv

### 运行环境
- **代码位置**: Windows `a:\Agent\RLLAB\`（Agent 在这里修改）
- **venv 位置**: WSL `~/dev/rllab/.venv/`（用户在这里运行）
- **运行方式**: WSL 中 `source ~/dev/rllab/.venv/bin/activate && cd /mnt/a/Agent/RLLAB && python run.py`
- **日志**: 终端只显示关键信息；详细日志在 `logs/evolution_*.log`（用 `--verbose` 还原全部输出）

### Checkpoint 恢复
- 程序自动检测 `checkpoints/registry.yaml`，从上次最佳位置恢复
- 如需从头开始：删除 `checkpoints/` 目录
- `get_best()` 返回的 key 是 `generation`（不是 `best_generation`）

### Robosuite Warnings
- "No private macro file" — 无害，可 `python .venv/.../robosuite/scripts/setup_macros.py` 消除
- "Could not import robosuite_models" — 没装额外模型包，Panda 自带的，不影响
- "Could not load mink-based IK" — 不用 GR1 机器人，无关

## CLI 参考

```bash
python run.py                              # 默认配置
python run.py --max-generations 5          # 限制代数
python run.py --timesteps 10000            # 快速测试
python run.py --two-stage                  # 两阶段训练
python run.py --verbose                    # 显示所有警告
python eval_video.py                       # 录制最优策略 MP4
python eval_video.py --live                # WSLg 实时渲染
```
