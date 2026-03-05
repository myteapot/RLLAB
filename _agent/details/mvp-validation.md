# MVP 管线验证

## 状态
- **优先级**: 🔴 紧急
- **负责人**: 协作
- **预估**: 30min
- **创建**: 2026-03-06

## 描述
首次完整跑通 LLM reward evolution 管线，验证所有模块端到端工作。

## 当前进展
- [x] LLM 生成 4 个 candidate reward 函数
- [x] Candidate 1 训练完成 (10K steps, 1.5min, success=0%)
- [x] 4 个 candidate 全部训练完成
- [x] 评估结果正确写入 checkpoint (best score=0.1496)
- [x] 生成报告 `reports/generation_0000.md`
- [x] LLM 分析结果
- [ ] 无报错完整运行（f-string bug 已修，待二次验证）

## 已修复的问题
1. `safe_exec_reward` — `__import__ not found`（import 行被 strip，builtins 扩展）
2. `libGL.so.1` — 用 `opencv-python-headless` 替代
3. `h5py` 缺失 — 加入 `requirements.txt`

## 备注
- 10K steps 训练结果 success=0% 是预期的（太少了），不代表代码有 bug
- PPO on GPU 警告可忽略，或改 config.yaml `device: "cpu"`
