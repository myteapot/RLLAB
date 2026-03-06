# 两阶段训练架构

## 状态
- **优先级**: 🟠 高
- **负责人**: Agent
- **预估**: 2h
- **创建**: 2026-03-06

## 描述
将当前的"每个 candidate 统一训练 N steps"改为两阶段筛选策略，大幅提升 LLM token 利用效率。

## 规格

### 第一阶段：快速筛选
- 每代生成 20 个 candidate（当前 4 个）
- 每个 candidate 只训练 **1K-2K steps**（约 10-15 秒）
- 按 composite_score 排序，选 top-3

### 第二阶段：精细训练
- Top-3 candidate 训练 **100K-500K steps**
- 完整评估 + checkpoint 记录
- LLM 分析失败案例

### 预期效果
- LLM 调用密度提升 5x（更多 candidate 被生成和评估）
- 算力集中在有潜力的 reward 函数上
- 单机 8 核即可达到 ~2M tokens/小时

## 验收标准
- [ ] `config.yaml` 支持 `stage1_timesteps` 和 `stage2_timesteps`
- [ ] `reward_evolver.py` 实现两阶段筛选逻辑
- [ ] 日志清晰显示筛选过程
