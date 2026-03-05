# 终端日志可读性优化

## 状态
- **优先级**: 🟡 中
- **负责人**: Agent
- **预估**: 1h
- **创建**: 2026-03-06

## 描述
当前终端输出被 robosuite warnings 和重复日志淹没，人类无法快速获取关键信息。

## 规格

### 终端输出（给人类看）
```
🤖 Gen 0 | 4 candidates
▶ [1/4] training ████████░░ 80% | 45s
✅ [1/4] score=0.32 | success=15% | 1.5min
▶ [2/4] training...
```

### 文件日志（给 Agent 看）
- 路径: `logs/evolution_{timestamp}.log`
- 包含完整 debug 信息、robosuite warnings、traceback
- Agent 可直接读取分析

### 实现要点
- 设置 `robosuite` logger level 为 ERROR（抑制 WARNING）
- 训练进度用 tqdm 或自定义进度条
- 关键指标用 emoji + 颜色区分

## 验收标准
- [ ] 终端输出不超过每个 candidate 3 行
- [ ] 详细日志写入 `logs/` 目录
- [ ] robosuite warnings 不出现在终端
