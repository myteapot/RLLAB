---
name: project-init-ob
description: 项目初始化 — 在项目根目录下创建 Obsidian 兼容的项目管理文件（Kanban 看板、详情卡片、Agent 通信日志）
---

# 项目初始化（Obsidian 本地管理）

## 概述

在项目根目录下创建一套本地项目管理文件，与 Obsidian 无缝集成。
无需云端服务、无需 Notion、无需数据库——纯 Markdown，人类和 Agent 共同维护。

## 何时触发

- 新项目需要任务追踪时
- 人类要求"创建 roadmap"或"初始化项目管理"时
- 检测到项目无 `_agent/` 目录且需要协作时

## 初始化步骤

### 1. 创建目录结构

```
{project_root}/
├── _agent/
│   ├── roadmap.md          ← Kanban 看板（Obsidian Kanban 插件兼容）
│   ├── dialog.md           ← Agent 间跨 session 通信
│   └── details/
│       └── {card-slug}.md  ← 每张卡片的详情文件
```

### 2. 创建 `roadmap.md`（Kanban 看板）

使用 [Obsidian Kanban 插件](https://github.com/mgmeyers/obsidian-kanban) 兼容格式：

```markdown
---
kanban-plugin: basic
---

## Backlog

- [ ] 任务名称 [[details/task-slug|📋]]

## In Progress

- [ ] 正在进行的任务 [[details/task-slug|📋]]

## Review

- [ ] 人类需要做的验证动作（如：跑 `python run.py` 确认终端干净）

## Done

- [x] 已完成的任务 [[details/task-slug|📋]]

%% kanban:settings
```
{"kanban-plugin":"basic","lane-width":280,"show-checkboxes":true}
```
%%
```

> [!CAUTION]
> settings 代码块**不要加语言标识符**（如 ` ```json `），否则 Kanban 插件会将 "json" 当做 JSON 内容解析导致报错。只用裸 ` ``` `。

**规则**:

- 列名用 `##` 标题，**固定四列**: `Backlog`, `In Progress`, `Review`, `Done`
- 卡片用 `- [ ]`（待办）或 `- [x]`（完成）
- 卡片后用 `[[details/slug|📋]]` 链接到详情文件

**Review 列语义**:

- Review 是**人类的 action queue**，不是任务本身
- 多个 In Progress 完成后，可能只产生一条简单的 Review（如 "跑通验证"）
- Agent 完成工作后，将人类需要做的最小验证动作放到 Review
- 人类确认通过 → 移到 Done；有问题 → Agent 重新处理
- 底部的 `kanban:settings` 块**不要删除**，这是插件识别标记
- 人类和 Agent 均可直接编辑此文件

### 3. 创建详情文件 `details/{slug}.md`

每张需要详细说明的卡片，在 `details/` 下创建对应文件：

```markdown
# {任务标题}

## 状态
- **优先级**: 🟠 高 / 🟡 中 / 🟢 低
- **负责人**: Human / Agent / 协作
- **预估**: 2h / 1d / 1w
- **创建**: YYYY-MM-DD

## 描述
简述任务目标和背景。

## 规格 / 技术细节
- 具体的技术要求
- 接口定义、参数约束等

## 验收标准
- [ ] 标准 1
- [ ] 标准 2

## 备注
人类或 Agent 均可在此追加笔记。
```

**命名规范**: 使用 kebab-case，例如 `two-stage-training.md`、`log-readability.md`

### 4. 创建 `dialog.md`（Agent 间通信）

```markdown
# Agent Dialog

跨 session 的 Agent 间通信记录。每条消息标注来源 session 和时间。

---

## [YYYY-MM-DD HH:MM] Session {short-id}
**任务**: {当前任务}
**状态**: {进展摘要}
**遗留**: {需要后续 session 处理的事项}

---
```

**规则**:

- 新消息**追加到文件末尾**，不修改历史记录
- 每条消息用 `---` 分隔
- `short-id` 取 conversation-id 前 8 位
- 保持简洁：每条消息不超过 10 行

## 维护规则

### 谁改什么

| 文件 | 人类 | Agent |
|------|:---:|:---:|
| `roadmap.md` | ✅ 拖拽卡片、添加任务 | ✅ 更新状态、添加任务 |
| `details/*.md` | ✅ 写规格、改需求 | ✅ 补充技术细节、更新状态 |
| `dialog.md` | ⚠️ 可读，一般不写 | ✅ 每次 session 结束时追加 |

### Agent 编辑 roadmap.md 的注意事项

1. 用 `replace_file_content` 精准编辑，**不要全文覆盖**
2. 移动卡片 = 从一个 `##` 区域剪切到另一个 `##` 区域
3. **不要删除** 底部的 `kanban:settings` 块
4. 新增卡片追加到对应列的末尾

### 何时更新 dialog.md

- Session 结束前（被用户终止或任务完成时）
- 发现需要跨 session 传递的信息时
- 遇到阻塞问题、需要下次 session 处理时
