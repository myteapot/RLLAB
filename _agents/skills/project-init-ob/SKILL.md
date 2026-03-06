---
name: project-init-ob
description: 项目初始化 — 在 Obsidian vault 内创建项目管理文件（Kanban 看板、详情卡片、Agent 通信日志）
---

# 项目初始化（Obsidian 本地管理）

## 概述

在用户的 Obsidian vault 内创建项目管理文件，纯 Markdown，人类和 Agent 共同维护。
文件位置由 `config.toml` 配置，Agent 读取配置后写入 vault 对应目录。

## 配置

读取同目录下的 `config.toml`，关键路径解析：

```
管理文件目录 = {vault.path} / {vault.projects_dir} / {project.name}
示例:        D:/Cortex/Projects/RLLAB/
```

首次使用时，Agent 应确认 `config.toml` 中的 `vault.path` 是否正确。

## 何时触发

- 新项目需要任务追踪时
- 人类要求"创建 roadmap"或"初始化项目管理"时
- 检测到项目无管理文件且需要协作时

## 初始化步骤

### 1. 创建目录结构

```
{vault.path}/{vault.projects_dir}/{project.name}/
├── roadmap.md          ← Kanban 看板（Obsidian Kanban 插件兼容）
├── dialog.md           ← Agent 间跨 session 通信
└── details/
    └── {card-slug}.md  ← 每张卡片的详情文件
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

## 子动作：`update`（Agent 可主动触发）

Agent 完成一组工作后，主动更新看板状态：

### 步骤

1. **移动完成项** — 将已完成的 In Progress 卡片移到 Done（或合并到已有 Done 条目）
2. **添加 Review 项** — 将人类需要做的最小验证动作放到 Review 列，描述要具体（如 "跑 `python run.py` 确认无报错"）
3. **更新 detail 文件** — 修改对应 `details/*.md` 的进展和状态
4. **追加 dialog.md** — 记录本次 session 的工作摘要
5. **提醒 git** — 提醒用户 commit（Agent 不自行 commit，因为用户可能还要改）：

```
📌 Roadmap 已更新，建议 commit：
  git add _agent/ && git commit -m "update: {简述}"
```

### 触发时机

- Session 结束前
- 一组相关任务全部完成后
- 用户要求 "更新 roadmap" 时

---

## 子动作：`clean`（仅人类触发）

> ⚠️ **此动作只能由人类明确要求时执行**（如 "clean roadmap"、"整理看板"）。Agent 不得自行触发。

### 步骤

1. **安全快照** — 先执行 `git add _agent/ && git commit -m "pre-clean snapshot"`，确保清理前有回滚点
2. **合并同类项** — 将相似的 Backlog 卡片合并到一条，保留所有事实细节到对应 detail 文件
3. **归档已完成** — Done 列中过多的细粒度条目合并为按里程碑分组的总结条目
4. **清理空 Review** — 如果 Review 列为空且无待审项，保留空列即可
5. **清理孤立 detail** — 检查 `details/` 下是否有不再被 `roadmap.md` 引用的文件，列出供用户决定是否删除
6. **通知用户** — 告知清理完成，提供回滚方式：

```
✅ Roadmap 已清理。清理前快照已 commit。
  查看变更：git diff HEAD~1
  回滚：    git checkout HEAD~1 -- _agent/
```

### 规则

- **不丢失信息** — 合并条目时，细节写入 detail 文件或 wiki.md，不要静默丢弃
- **不删除文件** — 只列出建议删除的孤立文件，由人类决定
- **幂等** — 连续执行两次 clean 不应产生额外变更
