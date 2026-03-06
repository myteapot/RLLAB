---

kanban-plugin: board

---

## Backlog

- [ ] GitLab 搭建 + Linear 绑定（替代自建 CI skill）
- [ ] [[details/meta-reward|Meta-Reward 验证系统（防 reward hacking）]]
- [ ] [[details/so101-adapter|SO-101 机械臂适配]]
- [ ] [[details/distillation|模型蒸馏 Pipeline]]
- [ ] 依赖清理（numpy 锁版本、imageio、robosuite macros）

## In Progress

- [ ] [[details/eval-video|可视化脚本：WSLg 测试]]

## Review


## Done

- [x] [[details/mvp-validation|MVP 管线验证（score=0.1498）]]
- [x] 核心架构（6 模块 + prompt + 沙箱 + checkpoint）
- [x] [[details/two-stage-training|两阶段训练]] + [[details/log-readability|日志分离]] + device:cpu
- [x] 环境搭建（WSL + venv + 依赖修复）
- [x] Obsidian Kanban 看板 + Wiki

%% kanban:settings
```
{"kanban-plugin":"board","lane-width":280,"show-checkboxes":true}
```
%%