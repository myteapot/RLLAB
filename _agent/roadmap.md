---

kanban-plugin: board

---

## Backlog

- [ ] GitLab 集成 + 自动 commit [[details/gitlab-integration|📋]]
- [ ] Meta-Reward 验证系统 [[details/meta-reward|📋]]
- [ ] SO-101 机械臂适配 [[details/so101-adapter|📋]]
- [ ] 模型蒸馏 Pipeline [[details/distillation|📋]]
- [ ] Robosuite macros 初始化（消除 WARNING）
- [ ] numpy 版本锁定（1.26.x，避免 2.x 兼容问题）
- [ ] imageio[ffmpeg] 加入 requirements.txt
- [ ] Skill 跨项目复用方案（symlink / submodule）
- [ ] 训练进度条（tqdm / 自定义）


## In Progress

- [ ] 可视化脚本：先尝试wslg [[details/eval-video|📋]]


## Review

- [ ] MVP 二次验证：跑 `python run.py --max-generations 1 --timesteps 10000`，确认终端干净、无报错
- [ ] 日志分离确认：检查 `logs/` 目录是否生成详细日志文件


## Done

- [x] safe_exec_reward 沙箱修复
- [x] PPO 改用 CPU（device: cpu）
- [x] Obsidian Kanban 看板搭建
- [x] LLM Reward 进化循环实现
- [x] 终端日志可读性优化（robosuite 静音 + 文件日志）
- [x] 项目骨架搭建（src/ 全部模块）
- [x] 两阶段训练架构实现
- [x] 环境配置（WSL + venv + 依赖）




%% kanban:settings
```
{"kanban-plugin":"board","lane-width":280,"show-checkboxes":true}
```
%%