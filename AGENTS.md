# AUV 项目：给 Codex 的仓库指令

## 当前任务与轻量版本管理（2026-10-06 用户授权修订）

先读 `docs/PROJECT_STATUS.md`、相关冻结规格和真实代码/证据，再按 LOCAL 科学 Stage 推进。
当前工程实现已到 PHASE B5.1；工程 Phase、论文方法 B5、科学 Stage 不能互相替代。
原 START_HERE/FIRST_TASK/CODEX_EXECUTION_PLAN 和旧 Gate 是历史交接资料，不代表当前状态。
不再运行额外文件摘要封存、fresh-copy/ZIP 验收或 provenance rebase；用 Git 提交、
版本化配置和真实运行记录定位版本。随机派生和稳定场景 ID 的算法行为保持不变。
当前授权为STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1，登记见docs/STAGE2_B0_REPAIR_R1.md。
先已有数据与有界冻结重放，后按独立证据只选择L1修复复测、唯一lr1e-4 L2对照或
关键冲突停止三者之一。新学习仅空场景seed11/22/33各100k，C可比时复用V1；
C不可比且登记后总新增不得超过600k。不得自动第二候选、重跑CV、改reward/边界/采样律。
V1已完成900k是真实历史，不作为本轮新计数；不进入Stage3、Test/OOD或联邦。

每个有实际修改的已授权任务结束时，只提交本任务文件并推送至
`https://github.com/Whsjbwy/FL-RL.git` 的工作分支；失败工作明确标 WIP/未通过。
未成功 push 不得声称已上传；不得强制推送、合并 main、重写历史或擅启下一科学 Stage。
公开发布仅包含必要代码、配置、测试和小体积证据；排除秘密、个人文件、未经授权全文、
虚拟环境、缓存、大数据、完整 Replay、重复包和大量 checkpoint。暂存区必须人工复核。

## 来源与状态

- 实现“已经做了什么”以最新源码、配置、实际日志为准。
- 算法“应该做什么”以 `handoff/protocol/LOCAL_v2_0.docx` 的冻结设计及已确认决议为准。
- 冲突必须列入 Code–Math Conflict Register，不能通过修改文档使错误代码合法化。
- `handoff/context/` 是历史上下文，不是本次运行结果。
- `handoff/protocol/history/FL_0_1_HISTORY_ONLY.*` 是旧建议；其方案 N 不用于主方法。
- 不能从聊天记忆恢复缺失公式。读取原生 OMML；无法可靠读出时记录 `Formula Extraction Pending Verification`。
- 项目没有独立完整版 FL-0.2 Word；不要谎称读过不存在的文件。

## 当前实有代码

已存在动力学、CV、感知/KF、234维输入、任务接口、风险/验证器、普通及约束SAC、
成本与lambda、TRAIN_SCENARIO_V1。以源码和本次验收为准，历史 PASS 不冒充当前结果。
本轮只授权登记的B0失败诊断及条件修复复测；历史准备smoke不冒充科研结果，联邦实现仍未授权。
不得删科学测试凑历史总数。

## 禁止静默改动的语义

- 三维导航、八维状态、三维速度/角率指令；不是推进器力或六维动作。
- 234 = 18 + 45 + 45 + 6×21；上一执行动作、剩余任务比例已经包含。
- 不加 client ID、联邦轮次、真流速、真障碍速度或计算风险标量到输入。
- 上条真障碍速度限制适用于有限感知输入；B0专用当前特权真值接口是LOCAL规定例外，
  只含当前位置/地速和零协方差，不含真未来、不使用执行安全过滤。
- CV-KF：P为6×6，未来位置Sigma为3×3；Joseph、过程噪声、发生时间戳重放。
- 半空间＋0.05秒扫掠＋时间/目标并集界，保留未截断U；不能改成max单项风险。
- 方向为(m0+m1)归一化，零和固定NED轴；小正方差不能当确定性零方差。
- 45候选加名义/上一动作，最近有效候选及冻结main/backup；不能换成argmax Q。
- 动力学和平滑reward用executed；未来reward/cost critics与SAC log-prob用nominal。
- 主成本只用LOCAL式(45)(46)(50)(51)(52)，不引入方案N、不静默clip cost target。
- fallback不是安全保证，不自动terminate；真实有限时域和外部truncated分开。
- 未来只联邦Actor；Critics/targets/alpha/lambda/Replay/KF/wrapper/Adam全部本地。

## 代码与测试工作纪律

先读相关源码和规则，复现问题，写根因/最小修改计划，再修改并回归。
保持现有模块职责；中文docstring/注释、类型标注、单位/坐标明确、随机流可追溯。
不得为通过检查删除测试、增加无理由skip/xfail、放松容差、关闭Ruff规则或手改Gate为GO。
禁止整仓`--fix`、批量格式化、依赖全面升级；静态修复逐项审查，无算法改变。
原始证据放在 `handoff/evidence/` 和 `handoff/archives/`，不得覆盖。
新输出放本任务独立 results 子目录；使用 audit 的 `--output-dir`，不得覆盖历史证据。
每次记录command/cwd/exit code/stdout/stderr/JUnit/Git提交/配置/解释器/依赖。
命令未执行写NOT RUN；工具缺失写BLOCKED；不能写PASS或把null写成0。
按当前用户授权推送工作分支；不删除用户文件、不泄露密钥，不为装工具关闭安全限制。

## 命令入口

在含pyproject.toml的仓库根目录，用同一个项目解释器运行：

```text
python -m pytest tests -q
python -m ruff check src tests scripts
python -m compileall -q src tests scripts
python scripts/run_stage0_integration_smoke.py
python scripts/run_stage0_audit.py --output-dir results/<task>/audit
```

外置13项须先设置 `AUV_AUDIT_PROJECT_ROOT` 为真实仓库绝对路径：

```text
python -m pytest handoff/reference_tests/test_phase_a_conformance.py -q
```

安装、PATH、独立日志和验收命令见 `handoff/commands/LOCAL_SETUP_AND_B1_CHECKS.md`。
Ruff建议固定0.6.0（满足原>=0.6）；不是原项目精确锁文件。不得缺工具即跳过。

## 停止条件与交付

本轮禁止登记之外的科研训练、F0/F1、FedAvg/FedProx、Test调参及1.5M正式实验。
不要添加Manifold/GNN/Transformer/Attention/PER/Diffusion/RRT*/APF；IMM/6-DOF仅登记备份。
如修lint需要改变风险/成本/动作/终止语义，暂停并请求新的明确决定。
本任务结束交付：差异、全部真实测试输出、工具版本、源码标识、Gate状态和下一允许阶段。
只按已有证据汇报，代码可运行不等于科研假设成立。

## 退役工程检查

旧 `tools/verify_handoff.py`、其摘要工具测试及历史 manifests 留在本机作为历史材料，
不再是工作前门禁，不计本轮 PASS。纯摘要断言可明确退役，混合文件的科学断言必须保留。
材料缺失必须报告，不能从记忆重建协议，也不能把无法验收写成 GO。
