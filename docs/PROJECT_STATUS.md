# 项目状态：B5.1 迁入 Git 与 LOCAL Stage 0／Stage 1 训练前验收

本轮验收执行中；本文件将在真实检查完成后填入结果，当前不声明新的科学 GO。

## 工程流程修订（2026-10-06 用户授权）

取消额外文件摘要清单、ZIP/fresh-copy 验收和专门 provenance rebase。
版本定位改用 Git 提交、版本化配置和真实测试记录；科学模型、参数、统计规则及 Gate 不变。
原 B5_SCENARIO_CLOSED 的额外封存要求由轻量 Git 流程替代，不伪造旧封存 PASS。
原历史报告及 manifests 留存，不追改历史结论。RNG 派生、可复现场景 ID 保持原实现。

## 身份及适用范围

真实源码来源：`D:\FL+RL\AUV_CODEX_HANDOFF_V2_COMPLETE`，原目录无 Git；
目标 `Whsjbwy/FL-RL` 原 main 只有 README，本轮分支以该 main 为祖先，不上传旧项目历史。
直接读取 LOCAL v2.0 Word 原件及相关章节/阶段卡、B5.1 SPEC 和源码。
工程 B5.1 是场景生成器实现；论文 B5 是完整方法；本轮科学验收是 Stage 0／Stage 1。
历史 B5.1 的 389 passed、100 warm-ups、未训练等均仍标为历史，不计入本轮结果。

## 额外工程检查退役

旧 `tools/verify_handoff.py` 及四个 `tools/tests/test_verify_handoff*.py` 是摘要封存工具，
留在本机，不再执行或计 PASS，不迁入公开运行路径。
混合科学测试中仅退役 12 个纯摘要用例：B1 fingerprint 1项、B5 frozen-files 11项。
Stage0 audit 的摘要函数/字段退役；公式映射、必需集成测试、质量/科学检查保留。
因此原 389 不是本轮硬目标：保留 377 项，再加本轮最小科学补测。

## 待写入

证据对应表、被测试提交、命令/退出码、JUnit、非学习轨迹/事件/传感日志、
环境漂移、科学 Gate、存储和 Git 发布状态均在验收后按实测记录。

## LOCAL 证据对应表（运行结果待补，先登记覆盖依据）

| 章节/要求 | 现有实现及科学断言 | 历史适用记录 | 本轮动作/证据 |
| --- | --- | --- | --- |
| §7、§25.1 坐标/动力学/积分 | frames、auv_kinematics、world；test_frames、test_dynamics、test_world_integration | B1内核及B2环境 | 重跑；新增恒速/稳态恒转/响应的独立参考，docs/STAGE1_ACCEPTANCE_SPEC.md |
| §8、§10 CV预测/Q | cv_model、predictor；test_cv_model的积分/半群 | B1数学核 | 重跑原测试，不只比较同函数重复调用 |
| §10 KF/Joseph | kalman_filter；test_kalman_filter；PhaseA PA09/PA10独立块/条件协方差 | B1、外置PhaseA | 真实重跑 |
| §12、§25.1 概率/扫掠/风险 | risk_bounds、gaussian_coverage、B1 risk/clearance；精确秩一/非中心卡方和模型内采样 | B1/PhaseA | 重跑；与采样容差比较，不声称导航安全证明 |
| §9—10 延迟/缺测/重放 | sensor_pipeline、delay_replay、B1 delay、B2 warmup | B1/B2 | 重跑，新增1/2 tick日志和独立CV缺测参考 |
| §13 终止/成本尾项 | cost_accounting、B2 environment、B4 cost_math、PhaseA PA11/13 | B2/B4 | 重跑；不改奖励、归一化或终止语义 |
| §14—15 真实环境/验证器接口 | world_integration、validator、data_flow、B2 environment/truth_boundary | B1/B2/B5 | 重跑真实接口及信息毒化测试 |
| §25.2 A 基本运动 | 新stage1_acceptance：10s三维直行、稳态圆弧、独立标量响应ODE | 历史同核步长减半不足以替代独立参考 | 新增并生成控制节点轨迹 |
| §25.2 B 几何/事件 | 原扫掠、B1最早事件及前缀、B2终止；新交叉/头对头/边界/姿态/目标/timeout | 已有多个局部单元 | 补首次事件时间表、终止后不可推进、净间距0.25/0.5/0.75边界 |
| §25.2 C 感知/跟踪 | rays、sonar、PerceptionSession；B2 rays/warmup | B2缺测进入已覆盖 | 新增range/FOV返回、遮挡解除、dropout恢复和同步日志；只是接口验收 |
| §25.2 D 信息边界 | B2 truth_boundary的真速度/未来毒化对象；静态禁止依赖 | B2/B5 | 重跑；真值只用于观测产生和离线诊断 |
| B5.1 训练场景规格 | scenario_generator、train_scenario_v1.yaml、26场景测试 | 历史389/100warmup/1万分布诊断 | 重跑26项含100warmup；复用历史分布程序/摘要，采样律不改 |
| §25.3 Stage2进入 | Stage1通过后还需任务可达性确认；B0当前真值普通SAC，不启用风险训练/执行过滤 | 当前尚无该训练入口 | 仅列下一任务，本轮不实现/运行 |
| §27、§30、附录B/D.3 | 有限L1修复、共享内核独立参考、冻结配置、Git提交与真实记录 | 原报告保留 | 新日志不覆盖历史；所有容差先登记；无新模型/预算 |

TRAIN_SCENARIO_V1 的均匀采样、偶数index条件垂向抽样等是原SPEC明确登记的实现选择，
不冒称LOCAL指定概率律。至少半数满足垂向差，不是恰好半数；保留操作盒外障碍中心和
初始重叠诊断、两个已允许的数值/stratum重试，不加policy/risk/TTC筛选。

## 已知环境限制（只读核对）

当前 `.venv-b1` 是 Python3.13.5、PyTorch2.14.0+cpu；声明为torch==2.11.0，
历史为2.11.0+cu130。RTX5060仍可由nvidia-smi识别，但当前Torch CUDA不可用。
未升级、降级或安装依赖；本轮环境数学验收用当前NumPy/SciPy，算法单测结果只适用此环境。
正式RL入口准备必须先解决声明版本/设备环境漂移，不能复用历史CUDA PASS。
