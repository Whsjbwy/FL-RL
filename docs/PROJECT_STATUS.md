# 项目状态：B5.1 迁入 Git 与 LOCAL Stage 0／Stage 1 训练前验收

2026-10-06 本轮科学验收：**Stage 0 GO；Stage 1 GO**。
Stage 2 仅可考虑准备工作，当前准备判断为 **CONDITIONAL GO**，没有训练授权。
当前Torch环境与声明版本不符，旧质量诊断未通过；具体限制见下文。

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

## 本次实际版本和运行结果

代码迁入提交 `0bcc2531de0691890c67766ef66cde968ab797d7`；修正验收临时目录的提交
`34f83f6238068f6558d5b08d0881fb904d458f5e`。两者的src/tests/configs/scripts/依赖声明
由Git diff确认无差异；第二提交只增加记录器的可写临时目录参数。
最终全仓检查对应后者，其他已通过检查对应前者；报告不要求引用包含自身的最终提交。

| 本次检查 | 真实结果/退出码 | 必要证据（results/stage01_readiness/ 下） |
| --- | --- | --- |
| 第一次全仓pytest | 355 passed、0 failed、42 errors、0 skipped；exit1。全部为pytest临时夹具目录WinError5 | pytest_all_initial.xml、pytest_all_initial.stdout.txt |
| 修正临时输出目录后完整重跑 | **397 passed、0 failed/errors/skipped；exit0** | pytest_all.xml、pytest_all.stdout.txt |
| 外置PhaseA | **13 passed、0 failed/errors/skipped；exit0** | pytest_phase_a.xml、stdout.txt |
| Stage1固定非学习验收 | **19案例PASS；231次world转移；exit0** | stage1/stage1_acceptance.json、trajectories.csv、events.json、sensor_records.csv |
| Stage0真实接口smoke | **exit0**；仅一个真实控制周期，非训练 | stage0_interface_smoke.stdout.txt |
| Ruff0.6.0 | **exit0**；未增加ignore，未批量--fix | ruff.stdout.txt、ruff_version.txt |
| compileall | **exit0**；命令级可写缓存，已清理本轮临时产物 | compileall.stdout/stderr.txt、commands.json |
| 旧质量规则诊断 | **exit1，未通过**；必需Stage0节点20项均存在且在全仓pytest中通过 | quality/code_quality_audit.csv、stage0_quality_diagnostic.stdout.txt |

JUnit按实际节点归组：original34、B1保留34、B2=75、B3=81、B4=73、B5保留54、
B5.1场景26、新Stage1测试20，总397。另有外置13项，不重复计入397。
历史389减12项纯摘要用例，再加20项独立非学习验收；不是“原样389全部通过”。
26项场景测试含本轮真实100个合法warm-up，全部通过。
历史一万场景分布审计的程序/摘要作为历史来源复用，没有新跑该完整分布程序或Actor smoke；
本轮测试自身的一万场景no-OOD断言真实执行。原历史报告没有被追改。

所有命令、cwd、开始/结束UTC、退出码与对应提交保存在commands.json/txt。
首次临时目录错误以及开发中的行宽/缓存权限诊断均保留，未将它们改写成PASS。

## LOCAL 证据对应表

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

## 科学判断与未解决问题

Stage0 GO依据：独立坐标/CV-Q/Joseph/概率/风险/延迟/成本断言、真实env与validator
集成以及新增独立运动参考均通过。Stage1 GO依据：预定案例误差符合先验容差，
最早物理事件停止无穿透；range/FOV、遮挡恢复、dropout与延迟的真实检测/估计记录可查，
动态毒化测试未发现真速度或真未来泄漏；非法终止后推进被拒绝且状态不变。
这些结论仅适用于冻结模型与验收案例，不是学习效果或真实导航安全保证。

三维直行最大位置误差约4.62e-13m，稳态圆弧约6.12e-6m；
执行器响应最大位置误差约7.94e-5m。头对头/交叉事件分别约0.04545454545s和
0.08891376643s，时间误差均小于预登记1e-10s。详情含每控制节点及离线真值→检测→估计记录。
物理碰撞不含0.30m额外裕量；near-miss严格为无碰撞且真实净间距<0.5m。
固定无障碍解析场景和短timeout夹具独立命名，未修改TRAIN_SCENARIO_V1支持或生产时域。

旧质量诊断有39项缺docstring、3条英文注释，以及两个文本启发式告警：
scenario配置合法性守卫被当Magic Number；Replay的显式seed被当隐藏RNG。
保留这些真实FAIL，未关闭规则或改生产源码来强造GO。该诊断是旧工程风格/文本扫描，
与§25.1/25.2的独立数值、几何及信息通路科学Gate分开报告；没有声称旧audit主程序GO。

本轮未运行throughput；旧 `results/safe_sac_throughput_benchmark/BENCHMARK_GATE.json`
仍是历史BLOCKED。取消摘要门禁并未把benchmark改成PASS，benchmark也不是Stage2科学进入条件。
本轮未新增联邦模块、训练网络或完整training harness；未来B0允许当前特权状态，
其信息规则不等于本轮有限感知接口规则。

## 修改、计算和存储

生产源码、科学模型、两份YAML及依赖声明：**本轮修改0**。
修改AGENTS/.gitignore/README；新增本状态及预登记规格；Stage0 audit仅退役摘要元数据并
增加独立输出目录；两份混合测试仅退役12项纯工程断言；新增Stage1脚本/20个用例。
结果目录中的验收记录器和计数插件只用于本轮记录，不是训练harness。

**科研RL训练步数0；科研梯度更新0。**完整单测确实执行合成/工程优化器操作：
最终397项记录到Adam384次、SGD1次；首次受权限阻断的运行另外有Adam147/SGD1次，
开发相关单测另行运行，未宣称全计算梯度为零，也未把这些操作当策略科研训练。
最终全仓单测world成功step2015次、LocalNavigationEnv成功step1047次，嵌套计数不可相加；
独立固定场景脚本231次非学习world转移另计，Stage0接口smoke1次另计。

新增持久Git数据、验收记录及验收代码约**1.31MiB**（证据提交前测量，后续Git对象略增）。
清理的约246MiB仅为本轮单测checkpoint夹具、编译缓存与重复开发诊断，清单见
temporary_cleanup.json。未复制源码/环境/交接包，未删除或移动原数据、必要模型、协议或历史报告。
公开仓库仅迁入必要代码、配置、测试和选定小文本；未上传原DOCX、私人上下文、虚拟环境、
历史副本、旧摘要门禁、replay或checkpoint。依赖安全校验与Git内部完整性机制未改变。

## 下一项允许考虑的任务（仅一项）

**Stage2准备：对齐声明Torch环境，准备B0 Full-State ordinary SAC入口、训练harness和任务可达性确认。**
尚需下一轮明确授权；不在本轮实现或运行。B0仅当前真值、普通SAC、共同动力学/动作/任务reward，
无风险训练、无执行动作安全过滤，先无动态障碍再简单CV交会。可达性确认和训练授权仍未完成。

## Git 发布定位

仓库：<https://github.com/Whsjbwy/FL-RL>；工作分支 `codex/b51-stage1-readiness`。
验收证据提交 `034f3434c4ac28e11c53c81442b16d183f596542` 已成功push，
远端工作分支引用核对相同，main仍为原 `be72a6b`；见PUBLICATION.json、push日志和
remote_confirmation.txt。此回执记录该时间点，随后回执本身的提交以最终远端分支为准。
未创建PR、自动合并main或改写远端历史；原目录中的忽略项仍本机留存。
