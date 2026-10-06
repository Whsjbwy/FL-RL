# 当前结果：STAGE2_B0_MVP_BATCH_V1（2026-10-07本机）

**BATCH_COMPLETED；Stage2科学Gate暂缓裁决，GO未建立；Stage3未进入。**
实际实验提交45f3cc87bb79f69ef5257d6bbd5c92bfbea8b898，工作分支codex/stage2-b0-mvp-v1。
三seed11/22/33各100k无障碍+200kCV，均300000真实transition/290001完整update；
总900000/870003，3480012 optimizer steps。原普通SAC/动力学/reward/采样律不变。
42固定验证点和2880完整episode齐全；无障碍Val300成功35/5/14，CV终点2/1/0。
18次learned固定诊断0成功（10超时、8姿态边界失败）；不能当作已学会简单任务。
本轮最终561+13真实通过、Ruff/compileall exit0；旧461+13仍为历史验收。
复用Python3.13.5/Torch2.11.0+cu130/RTX5060；未重装或改变默认参数。
科研有限性检查无异常，源/配置/测试/登记相对45f normal Git diff无变化；结果/工具提交独立。
完整回执、计数、曲线、失败证据与存储见results/stage2_b0_mvp_v1/MVP_RESULT_REPORT.md和MVP_RESULT.json。
本机模型/Replay/原始日志保留；公开轻量CSV/gzip/图及必要测试，发布回执见PUBLICATION.json。
没有额外摘要/ZIP封存、throughput正式benchmark、预算追加、Stage3或联邦工作。
GO证据不足；CONDITIONAL GO原因未确认；NO-GO有限修复前提未完成，不宣告研究假设失败。
**唯一下一任务建议：Stage2 B0姿态边界及近目标超时的有界失败诊断。**
不自动调参、重训或进入下一科学阶段。以下旧状态与“科研0/尚未授权”均为当时历史记录。

---

# 历史准备和迁入记录（保留原结论）
# 项目状态：STAGE2_B0_PREPARATION

2026-10-06 最新工程结论：**PREPARATION_READY**。LOCAL Stage2 科研仍为 **NOT RUN**。
没有启动300k/500k/1.5M、多seed科研训练、正式throughput、OOD、Stage3或联邦工作。
本轮从上轮已验收提交 `a836922b2e1f81689d09e811c29be3badf2462aa` 继续，未回退main重建。
实际目录仍为 `D:\FL+RL\AUV_CODEX_HANDOFF_V2_COMPLETE`。
工作分支 `codex/stage2-b0-preparation`；最终被测代码提交
`6840dfbbf151ff38009487a1e8f13ac74d40c4c1`。后续证据/回执提交不改变被测代码。
实际远端发布结果记录在本轮 `PUBLICATION.json`，未成功push前不视为上传。
验收证据提交 `b645dca7799cca536ebbec8538eb45d61b70d6cb` 已实际push并核对远端相同；
main仍为be72a6b、旧验收分支仍为a836922，未创建PR/合并main/强推。
随后仅文档回执提交以工作分支最终引用为准，不改变已测源码和测试。

## 本轮工程准备及科学边界

| 工程字段 | 当前结果 | 依据 |
| --- | --- | --- |
| ENVIRONMENT_READY | READY | 同一项目venv实际Torch2.11.0+cu130、RTX5060；GPU张量、前反向/Adam/目标更新和完整恢复测试通过；pip check exit0 |
| B0_INTERFACE_READY | READY | 当前真值234D、零协方差、无未来、无风险训练/执行过滤；17项接口测试和8项方法身份防误用测试通过 |
| HARNESS_READY | READY | 固定两环境轮转/连续场景索引、10000起步、25000触发、独立验证、完整checkpoint/恢复、身份守卫；31项harness测试通过 |
| REACHABILITY_CONFIRMED | READY，限登记案例 | 6个固定非学习案例各1次尝试找到合法到达见证；1488次正式环境转移，不证明整个随机训练分布可达 |
| PREPARATION_READY | READY | 最终461项全仓、外置13项、Stage0/1必要回归和静态检查通过；科研运行仍待下一次授权及预登记 |

冻结规格：`docs/STAGE2_B0_PREPARATION.md`。已直接读取本机LOCAL原件相关章节，Word及
私人上下文仍留本机，不上传。工程B5.1、论文方法B5、科学Stage0/1/2继续分别记录。
原Stage0/1数值、几何、信息边界断言和TRAIN_SCENARIO_V1采样律完整保留。

## 本轮实现与环境修复

新增 `env/b0_navigation.py`，独立于有限感知/过滤环境。目标槽读取当前World位置和NED
地速，以Body表达，协方差零、年龄0，当前距离/稳定ID排序；位置25m和速度1m/s尺度
按本轮工程选择登记。自身/任务/射线复用原编码，不读取真未来/KF预测目标。
动作直接经过原映射与World，nominal=executed指令；执行器滞后/变化率、碰撞/边界、
reward和终止仍有效。成本字段只记物理失败诊断，普通SAC损失不读取成本；风险/验证指标
使用None/禁用语义。运行时异常spy确认没有风险筛选、候选或fallback，原有限感知测试仍通过。

新增 `training/` 组合式harness、`run_b0_training.py` 和两份版本化配置；没有重写SAC、
动力学、KF或风险核。原 `rl/`、`LocalNavigationEnv`、`ObservationBuilder`、场景生成器、
stage0/train-v1配置及pyproject与上轮基点由Git diff确认无修改。科研入口默认preflight，
未授权scientific_training执行仍被拒绝；工程产物与科研配置不能静默互换。
两个任务profile区分无障碍与原1—4障碍CV，训练/验证/工程随机流独立。
验证直接比较Agent/Replay/训练环境/观察/场景索引和随机流；不靠“没backward”代替隔离验证。
完整恢复覆盖模型/targets/优化器/alpha、Replay/随机流、全部环境与感知历史、待reset、
累积日志/上一指令、场景发行/轮转/评估和保存触发位置、配置/run_kind/Git/Torch身份。
仅接受本项目明确可信本机checkpoint，原B3/B4/B5接口未修改。

项目解释器 `.venv-b1/Scripts/python.exe`，Python3.13.5；只在此venv将误装Torch2.14.0+cpu
修复为声明的2.11.0 cu130。官方源直接pip下载停滞及首个下载失败均保留，随后从官方源
取得兼容wheel、校验官方包完整性并本地安装成功；没有关闭依赖安全校验、改全局Python、
驱动、CUDA Toolkit或顺带安装vision/audio。RTX5060能力12.0、驱动596.21、CUDA构建13.0。
网络float32，环境/几何/KF原精度不变；没引入AMP/compile或性能优化。

## 本轮实际运行及失败处理

所有相对证据路径均位于 `results/stage2_b0_preparation/`；原始命令/cwd/UTC/解释器、
Git版本、退出码及stdout/stderr保存在 `commands.jsonl`，没有用历史PASS冒充当前结果。

| 检查 | 当前真实结果 | 证据 |
| --- | --- | --- |
| 最终全仓pytest | **461 passed，0 failed/errors/skipped；exit0** | pytest_all_acceptance.xml、stdout、counts.json；被测6840dfb |
| 外置PhaseA | **13 passed，0 failed/errors/skipped；exit0** | pytest_phase_a_acceptance.xml；被测6840dfb |
| 新增64项 | B0接口17、harness31、reachability纯单测8、入口5、设备3；均包含在461中 | 同一JUnit，未与全仓重复相加 |
| Stage1固定非学习回归 | **19案例PASS，231次World转移；exit0** | stage1_regression/轨迹、传感记录、事件及JSON |
| Stage0真实env/validator接口 | **1次控制周期；exit0** | stage0_interface.stdout.txt |
| Ruff0.6.0 | **exit0**；无新增ignore、无批量fix | ruff_acceptance.stdout.txt |
| compileall | **src/tests/scripts及本轮真实Python记录程序exit0** | compileall_acceptance_scoped.stdout/stderr.txt |
| 旧风格诊断 | **exit1，保留FAIL** | quality_final/code_quality_audit.csv、current_quality_diagnostic_final输出 |
| CUDA恢复对照 | **PASS，完整状态直接比较精确一致** | engineering_smoke_2/smoke_resume.json、实际更新/episode/验证日志 |
| 非学习可达性 | **6/6合法见证；每例1次；exit0** | reachability/reachability.json、trajectories.csv、obstacles.csv |

必要修复及真实失败均留存：首次CUDA恢复把Adam的CPU步数标量映射到CUDA，
`gpu_resume_failure.xml` 和首个 `engineering_smoke/failure.json` 记录失败。
最小修复仅为可信checkpoint先CPU反序列化，模型与矩仍恢复CUDA；未改变数学或容差。
修复后的独立失败案例回归、连续264与256保存/260再保存恢复至264均通过；
两环境的场景、RNG、Replay、计数器及浮点状态实际精确一致，不声称跨设备/Torch版本一致。
代码审核还发现可注入成本/约束Agent冒名B0；`agent_identity_failure.json` 在旧Git代码上
实际复现，0环境/0梯度；现已拒绝非普通Agent及过滤环境，8项无推进拒绝测试通过。
先前453项通过记录保留；新增8项检查后完整重跑为461，不把它们当成第一次已测。
一次compileall错误地递归编译可再生成pytest临时夹具，触发Windows路径错误；失败输出保留。
随后明确检查全部生产/测试/脚本及真实记录程序，exit0，未跳过任何科学源码。

旧诊断仍有39项缺docstring、3条英文注释及原文本Magic Number/隐藏RNG告警。
本轮新写/改动模块没有这些新增告警；Replay使用显式seed，场景随机派生/隔离测试通过。
未为旧风格告警全面重构或关闭规则，不将风格FAIL冒称质量PASS，也不将其冒称科学假设失败。
旧throughput blocker仍保持历史BLOCKED，没有运行正式benchmark或据此自动批准科研预算。

可达性只证明登记的水平/上升/下降无障碍、偏置头对头/横向交叉/垂向差CV夹具有可执行解。
所有控制通过真实B0物理世界、一秒warm-up、原动作范围/时域/成功半径及碰撞限制；
没有改TRAIN_SCENARIO_V1或用成功见证筛选随机分布。reachability被测3e0d0d8；
Stage0/1回归与完整工程恢复被测039d93a；之后仅增加方法身份拒绝守卫/观察式计数，
默认B0转移、普通SAC数学、可达性脚本及共享Stage0/1内核未改变；最终461项回归覆盖新守卫。

## 实际计算与轻量存储

科研 `scientific_training_steps=0`、`scientific_training_updates=0`。
新真实工程学习smoke（首次失败仍计账）：`engineering_env_transitions=1120`，
其中1056次训练Replay采样、64次独立短验证；`engineering_sac_updates=36`，144次Adam。
包含warm-up的World调用1280次，属于嵌套计数，不与环境转移相加；上限仍为2048/128。
本轮合成单测 `synthetic_test_updates=151`（成功完成普通SAC core的合成batch/Replay更新，
包括两次全仓和CUDA失败/修复probe）；旧真实/混合环境单元夹具完整core更新另22次。
两次全仓及CUDAprobe合计Adam820、SGD2；Actor-only/成本优化器操作包含在优化器计数中，
不伪装成完整UTD更新。节点来源及嵌套World/有限感知/B0计数保存在两个counts JSON。
非学习可达性1488正式转移及30 warm-up、Stage1的231和Stage0的1均另计。
开发单测也有非学习环境调用，保留各次实际记录，不宣称所有环境和梯度计算均为0。

仅清理本轮可再生成wheel、编译缓存和pytest checkpoint夹具 **2610777971字节**；
清单 `temporary_cleanup.json`。原数据、原模型、协议、历史Gate和manifests未删除/重写。
Torch包净增加 **2395657030字节（约2.23GiB）**；结果目录测量约46.23MiB，
其中约44.86MiB为必要工程恢复/失败checkpoint，只在本机、全部排除Git。
公开小体积文本证据约1.4MiB；精确测量时点/范围见storage_usage.json，Git发布对象略有增长。
未复制源码/venv/交接包，未生成额外源码摘要清单、ZIP或provenance rebase。
本轮没有新增退役科学断言；上轮12个纯摘要检查的退役仍按原记录，不计本轮PASS。

## 下一项唯一任务

**登记并申请Stage2 B0首个受控无障碍科研训练运行授权。**
先明确该子任务配置/预算/停止安排，以及随后CV的预算分配、是否重新初始化和Replay携带；
不默认无障碍300k再自动CV300k。独立配对验证集合及最终checkpoint主比较也须登记。
此时工程准备已就绪，但尚无多seed学习简单任务证据，因此Stage2科研仍为NOT RUN，
不能写Stage2科研GO；本轮到此停止。

---

# 历史记录：B5.1 迁入 Git 与 LOCAL Stage 0／Stage 1 训练前验收

以下为上轮记录，保留其当时结论和环境限制；不作为当前环境状态。

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
# 当前任务：STAGE2_B0_MVP_BATCH_V1（运行前登记）

2026-10-06 在已发布34f2d4a准备版本上继续，工作分支codex/stage2-b0-mvp-v1。
用户已明确授权seed11/22/33各100k无障碍+200kCV连续课程，共900k科研transition。
本轮冻结登记见docs/STAGE2_B0_MVP_V1.md和configs/stage2_b0_mvp_v1.yaml。
普通SAC和B0物理环境不重新实现；补固定Val、科研入口守卫、课程/批次及恢复日志。
当前记录位置results/stage2_b0_mvp_v1；运行结果只认该批真实日志及固定实验Git版本。
历史461+13属于上一轮验收；本轮最终回归与训练结果将在执行后分别登记。
登记/代码测试前和真实训练前，Stage2科研状态均为NOT RUN，不因入口可执行就记GO。
上轮397/461测试、非学习可达性和旧throughput blocker保留原历史结论。
旧39项docstring、3条英文注释及词法告警不借本轮全面重构；新代码执行既有Ruff规则。

## 本轮验收与实际科研启动回执

实验源码提交：`45f3cc87bb79f69ef5257d6bbd5c92bfbea8b898`（已推送任务分支）。
本轮最终全仓561passed、0failed/errors/skipped；外置PhaseA13passed；Ruff/compileall exit0。
测试命令实际在前驱34f2d4a加本轮工作树执行；相同被测源/配置/测试随后提交为45f3cc8，未在验收后改变算法或测试。
此前一个定向入口命令因新basetemp父目录未建立产生11 setup errors，创建本轮目录后19pass；原失败日志保留，科学断言未改。
批次已于2026-10-06T12:56:55.891120+00:00实际启动，真实worker PID `43736`。固定代码版本不随结果/文档提交改变。
当前Stage2科研状态RUNNING，未完成整批、未作GO判断。实时事实读取 `results/stage2_b0_mvp_v1/batch_state.json`，不要把本回执的瞬时步数当最终预算。
完成后自动写统计、固定策略诊断与图；科研判断和公开结果提交仍须依据实际完整证据。旧throughput blocker未改PASS。
