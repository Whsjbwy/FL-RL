# STAGE2_B0_MVP_BATCH_V1 — 运行前登记

登记日期：2026-10-06。结果产生前冻结；改动须另有决议，不能追改成有利日程。

## A. LOCAL 原文依据

本机 LOCAL_v2_0.docx v2.0（2026-09-17）：§13/18 的 B0 当前真值、普通 SAC、
无执行过滤；§22/24 与附录 B 的 MVP seeds 11/22/33、Val300、25k 验证和最终
checkpoint 主比较；§25.3 先无动态障碍再简单 CV，300k/seed（必要时最多500k）。
本轮只批准300k。Stage2 GO 为多个 seed 完成简单可达任务且行为与任务效用一致；
CONDITIONAL GO 为可学但不稳定且有明确动作尺度/reward/预算问题；NO-GO 须有限
修复仍无效。本轮不添加 SR 数值阈值。§27 有限失败诊断，§29–30/附录D.3运行记录。
Word 留本机，未授权公开；不把工程 Phase B5.1 当作科学 Stage2。

## B. 本轮预先决定的安排

机器可读登记为 configs/stage2_b0_mvp_v1.yaml，必须逐字段匹配；科研入口默认
preflight，只有显式 --execute、run_kind、登记路径、总预算和输出目录匹配才能执行。
被测代码和本登记运行前提交 Git；experiment_code_commit 在批次启动记录一次，
恢复使用相同标识并检查该提交到当前的源码/配置/登记差异，不做额外文件摘要。

- 独立 seed 11、22、33。执行11/22/33各至100k，然后各自恢复到300k。
- 每 seed 1..100000 obstacle_free；100001..300000 cv_train_v1。连续模型、
  Q/targets/Adam/alpha/Replay/RNG；不清Replay、不重新学习起步。两环境固定轮转，
  场景全局连续索引跨课程不重置，新环境仍使用原合法1秒warm-up。
- 在100k完整transition/update边界先空场景Val300并保存100k模型，再显式switch。
  未完成episode记录phase_boundary片段；不添加奖励、不改已存Replay的terminated/
  truncated、不标物理timeout、不额外推进。切换后CV monitor30，尚无CV更新。
- 单步store后eligible：9999无update；10000首次；10001第二次。
  预期90001+200000=290001完整update/seed；批次870003，最终以实测计数核对。
- 不改生产gamma .999/tau .005/lr3e-4/batch256/replay500k/starts10k/UTD1/
  alpha .2/entropy -3/256–256；num_envs2；dt .2/substep .05/horizon1000/goal2m。
  训练动作沿用sample_action，不新增随机探索规则，不启用AMP/compile/PER。
- TRAIN_SCENARIO_V1原采样律不变；无障碍使用既有独立profile并移除生成基底的障碍，
  不改train-v1的1–4支持。CV随机场景不是全部简单交会，也不保证全部可达。
- 固定验证root20261006、base索引0..299、monitor0..29。同一原train-v1生成基底派生
  两profile，保留base_scenario_id和实际ID；独立固定环境随机seed，不关联训练seed。
  验证专用adapter按环境seed×整数世界tick（含warm-up）×稳定障碍ID派生独立
  noise/dropout流；仍调用原遮挡/可见性/高斯噪声函数，检测序列可随策略不同。
  只在同步验证上下文替换采集入口，退出恢复；默认训练感知路径不改。此命名空间
  是工程选择，用于配对潜在随机数，不把相同envseed误称为顺序RNG逐时配对。
- 空场景step0/25k/50k/75k monitor30，100k Val300；CV100k起点monitor30，
  125k/150k/175k/200k/225k/250k/275k monitor30，300k Val300。
  deterministic_action，真实1000步/提前物理终止，不带工程截断。
  验证前后直接比较完整训练状态与随机流，不写Replay、不推进训练环境/场景索引。
- 固定轨迹索引0、1、2：每验证保留，成功失败都保存；另按索引升序保存该评估
  最早失败一例。训练全episode必要原始记录保存，非选定轨迹不附巨量逐周期快照。
  更新逐条原始日志和既有有限性检查保留；文件句柄缓冲，每100次记录flush，
  checkpoint前flush+fsync。汇总不替代原始日志，不删除失败/慢段。
- 本批输出仅results/stage2_b0_mvp_v1。每seed保留latest完整恢复点、100k/300k模型、
  最终完整恢复状态；25k覆盖latest，不保留每点Replay副本。安全partial写完再替换。
  单PID互斥锁+batch_state，失败保留日志，恢复开新segment并登记parentcheckpoint，
  未确认旧尾部不混入新段。训练/评估/保存/暖机/失败重算分别记账。
- 固定6例旧可达性案例：每seed100k策略跑R01–03、300k策略跑R04–06，各一次，
  独立诊断，不用LOS控制器、不写训练Replay。适用范围仅这6个固定案例。

## C. 结果出来后才能判断的结论

BATCH_COMPLETED只表示日程完成，不能代替Stage2科学GO。保存所有seed、所有物理
完整episode及未完成片段；无障碍净间距null；SR/事件率分母为完整物理episode，
阶段片段/预算停止另计。训练曲线按profile分开，标100k切换、monitor30与Val300。
主比较最终checkpoint；前后对照使用相同monitor子集。跨seedN=3，不把episode当独立seed。
正常低reward/碰撞/超时不能触发换seed/增预算/调参。非有限/CUDA/身份/磁盘/日志故障
停止并保留最小证据；不把损坏半步状态作恢复点。批次完成后只判断Stage2，不启动Stage3。

## 预登记的测试与数值比较

调度边界用mock夹具，不额外运行100k。离散索引/RNG/配置/状态精确一致；同设备恢复
模型/优化器张量沿用准备阶段已有exact比较，不宣称跨版本/设备逐位一致。
定向入口/固定验证/隔离/课程/恢复测试后一次全仓、外置PhaseA、Ruff0.6.0、compileall。
历史461+13不是本轮结果。资源按真实Replay dtype和本机余量估算，耗时仅实测后估计。
