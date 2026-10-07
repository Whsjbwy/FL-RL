# Stage2 B0 R1 诊断与单因素修复检验结果

**批次真实完成；R1学习率候选未改善固定100k任务完成能力。Stage2 GO未建立，Stage3暂停。**

本机目录 `D:\FL+RL\AUV_CODEX_HANDOFF_V2_COMPLETE`。实验代码 `9029d60a3d88210e9113936b5da4f604bbf8c202`；冻结登记 `docs/STAGE2_B0_REPAIR_R1.md`，
配置 `configs/stage2_b0_repair_r1.yaml`。被测代码ca4e67b与实际实验9029d60的
src/tests/scripts/configs/登记及依赖Git差异为空；训练后相同保护范围亦无变化。
公开分支codex/stage2-b0-repair-r1-public；实际发布回执见PUBLICATION.json。

## 分支选择、可比性和修改

先冻结假设/选样规则，完成已有日志、原生公式、独立数学及有界策略重放后选择B。
未在已审阅范围发现确认的科学L1，也无未解决的关键协议冲突；这不是全域无bug证明。
仅公共学习率3e-4→1e-4，原默认与V1仍3e-4。Actor/Q/alpha共用原优化器配置；
从零、空场景、双环境、batch256、Replay500k、starts10000、UTD1及所有其他科学参数保持。
C复用V1前三个100k，不重新跑C，不接续旧模型，不跑CV。初始化网络/targets/alpha、
Actor/Replay/场景/环境派生种子和固定Val几何/潜在噪声逐项相同；
15个验证点的直接场景身份比较全部一致，细节见control_comparability.json。
不同参数导致episode长度变化后，训练场景发放时间线可以不同；不强制假配对。
原SAC、动力学、事件、reward、B0观察、TRAIN_SCENARIO_V1内核未改。
新增/修改仅登记守卫、版本化LR配置、薄单课程运行管理、可撤销只读诊断及相关测试。
诊断observer目标root舍入、结果JSON及回执时戳修复是诊断/记录工具修复，
不改变世界转移或优化数学，不能冒称“发现并修好了V1科学L1”。

## Q1—Q4：已确认事实与不足

| 问题 | 证据与结论 |
| --- | --- |
| Q1 首发boundary | 原18固定诊断确为8pitch、10timeout；终态±30°与最早事件插值一致。所选其他轨迹也有位置包络失败。R1全900个终点Val有16 pitch_lower和119位置限制，无未分类。C全Val旧子类NOT_RECORDED，不能补造。 |
| Q2 是否进入2m球 | 独立审计213578个0.05s实际分段无首次事件不一致或漏判；R1的26成功均有first2m，874失败均无first2m。固定九例最近距离均大于2m，接近不等于成功。 |
| Q3 更新/执行/评价是否一致 | 六个真实失败234D输入独立误差0；18项解析/梯度检查通过。动作顺序/Body与NED/执行器响应/终止mask/真实next_obs/reward一致。Word式53的exp(log_alpha)形式正确，不能因公开库代理损失不同而替换。 |
| Q4 50k—100k退化为何 | 原monitor在50k或75k后下降，先于课程切换；alpha下降与Q-loss增加是事实而非完整因果解释。降低LR后所有seed终点成功数仍下降，具体候选不受支持。历史中间critic缺失，完整机制仍未建立。 |

机制证据等级和反对证据详见R1_RESULT.json。成功episode经验占旧前100k的
9.932%/5.714%/8.370%，不能只数17/12/15个成功终点就称Replay没有成功经验。
六个快照各五条预登记反事实共30条，接管8成功，但不能算learned策略成功；
yaw/pitch替换对不同案例作用不同，支持多组件目标捕获困难，未证明全部只因pitch。
60条300k critic随机50步soft-return诊断未测尾项，不能当完整Q校准真值；
V1历史50k/75k模型及100k critic不存在，不重跑V1伪造它们。
reward四项独立累加及进展首尾抵消核对一致；gamma加权进展另计。
没有额外失败惩罚属于冻结设计；没有改成功半径/姿态界/reward或加入风险过滤。

## 固定100k终点：每行Val300，N=300

| 组 | seed | 成功 | SR | 边界 | timeout | 平均reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C_3e-4 | 11 | 35 | 11.67% | 88 | 177 | 51.0566 |
| C_3e-4 | 22 | 5 | 1.67% | 295 | 0 | 50.2044 |
| C_3e-4 | 33 | 14 | 4.67% | 37 | 249 | 42.0407 |
| R1_1e-4 | 11 | 19 | 6.33% | 23 | 258 | 41.8531 |
| R1_1e-4 | 22 | 1 | 0.33% | 2 | 297 | 26.6261 |
| R1_1e-4 | 33 | 6 | 2.00% | 110 | 184 | 35.9608 |

跨三个独立训练seed的SR均值±样本SD：C **6.00%±5.13pp**，
R1 **2.89%±3.10pp**。总边界比例46.67%→15.00%，timeout47.33%→82.11%；
不能只取边界减少声称导航或安全改善。三个配对seed的终点SR均下降。
reward四项及分层均值/样本SD见endpoint_comparison.csv和结构化回执。
这是开发Val，不是独立Test；数百episode不是数百训练重复，不生成虚假p值。

### R1终点失败细分与接近目标

| seed | 首发边界子类 | timeout N | timeout最小目标距离均值 | timeout进入10m / 2m |
| --- | --- | ---: | ---: | ---: |
| 11 | {'pitch_lower': 16, 'position_D_upper': 6, 'position_D_lower': 1} | 258 | 6.2504 m | 240 / 0 |
| 22 | {'position_D_upper': 1, 'position_E_lower': 1} | 297 | 13.5238 m | 76 / 0 |
| 33 | {'position_D_upper': 108, 'position_N_upper': 2} | 184 | 8.5641 m | 123 / 0 |

进入10m是接近诊断，不是2m成功；其首次进入时间只对真正进入的episode求均值。
细分操作边界和最小目标距离按0.05s已执行分段采集；未执行提议仍留在选定详细轨迹。
无障碍净间距不适用，不以0冒充；训练完整episode与预算未完成片段分开。

### 原定monitor30曲线（100k为Val300中的同indices0..29子集）

| 组 / seed | step0 | 25k | 50k | 75k | 100k |
| --- | ---: | ---: | ---: | ---: | ---: |
| C_3e-4 / 11 | 0 | 4 | 7 | 11 | 3 |
| C_3e-4 / 22 | 0 | 0 | 9 | 1 | 0 |
| C_3e-4 / 33 | 0 | 1 | 10 | 5 | 1 |
| R1_1e-4 / 11 | 0 | 0 | 3 | 0 | 3 |
| R1_1e-4 / 22 | 0 | 1 | 0 | 3 | 0 |
| R1_1e-4 / 33 | 0 | 3 | 0 | 5 | 1 |

每个值为成功数/30；不把监测子集说成Val300，不选择最好25k模型代替终点。
真实曲线见analysis/figures/paired_monitor30_learning_curves.svg、
fixed_100k_val300_raw_seeds.svg和update_diagnostics_comparison.svg；未平滑或删除失败点。

### 固定R01—R03 learned策略诊断

| seed | 固定案例 | 原V1 100k结果 | R1 100k结果 | R1最小目标距离 |
| --- | --- | --- | --- | ---: |
| 11 | R01_horizontal_empty | task_horizon | task_horizon | 5.5168 m |
| 11 | R02_deeper_empty | task_horizon | task_horizon | 8.6166 m |
| 11 | R03_shallower_empty | task_horizon | task_horizon | 4.9152 m |
| 22 | R01_horizontal_empty | operational_boundary_failure | task_horizon | 11.4118 m |
| 22 | R02_deeper_empty | operational_boundary_failure | task_horizon | 14.2875 m |
| 22 | R03_shallower_empty | operational_boundary_failure | task_horizon | 17.9048 m |
| 33 | R01_horizontal_empty | task_horizon | task_horizon | 3.3802 m |
| 33 | R02_deeper_empty | task_horizon | task_horizon | 11.1390 m |
| 33 | R03_shallower_empty | task_horizon | operational_boundary_failure | 20.8753 m |

R1九例0成功、8timeout、1position_D_upper；各一次，不写Replay/不更新参数。
全八维状态/234D观察/Actor分布/动作/提议/事件链的gzip留本机diagnostic_replay/。

## 真实计数、执行结束与检查

| 项目 | 实测 |
| --- | ---: |
| 新R1科研训练transition | 300000（11/22/33各100000） |
| 新完整SAC update | 270003（各90001，10000第一次更新） |
| 新optimizer step | 1080012（各360004；不能当完整update） |
| 独立固定验证transition | 1091711 |
| 训练合法warm-up控制步 | 2505 |
| 验证合法warm-up控制步 | 6300 |
| 全部冻结诊断轨迹 / 控制步 / warm-up | 167 / 65056 / 385 |
| 失败科研attempt / 科研重算 / 新CV | 0 / 0 / 0 |

V1历史900000不是本轮新增；复用C只是其首300000，不与R1拼一条学习曲线。
实际训练累计3599.756s；固定验证1111.927s；
保存15.385s。不是正式throughput测量，也不据此改算法。
实际计算PID37516结束码0，启动器PID38168结束码0分别保存并核对；
结束时间2026-10-07T16:24:49 UTC（本机2026-10-08凌晨），不只报launcher。
batch_state.json与commands原始记录确认3seed COMPLETED，没有仍在运行的训练。

本轮最终全仓**621 passed, 0 failed, 0 errors, 0 skipped**；外置Phase A **13 passed**。
新增科学/调度测试60项，其中独立数值18；未删科学断言、未改容差、无skip/xfail。
Ruff0.6.0与compileall均exit0；最终被测代码ca4e67b，与训练代码保护范围相同。
运行后仅结果/绘图/回执工具有改动，另查其Ruff/compile，不重复已未变的621项。
最终全仓观察器实测84次完整ordinary update，其中62次显式合成batch，
401次Adam和1次SGD；另有单独数学检查Actor步。早期定向调试未全程安装
计数观察器，因此不伪称全轮所有合成操作精确为84；嵌套World/env计数不能相加。
两次结果工具失败（NumPy JSON、分开时戳读取）保留首次stdout/stderr/exit1；
最小结果工具修复后exit0，九条已完成重放从summary复用，没有重训/重复轨迹。
旧docstring等无关风格告警不做全面重构；原throughput blocker仍为历史未解决状态。

## 存储、证据与公开边界

每seed latest_resume.pt仅保留最近完整点；models/保存25k/50k/75k/100k
小模型（Actor/两个Q/targets/alpha），12份均实际读取核验有限、身份及计数正确。
模型、Replay、完整update JSONL、完整234D细轨迹和Word原件只在本机；
原V1唯一数据/模型/报告未删除或覆盖。没有复制整套项目/venv或生成额外摘要/ZIP。
公开保留全部确认episode的精简CSV.gz、固定index0三对终点轨迹、
终点与分箱CSV、三幅SVG、复算脚本、必要JUnit/结束回执；原未压缩表留本机。
公开轨迹减量只按已登记index0，不按好坏选取；原indices0..2及最早失败全集仍留本机。
原始commands.jsonl留在本机并解除新版本Git跟踪，历史提交保留；
公开COMMANDS.txt保留实际命令、退出码、时间和代码版本，绘图解释器的个人路径明确代称。
实际新增本轮results体积见storage_usage.json（现场字节统计，无摘要）；
不将未能单独归因的Git内部空间或已有venv写成新增数据。
首次过宽公开push被自动审查拒绝后，没有绕过；完整ca4本地提交仍保留，
另建不含ca4祖先的小公开提交9029，已实际push。最终结果同样只上传明确小文件。

## 科学判断与停止位置

**执行完成不等于科学通过。** 预登记第一轮L2候选未支持改善终点完成能力；
仅空场景复测不能把完整CV/Stage2标为通过。LOCAL §25.3多seed简单任务
及任务效用依据仍不足，GO未建立；CONDITIONAL GO的明确可修复原因尚未确认。
进入Stage3为NO-GO/HOLD；第一轮后有限L2/L3修复尚有一轮，未自动授权。
不提前宣告所有修复耗尽或预测/风险/联邦假设NO-GO。
**唯一下一任务建议：使用已保存的中间Actor/critic完成有界近目标价值/控制诊断，
预登记最后一轮单因素R2，再申请用户授权；不自动训练R2、CV或进入Stage3。**
本轮到此停止。
