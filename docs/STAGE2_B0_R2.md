# STAGE2_B0_R2_CONTROLLED_REWARD_AND_BUDGET — 运行前登记

2026-10-09 Asia/Shanghai。本登记在任何R2新诊断数值及训练结果之前提交Git。
这是LOCAL Stage2最后一轮常规L2/L3修复，不是Stage3；旧V1/R1不重跑、不覆盖。
历史R1公开代码9029d60、结果000c4c3及本地发布回执f476b1b保持。

## 原文、历史与待检验假设

本机LOCAL-v2.0原件为科学依据，不公开Word全文。原生奖励表（document.xml
第54个table，0-based）到达权重Nominal100、允许Validation区间50–200；
不是100–200。共同任务reward适用于各算法，不增加碰撞或不确定性奖励。
原文Stage2表（table88）最多两轮L2/L3；第一轮R1已用、无关键L1被确认。
V1终点空场景SR均值6%，R1公共lr1e-4终点2.89%，均属历史开发验证证据。

H-BUDGET：100k后转CV过早可能限制基础到达学习；同reward训练至300k能否更稳定？
H-GOAL：到达奖励100的相对强度可能不足；其他条件相同的200是否改善真正2m到达？
两者均是待检验假设，回报增加或预算执行完成不等于学会；不追加第三候选。

## 冻结科学配置和执行选择

C300：普通B0 SAC，公共lr3e-4，goal100；G200仅goal改200。
两组从零、seed11/22/33、全程obstacle_free、每run300000，共1800000 transition。
顺序C300_11→22→33→G200_11→22→33；一次仅一个GPU训练run、两个环境固定轮转。
不加载旧模型/Replay，不清同run Replay/优化器，不切换CV。
gamma .999、tau .005、batch256、Replay500000、starts10000、UTD1、alpha .2、
target_entropy -3、256×256；progress1、time.01、smooth.02；234D当前真值、零协方差、
无未来/风险/执行过滤；.2s控制/.05s积分、1000控制步任务、2m成功球、原3D动作及边界。
这些科学默认不因速度或中间结果改变。goal200是本轮唯一reward候选，不改全局默认。

登记/组名/输出不进入随机派生；同seed两组初始化、Actor、Replay、场景、环境流相同。
策略不同造成episode长度与场景实际发行时刻不同是正常事实，不筛除困难样本。
原TRAIN_SCENARIO_V1不修改；无障碍使用已有独立profile，不冒充1–4障碍分布。

## 新结果前的有限审核选择

复用R1独立数学、事件和reward核验，不重放已充分记录的轨迹。
读取R1已保存25/50/75/100k模型清单、固定失败R01–R03和Val0/1/2详细轨迹：
按原case身份顺序提取首次10m/5m/3m（未达到为null）、动作/响应/目标向量/最小距离/
最早事件。对已有全部完整R1训练和固定评价episode离线核对reward100/200：
失败其他分量不变，成功终点增加100；选原顺序最早成功轨迹比较折扣终点和进展。
统计成功episode所有transition与单个成功终点分别记账；不虚构历史Replay抽样频率。
发现关键L1/实质冲突时先停止科研启动并标影响；离线改reward不当作新行为证据。

## 固定验证、轨迹与终点

Val root20261006、base indices0..299、monitor0..29，所有组/seed同几何及潜在噪声。
0、25k、50k、75k、100k、125k、150k、175k、200k、225k、250k、275k、300k
每点monitor30；100k和300k**额外**Val300（分别保留两条评价身份，不替代monitor）。
deterministic，完整1000步/真实提前终止；直接比较训练Agent/Replay/环境/观察/发行器/
随机流，验证不写Replay或更新。验证集已用于开发，非Test/OOD/Calibration。
固定Val indices0/1/2及按场景索引最早失败保留真实轨迹；终点R01水平/R02下降/R03
上升三个既有可达案例每seed/组各一次，学习策略执行，不用LOS成功冒充策略能力。
每25k小模型保留五网络及alpha，不按best挑点；主要终点固定300k，100k作开发核对。

核心SR、边界/子型、timeout、最小目标距离、2m首次进入、完成时间和3D轨迹/角率。
逐seed结果及跨3seed均值/样本SD；episode不是独立训练seed，不制造显著性。
两reward版本原始return分开；从真实分项离线计算共同goal100效用，不修改实际回报。
学习曲线无平滑挑点，100/200/300k对照；NE/ND与NED三维坐标一致。

## 工程质量目标与科学判断

提前登记工程目标：300k三seed Val300平均SR≥70%，每seed≥50%，边界不主导，
200–300k无无法解释的严重完成能力崩溃，固定水平/上升/下降有真实策略成功轨迹。
这是本轮工程目标，非LOCAL原文阈值，不事后降低，也不替代LOCAL25.3科学Gate。
两组均失败则如实记录有限修复耗尽及不确定性，停止自动长训；基础不稳不进CV/Stage3。

## 计数、恢复、资源和发布

一次env.step一条transition，暖机/验证/合成操作另计。store后eligible沿用：
9999无update、10000首update；预计每run290001完整update、两组1740006，最终按实测。
每完整update包含4个optimizer step，非一条经验四次训练。无正常失败episode筛除。
完整控制步边界保存，latest完整恢复点+原子临时替换，组/reward/日程/代码/Git/Torch/
Replay/RNG/场景/评估触发/有效日志身份不符拒绝。失败不自动retry或追加1800000预算。
每25k只保存小模型，不复制完整Replay；旧唯一数据保留。run结束捕获真实worker退出码。
使用已验收本机venv/Torch2.11+cu130/RTX5060；不重装、AMP、compile或优化算法。
运行前最终全仓、PhaseA、Ruff0.6和compileall；结果不得冒充旧PASS。
Git提交科学代码/配置/本登记后固定experiment_code_commit；后续报告提交不改科学身份。
小证据公开、模型/Replay/完整原始轨迹/Word留本机；不hash封存、ZIP、强推或合并main。

结果目录results/stage2_b0_r2。完整执行后停在Stage2科学判断，不自动开展后续研究。
