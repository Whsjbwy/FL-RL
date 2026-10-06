# STAGE2_B0_PREPARATION：实施前规格（revision 1）

冻结时间：2026-10-06。代码基点：a836922b2e1f81689d09e811c29be3badf2462aa。
分支：codex/stage2-b0-preparation。本轮仅工程准备，LOCAL Stage2 科研 NOT RUN。

## 现有能力—缺项—最小修改

已有普通/约束SAC、双动作Replay、完整Agent状态、共同动力学/事件/reward、
234维有限感知环境、合法warm-up、TRAIN_SCENARIO_V1和Stage0/1验收。
缺项是声明Torch与实际环境漂移、B0特权入口、跨episode/多环境调度与完整恢复、
独立验证及固定任务的可执行可达见证。新增B0适配层及组合式harness，复用上述模块；
不修改普通SAC数学、原有限感知观察、验证器、训练分布和科学参数。

## 科学依据与边界

本机LOCAL_v2_0.docx v2.0（2026-09-17），直接读取原Word XML/原生公式。
§7–10共同动力学/CV/感知/KF；§13共同234维、三维动作、四项reward和终止；
§18表18-1 B0为当前真值位置/速度、零协方差、不限可见性、无真实未来、
无风险训练/执行验证；§22独立配对评价；§24与§25.3先无障碍再简单CV，
可达性脚本不替代多seed学习证据；§27有限修复；§29–30及附录B/D.3配置与运行记录。
协议全文留本机，不公开上传。工程B5.1、论文B5与LOCAL Stage编号分别记录。

## B0观察和动作

18+45+45+6×21=234。自身/任务及射线复用原ObservationBuilder的空目标槽输出，
包括既有固定epsilon配置编码；它不是计算风险输入。每个B0目标槽：
当前Body相对位置/25m；当前Body表达的障碍NED地速/1m/s；12个协方差编码为0；
物理半径；年龄0；存在mask。速度不是相对地速，也不是旋转坐标相对位置导数。
速度归一化1m/s、年龄0、当前位置代替未来均值是本轮工程选择，非冒称Word具体规定。
按当前距离/稳定ID排序，缺槽padding；每步读取当前World真值，不调用未来查询/预测器。
真值槽不受FOV/range/dropout/KF建轨限制；有限感知接口原禁止真速度/未来的规则不变。

B0独立环境适配层复用原reset的一秒合法感知warm-up、动作映射、执行器响应/变化率、
World扫掠物理事件、四项task reward和终止。step不调用validator、候选/fallback或risk。
nominal=executed指令，不意味着实际速度瞬时等于指令。Replay cost仅真实物理失败的
诊断0/1；普通SAC损失不使用它，未启用风险/验证指标为None，不伪造安全结论。

## 配置与调度

生产默认gamma=.999、tau=.005、lr=3e-4、batch256、Replay500000、starts10000、UTD1、
alpha初值.2、目标熵-3、隐藏层256–256；.2s控制/.05s积分、1000步、成功半径2m、
既有四项reward。准备MVP seeds11/22/33，验证间隔25000，以最终checkpoint作主比较。
生产num_envs=2，固定round-robin，每个真实step只计一个全局transition/一次Replay写入。
新episode按固定槽顺序发放连续scenario_index，保存slot/episode/scenario ID与显式split。
CV严格复用train-v1；obstacle_free复用其起终点律但明确去除障碍的独立任务身份，
不把train-v1改成0–4，不按policy/risk/TTC/成功拒绝样本。训练/验证/smoke独立派生随机流。

warm-up不计Replay/起步。store后eligible为全局transitions>=starts且Replay>=batch：
9999无更新，10000首次一次完整update，10001第二次。每次完整update有4个Adam.step，
不是4个UTD。起步前也使用原Actor随机策略，不添加新随机动作采样律。
终止next_obs先写Replay后等待下次reset；外部截断、物理结束、预算耗尽分别记录。
预算耗尽保留未完整episode，不能伪造成功/物理timeout。

验证仅deterministic_action，独立环境/场景/RNG；不得改变Agent/Replay/训练环境、
训练场景索引或随机流。运行前后直接递归比较状态，无摘要。少量工程验证不作科研结论。

## checkpoint与恢复

在完整transition/update边界保存Agent所有状态、Replay/随机流、各环境/当前观察、
待reset、上一执行指令/累积日志、全局计数/场景index/轮转槽、评估/保存触发位置、
完整配置/run_kind/B0方法和Git代码版本。旧source_fingerprint兼容字段填Git标识，不计算摘要。
可信本机文件显式加载；拒绝run_kind/配置/版本不一致，工程产物不能作为科研续点。
同设备/同Torch条件，标识/计数/RNG/Replay索引要求精确；环境float64和网络float32
数值比较预登记rtol=1e-6、atol=1e-7（不得为结果放宽）；不宣称跨设备/版本逐位一致。
临时文件完整写入后replace，只保留最近完整恢复点和最终模型，不默认每25k保存全Replay。
实现补充：保存触发通过harness callback落实；失败的半transition不允许保存恢复点。
入口身份补充：真实Agent只能为原OrdinarySACAgent，环境只能为B0NavigationEnv；
成本/约束Agent及有限感知过滤环境在创建时拒绝，纯调度夹具必须明确标记且不含Actor模块。
checkpoint登记并检查Torch版本。路径长度日志标记CONTROL_NODE_POLYLINE，非连续曲线弧长。
网络三个随机流由training_seed/run_kind分别派生；YAML中的旧种子字段仅兼容输入，
preflight和实际checkpoint记录解析后真正使用的派生值。

## 预登记非学习可达性（每例仅1次，最多1000正式步）

共同起点N20/E50、目标N80/E50；初始surge=.3、角率0、姿态LOS；零海流。

|案例/独立seed|起点D→目标D|CV障碍初态(N,E,D)/地速/半径|
|---|---|---|
|R01/810001|20→20|无|
|R02/810002|12→28|无|
|R03/810003|28→12|无|
|R04/810004|20→20|(60,53,20)/(-.4,0,0)/.6，偏置头对头|
|R05/810005|20→20|(50,36,20)/(0,.4,0)/.6，横向交叉|
|R06/810006|20→20|(60,50,24)/(-.4,0,0)/.6，垂向差头对头|

预定有界LOS控制：surge指令1.2m/s，yaw/pitch误差增益1s^-1，按共同动作限幅；
不读取障碍/risk/TTC，不加新规划系统。成功须真实事件、距离<=2+1e-8m、无碰撞/边界、
动作有限且合法、无操作限制违背（仅浮点roundoff1e-10）。无障碍净间距为null。
记录warm-up后初态、指令/轨迹/事件和间距；失败只表示此控制器未建立可达见证。
这些夹具不能筛选训练场景，也不能证明整个随机分布都可达。

## 本轮有限工程计算预算（运行前登记）

调度边界用mock/synthetic，不真实跑10000/25000步。
CUDA合成网络probe 1次完整普通SAC update（4Adam）及张量运算，非导航科研训练。
真实工程演示使用CUDA、batch256/隐藏256–256、starts256、Replay1024、num_envs2、
264 transitions/次，外部episode截断64步。连续264与256保存→恢复8的对照共528个采样、
18次完整更新；独立短验证最多每例8步，不更新不写Replay。其余单测真实环境也独立计数。
恢复对照在256保存待reset状态，再推进4步保存episode中途状态，恢复继续至264；
这些保存/恢复不增加额外采样预算。
L1修复登记（首次工程运行后）：CUDA加载将Adam非capturable的CPU步数标量映射为CUDA，
独立失败用例已保存。修复为可信checkpoint先反序列化到CPU，再由原模型/优化器接口
恢复目标CUDA参数与一二阶矩；不改变数学或容差。首次528采样/18更新及32独立验证仍计账，
最多再进行一次相同528/18恢复对照；累计预计1120真实工程环境步/36完整更新，仍在原上限内。
全轮真实学习smoke及重试累计硬上限2048 transitions/128完整update；失败保留并计账。
所有科研training_steps/updates=0；非学习可达性、单元合成更新与工程真实采样分别记录。
入口默认help/preflight，必须显式run_kind与预算；科研执行还需下一轮注册运行安排，
本轮不启动scientific_training。

## 尚待科研运行前决定

Stage2每seed300k、必要时最多500k；无障碍/CV预算分配、是否重初始化、是否保留Replay
必须在下一轮明确登记；本轮不默认300k+300k，也不把继续训练称为独立CV实验。
本轮不测正式吞吐、不启动研究训练/Stage3/联邦；原benchmark blocker保留历史状态。
