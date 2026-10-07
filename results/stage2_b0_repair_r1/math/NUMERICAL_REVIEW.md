# R1 独立数学与实现核验

本记录限定于本轮预登记的标量/梯度、最早事件、实际失败输入和冻结critic核验，
不能由测试通过反推策略已学会。以下统计只覆盖本文件列出的本轮真实运行。

## 直接读取的权威来源

直接打开本机`handoff/protocol/LOCAL_v2_0.docx`的`word/document.xml`，
读取原生OMML表52–64（零基索引）对应式43、44、47、48、49、52、53、54。
没有依据公开库反向替换本项目公式，也没有生成文件摘要。

特别是式53位于Word表63：外层期望括号中为负`alpha_ent`乘以内层
`(log pi + H_tar)`，随后明确`alpha_ent=exp(ell_alpha)`、`H_tar=-3`。
独立标量转写为`L_alpha = -exp(ell_alpha) * (log_pi - 3)`。
保持logpi停止梯度时，`dL/dell_alpha = -exp(ell_alpha)*(log_pi-3)`。
因此当前`rl/losses.py:41`的exp形式匹配原文；常见log-alpha代理损失的
梯度幅值不同，不能因为它更常见就作为修复。

## 真实执行

项目解释器`.venv-b1/Scripts/python.exe`，命令、cwd、实际退出码在`commands.json`。
新增`tests/test_b0_r1_numerics.py`：18 passed，0 failed/errors/skipped，19.88秒；
本文件定向Ruff退出码0。原始stdout/stderr及`numerics.xml`保留。
解释器启动打印的既有F盘真实位置警告保留在stderr；后续进程实际退出0。

参考依据在执行前固定：float64解析绝对误差1e-10，事件fraction1e-8；
float32网络标量/梯度atol/rtol均1e-5；中心差分h=1e-3，绝对误差1e-3。
未放宽旧科学断言、未关闭检查规则、未改生产数学。

|对象|独立参考与本轮结果|证据等级及适用边界|
|---|---|---|
|Eq48 tanh logprob|直接标量高斯密度减log(1-tanh²)，解析导数与中心差分；4点通过|未支持Jacobian符号错误；非全参数空间证明|
|Eq47重参数化|固定独立epsilon，解析da/dmean与da/dlogstd、随机流计数；通过|支持正确动作梯度路径|
|Actor更新|生产`update_actor`搭配可手算动作线性Q；梯度匹配、Q参数不变且无grad、alpha无Actor梯度|未支持冻结critic切断dQ/da的假设|
|Eq49|逐行标量目标、两个真终止NaN行不求值，外部截断对应非终止行续接，shape(B,1)|支持当前终止mask与目标尺度；不能说明学习后Q准确|
|Critic MSE|手算2(q-target)/B，目标detach|未支持广播/目标梯度泄漏|
|Eq53|4个logpi值，原生exp公式值、解析梯度、有限差分与下降方向|未支持exp形式是L1错误|
|Eq54|目标旧值×.995+在线×.005的手算权重/偏置|未支持软更新方向错误|
|Body/NED与动作|yaw90/pitch30的独立分量式；固定三维动作映射|当前编码尺度小不等于编码错误|
|目标事件|稳态1.5m/s从距目标2.06m开始，t=.04s入球，当即终止，拒绝后续step|支持扫掠到达能在控制节点之前触发；实际失败另核验|
|pitch边界|稳态pitchrate=.2、离界.006rad，t=.03s首次违规；姿态及位置同时插值|终态等于30°可由正确定位产生，不是自动clip证据|
|Eq44 reward|逐项手算四项；60→52→54→50未折扣进展抵消为10m，折扣和单列|正return本身不能证明无限刷进度或到达|

本次定向运行计算单独记账：1次合成Actor Adam step，0次完整SAC update；
2个非学习World控制step成功调用，以及1个终止后拒绝调用；0科研transition/update。
这不是导航训练，也不代表本轮之后所有测试/重放的总计算计数。

## 代码审阅范围与限制

原始B0训练及deterministic验证均调用相同观察适配层；`B0NavigationEnv.step`
的名义与执行指令相同，但仍经过真实执行器响应/变化率限制及World物理事件。
Replay在reset前存next_obs，普通SAC损失白名单只读取obs/nominal/reward/
next_obs/terminated；cost/executed仅存储，不参与普通SAC目标。
世界比较每0.05s小步的扫掠碰撞、球包络位置、pitch/surge/角率约束和目标事件，
按fraction与既有优先级取最早事件；最终插值快照不是把越界状态夹回去继续运行。

100k小模型实际只保存Actor，不能用300k完整恢复点中的critics/Replay/RNG
冒称100k时刻状态；50k/75k权重缺失应由原始评价日志反映，不能重新造一个历史模型。
核心数值核验未复现L1；实际失败输入/冻结critic的后续核验如下。本结论仅限审阅范围，
不能证明全部参数/场景无错误，也不能把近似critic的不准确直接认定为实现bug。

## 六个实际失败快照独立核验

按登记的首次失败及pitch/距离/第50步选择规则，读取三个seed各100k空场景、
300k CV的六个真实快照。`actual_failure_input_review.json`记录独立三角式重建
Body目标向量、18维自身/任务、射线、当前真实目标槽和padding。
六例原始234维输入与独立float32参考最大绝对误差均为0；协方差编码均为0。
控制时刻读取当前状态，没有把reset旧目标状态重用为当前输入的反例。
相关物理指令映射、reward四项与首次小步事件由冻结重放证据单独记录。

## 最终300k critic：有界动作排序与随机后续

直接读取本机V1三个`latest_resume.pt`，核对实验代码、seed、300000transition、
290001update及Torch版本，抽取Actor/Q1/Q2和alpha后释放Replay对象。
最终恢复点Actor与原300k小模型逐张量一致；这些critic不是100k/50k/75k权重。
只对每seed登记的第一个CV Val0/1/2失败快照，比较固定五种初始动作。
后续采用同一冻结随机策略与固定alpha，四个独立诊断策略流，各最多50控制步。
各动作复用相同四个流，初始固定动作不消耗该流；实际60条×50步=3000步，
共享`diag_budget.json`计账，无warm-up、梯度、优化器步骤或Replay写入。
程序真实退出码0，网络参数和原快照不变，未出现非有限值。

定义明确为`r0 + sum(t=1..49, gamma**t*(r_t-alpha*logpi_t))`。
初始固定动作不带熵项，后续熵项位于后续状态动作时刻。三seed固定alpha分别约
0.0125663、0.0174239、0.0158986。60条均未在50步内物理终止，剩余任务尾项未知，
因此不能将这些部分soft return直接当成完整soft Q精确真值，不能据此宣布Q偏差数值。
LOS仅替换第一个固定动作，后续仍是冻结随机策略，不是LOS回报的Q标签。

min(Q1,Q2)排序如下（高到低）：

- seed11：surge-only、original、pitch-only、yaw-only、LOS；
- seed22：surge-only、original、yaw-only、LOS、pitch-only；
- seed33：LOS、yaw-only、pitch-only、original、surge-only。

部分后续轨迹的平均目标进展约seed11 1.74–1.86m、seed22 10.71–10.78m、
seed33 11.78–12.02m；但都没有到达/失败事件，不能以微小短期差别认定长期排序错误。
证据等级为`INSUFFICIENT_EVIDENCE`用于critic准确性/50k–100k退化的因果归因；
保留为后续机制诊断数据，不能用loss有限或短期进展代替策略任务完成。
完整记录为`critic_result.json`，紧凑15行动作表为`critic_action_summary.csv`。
复现仅使用可信本机数据，禁止自动重复已存在attempt以避免重复消耗预算。
