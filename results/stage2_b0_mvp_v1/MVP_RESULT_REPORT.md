# STAGE2_B0_MVP_BATCH_V1：真实批次结果与科学审查

**BATCH_COMPLETED；Stage2科学判断暂缓（待有界失败诊断），不进入Stage3。**

真实项目 `D:\FL+RL\AUV_CODEX_HANDOFF_V2_COMPLETE`；本机Python3.13.5、
Torch2.11.0+cu130、RTX5060，复用已验收环境，未重装、升级或更改科学参数。
实际实验代码 `45f3cc87bb79f69ef5257d6bbd5c92bfbea8b898`；登记/配置在科研运行前提交并推送。
分支 `codex/stage2-b0-mvp-v1`。结果/绘图工具提交不改变此实验代码身份；
最终公开提交见PUBLICATION.json及远端分支引用，不要求报告包含自身提交ID。

## 实际执行与计数

| seed | 无障碍transition/update | CV transition/update | 合计transition/update | optimizer steps |
| --- | ---: | ---: | ---: | ---: |
| 11 | 100000/90001 | 200000/200000 | 300000/290001 | 1160004 |
| 22 | 100000/90001 | 200000/200000 | 300000/290001 | 1160004 |
| 33 | 100000/90001 | 200000/200000 | 300000/290001 | 1160004 |

实测总科研训练 **900000 transitions、870003完整SAC update、
3480012 optimizer steps**。计数与确认后的逐条update、episode日志一致。
同seed保留网络/Adam/alpha/Replay/RNG跨100k课程；其他seed独立初始化。
每seed9999步不更新，10000首次更新；无重复learning_starts；2环境共用预算。
完整训练episode 1565，另6个phase_boundary及6个budget_stop片段；
未完成片段不作事件分母、不补reward、不改Replay终止标记、不额外推进。
验证 **1696646** transitions（42点、2880完整episode）；
训练warm-up 7885、验证warm-up 14400控制transition，均不写训练Replay、不计学习起步。
独立learned策略诊断 12580 transitions、90warm-up，0更新/Replay写入。
科研失败/意外中断/重算0；3次100k→CV加载是预登记恢复，不是失败重试。
实测训练计时 3.5827h、验证计时 0.7946h、checkpoint保存 116.383s；
批次实际elapsed 4.4287h（不含最后分析）。
warm-up数量独立计账，其wall-clock没有单独instrument：部分包含在step/reset计时中，
不能把一秒模拟暖机写成一秒实测墙钟。计时用于运行记账，未开展正式throughput benchmark。

## 固定终点Val300（每行分母300，普通B0，不作安全方法比较）

| seed | profile/终点 | 成功 | 碰撞 | 操作边界失败 | 超时 | SR | mean reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | cv_train_v1/300000 | 2 | 12 | 224 | 62 | 0.6667% | 26.541731 |
| 22 | cv_train_v1/300000 | 1 | 8 | 271 | 20 | 0.3333% | 21.179733 |
| 33 | cv_train_v1/300000 | 0 | 10 | 274 | 16 | 0.0000% | 25.904788 |
| 11 | obstacle_free/100000 | 35 | 0 | 88 | 177 | 11.6667% | 51.056649 |
| 22 | obstacle_free/100000 | 5 | 0 | 295 | 0 | 1.6667% | 50.204424 |
| 33 | obstacle_free/100000 | 14 | 0 | 37 | 249 | 4.6667% | 42.040679 |

跨独立训练seed N=3；样本SD分母n−1，不把episode当独立seed。
- cv_train_v1 SR **0.3333% ± 0.3333个百分点**；reward 24.542084 ± 2.929245。
- obstacle_free SR **6.0000% ± 5.1316个百分点**；reward 47.767251 ± 4.977629。

## 配对学习前后与全部曲线

Val固定root20261006、indices0..299，monitor0..29；三seed相同场景ID、外生CV运动与潜在传感流。
全部点直接训练状态/RNG隔离检查通过；未用Test-ID/OOD/校准，没有滚动换场景。
完整Val300端点的monitor比较仍只取同一0..29子集。

| seed | 空场景step0→100k monitor成功数/30 | CV step100k→300k monitor成功数/30 |
| --- | ---: | ---: |
| 11 | 0→3 | 2→0 |
| 22 | 0→0 | 0→0 |
| 33 | 0→1 | 1→0 |

所有训练/验证点与reward、碰撞/边界/超时、loss/alpha图见analysis/figures，
任务两段分别画图，标100k切换和样本30/300，不平滑、不挑最好25k模型。
图已从公开CSV/gzip重新生成并检查版面；NED Down/俯视/三维坐标一致。
训练路径长度定义为实际控制节点折线；无障碍净间距null/N/A，碰撞0不代表避障能力。

## learned策略固定可达性诊断及失败证据

上轮有界非学习脚本曾为六例找到可达见证；本轮仅用真实100k/300k模型，每例每seed一次。
**0/18成功，10超时、8操作边界失败**。不是拿旧脚本成功冒充策略学会。
8个边界失败诊断的实际终态pitch均触及±30°，位置仍在操作盒内；具体CSV引用见MVP_RESULT.json。
验证的boundary还包含位置球包络、姿态、速度/角率限制；没有完整终态的验证点不能全部归因为位置越界。
seed11空场景Val index0/1超时时距目标约9.0/8.7m；seed22 index0边界终止仍距目标3.59m，未进入2m目标球。
固定轨迹indices0/1/2+最早失败index共127条、72864节点，成功与失败均保留。
高正reward/局部进展不能替代到达：冻结reward只有progress、goal、时间和动作平滑项，
无额外失败奖励惩罚；这是冻结语义，不据此擅自改reward或宣布实现bug。

## 真实验收、范围及缺陷

**本轮全仓561 passed，0 failed/errors/skipped；外置PhaseA13 passed；Ruff0.6.0/compileall exit0。**
实际测试发生在34f2d4a前驱加工作树，完全相同的被测源码/配置/测试随后提交45f3cc8；
登记与被测代码已在训练前提交推送。历史461+13及Stage0/1记录不当成本轮新运行。
一次定向入口命令因新basetemp父目录不存在，8pass/11setup errors；建立可写目录后19pass，
失败原日志保留，科学断言/容差未改。注册、固定Val隔离、课程边界、恢复、UTD和日志单测覆盖在561中。
结果导出9项纯解析、压缩绘图最新11项纯解析和其Ruff通过；不与561重复合计。
结果报告helper首次Ruff15处E501、逐项修后余2处、最终0；失败原输出/退出码保留，没有放宽规则或改变算法。一次日志预览GBK编码错，外层改UTF-8后可读。
最终561项观察器记录84次普通update返回、62次显式batch完整update返回、Adam400/SGD1；
它们有嵌套且只覆盖这次回归，不能相加或冒充本轮全部合成操作。早期定向操作未全量instrument。
训练逐更新有限性检查及全部已确认日志无NaN/Inf异常；未关闭检查、AMP/compile/PER/风险过滤均未新增。
旧39项docstring、3条英文注释/词法告警及throughput blocker仍是历史遗留；未将其改PASS。
轻微状态标签局限：CV恢复初段operation显示INITIALIZING到首个125k点，但真实计数/日志正常前进。
启动记录原样保留；launcher工具会话exit0，worker终码未独立捕获，
但COMPLETED/完整确认日志、两自动分析命令exit0、stderr空和进程退出均实际核对。

## 本机保留、公开发布与科学判断

本机models/seed_{11,22,33}_{100000,300000}.pt以及seed_*/latest_resume.pt保留，
有效原始episode/update/validation与segment cutoffs保留；最近全状态可恢复，不存所有25k Replay副本。
公开两个episode CSV及选定轨迹CSV.gz、Val清单、统计/曲线、18固定诊断、必要测试与命令。
压缩21,450,928→7,511,523字节，经直接字节流往返一致检查；原CSV留本机，非项目ZIP/摘要封存。
不上传模型/Replay/虚拟环境/Word原件/私人上下文/大update和rawvalidation日志。
本轮结果目录当前 2833666511 bytes（2.639GiB，测量范围/时点见JSON）；
仅清理自生成可再生单测夹具 386956514 bytes；原数据/模型未删除。
已观测进程峰值RAM 5144461312 bytes；GPU为资源快照，未instrument分配峰值。
最后normal Git diff确认科研源码、测试、配置、依赖、冻结登记相对45f无漂移。

严格按LOCAL§25.3：**GO证据未建立**。固定可达任务0/18、随机CV终点极少成功，
不能只因跑满预算或reward为正晋级。CONDITIONAL GO所需的可学不稳定及具体原因尚未确认；
NO-GO所需的限定修复无效也未执行。因此科学分类**DEFERRED（待有界诊断）**，
不是科学假设被否定，不伪造三类Gate中的通过结论。没有改变预算/奖励/采样律或追加seed。
唯一下一建议：**Stage2 B0姿态限制与接近目标后超时的有界失败诊断**；
固定少数本批失败ID/最终模型，先非学习核对首次终止完整状态、观察归一化、
动作滞后/边界/reward链，再按L1/L2/L3提交具体依据。未经新决定不调参、不重训、不进Stage3。
