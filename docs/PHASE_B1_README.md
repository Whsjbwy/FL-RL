# Phase B1 使用与证据边界

本包是修复后的生产源码，不是仅含审计脚本的包。建议解压到单独目录，不覆盖你未备份的工作区。
原配置和依赖声明未改；pyproject中的项目版本仍0.2.0，B1快照以源码指纹与修复登记区分。

## 已修范围
CM02冻结线段方向及统一相对符号；CM03单侧负方差校正；CM04明确异常与日志；
CM05原backup字典序；CM06真实执行净间距；CM07component日志；CM08整数tick延迟。
CM13仅指纹/映射存在性/Ruff缺失门控。详细源差异在独立patch与报告中。

## 本轮实际结果
仓库69项（原34＋新增35）通过；外置Phase A原13项通过；smoke推进0.2秒；compileall通过。
所有结果来自本容器，不是用户工作站吞吐，不是导航策略结果。
Ruff：pyproject仅声明dev依赖ruff>=0.6、未锁版本；尝试隔离安装0.6.0失败（网络不可达）。
未执行Ruff lint，不假装通过。Stage0 audit正确返回NO-GO。B1为CONDITIONAL PASS。

## 本地复核
在已激活的项目虚拟环境、项目根目录执行：

```text
python -m pip install -e .
python -m pip install ruff==0.6.0
python -m pytest -q
python -m ruff check src tests scripts
python scripts/run_stage0_integration_smoke.py
python scripts/run_stage0_audit.py
```

0.6.0是本轮拟采用的独立工具固定版本，满足原>=0.6声明；并非声称原项目锁定0.6.0。
不运行任何--fix来隐式改动算法。若Ruff报告问题，先保存报告并逐项审查差异。
Stage0 audit用PATH查找ruff，请先激活对应虚拟环境；不要混用其他环境的工具。

## API迁移
主运行链：DetectionDelayQueue(control_dt_s=配置值, clock_origin_s=时钟原点)，
pop_arrived(current_control_tick=整数)。SensorDetection增加measurement/arrival_control_tick。
KF仍读取measurement_timestamp_s，arrival_timestamp_s只作为名义时间记录。
旧秒制调用和旧SensorDetection构造保留显式兼容入口；控制节点32 ULP窗口仅解释舍入，
非节点查询向下取整。不可借这个旧适配器区分窗口内物理上不同的亚步时刻。

LogContext规范字段由module改为component。旧adapter字典module显式迁移并移除，
不传给LogRecord。标准LogRecord.module保留Python原义。

## 尚未完成
完整感知/234维、reward/SAC/cost critic/Replay/Trainer/F0/F1均不在B1。
本包不批准Phase B2实施、MVP或120M预算。补齐静态证据后再按用户Gate决定。
