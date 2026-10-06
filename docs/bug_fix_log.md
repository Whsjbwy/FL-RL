# Bug Fix Log

## BUG-S0-001：Validator 空间边界只检查中心点

- **bug_id**：BUG-S0-001
- **affected_module**：`src/auv_risk_rl/safety/validator.py`
- **root_cause**：上一版 `_is_state_within_operational_bounds()` 直接比较 AUV 中心位置与环境
  NED 边界，没有扣除 `auv_radius_m`。这与技术协议 Eq. (57)“AUV 完整包络在边界内”不一致。
- **fix**：环境边界统一移动到 `EnvironmentConfig`；env 与 validator 都使用
  `first_sphere_boundary_violation_fraction()` 检查完整物理球包络。
- **tests_added**：
  - `test_boundary_check_uses_complete_auv_sphere`
  - `test_validator_rejects_center_inside_box_when_auv_sphere_is_outside`
- **affected_outputs**：上一版若 AUV 中心仍在盒内但物理球已穿出边界，validator 可能错误认为操作约束
  合法。本版相关候选通过/拒绝结果会发生预期变化。

## INT-S0-002：Stage 0 缺少真实 env 集成证据

- **bug_id**：INT-S0-002
- **affected_module**：Stage 0 工程集成范围
- **root_cause**：上一交付只包含数学、KF、risk 与 validator 内核，因此 Stage-Gate 只能是
  `CONDITIONAL GO`；缺少真实环境推进、扫掠碰撞、传感延迟数据流和真值隔离集成测试。
- **fix**：新增 `env/`、`sensors/visibility.py`、`sensors/sonar.py`、`sensors/delay_queue.py`、
  `tracking/pose_history.py`、`tracking/track_manager.py` 和 `runtime/stage0_cycle.py`。
- **tests_added**：环境几何、世界推进、FOV/dropout/delay、measurement-time 位姿、数据隔离与
  Stage 0 control-cycle 集成测试。
- **affected_outputs**：Stage 0 在全量回归和 CODE QUALITY AUDIT 无 FAIL 时可按协议判定为
  `GO`，允许进入 Stage 1；该状态不代表导航性能或论文创新成立。

## MATH-S0-003：Stage 0 缺少有限时域失败尾项可执行实现

- **bug_id**：MATH-S0-003
- **affected_module**：有限时域成本数学账本
- **root_cause**：技术协议 Stage 0 Required outputs 包含终止尾项，但上一版仓库尚未将 Eq. (46)、
  Eq. (51) 写成可独立测试的 reference implementation。
- **fix**：新增 `src/auv_risk_rl/costs/finite_horizon.py`，实现有限时域归一化系数、失败吸收尾项
  和直接成本账本。
- **tests_added**：
  - `test_normalized_cost_constant_one_equals_one`
  - `test_failure_tail_matches_explicit_absorbing_sequence`
  - `test_discount_normalizer_matches_closed_geometric_sum`
- **affected_outputs**：后续 Stage 7 cost critic 必须与该 reference 数学语义一致，不得重新定义
  失败尾项或把成本预算解释为任务碰撞率。


## Phase B1：CM02–CM08现有内核修复（2026-09-18）

修复前重新复现：原34通过、外置Phase A 8失败/5通过。生产修改前已登记根因。
修复后：原测试未改；正式新增PA01–PA08对应测试和相关边界测试，共新增35项。
当前仓库69通过、外置原13通过；Ruff网络安装失败，静态门未完成，B1为CONDITIONAL PASS。

| ID | 最小修复 | 主要正式测试 |
|---|---|---|
|CM02|恢复两端均值和方向，精确零和North轴；风险使用AUV减障碍|test_segment_direction_matches_frozen_spec|
|CM03|v>0始终Gaussian；仅v=0/负向舍入校正为确定事件|test_positive_tiny_projection_variance_not_deterministic|
|CM04|只捕获明确数值类型，记录candidate/type/reason并排除无效候选|test_invalid_covariance_enters_unverified_fallback|
|CM05|记录首硬违反时刻，按最晚时间/U/距离/id选择backup|test_backup_fallback_uses_lexicographic_rule|
|CM06|全实际周期运行最小值，终止只计算执行前缀|test_world_reports_whole_interval_minimum_clearance；test_collision_clearance_excludes_unexecuted_suffix|
|CM07|component替代保留键module，旧字典显式迁移|test_logger_accepts_research_context|
|CM08|整数测量/到达/当前tick调度，发生时间仍供KF使用|test_one_step_delay_releases_on_next_control_tick|
|CM13|指纹排除派生文件、POSIX路径；AST核实测试名；Ruff缺失FAIL|tests/test_b1_audit_integrity.py|

无候选、异常fallback仍是未验证执行，不是安全保证。外置秒制测试通过兼容入口，
主控制链在真实整数tick上运行。没有更改动作库、风险预算、成本数学或动力学。
