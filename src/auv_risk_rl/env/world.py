"""
Stage 0 无 RL 世界真值推进与终止事件模块。

功能：
1. 使用与 validator 共用的 AUV RK2 动力学推进真实 AUV；
2. 使用严格 CV 模型推进障碍真值；
3. 在每个 0.05 s 积分小步内做扫掠球碰撞与完整 AUV 包络边界检测；
4. 定位最早事件并区分 collision、boundary、success、timeout。

说明：
本模块不负责传感器噪声、KF、策略 Observation 或安全验证器动作选择。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.dynamics.auv_kinematics import integrate_rk2_step, validate_control_command
from auv_risk_rl.env.geometry import (
    first_moving_sphere_collision_fraction,
    first_scalar_limit_violation_fraction,
    first_sphere_boundary_violation_fraction,
    moving_sphere_minimum_clearance,
)
from auv_risk_rl.env.obstacles import propagate_cv_obstacle_truth
from auv_risk_rl.env.types import EnvironmentEvent, TerminationReason, WorldStepResult
from auv_risk_rl.exceptions import InvalidEnvironmentStateError
from auv_risk_rl.types import AUVState, ControlCommand, GroundTruthObstacleState


@dataclass(frozen=True)
class _CandidateEvent:
    """环境内部最早事件候选；只在一个积分小步内使用。"""

    fraction: float
    priority: int
    event: EnvironmentEvent


@dataclass(frozen=True)
class _IntegrationStepProposal:
    """一个积分小步的未提交真值推进结果。"""

    auv_end_state: AUVState
    obstacle_end_states: tuple[GroundTruthObstacleState, ...]
    minimum_clearance_m: float
    event: _CandidateEvent | None


def _interpolate_auv_state(
    start_state: AUVState,
    end_state: AUVState,
    fraction: float,
) -> AUVState:
    """
    在线性事件时刻插值 AUV 状态。

    位置插值与协议 A8 完全一致；姿态/响应状态插值只用于把终止快照对齐到同一事件时刻，
    不替代 RK2 连续轨迹模型。
    """

    interpolation_fraction = float(np.clip(fraction, 0.0, 1.0))
    return AUVState(
        position_ned_m=(
            start_state.position_ned_m
            + interpolation_fraction * (end_state.position_ned_m - start_state.position_ned_m)
        ),
        yaw_rad=float(
            start_state.yaw_rad
            + interpolation_fraction * (end_state.yaw_rad - start_state.yaw_rad)
        ),
        pitch_rad=float(
            start_state.pitch_rad
            + interpolation_fraction * (end_state.pitch_rad - start_state.pitch_rad)
        ),
        surge_speed_mps=float(
            start_state.surge_speed_mps
            + interpolation_fraction * (end_state.surge_speed_mps - start_state.surge_speed_mps)
        ),
        yaw_rate_rad_s=float(
            start_state.yaw_rate_rad_s
            + interpolation_fraction * (end_state.yaw_rate_rad_s - start_state.yaw_rate_rad_s)
        ),
        pitch_rate_rad_s=float(
            start_state.pitch_rate_rad_s
            + interpolation_fraction
            * (end_state.pitch_rate_rad_s - start_state.pitch_rate_rad_s)
        ),
    )


def _interpolate_obstacle_state(
    start_state: GroundTruthObstacleState,
    end_state: GroundTruthObstacleState,
    fraction: float,
) -> GroundTruthObstacleState:
    """将严格 CV 障碍真值插值到积分小步内的事件时刻。"""

    interpolation_fraction = float(np.clip(fraction, 0.0, 1.0))
    return GroundTruthObstacleState(
        obstacle_id=start_state.obstacle_id,
        position_ned_m=(
            start_state.position_ned_m
            + interpolation_fraction * (end_state.position_ned_m - start_state.position_ned_m)
        ),
        velocity_ned_mps=start_state.velocity_ned_mps.copy(),
        radius_m=start_state.radius_m,
    )


def _operation_boundary_fraction(
    start_state: AUVState,
    end_state: AUVState,
    config: ProjectConfig,
) -> float | None:
    """
    求一个积分小步内最早确定性操作边界违规时刻。

    空间边界检查 AUV 完整物理球包络；pitch、速度和 Euler 角率使用相同小步线性事件定位。
    """

    lower_bound_ned_m = np.asarray(config.environment.position_lower_bound_ned_m, dtype=np.float64)
    upper_bound_ned_m = np.asarray(config.environment.position_upper_bound_ned_m, dtype=np.float64)
    candidate_fractions: list[float] = []
    position_fraction = first_sphere_boundary_violation_fraction(
        start_state.position_ned_m,
        end_state.position_ned_m,
        lower_bound_ned_m,
        upper_bound_ned_m,
        config.risk.auv_radius_m,
    )
    if position_fraction is not None:
        candidate_fractions.append(position_fraction)

    scalar_checks = (
        (
            start_state.pitch_rad,
            end_state.pitch_rad,
            -config.dynamics.max_pitch_rad,
            config.dynamics.max_pitch_rad,
        ),
        (
            start_state.surge_speed_mps,
            end_state.surge_speed_mps,
            config.dynamics.min_surge_speed_mps,
            config.dynamics.max_surge_speed_mps,
        ),
        (
            start_state.yaw_rate_rad_s,
            end_state.yaw_rate_rad_s,
            -config.dynamics.max_yaw_rate_rad_s,
            config.dynamics.max_yaw_rate_rad_s,
        ),
        (
            start_state.pitch_rate_rad_s,
            end_state.pitch_rate_rad_s,
            -config.dynamics.max_pitch_rate_rad_s,
            config.dynamics.max_pitch_rate_rad_s,
        ),
    )
    for start_value, end_value, lower_limit, upper_limit in scalar_checks:
        fraction = first_scalar_limit_violation_fraction(
            start_value,
            end_value,
            lower_limit,
            upper_limit,
        )
        if fraction is not None:
            candidate_fractions.append(fraction)
    return min(candidate_fractions) if candidate_fractions else None


def _goal_entry_fraction(
    start_position_ned_m: np.ndarray,
    end_position_ned_m: np.ndarray,
    goal_position_ned_m: np.ndarray,
    goal_radius_m: float,
) -> float | None:
    """
    求 AUV 中心线段首次进入目标球的时刻。

    该任务到达定义属于协议预注册仿真设置，不是安全几何或概率公式。
    """

    return first_moving_sphere_collision_fraction(
        auv_start_position_ned_m=start_position_ned_m,
        auv_end_position_ned_m=end_position_ned_m,
        obstacle_start_position_ned_m=goal_position_ned_m,
        obstacle_end_position_ned_m=goal_position_ned_m,
        combined_physical_radius_m=goal_radius_m,
    )


def _minimum_clearance_over_step(
    auv_start_state: AUVState,
    auv_end_state: AUVState,
    obstacle_start_states: tuple[GroundTruthObstacleState, ...],
    obstacle_end_states: tuple[GroundTruthObstacleState, ...],
    auv_radius_m: float,
) -> float:
    """计算一个积分小步内对全部障碍的最小真实物理净间距。"""

    if not obstacle_start_states:
        return float("inf")
    end_by_id = {state.obstacle_id: state for state in obstacle_end_states}
    minimum_clearance_m = float("inf")
    for obstacle_start_state in obstacle_start_states:
        obstacle_end_state = end_by_id[obstacle_start_state.obstacle_id]
        clearance_m, _ = moving_sphere_minimum_clearance(
            auv_start_state.position_ned_m,
            auv_end_state.position_ned_m,
            obstacle_start_state.position_ned_m,
            obstacle_end_state.position_ned_m,
            auv_radius_m + obstacle_start_state.radius_m,
        )
        minimum_clearance_m = min(minimum_clearance_m, clearance_m)
    return minimum_clearance_m


class AUVWorld:
    """
    无 RL 的 AUV + 动态障碍真值世界。

    对应技术协议：
        Eq. (9)、Eq. (11)–(12)、Eq. (27)、Eq. (57) 与第 15、25.1–25.2 章。

    输入/输出：
        输入执行动作 ControlCommand；输出 WorldStepResult。

    shape/单位/坐标系：
        AUV/障碍位置均为 shape=(3,) NED 米制向量；速度为 m/s；时间为 s。

    关键假设：
        A1、A3、A8；Stage 0 主环境零海流，障碍严格 CV。

    重要限制：
        本类不是 Gymnasium Env，也不包含 reward、Observation、SAC 或真实声呐信号链。
    """

    def __init__(
        self,
        config: ProjectConfig,
        initial_auv_state: AUVState,
        initial_obstacle_states: tuple[GroundTruthObstacleState, ...],
        goal_position_ned_m: np.ndarray,
        initial_timestamp_s: float = 0.0,
    ) -> None:
        """
        初始化无 RL 世界真值。

        参数：
            config:
                Stage 0 冻结配置。
            initial_auv_state:
                AUV 初始八维状态，位置 NED。
            initial_obstacle_states:
                障碍真值元组；位置/速度 NED。
            goal_position_ned_m:
                目标位置，shape=(3,)，单位 m，NED。
            initial_timestamp_s:
                初始世界时间，单位 s。

        返回：
            无。

        关键假设：
            初始状态合法且不存在重复 obstacle_id。

        重要限制：
            初始真值只存于环境内部；策略侧数据需通过专用感知接口构造。
        """

        if goal_position_ned_m.shape != (3,) or not np.all(np.isfinite(goal_position_ned_m)):
            raise InvalidEnvironmentStateError(
                "goal_position_ned_m 必须是有限 shape=(3,) NED 向量。"
            )
        obstacle_ids = [state.obstacle_id for state in initial_obstacle_states]
        if len(obstacle_ids) != len(set(obstacle_ids)):
            raise InvalidEnvironmentStateError(f"障碍 ID 必须唯一，actual={obstacle_ids}。")
        if not np.isfinite(initial_timestamp_s):
            raise InvalidEnvironmentStateError("initial_timestamp_s 必须有限。")
        self._config = config
        self._auv_state = initial_auv_state
        self._obstacle_states = tuple(initial_obstacle_states)
        self._goal_position_ned_m = goal_position_ned_m.astype(np.float64, copy=True)
        self._timestamp_s = float(initial_timestamp_s)
        self._control_step_index = 0
        self._is_terminated = False
        self._validate_initial_state()

    @property
    def timestamp_s(self) -> float:
        """返回当前世界时间，单位 s。"""

        return self._timestamp_s

    @property
    def auv_state(self) -> AUVState:
        """返回当前 AUV 真值状态；仅环境/评价侧可直接使用。"""

        return self._auv_state

    @property
    def obstacle_states(self) -> tuple[GroundTruthObstacleState, ...]:
        """返回当前障碍真值元组；禁止直接进入普通策略或 KF。"""

        return self._obstacle_states

    @property
    def control_step_index(self) -> int:
        """返回已完成的控制周期计数。"""

        return self._control_step_index

    def _validate_initial_state(self) -> None:
        """检查初始 AUV 完整包络、响应状态和目标位置是否处于协议定义域。"""

        lower_bound_ned_m = np.asarray(
            self._config.environment.position_lower_bound_ned_m,
            dtype=np.float64,
        )
        upper_bound_ned_m = np.asarray(
            self._config.environment.position_upper_bound_ned_m,
            dtype=np.float64,
        )
        position_violation = first_sphere_boundary_violation_fraction(
            self._auv_state.position_ned_m,
            self._auv_state.position_ned_m,
            lower_bound_ned_m,
            upper_bound_ned_m,
            self._config.risk.auv_radius_m,
        )
        if position_violation is not None:
            raise InvalidEnvironmentStateError("AUV 初始完整物理包络已越出环境边界。")
        if abs(self._auv_state.pitch_rad) > self._config.dynamics.max_pitch_rad:
            raise InvalidEnvironmentStateError("AUV 初始 pitch 超出操作边界。")
        if not (
            self._config.dynamics.min_surge_speed_mps
            <= self._auv_state.surge_speed_mps
            <= self._config.dynamics.max_surge_speed_mps
        ):
            raise InvalidEnvironmentStateError("AUV 初始 surge_speed_mps 超出操作边界。")
        if np.any(self._goal_position_ned_m < lower_bound_ned_m) or np.any(
            self._goal_position_ned_m > upper_bound_ned_m
        ):
            raise InvalidEnvironmentStateError("目标中心位置必须位于环境 NED 边界内。")

    def _collision_candidates(
        self,
        auv_start_state: AUVState,
        auv_end_state: AUVState,
        obstacle_start_states: tuple[GroundTruthObstacleState, ...],
        obstacle_end_states: tuple[GroundTruthObstacleState, ...],
        step_start_timestamp_s: float,
        minimum_clearance_m: float,
    ) -> list[_CandidateEvent]:
        """构造一个积分小步内全部真实物理碰撞候选事件。"""

        end_by_id = {state.obstacle_id: state for state in obstacle_end_states}
        candidates: list[_CandidateEvent] = []
        for obstacle_start_state in obstacle_start_states:
            obstacle_end_state = end_by_id[obstacle_start_state.obstacle_id]
            fraction = first_moving_sphere_collision_fraction(
                auv_start_state.position_ned_m,
                auv_end_state.position_ned_m,
                obstacle_start_state.position_ned_m,
                obstacle_end_state.position_ned_m,
                self._config.risk.auv_radius_m + obstacle_start_state.radius_m,
            )
            if fraction is None:
                continue
            candidates.append(
                _CandidateEvent(
                    fraction=fraction,
                    priority=0,
                    event=EnvironmentEvent(
                        reason="collision",
                        event_timestamp_s=(
                            step_start_timestamp_s
                            + fraction * self._config.dynamics.integration_dt_s
                        ),
                        event_fraction_of_integration_step=fraction,
                        obstacle_id=obstacle_start_state.obstacle_id,
                        minimum_clearance_m=minimum_clearance_m,
                        detail="physical_swept_sphere_collision",
                    ),
                )
            )
        return candidates

    def _boundary_and_success_candidates(
        self,
        auv_start_state: AUVState,
        auv_end_state: AUVState,
        step_start_timestamp_s: float,
        minimum_clearance_m: float,
    ) -> list[_CandidateEvent]:
        """构造边界与到达事件候选；同一 fraction 时安全失败优先于 success。"""

        candidates: list[_CandidateEvent] = []
        boundary_fraction = _operation_boundary_fraction(
            auv_start_state,
            auv_end_state,
            self._config,
        )
        if boundary_fraction is not None:
            candidates.append(
                self._make_noncollision_event(
                    "boundary",
                    boundary_fraction,
                    1,
                    step_start_timestamp_s,
                    minimum_clearance_m,
                    "operational_boundary_violation",
                )
            )
        success_fraction = _goal_entry_fraction(
            auv_start_state.position_ned_m,
            auv_end_state.position_ned_m,
            self._goal_position_ned_m,
            self._config.environment.goal_radius_m,
        )
        if success_fraction is not None:
            candidates.append(
                self._make_noncollision_event(
                    "success",
                    success_fraction,
                    2,
                    step_start_timestamp_s,
                    minimum_clearance_m,
                    "goal_region_entered",
                )
            )
        return candidates

    def _make_noncollision_event(
        self,
        reason: TerminationReason,
        fraction: float,
        priority: int,
        step_start_timestamp_s: float,
        minimum_clearance_m: float,
        detail: str,
    ) -> _CandidateEvent:
        """统一构造不带 obstacle_id 的边界/成功候选事件。"""

        return _CandidateEvent(
            fraction=fraction,
            priority=priority,
            event=EnvironmentEvent(
                reason=reason,
                event_timestamp_s=(
                    step_start_timestamp_s
                    + fraction * self._config.dynamics.integration_dt_s
                ),
                event_fraction_of_integration_step=fraction,
                obstacle_id=None,
                minimum_clearance_m=minimum_clearance_m,
                detail=detail,
            ),
        )

    def _propose_integration_step(
        self,
        auv_start_state: AUVState,
        obstacle_start_states: tuple[GroundTruthObstacleState, ...],
        executed_action: ControlCommand,
        step_start_timestamp_s: float,
    ) -> _IntegrationStepProposal:
        """计算一个积分小步的未提交真值终点与最早事件。"""

        auv_end_state = integrate_rk2_step(
            state=auv_start_state,
            command=executed_action,
            current_velocity_ned_mps=np.zeros(3, dtype=np.float64),
            integration_dt_s=self._config.dynamics.integration_dt_s,
            config=self._config.dynamics,
        )
        obstacle_end_states = tuple(
            propagate_cv_obstacle_truth(
                obstacle_state,
                self._config.dynamics.integration_dt_s,
            )
            for obstacle_state in obstacle_start_states
        )
        minimum_clearance_m = _minimum_clearance_over_step(
            auv_start_state,
            auv_end_state,
            obstacle_start_states,
            obstacle_end_states,
            self._config.risk.auv_radius_m,
        )
        candidates = self._collision_candidates(
            auv_start_state,
            auv_end_state,
            obstacle_start_states,
            obstacle_end_states,
            step_start_timestamp_s,
            minimum_clearance_m,
        )
        candidates.extend(
            self._boundary_and_success_candidates(
                auv_start_state,
                auv_end_state,
                step_start_timestamp_s,
                minimum_clearance_m,
            )
        )
        earliest = (
            min(candidates, key=lambda item: (item.fraction, item.priority))
            if candidates
            else None
        )
        return _IntegrationStepProposal(
            auv_end_state=auv_end_state,
            obstacle_end_states=obstacle_end_states,
            minimum_clearance_m=minimum_clearance_m,
            event=earliest,
        )

    def _commit_termination(
        self,
        auv_start_state: AUVState,
        obstacle_start_states: tuple[GroundTruthObstacleState, ...],
        proposal: _IntegrationStepProposal,
        running_min_clearance_m: float,
    ) -> WorldStepResult:
        """提交最早事件快照；净间距只覆盖此前已执行部分及当前事件前缀。"""

        if proposal.event is None:
            raise InvalidEnvironmentStateError("内部错误：无事件 proposal 不能提交终止。")
        fraction = proposal.event.fraction
        self._auv_state = _interpolate_auv_state(
            auv_start_state,
            proposal.auv_end_state,
            fraction,
        )
        end_by_id = {state.obstacle_id: state for state in proposal.obstacle_end_states}
        self._obstacle_states = tuple(
            _interpolate_obstacle_state(
                obstacle_state,
                end_by_id[obstacle_state.obstacle_id],
                fraction,
            )
            for obstacle_state in obstacle_start_states
        )
        executed_prefix_clearance_m = _minimum_clearance_over_step(
            auv_start_state,
            self._auv_state,
            obstacle_start_states,
            self._obstacle_states,
            self._config.risk.auv_radius_m,
        )
        actual_event = replace(
            proposal.event.event,
            minimum_clearance_m=min(running_min_clearance_m, executed_prefix_clearance_m),
        )
        self._timestamp_s = proposal.event.event.event_timestamp_s
        self._control_step_index += 1
        self._is_terminated = True
        return WorldStepResult(
            auv_state=self._auv_state,
            obstacle_states=self._obstacle_states,
            timestamp_s=self._timestamp_s,
            control_step_index=self._control_step_index,
            is_terminated=True,
            event=actual_event,
        )

    def _finalize_full_control_cycle(self, minimum_clearance_m: float) -> WorldStepResult:
        """完成无中途事件的控制周期，并在计划时域耗尽时产生 timeout。"""

        self._control_step_index += 1
        if self._control_step_index >= self._config.environment.max_episode_control_steps:
            self._is_terminated = True
            reason: TerminationReason = "timeout"
            detail = "planned_horizon_exhausted"
        else:
            reason = "none"
            detail = "control_cycle_completed"
        event = EnvironmentEvent(
            reason=reason,
            event_timestamp_s=self._timestamp_s,
            event_fraction_of_integration_step=1.0,
            obstacle_id=None,
            minimum_clearance_m=minimum_clearance_m,
            detail=detail,
        )
        return WorldStepResult(
            auv_state=self._auv_state,
            obstacle_states=self._obstacle_states,
            timestamp_s=self._timestamp_s,
            control_step_index=self._control_step_index,
            is_terminated=self._is_terminated,
            event=event,
        )

    def step(self, executed_action: ControlCommand) -> WorldStepResult:
        """
        使用已决定的执行动作推进一个控制周期或推进到最早终止事件。

        对应技术协议：
            Eq. (9)、Eq. (11)–(12)、Eq. (27) 与第 15 章步骤 5–7。

        参数：
            executed_action:
                已由上层决定的物理执行动作；单位 m/s、rad/s、rad/s。

        返回：
            WorldStepResult：真值推进结果和最早事件。

        shape/单位/坐标系：
            所有位置/速度统一 NED；积分与事件时间单位 s。

        关键假设：
            A1、A3、A8；主环境零流。

        重要限制：
            验证失败不会在此直接结束 episode；只有真实碰撞、操作边界、成功或超时才终止。
        """

        if self._is_terminated:
            raise InvalidEnvironmentStateError("环境已经终止，禁止继续 step。")
        validate_control_command(executed_action, self._config.dynamics)
        integration_step_count = int(
            round(self._config.dynamics.control_dt_s / self._config.dynamics.integration_dt_s)
        )
        running_min_clearance_m = float("inf")
        for _ in range(integration_step_count):
            auv_start_state = self._auv_state
            obstacle_start_states = self._obstacle_states
            proposal = self._propose_integration_step(
                auv_start_state,
                obstacle_start_states,
                executed_action,
                self._timestamp_s,
            )
            if proposal.event is not None:
                return self._commit_termination(
                    auv_start_state,
                    obstacle_start_states,
                    proposal,
                    running_min_clearance_m,
                )
            # 仅提交完整执行小步后更新运行最小值；有事件的未执行后缀绝不计入。
            running_min_clearance_m = min(running_min_clearance_m, proposal.minimum_clearance_m)
            self._auv_state = proposal.auv_end_state
            self._obstacle_states = proposal.obstacle_end_states
            self._timestamp_s += self._config.dynamics.integration_dt_s
        return self._finalize_full_control_cycle(running_min_clearance_m)
