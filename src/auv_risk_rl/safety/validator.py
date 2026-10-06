"""
基于固定候选动作与模型条件风险上界的在线安全验证器。

功能：
1. 对候选指令执行 Eq. (56) 固定动作滚动；
2. 检查操作边界和 Eq. (37)–(39) 风险上界；
3. 根据 Eq. (57)–(58) 保留名义动作或选择最小修改候选；
4. 在无通过候选时执行协议定义的主/备用 fallback。

说明：
验证器只提供短时域、模型条件动作检验，不提供递归可行性、不变集或全任务无碰撞证明。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.dynamics.auv_kinematics import rollout_constant_command
from auv_risk_rl.env.geometry import first_sphere_boundary_violation_fraction
from auv_risk_rl.env.world import _operation_boundary_fraction
from auv_risk_rl.exceptions import InvalidCovarianceError, NumericalRiskError
from auv_risk_rl.prediction.predictor import predict_position_distribution
from auv_risk_rl.risk.gaussian_bounds import aggregate_union_bounds, segment_collision_upper_bound
from auv_risk_rl.safety.candidates import build_candidate_actions
from auv_risk_rl.types import (
    AUVState,
    ControlCommand,
    RiskResult,
    TrackedObstacle,
    ValidationDecision,
)

_LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class CandidateEvaluation:
    """
    单个候选动作的验证结果。

    对应技术协议：
        Eq. (57)–(58)

    参数：
        action:
            被检查动作。
        candidate_id:
            固定候选序号，用于确定性打破平局。
        satisfies_operational_constraints:
            固定时域滚动是否始终满足确定性操作边界。
        risk_result:
            模型条件风险聚合结果。
        passes_validator:
            是否同时满足操作约束和短时风险预算。
        first_hard_constraint_violation_time_s:
            相对当前时刻的最早硬约束违反时间，单位 s；None 表示整个滚动内未违反。
            数值有效性独立由 risk_result 标记，不能因无硬约束违反而称为已验证。

    单位/坐标系：
        动作属于指令空间；风险无量纲；轨迹约束内部在 NED 检查。

    关键假设：
        A3–A10。

    重要限制：
        passes_validator 仅表示协议定义的短时模型检验通过。
    """

    action: ControlCommand
    candidate_id: int
    satisfies_operational_constraints: bool
    risk_result: RiskResult
    passes_validator: bool
    first_hard_constraint_violation_time_s: float | None = None

    @property
    def evaluation_status(self) -> str:
        """显式区分数值无效、确定性越界、风险超预算和通过；底层标志均保留。"""

        if not self.risk_result.is_numerically_valid:
            return "numerical_invalid"
        if not self.satisfies_operational_constraints:
            return "deterministic_boundary_violating"
        if not self.passes_validator:
            return "risk_invalid"
        return "verified"


def _action_to_normalized_vector(action: ControlCommand, config: ProjectConfig) -> np.ndarray:
    """把混合单位物理动作映射到无量纲 [-1,1] 坐标，供 Eq. (58) 距离比较。"""

    speed_span_mps = config.dynamics.max_surge_speed_mps - config.dynamics.min_surge_speed_mps
    normalized_speed = (
        2.0
        * (action.surge_speed_command_mps - config.dynamics.min_surge_speed_mps)
        / speed_span_mps
        - 1.0
    )
    normalized_yaw_rate = action.yaw_rate_command_rad_s / config.dynamics.max_yaw_rate_rad_s
    normalized_pitch_rate = (
        action.pitch_rate_command_rad_s / config.dynamics.max_pitch_rate_rad_s
    )
    return np.array(
        [normalized_speed, normalized_yaw_rate, normalized_pitch_rate],
        dtype=np.float64,
    )


def _is_state_within_operational_bounds(state: AUVState, config: ProjectConfig) -> bool:
    """
    检查位置、pitch、速度和角率是否满足冻结的确定性操作边界。

    位置约束检查 AUV 完整物理球包络，而不是只看中心点；这与 Eq. (57) 和真实 env 保持
    同一几何定义，避免 validator 通过而环境立即因包络越界终止。
    """

    lower_bound_ned_m = np.asarray(
        config.environment.position_lower_bound_ned_m,
        dtype=np.float64,
    )
    upper_bound_ned_m = np.asarray(
        config.environment.position_upper_bound_ned_m,
        dtype=np.float64,
    )
    position_violation_fraction = first_sphere_boundary_violation_fraction(
        start_position_ned_m=state.position_ned_m,
        end_position_ned_m=state.position_ned_m,
        lower_bound_ned_m=lower_bound_ned_m,
        upper_bound_ned_m=upper_bound_ned_m,
        sphere_radius_m=config.risk.auv_radius_m,
    )
    is_position_valid = position_violation_fraction is None
    return bool(
        is_position_valid
        and abs(state.pitch_rad) <= config.dynamics.max_pitch_rad
        and config.dynamics.min_surge_speed_mps
        <= state.surge_speed_mps
        <= config.dynamics.max_surge_speed_mps
        and abs(state.yaw_rate_rad_s) <= config.dynamics.max_yaw_rate_rad_s
        and abs(state.pitch_rate_rad_s) <= config.dynamics.max_pitch_rate_rad_s
    )


def _rollout_candidate_states(
    auv_state: AUVState,
    action: ControlCommand,
    config: ProjectConfig,
) -> list[AUVState]:
    """
    按 Eq. (56) 生成固定指令延续轨迹。

    主验证器使用零流，以保持与技术协议 v2.0 的主设置一致；未知海流留到 OOD 压力测试。
    """

    return rollout_constant_command(
        initial_state=auv_state,
        command=action,
        current_velocity_ned_mps=np.zeros(3, dtype=np.float64),
        horizon_s=config.validator.validation_horizon_s,
        integration_dt_s=config.dynamics.integration_dt_s,
        config=config.dynamics,
    )


def _first_hard_constraint_violation_time_s(
    states: list[AUVState], config: ProjectConfig
) -> float | None:
    """按现有真值线段事件函数定位首硬约束违反；时间为相对当前的秒数。"""

    if not _is_state_within_operational_bounds(states[0], config):
        return 0.0
    for segment_index, (start, end) in enumerate(zip(states, states[1:], strict=False)):
        # 复用已有环境几何，避免 backup 与真实终止使用两套边界定义。
        fraction = _operation_boundary_fraction(start, end, config)
        if fraction is not None:
            return (segment_index + fraction) * config.dynamics.integration_dt_s
    return None


def _predict_obstacle_nodes(
    tracked_obstacle: TrackedObstacle,
    current_timestamp_s: float,
    node_count: int,
    config: ProjectConfig,
) -> list:
    """
    为一个已维护目标预测与 AUV 积分节点对齐的位置分布。

    若 KF 后验时间晚于验证节点，说明时间接口错误；禁止通过负预测时域或未来信息修补。
    """

    spectral_density_ned_m2_s3 = np.diag(
        np.asarray(config.kf.acceleration_spectral_density_m2_s3, dtype=np.float64)
    )
    predictions = []
    for node_index in range(node_count):
        node_timestamp_s = current_timestamp_s + node_index * config.dynamics.integration_dt_s
        prediction_horizon_s = node_timestamp_s - tracked_obstacle.track_state.state_timestamp_s
        if prediction_horizon_s < 0.0:
            raise NumericalRiskError(
                "KF 后验时间戳晚于验证节点："
                f"track_time={tracked_obstacle.track_state.state_timestamp_s}, "
                f"node_time={node_timestamp_s}。"
            )
        predictions.append(
            predict_position_distribution(
                track_state=tracked_obstacle.track_state,
                prediction_horizon_s=prediction_horizon_s,
                acceleration_spectral_density_ned_m2_s3=spectral_density_ned_m2_s3,
                symmetry_tolerance=config.kf.covariance_symmetry_tolerance,
                psd_tolerance=config.kf.covariance_psd_tolerance,
            )
        )
    return predictions


def _calculate_rollout_risk(
    rollout_states: list[AUVState],
    current_timestamp_s: float,
    tracked_obstacles: list[TrackedObstacle],
    config: ProjectConfig,
) -> RiskResult:
    """
    计算一个候选轨迹对全部已维护目标的 Eq. (37)–(39) 联合风险上界。

    返回的 U 可能大于 1；保留未截断值用于候选排序和保守性诊断。
    """

    component_upper_bounds: list[float] = []
    for tracked_obstacle in tracked_obstacles:
        safe_radius_m = (
            config.risk.auv_radius_m
            + tracked_obstacle.radius_m
            + config.risk.extra_margin_m
        )
        predictions = _predict_obstacle_nodes(
            tracked_obstacle,
            current_timestamp_s,
            len(rollout_states),
            config,
        )
        for segment_index in range(len(rollout_states) - 1):
            relative_start_ned_m = (
                rollout_states[segment_index].position_ned_m
                - predictions[segment_index].position_mean_ned_m
            )
            relative_end_ned_m = (
                rollout_states[segment_index + 1].position_ned_m
                - predictions[segment_index + 1].position_mean_ned_m
            )
            component_upper_bounds.append(
                segment_collision_upper_bound(
                    relative_mean_start_ned_m=relative_start_ned_m,
                    relative_mean_end_ned_m=relative_end_ned_m,
                    position_covariance_start_ned_m2=(
                        predictions[segment_index].position_covariance_ned_m2
                    ),
                    position_covariance_end_ned_m2=(
                        predictions[segment_index + 1].position_covariance_ned_m2
                    ),
                    safe_radius_m=safe_radius_m,
                    mean_norm_tolerance_m=config.risk.mean_norm_tolerance_m,
                    variance_tolerance_m2=config.risk.variance_tolerance_m2,
                    covariance_symmetry_tolerance=config.kf.covariance_symmetry_tolerance,
                    covariance_psd_tolerance=config.kf.covariance_psd_tolerance,
                )
            )
    return aggregate_union_bounds(component_upper_bounds)


def evaluate_candidate_action(
    auv_state: AUVState,
    current_timestamp_s: float,
    action: ControlCommand,
    candidate_id: int,
    tracked_obstacles: list[TrackedObstacle],
    config: ProjectConfig,
) -> CandidateEvaluation:
    """
    对单个固定动作延续执行短时安全验证。

    对应技术协议：
        Eq. (27)、Eq. (37)–(39)、Eq. (56)–(57)

    参数：
        auv_state: 当前 AUV 八维状态；位置 NED。
        current_timestamp_s: 当前世界时刻，单位 s。
        action: 待验证物理动作。
        candidate_id: 固定候选序号，无单位。
        tracked_obstacles: 全部已维护目标；包含 KF 后验和物理包络半径。
        config: Stage 0 冻结配置。

    返回：
        CandidateEvaluation。

    shape/单位/坐标系：
        AUV/障碍位置均在 NED，风险无量纲，时间单位 s。

    关键假设：
        A3–A10。

    重要限制：
        只覆盖已维护目标；未知海流、自身定位误差和错误关联不由障碍 KF 协方差覆盖。
    """

    rollout_states = _rollout_candidate_states(auv_state, action, config)
    first_violation_time_s = _first_hard_constraint_violation_time_s(rollout_states, config)
    satisfies_operational_constraints = first_violation_time_s is None
    try:
        risk_result = _calculate_rollout_risk(
            rollout_states,
            current_timestamp_s,
            tracked_obstacles,
            config,
        )
    except (InvalidCovarianceError, NumericalRiskError, np.linalg.LinAlgError) as error:
        # 数值异常不能被当成安全；显式标记无效，由 fallback 规则处理并留存原因。
        _LOGGER.warning(
            "候选数值无效：candidate_id=%s exception_type=%s reason=%s",
            candidate_id, type(error).__name__, str(error),
            extra={
                "component": "safety.validator",
                "candidate_id": candidate_id,
                "exception_type": type(error).__name__,
                "reason": str(error),
            },
        )
        risk_result = RiskResult(
            risk_upper_bound=1.0,
            untruncated_union_bound=float("inf"),
            is_numerically_valid=False,
            detail=f"numerical_error:{type(error).__name__}:{error}",
        )
    passes_validator = bool(
        satisfies_operational_constraints
        and risk_result.is_numerically_valid
        and risk_result.risk_upper_bound <= config.risk.short_horizon_risk_budget
    )
    return CandidateEvaluation(
        action=action,
        candidate_id=candidate_id,
        satisfies_operational_constraints=satisfies_operational_constraints,
        risk_result=risk_result,
        passes_validator=passes_validator,
        first_hard_constraint_violation_time_s=first_violation_time_s,
    )


def _normalized_action_distance(
    first_action: ControlCommand,
    second_action: ControlCommand,
    config: ProjectConfig,
) -> float:
    """计算 Eq. (58) 使用的无量纲动作欧氏距离，避免直接混加 m/s 与 rad/s。"""

    return float(
        np.linalg.norm(
            _action_to_normalized_vector(first_action, config)
            - _action_to_normalized_vector(second_action, config)
        )
    )


def _decision_from_evaluation(
    nominal_action: ControlCommand,
    selected: CandidateEvaluation,
    decision_type: Literal["nominal", "modified", "fallback"],
    reason: str,
) -> ValidationDecision:
    """把候选评价稳定转换为对外 ValidationDecision，避免选择分支重复组装字段。"""

    return ValidationDecision(
        nominal_action=nominal_action,
        executed_action=selected.action,
        decision_type=decision_type,
        selected_risk_upper_bound=selected.risk_result.risk_upper_bound,
        selected_untruncated_union_bound=selected.risk_result.untruncated_union_bound,
        reason=reason,
    )


def _select_nearest_passing_evaluation(
    evaluations: list[CandidateEvaluation],
    nominal_action: ControlCommand,
    config: ProjectConfig,
) -> CandidateEvaluation | None:
    """按 Eq. (58) 的归一化动作距离、未截断风险 U 和候选编号选择通过候选。"""

    passing = [evaluation for evaluation in evaluations if evaluation.passes_validator]
    if not passing:
        return None
    return min(
        passing,
        key=lambda evaluation: (
            _normalized_action_distance(evaluation.action, nominal_action, config),
            evaluation.risk_result.untruncated_union_bound,
            evaluation.candidate_id,
        ),
    )


def _select_primary_fallback_evaluation(
    evaluations: list[CandidateEvaluation],
    nominal_action: ControlCommand,
    config: ProjectConfig,
) -> CandidateEvaluation | None:
    """在满足确定性操作边界的候选中按最低 U、名义距离和候选编号选择主 fallback。"""

    hard_valid = [
        evaluation for evaluation in evaluations
        if evaluation.satisfies_operational_constraints
        and evaluation.risk_result.is_numerically_valid
    ]
    if not hard_valid:
        return None
    return min(
        hard_valid,
        key=lambda evaluation: (
            evaluation.risk_result.untruncated_union_bound,
            _normalized_action_distance(evaluation.action, nominal_action, config),
            evaluation.candidate_id,
        ),
    )


def _select_backup_fallback_evaluation(
    evaluations: list[CandidateEvaluation],
    nominal_action: ControlCommand,
    config: ProjectConfig,
) -> CandidateEvaluation | None:
    """按最晚硬违反、最低未截断U、最小名义距离、固定编号选择未验证备份。"""

    calculable = [
        evaluation for evaluation in evaluations
        if evaluation.risk_result.is_numerically_valid
        and evaluation.first_hard_constraint_violation_time_s is not None
    ]
    if not calculable:
        return None
    return min(
        calculable,
        key=lambda evaluation: (
            -float(evaluation.first_hard_constraint_violation_time_s),
            evaluation.risk_result.untruncated_union_bound,
            _normalized_action_distance(evaluation.action, nominal_action, config),
            evaluation.candidate_id,
        ),
    )


def validate_nominal_action(
    auv_state: AUVState,
    current_timestamp_s: float,
    nominal_action: ControlCommand,
    previous_executed_action: ControlCommand,
    tracked_obstacles: list[TrackedObstacle],
    config: ProjectConfig,
) -> ValidationDecision:
    """
    验证名义动作，并在必要时选择最近的可验证备选动作。

    对应技术协议：
        Eq. (55)–(58)

    参数：
        auv_state: 当前 AUV 状态；位置 NED。
        current_timestamp_s: 当前世界时刻，单位 s。
        nominal_action: 名义物理动作。
        previous_executed_action: 上一实际执行动作。
        tracked_obstacles: 全部已维护目标，不只网络前 K 个槽位。
        config: 冻结项目配置。

    返回：
        ValidationDecision：名义通过、修改或 fallback。

    shape/单位/坐标系：
        动作 3 维混合物理单位；风险无量纲；轨迹与障碍计算统一 NED。

    关键假设：
        A3–A10。

    重要限制：
        候选选择是工程启发式，不是连续 QP；fallback 不构成安全保证。
    """

    candidates = build_candidate_actions(
        nominal_action,
        previous_executed_action,
        config.validator,
    )
    evaluations = [
        evaluate_candidate_action(
            auv_state,
            current_timestamp_s,
            candidate_action,
            candidate_id,
            tracked_obstacles,
            config,
        )
        for candidate_id, candidate_action in enumerate(candidates)
    ]
    nominal_evaluation = evaluations[0]
    if nominal_evaluation.passes_validator:
        return _decision_from_evaluation(
            nominal_action,
            nominal_evaluation,
            "nominal",
            "nominal_passed_validator",
        )

    selected_passing = _select_nearest_passing_evaluation(evaluations, nominal_action, config)
    if selected_passing is not None:
        return _decision_from_evaluation(
            nominal_action,
            selected_passing,
            "modified",
            "nearest_validated_candidate",
        )

    selected_fallback = _select_primary_fallback_evaluation(evaluations, nominal_action, config)
    if selected_fallback is not None:
        return _decision_from_evaluation(
            nominal_action,
            selected_fallback,
            "fallback",
            "no_candidate_passed_risk_budget",
        )

    selected_backup = _select_backup_fallback_evaluation(evaluations, nominal_action, config)
    if selected_backup is not None:
        return _decision_from_evaluation(
            nominal_action,
            selected_backup,
            "fallback",
            "backup_latest_hard_constraint_violation_unverified",
        )

    # 没有任何有效评估才使用最后定义域内指令；不是安全停船，也不终止真实 episode。
    final_fallback = ControlCommand(config.dynamics.min_surge_speed_mps, 0.0, 0.0)
    return ValidationDecision(
        nominal_action=nominal_action,
        executed_action=final_fallback,
        decision_type="fallback",
        selected_risk_upper_bound=1.0,
        selected_untruncated_union_bound=float("inf"),
        reason="no_valid_evaluation_unverified_fallback",
    )
