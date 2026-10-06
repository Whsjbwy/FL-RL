"""验证无 RL 世界推进、事件终止以及 env/validator 共享动力学一致性。"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.dynamics.auv_kinematics import rollout_constant_command
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.safety.validator import evaluate_candidate_action
from auv_risk_rl.types import AUVState, ControlCommand, GroundTruthObstacleState


def _base_auv_state(position_ned_m: np.ndarray, speed_mps: float = 0.8) -> AUVState:
    """构造测试用水平 AUV 状态，所有单位遵循 AUVState。"""

    return AUVState(
        position_ned_m=position_ned_m.astype(np.float64),
        yaw_rad=0.0,
        pitch_rad=0.0,
        surge_speed_mps=speed_mps,
        yaw_rate_rad_s=0.0,
        pitch_rate_rad_s=0.0,
    )


def test_world_matches_shared_rk2_rollout_for_one_control_cycle(project_config) -> None:
    """
    验证真实 env 与 validator 共用的 RK2 实现产生完全一致的一控制周期无障碍轨迹终点。

    若失败说明环境和安全验证器已经出现模型复制或时间步语义分叉，Stage 0 不允许 GO。
    """

    initial_state = _base_auv_state(np.array([20.0, 20.0, 20.0]), speed_mps=0.8)
    command = ControlCommand(0.9, 0.05, -0.02)
    expected_state = rollout_constant_command(
        initial_state=initial_state,
        command=command,
        current_velocity_ned_mps=np.zeros(3, dtype=np.float64),
        horizon_s=project_config.dynamics.control_dt_s,
        integration_dt_s=project_config.dynamics.integration_dt_s,
        config=project_config.dynamics,
    )[-1]
    world = AUVWorld(
        config=project_config,
        initial_auv_state=initial_state,
        initial_obstacle_states=tuple(),
        goal_position_ned_m=np.array([90.0, 90.0, 20.0], dtype=np.float64),
    )
    result = world.step(command)

    position_error = float(
        np.max(np.abs(result.auv_state.position_ned_m - expected_state.position_ned_m))
    )
    scalar_errors = [
        abs(result.auv_state.yaw_rad - expected_state.yaw_rad),
        abs(result.auv_state.pitch_rad - expected_state.pitch_rad),
        abs(result.auv_state.surge_speed_mps - expected_state.surge_speed_mps),
        abs(result.auv_state.yaw_rate_rad_s - expected_state.yaw_rate_rad_s),
        abs(result.auv_state.pitch_rate_rad_s - expected_state.pitch_rate_rad_s),
    ]
    tolerance = 1.0e-12
    assert result.event.reason == "none", f"expected no event, actual={result.event}"
    assert position_error <= tolerance, f"expected <= {tolerance}, actual={position_error}"
    assert max(scalar_errors) <= tolerance, (
        f"expected <= {tolerance}, actual={max(scalar_errors)}"
    )


def test_world_stops_at_swept_physical_collision(project_config) -> None:
    """
    验证真实环境在积分节点之间发生物理包络碰撞时于最早事件时刻终止。

    extra_margin_m 不参与 collision 半径，避免把验证裕量接触误标为真实碰撞。
    """

    initial_state = _base_auv_state(np.array([50.0, 50.0, 20.0]), speed_mps=0.3)
    obstacle = GroundTruthObstacleState(
        obstacle_id=3,
        position_ned_m=np.array([51.30, 50.0, 20.0], dtype=np.float64),
        velocity_ned_mps=np.array([-0.8, 0.0, 0.0], dtype=np.float64),
        radius_m=0.5,
    )
    world = AUVWorld(
        config=project_config,
        initial_auv_state=initial_state,
        initial_obstacle_states=(obstacle,),
        goal_position_ned_m=np.array([90.0, 90.0, 20.0], dtype=np.float64),
    )
    result = world.step(ControlCommand(0.3, 0.0, 0.0))

    assert result.is_terminated, "expected collision to terminate the world"
    assert result.event.reason == "collision", f"expected collision, actual={result.event.reason}"
    assert result.event.obstacle_id == 3, f"expected obstacle 3, actual={result.event.obstacle_id}"
    assert 0.0 <= result.event.event_timestamp_s <= project_config.dynamics.integration_dt_s, (
        "expected first collision within first integration step, "
        f"actual={result.event.event_timestamp_s}"
    )


def test_validator_rejects_center_inside_box_when_auv_sphere_is_outside(project_config) -> None:
    """
    验证 validator 与 env 一致使用完整 AUV 物理包络检查边界。

    这是本版修复的集成缺陷回归测试：中心 N=99.5 m 虽小于 100 m，但 0.75 m 包络已经越界。
    """

    invalid_state = _base_auv_state(np.array([99.5, 50.0, 20.0]), speed_mps=0.8)
    evaluation = evaluate_candidate_action(
        auv_state=invalid_state,
        current_timestamp_s=0.0,
        action=ControlCommand(0.8, 0.0, 0.0),
        candidate_id=0,
        tracked_obstacles=[],
        config=project_config,
    )
    assert not evaluation.satisfies_operational_constraints, (
        "expected complete AUV sphere boundary violation to fail operational constraints"
    )
    assert not evaluation.passes_validator, "expected invalid operational state to fail validator"
