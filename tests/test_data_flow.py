"""验证 Ground Truth、Sensor、KF 与策略侧感知接口严格隔离。"""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from auv_risk_rl.env.types import PolicyPerceptionFrame
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.runtime.stage0_cycle import Stage0ControlCycleRunner
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.types import AUVState, ControlCommand, GroundTruthObstacleState


def test_policy_perception_frame_has_no_ground_truth_field() -> None:
    """
    验证策略侧感知帧结构中不存在障碍真值或 Oracle 未来字段。

    这是接口级防泄漏测试，不以“训练时没有碰巧读取”为证据。
    """

    field_names = {field.name for field in fields(PolicyPerceptionFrame)}
    forbidden_fields = {"obstacle_states", "ground_truth_obstacles", "oracle_future_states"}
    overlap = field_names & forbidden_fields
    assert not overlap, f"expected no ground-truth fields, actual forbidden overlap={overlap}"


def test_stage0_cycle_keeps_truth_in_diagnostics_only(project_config) -> None:
    """
    验证集成 runner 同时提供策略侧感知与独立真值诊断，但二者不会合并为同一数据结构。
    """

    initial_auv_state = AUVState(
        position_ned_m=np.array([20.0, 20.0, 20.0], dtype=np.float64),
        yaw_rad=0.0,
        pitch_rad=0.0,
        surge_speed_mps=0.8,
        yaw_rate_rad_s=0.0,
        pitch_rate_rad_s=0.0,
    )
    obstacle = GroundTruthObstacleState(
        obstacle_id=1,
        position_ned_m=np.array([40.0, 20.0, 20.0], dtype=np.float64),
        velocity_ned_mps=np.array([0.2, 0.0, 0.0], dtype=np.float64),
        radius_m=0.5,
    )
    world = AUVWorld(
        project_config,
        initial_auv_state,
        (obstacle,),
        np.array([90.0, 90.0, 20.0], dtype=np.float64),
    )
    runner = Stage0ControlCycleRunner(
        config=project_config,
        world=world,
        sensor_rng=SeedManager(20260917).get_rng("sensor"),
        obstacle_radius_by_id_m={1: 0.5},
    )
    initial_frame = runner.initialize_perception()
    result = runner.run_control_cycle(
        nominal_action=ControlCommand(0.8, 0.0, 0.0),
        previous_executed_action=ControlCommand(0.8, 0.0, 0.0),
    )

    assert len(initial_frame.track_states) == 1, (
        f"expected one initialized track, actual={len(initial_frame.track_states)}"
    )
    assert not hasattr(result.perception_frame, "obstacle_states"), (
        "expected policy-facing perception frame to hide obstacle Ground Truth"
    )
    assert len(result.world_diagnostics.obstacle_states) == 1, (
        "expected environment/evaluation diagnostics to retain Ground Truth separately"
    )
