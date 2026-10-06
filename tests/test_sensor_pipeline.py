"""验证简化声呐 FOV、dropout、固定延迟与 measurement-time 位姿语义。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from auv_risk_rl.sensors.delay_queue import DetectionDelayQueue
from auv_risk_rl.sensors.sonar import generate_sonar_detections
from auv_risk_rl.sensors.visibility import is_obstacle_geometrically_visible
from auv_risk_rl.tracking.pose_history import AUVPoseHistory
from auv_risk_rl.tracking.track_manager import MultiTargetCVTracker
from auv_risk_rl.types import AUVState, GroundTruthObstacleState, SensorDetection


def _auv_at(position_ned_m: np.ndarray) -> AUVState:
    """构造 yaw=pitch=0 的测试 AUV 状态。"""

    return AUVState(
        position_ned_m=position_ned_m.astype(np.float64),
        yaw_rad=0.0,
        pitch_rad=0.0,
        surge_speed_mps=0.8,
        yaw_rate_rad_s=0.0,
        pitch_rate_rad_s=0.0,
    )


def test_sonar_visibility_respects_forward_fov(project_config) -> None:
    """验证前方目标可见、同距离正后方目标因水平 FOV 被拒绝。"""

    auv_state = _auv_at(np.array([50.0, 50.0, 20.0]))
    front = GroundTruthObstacleState(
        obstacle_id=1,
        position_ned_m=np.array([60.0, 50.0, 20.0]),
        velocity_ned_mps=np.zeros(3, dtype=np.float64),
        radius_m=0.5,
    )
    behind = GroundTruthObstacleState(
        obstacle_id=2,
        position_ned_m=np.array([40.0, 50.0, 20.0]),
        velocity_ned_mps=np.zeros(3, dtype=np.float64),
        radius_m=0.5,
    )
    front_visible, _ = is_obstacle_geometrically_visible(auv_state, front, project_config.sensor)
    behind_visible, _ = is_obstacle_geometrically_visible(auv_state, behind, project_config.sensor)
    assert front_visible, "expected obstacle in front and within range/FOV to be visible"
    assert not behind_visible, "expected obstacle behind AUV to be outside forward-looking FOV"


def test_sonar_dropout_probability_one_produces_no_detection(project_config) -> None:
    """验证 dropout=1 时即使几何可见也不会产生检测，且随机源显式传入。"""

    sensor_config = replace(project_config.sensor, dropout_probability=1.0)
    auv_state = _auv_at(np.array([50.0, 50.0, 20.0]))
    obstacle = GroundTruthObstacleState(
        obstacle_id=1,
        position_ned_m=np.array([60.0, 50.0, 20.0]),
        velocity_ned_mps=np.zeros(3, dtype=np.float64),
        radius_m=0.5,
    )
    detections = generate_sonar_detections(
        auv_state=auv_state,
        obstacle_states=(obstacle,),
        measurement_timestamp_s=0.0,
        sensor_config=sensor_config,
        dynamics_config=project_config.dynamics,
        sensor_rng=np.random.default_rng(1234),
    )
    assert detections == tuple(), f"expected no detection at dropout=1, actual={detections}"


def test_delay_queue_releases_only_at_arrival_time(project_config) -> None:
    """
    验证固定延迟队列不会提前释放测量，并保留原 measurement_timestamp_s。

    测试使用 2 个控制步延迟，因此到达时间应为 2*control_dt_s。
    """

    delay_steps = 2
    sensor_config = replace(
        project_config.sensor,
        measurement_delay_control_steps=delay_steps,
        dropout_probability=0.0,
        measurement_std_body_m=(0.0, 0.0, 0.0),
    )
    auv_state = _auv_at(np.array([50.0, 50.0, 20.0]))
    obstacle = GroundTruthObstacleState(
        obstacle_id=1,
        position_ned_m=np.array([60.0, 50.0, 20.0]),
        velocity_ned_mps=np.zeros(3, dtype=np.float64),
        radius_m=0.5,
    )
    detections = generate_sonar_detections(
        auv_state,
        (obstacle,),
        0.0,
        sensor_config,
        project_config.dynamics,
        np.random.default_rng(7),
    )
    queue = DetectionDelayQueue()
    queue.enqueue_many(detections)
    expected_arrival_s = delay_steps * project_config.dynamics.control_dt_s

    assert queue.pop_arrived(expected_arrival_s - 0.01) == tuple()
    arrived = queue.pop_arrived(expected_arrival_s)
    assert len(arrived) == 1, f"expected one arrived detection, actual={len(arrived)}"
    assert arrived[0].measurement_timestamp_s == 0.0, (
        "expected original measurement timestamp to be preserved, "
        f"actual={arrived[0].measurement_timestamp_s}"
    )
    assert arrived[0].arrival_timestamp_s == expected_arrival_s, (
        f"expected arrival={expected_arrival_s}, actual={arrived[0].arrival_timestamp_s}"
    )


def test_tracker_uses_measurement_time_auv_pose_for_delayed_detection(project_config) -> None:
    """
    验证迟到检测使用测量发生时刻 AUV 位姿，而不是到达时刻位姿做 Body→NED 转换。

    若误用到达位姿，本测试目标 N 位置会从 5 m 错成 15 m，属于系统性时间戳 bug。
    """

    pose_history = AUVPoseHistory()
    pose_history.add(0.0, _auv_at(np.array([0.0, 0.0, 10.0])))
    pose_history.add(0.2, _auv_at(np.array([10.0, 0.0, 10.0])))
    delayed_detection = SensorDetection(
        obstacle_id=9,
        relative_position_body_m=np.array([5.0, 0.0, 0.0], dtype=np.float64),
        measurement_covariance_body_m2=np.diag([0.04, 0.04, 0.09]).astype(np.float64),
        measurement_timestamp_s=0.0,
        arrival_timestamp_s=0.2,
    )
    tracker = MultiTargetCVTracker(project_config)
    tracker.add_arrived_detections((delayed_detection,), pose_history)
    track = tracker.get_track_states(current_timestamp_s=0.2)[0]

    expected_position_ned_m = np.array([5.0, 0.0, 10.0], dtype=np.float64)
    error_m = float(np.max(np.abs(track.state_mean_ned[0:3] - expected_position_ned_m)))
    tolerance_m = 1.0e-12
    assert error_m <= tolerance_m, (
        f"expected measurement-time transformed position={expected_position_ned_m}, "
        f"actual={track.state_mean_ned[0:3]}, error={error_m}, tolerance={tolerance_m}"
    )
