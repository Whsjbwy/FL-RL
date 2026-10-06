"""CM08：整数控制tick主链及旧浮点时间接口的限定兼容回归。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.exceptions import TimestampOrderError
from auv_risk_rl.runtime.stage0_cycle import Stage0ControlCycleRunner
from auv_risk_rl.sensors.delay_queue import DetectionDelayQueue
from auv_risk_rl.sensors.sonar import generate_sonar_detections
from auv_risk_rl.types import AUVState, ControlCommand, GroundTruthObstacleState, SensorDetection


def _state() -> AUVState:
    """构造合法的无转向状态。"""

    return AUVState(np.array([20.0, 20.0, 20.0]), 0.0, 0.0, 0.8, 0.0, 0.0)


def _obstacle() -> GroundTruthObstacleState:
    """构造持续可见CV目标，避免将FOV变化混入到达测试。"""

    return GroundTruthObstacleState(1, np.array([40.0, 20.0, 20.0]), np.array([0.2, 0.0, 0.0]), 0.5)


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_delay_releases_exactly_on_integer_control_tick(project_config, delay) -> None:
    """delay=0/1/2恰在对应tick释放，测量时间仍为原发生时刻。"""

    cfg = replace(project_config.sensor, measurement_delay_control_steps=delay,
                  dropout_probability=0.0, measurement_std_body_m=(0.0, 0.0, 0.0))
    detection = generate_sonar_detections(
        _state(), (_obstacle(),), 7.2, cfg, project_config.dynamics, np.random.default_rng(11),
        measurement_control_tick=1,
    )[0]
    assert detection.arrival_control_tick == 1 + delay
    queue = DetectionDelayQueue(project_config.dynamics.control_dt_s, clock_origin_s=7.0)
    queue.enqueue_many((detection,))
    for tick in range(1, 1 + delay):
        assert queue.pop_arrived(current_control_tick=tick) == ()
    arrived = queue.pop_arrived(current_control_tick=1 + delay)
    assert len(arrived) == 1
    assert arrived[0].measurement_timestamp_s == 7.2
    assert arrived[0].measurement_control_tick == 1
    assert queue.pending_count == 0


def test_one_step_delay_releases_on_next_control_tick(project_config) -> None:
    """PA08：八次小步得到0.399999...时，主tick与旧兼容查询都不会多等一周期。"""

    queue = DetectionDelayQueue(project_config.dynamics.control_dt_s)
    detection = SensorDetection(1, np.ones(3), np.eye(3), 0.2, 0.4, 1, 2)
    queue.enqueue_many((detection,))
    assert queue.pop_arrived(current_control_tick=1) == ()
    assert len(queue.pop_arrived(current_control_tick=2)) == 1
    legacy_queue = DetectionDelayQueue(project_config.dynamics.control_dt_s)
    legacy_queue.enqueue_many((SensorDetection(1, np.ones(3), np.eye(3), 0.2, 0.4),))
    assert legacy_queue.pop_arrived(0.4 - 1e-10) == ()
    clock = 0.0
    for _ in range(8):
        clock += project_config.dynamics.integration_dt_s
    assert clock < 0.4
    assert len(legacy_queue.pop_arrived(clock)) == 1


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_runtime_uses_ticks_and_kf_keeps_measurement_time(project_config, delay) -> None:
    """运行真实无RL控制链，确认首测量按tick到达，KF观测年龄不被到达时刻覆盖。"""

    cfg = replace(project_config, sensor=replace(
        project_config.sensor, measurement_delay_control_steps=delay, dropout_probability=0.0,
    ))
    world = AUVWorld(cfg, _state(), (_obstacle(),), np.array([90.0, 90.0, 20.0]))
    runner = Stage0ControlCycleRunner(cfg, world, np.random.default_rng(22), {1: 0.5})
    frame = runner.initialize_perception()
    if delay == 0:
        assert frame.arrived_detections[0].measurement_control_tick == 0
        assert frame.track_states[0].last_measurement_timestamp_s == 0.0
    else:
        assert frame.arrived_detections == ()
    action = ControlCommand(0.8, 0.0, 0.0)
    for tick in range(1, 4):
        frame = runner.run_control_cycle(action, action).perception_frame
        if tick < delay:
            assert frame.arrived_detections == ()
            continue
        assert len(frame.arrived_detections) == 1
        received = frame.arrived_detections[0]
        assert received.arrival_control_tick == tick
        assert received.measurement_control_tick == tick - delay
        assert (
            frame.track_states[0].last_measurement_timestamp_s == received.measurement_timestamp_s
        )


def test_delay_queue_rejects_invalid_or_backwards_ticks() -> None:
    """非法tick和倒退查询不得靠时间容差静默纠正。"""

    queue = DetectionDelayQueue()
    with pytest.raises(TimestampOrderError):
        queue.pop_arrived(current_control_tick=1.5)
    with pytest.raises(TimestampOrderError):
        queue.pop_arrived(current_control_tick=True)
    queue.pop_arrived(current_control_tick=2)
    with pytest.raises(TimestampOrderError):
        queue.pop_arrived(current_control_tick=1)
    bad = SensorDetection(1, np.ones(3), np.eye(3), 0.2, 0.4, 2, 1)
    with pytest.raises(TimestampOrderError):
        queue.enqueue_many((bad,))
    assert queue.pending_count == 0
