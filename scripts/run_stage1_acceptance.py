"""LOCAL Stage 1 固定场景非学习验收；不运行策略训练或摘要封存。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from auv_risk_rl.config import ProjectConfig, load_project_config
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.exceptions import InvalidEnvironmentStateError
from auv_risk_rl.runtime.local_perception import PerceptionSession
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.types import AUVState, ControlCommand, GroundTruthObstacleState

ROOT = Path(__file__).resolve().parents[1]

MOTION_CASES = ("straight_3d", "steady_turn", "actuator_response")
EVENT_CASES = ("head_on", "crossing", "boundary", "pitch_boundary", "success", "timeout")
SENSOR_CASES = ("range_entry", "range_exit", "fov_return", "occlusion_release",
                "dropout_return", "delay_1", "delay_2")


def _state(position=(50.0, 50.0, 20.0), yaw=0.0, pitch=0.0,
           speed=0.8, yaw_rate=0.0, pitch_rate=0.0) -> AUVState:
    """创建明确命名的固定验收初值，单位为m、rad及秒。"""
    return AUVState(np.asarray(position, dtype=np.float64), yaw, pitch,
                    speed, yaw_rate, pitch_rate)


def _vector(state: AUVState) -> np.ndarray:
    """只用于离线数值比较，不向任何策略暴露真值。"""
    return np.array([*state.position_ned_m, state.yaw_rad, state.pitch_rad,
                     state.surge_speed_mps, state.yaw_rate_rad_s, state.pitch_rate_rad_s])


def _trajectory_row(case: str, world: AUVWorld, reference=None) -> dict:
    """保存小体积控制节点轨迹及独立参考误差。"""
    vector = _vector(world.auv_state)
    row = dict(case=case, timestamp_s=world.timestamp_s,
               control_step=world.control_step_index,
               north_m=vector[0], east_m=vector[1], down_m=vector[2],
               yaw_rad=vector[3], pitch_rad=vector[4], speed_mps=vector[5],
               yaw_rate_rad_s=vector[6], pitch_rate_rad_s=vector[7],
               offline_truth_obstacles=[dict(obstacle_id=obstacle.obstacle_id,
                                            position_ned_m=obstacle.position_ned_m.tolist())
                                        for obstacle in world.obstacle_states])
    if reference is not None:
        row.update(reference_north_m=float(reference[0]),
                   reference_east_m=float(reference[1]), reference_down_m=float(reference[2]),
                   max_position_error_m=float(np.max(np.abs(vector[:3]-reference[:3]))))
    return row


def _independent_response_rhs(config: ProjectConfig, command: ControlCommand):
    """独立标量Eq.(6)-(8)，不调用生产动力学、旋转或积分函数。"""
    dynamics = config.dynamics

    def rhs(_time, vector):
        _, _, _, yaw, pitch, speed, yaw_rate, pitch_rate = vector
        acceleration = (command.surge_speed_command_mps-speed)/dynamics.surge_time_constant_s
        yaw_acceleration = (
            command.yaw_rate_command_rad_s-yaw_rate)/dynamics.yaw_rate_time_constant_s
        pitch_acceleration = (
            command.pitch_rate_command_rad_s-pitch_rate)/dynamics.pitch_rate_time_constant_s
        # 这里的min/max是参考模型的执行器导数限制，不是结果裁剪。
        return [speed*math.cos(yaw)*math.cos(pitch),
                speed*math.sin(yaw)*math.cos(pitch), -speed*math.sin(pitch),
                yaw_rate, pitch_rate,
                max(-dynamics.max_surge_accel_mps2,
                    min(dynamics.max_surge_accel_mps2, acceleration)),
                max(-dynamics.max_yaw_accel_rad_s2,
                    min(dynamics.max_yaw_accel_rad_s2, yaw_acceleration)),
                max(-dynamics.max_pitch_accel_rad_s2,
                    min(dynamics.max_pitch_accel_rad_s2, pitch_acceleration))]
    return rhs


def run_motion_case(config: ProjectConfig, case: str) -> dict:
    """在真实世界运行50个控制周期，与预登记独立解析/数值轨迹逐点比较。"""
    if case == "straight_3d":
        initial = _state(yaw=0.4, pitch=0.2)
        command = ControlCommand(0.8, 0.0, 0.0)
    elif case == "steady_turn":
        initial = _state(yaw=0.2, yaw_rate=0.1)
        command = ControlCommand(0.8, 0.1, 0.0)
    elif case == "actuator_response":
        initial = _state(yaw=0.2, pitch=0.1, speed=0.3)
        command = ControlCommand(0.9, 0.1, -0.04)
    else:
        raise ValueError(f"未知运动案例：{case}")
    world = AUVWorld(config, initial, (), np.array([90.0, 90.0, 20.0]))
    times = np.arange(51)*config.dynamics.control_dt_s
    if case == "actuator_response":
        solution = solve_ivp(_independent_response_rhs(config, command), (0.0, 10.0),
                             _vector(initial), method="DOP853", t_eval=times,
                             rtol=1e-11, atol=1e-13)
        assert solution.success, solution.message
        references = solution.y.T
    else:
        references = np.tile(_vector(initial), (51, 1))
        if case == "straight_3d":
            direction = np.array([math.cos(0.4)*math.cos(0.2),
                                  math.sin(0.4)*math.cos(0.2), -math.sin(0.2)])
            references[:, :3] += times[:, None]*0.8*direction
        else:
            references[:, 0] += 8.0*(np.sin(0.2+0.1*times)-math.sin(0.2))
            references[:, 1] += 8.0*(math.cos(0.2)-np.cos(0.2+0.1*times))
            references[:, 3] += 0.1*times
    actual = [_vector(initial)]
    rows = [_trajectory_row(case, world, references[0])]
    for reference in references[1:]:
        result = world.step(command)
        assert result.event.reason == "none", result.event
        actual.append(_vector(result.auv_state))
        rows.append(_trajectory_row(case, world, reference))
    errors = np.max(np.abs(np.asarray(actual)-references), axis=0)
    if case == "straight_3d":
        assert max(errors[:3]) <= 1e-11, errors
        assert world.auv_state.position_ned_m[2] < initial.position_ned_m[2]
    elif case == "steady_turn":
        assert max(errors[:3]) <= 1e-5, errors
        assert errors[3] <= 1e-12, errors
    else:
        limits = np.array([2e-3, 2e-3, 2e-3, 2e-4, 2e-4, 5e-4, 2e-4, 2e-4])
        assert np.all(errors <= limits), (errors, limits)
        assert actual[1][5] < command.surge_speed_command_mps
        assert actual[1][6] < command.yaw_rate_command_rad_s
    return dict(case=case, status="PASS", transitions=50,
                maximum_absolute_state_errors=errors.tolist(), trajectories=rows)


def _first_contact_time(relative_position, relative_velocity, radius: float) -> float:
    """独立标量连续CV二次根，不调用生产几何函数。"""
    x, y, z = relative_position
    vx, vy, vz = relative_velocity
    a = vx*vx+vy*vy+vz*vz
    b = 2.0*(x*vx+y*vy+z*vz)
    c = x*x+y*y+z*z-radius*radius
    return (-b-math.sqrt(b*b-4*a*c))/(2*a)


def run_event_case(config: ProjectConfig, case: str) -> dict:
    """按物理球包络计算首次事件；终止后再次step不能改变真实世界。"""
    initial = _state(speed=0.3)
    command = ControlCommand(0.3, 0.0, 0.0)
    obstacles = ()
    goal = np.array([90.0, 90.0, 20.0])
    expected_reason = case
    if case in ("head_on", "crossing"):
        initial = _state(speed=0.3 if case == "head_on" else 0.8)
        command = ControlCommand(initial.surge_speed_mps, 0.0, 0.0)
        relative = np.array([1.3, 0.0 if case == "head_on" else 0.3, 0.0])
        velocity = np.array([-0.8, 0.0, 0.0] if case == "head_on" else [0.0, -0.8, 0.0])
        obstacles = (GroundTruthObstacleState(1, initial.position_ned_m+relative, velocity, 0.5),)
        expected_time = _first_contact_time(
            relative, velocity-np.array([initial.surge_speed_mps, 0.0, 0.0]),
            config.risk.auv_radius_m+0.5)
        expected_reason = "collision"
    elif case == "boundary":
        initial = _state(position=(99.2, 50.0, 20.0), speed=0.8)
        command = ControlCommand(0.8, 0.0, 0.0)
        expected_time = (100.0-config.risk.auv_radius_m-99.2)/0.8
    elif case == "pitch_boundary":
        initial = _state(pitch=config.dynamics.max_pitch_rad-0.012, pitch_rate=0.2)
        command = ControlCommand(0.3, 0.0, 0.2)
        expected_time = 0.012/0.2
        expected_reason = "boundary"
    elif case == "success":
        goal = np.array([52.03, 50.0, 20.0])
        expected_time = (2.03-config.environment.goal_radius_m)/0.3
    elif case == "timeout":
        # 只缩短命名固定验收夹具，成本时域同步；不写入生产配置。
        config = replace(config,
                         environment=replace(config.environment, max_episode_control_steps=2),
                         cost=replace(config.cost, planned_horizon_control_steps=2))
        expected_time = 2*config.dynamics.control_dt_s
    else:
        raise ValueError(f"未知事件案例：{case}")
    world = AUVWorld(config, initial, obstacles, goal)
    rows = [_trajectory_row(case, world)]
    for _ in range(2):
        result = world.step(command)
        rows.append(_trajectory_row(case, world))
        if result.is_terminated:
            break
    assert result.is_terminated and result.event.reason == expected_reason, result.event
    assert abs(result.timestamp_s-expected_time) <= 1e-10, (result.timestamp_s, expected_time)
    if expected_reason == "collision":
        assert abs(result.event.minimum_clearance_m) <= 1e-10
    if case == "boundary":
        assert abs(world.auv_state.position_ned_m[0]-(100-config.risk.auv_radius_m)) <= 1e-10
    if case == "pitch_boundary":
        assert abs(world.auv_state.pitch_rad-config.dynamics.max_pitch_rad) <= 1e-10
    before = _vector(world.auv_state).copy()
    before_obstacles = [obstacle.position_ned_m.copy() for obstacle in world.obstacle_states]
    before_time, before_count = world.timestamp_s, world.control_step_index
    try:
        world.step(command)
    except InvalidEnvironmentStateError:
        pass
    else:
        raise AssertionError("真实终止后没有拒绝额外step")
    np.testing.assert_array_equal(_vector(world.auv_state), before)
    for obstacle, position in zip(world.obstacle_states, before_obstacles, strict=True):
        np.testing.assert_array_equal(obstacle.position_ned_m, position)
    assert (world.timestamp_s, world.control_step_index) == (before_time, before_count)
    clearance = result.event.minimum_clearance_m
    return dict(case=case, status="PASS", transitions=world.control_step_index,
                trajectories=rows, event=dict(case=case, reason=expected_reason,
                expected_time_s=expected_time, actual_time_s=result.timestamp_s,
                minimum_clearance_m=clearance if math.isfinite(clearance) else None,
                no_advance_after_terminal=True))


def run_nearmiss_case(config: ProjectConfig, clearance: float) -> dict:
    """真实净间距与near-miss的严格阈值；不使用额外验证裕量作为碰撞半径。"""
    initial = _state(speed=0.3)
    lateral = config.risk.auv_radius_m+0.5+clearance
    obstacle = GroundTruthObstacleState(1, np.array([50.03, 50.0+lateral, 20.0]),
                                        np.array([-0.8, 0.0, 0.0]), 0.5)
    world = AUVWorld(config, initial, (obstacle,), np.array([90.0, 90.0, 20.0]))
    rows = [_trajectory_row(f"clearance_{clearance}", world)]
    result = world.step(ControlCommand(0.3, 0.0, 0.0))
    rows.append(_trajectory_row(f"clearance_{clearance}", world))
    assert result.event.reason == "none" and not result.is_terminated
    actual = result.event.minimum_clearance_m
    assert abs(actual-clearance) <= 1e-10, (actual, clearance)
    near_miss = result.event.reason != "collision" and actual < 0.5
    assert near_miss == (clearance < 0.5), (actual, near_miss)
    return dict(case=f"clearance_{clearance}", status="PASS", transitions=1,
                trajectories=rows, minimum_clearance_m=actual, near_miss=near_miss,
                physical_collision=False, extra_validator_margin_m=config.risk.extra_margin_m)


def _assert_missing_prediction(previous, current, config: ProjectConfig) -> None:
    """独立构造连续白加速度的CV F/Q，核对缺测只预测而无伪造更新。"""
    dt = current.state_timestamp_s-previous.state_timestamp_s
    transition = np.eye(6)
    transition[:3, 3:] = dt*np.eye(3)
    density = np.diag(config.kf.acceleration_spectral_density_m2_s3)
    process = np.block([[dt**3/3*density, dt**2/2*density],
                        [dt**2/2*density, dt*density]])
    np.testing.assert_allclose(current.state_mean_ned, transition@previous.state_mean_ned,
                               rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(current.state_covariance_ned,
                               transition@previous.state_covariance_ned@transition.T+process,
                               rtol=0.0, atol=1e-11)
    assert current.last_measurement_timestamp_s == previous.last_measurement_timestamp_s


def run_sensor_case(config: ProjectConfig, case: str) -> dict:
    """真实CV世界及生产感知/KF路径；固定夹具隔离几何、缺测和延迟。"""
    config = replace(config, sensor=replace(config.sensor, dropout_probability=0.0))
    initial = _state(position=(20.0, 50.0, 20.0), speed=0.3)
    target = GroundTruthObstacleState(1, np.array([30.0, 50.0, 20.0]),
                                     np.array([0.2, 0.0, 0.0]), 1.0)
    obstacles, intervals = (target,), 5
    if case in ("range_entry", "range_exit"):
        position = [45.1 if case == "range_entry" else 44.9, 50.0, 20.0]
        velocity = [-0.8 if case == "range_entry" else 0.8, 0.0, 0.0]
        obstacles = (GroundTruthObstacleState(1, np.array(position), np.array(velocity), 1.0),)
    elif case == "fov_return":
        initial = _state(position=(20.0, 50.0, 20.0), yaw=0.99, speed=0.3, yaw_rate=0.35)
        intervals = 30
    elif case == "occlusion_release":
        blocker = GroundTruthObstacleState(2, np.array([25.0, 49.0, 20.0]),
                                          np.array([0.0, 0.8, 0.0]), 0.5)
        obstacles, intervals = (target, blocker), 15
    elif case == "dropout_return":
        intervals = 4
    elif case in ("delay_1", "delay_2"):
        config = replace(config, sensor=replace(config.sensor,
                         measurement_delay_control_steps=int(case[-1])))
        intervals = 6
    else:
        raise ValueError(f"未知感知案例：{case}")
    world = AUVWorld(config, initial, obstacles, np.array([80.0, 50.0, 20.0]))
    session = PerceptionSession(config, SeedManager(71001),
                                {obstacle.obstacle_id: obstacle.radius_m for obstacle in obstacles})
    rows, trajectories, previous_tracks = [], [], {}
    for tick in range(intervals+1):
        if tick:
            yaw_command = 0.35 if tick <= 5 else -0.35
            command = ControlCommand(0.3, yaw_command if case == "fov_return" else 0.0, 0.0)
            result = world.step(command)
            assert not result.is_terminated, (case, tick, result.event)
        if case == "dropout_return":
            session.config = replace(config, sensor=replace(
                config.sensor, dropout_probability=1.0 if 1 <= tick <= 3 else 0.0))
        before_generated, before_arrived = len(session.generated), len(session.arrived)
        session.capture(world, tick)
        generated = session.generated[before_generated:]
        arrived = session.arrived[before_arrived:]
        tracks = {track.track_state.obstacle_id: track.track_state
                  for track in session.tracks(tick*config.dynamics.control_dt_s)}
        arrived_ids = {detection.obstacle_id for detection in arrived}
        for identity, current in tracks.items():
            if identity in previous_tracks and identity not in arrived_ids:
                _assert_missing_prediction(previous_tracks[identity], current, config)
            elif identity in arrived_ids:
                newest = max(d.measurement_timestamp_s for d in arrived
                             if d.obstacle_id == identity)
                assert abs(current.last_measurement_timestamp_s-newest) <= 1e-12
        for detection in arrived:
            assert detection.arrival_control_tick == tick
            assert detection.measurement_control_tick == (
                tick-config.sensor.measurement_delay_control_steps)
            assert abs(detection.arrival_timestamp_s-detection.measurement_timestamp_s
                       -config.sensor.measurement_delay_control_steps*0.2) <= 1e-12
        target_track = tracks.get(1)
        rows.append(dict(case=case, tick=tick, timestamp_s=tick*0.2,
                         detected_ids=[d.obstacle_id for d in generated],
                         arrived_ids=[d.obstacle_id for d in arrived],
                         measurement_times_s=[d.measurement_timestamp_s for d in arrived],
                         generated_detections=[dict(
                             obstacle_id=detection.obstacle_id,
                             relative_position_body_m=detection.relative_position_body_m.tolist(),
                             covariance_body_m2=detection.measurement_covariance_body_m2.tolist(),
                             measurement_timestamp_s=detection.measurement_timestamp_s,
                             arrival_timestamp_s=detection.arrival_timestamp_s)
                             for detection in generated],
                         offline_target_truth_position_ned_m=(
                             next(obstacle.position_ned_m.tolist()
                                  for obstacle in world.obstacle_states
                                  if obstacle.obstacle_id == 1)),
                         target_track_time_s=(target_track.state_timestamp_s
                                              if target_track is not None else None),
                         target_last_measurement_s=(target_track.last_measurement_timestamp_s
                                                   if target_track is not None else None),
                         target_mean=(target_track.state_mean_ned.tolist()
                                      if target_track is not None else None),
                         target_position_covariance_trace=(
                             float(np.trace(target_track.state_covariance_ned[:3, :3]))
                             if target_track is not None else None)))
        trajectories.append(_trajectory_row(case, world))
        previous_tracks = tracks
    visible = [1 in row["detected_ids"] for row in rows]
    if case == "range_entry":
        assert not visible[0] and all(visible[1:]), visible
    elif case == "range_exit":
        assert visible[0] and not any(visible[2:]), visible
    elif case in ("fov_return", "occlusion_release"):
        assert visible[0] and not visible[5] and visible[-1], visible
        assert rows[5]["target_last_measurement_s"] < rows[5]["timestamp_s"]
        assert rows[-1]["target_last_measurement_s"] == rows[-1]["timestamp_s"]
    elif case == "dropout_return":
        assert visible == [True, False, False, False, True], visible
        assert rows[3]["target_position_covariance_trace"] > rows[0][
            "target_position_covariance_trace"]
    else:
        delay = config.sensor.measurement_delay_control_steps
        assert all(not row["arrived_ids"] for row in rows[:delay])
        assert all(row["arrived_ids"] == [1] for row in rows[delay:])
    return dict(case=case, status="PASS", transitions=intervals,
                trajectories=trajectories, sensor_records=rows)


def run_acceptance(config: ProjectConfig) -> dict:
    """执行预登记固定案例；任何科学断言失败都会立即向上抛出。"""
    results = [run_motion_case(config, case) for case in MOTION_CASES]
    results += [run_event_case(config, case) for case in EVENT_CASES]
    results += [run_nearmiss_case(config, clearance) for clearance in (0.25, 0.5, 0.75)]
    results += [run_sensor_case(config, case) for case in SENSOR_CASES]
    return dict(status="PASS", scope="LOCAL_STAGE_1_NONLEARNING_FIXED_SCENARIOS",
                specification="docs/STAGE1_ACCEPTANCE_SPEC.md", cases=results,
                nonlearning_world_transitions=sum(case["transitions"] for case in results),
                scientific_rl_training_steps=0, scientific_gradient_updates=0,
                synthetic_gradient_operations_in_this_script=0,
                throughput_benchmark_run=False)


def write_evidence(output_dir: Path, report: dict) -> None:
    """只创建新证据文件，不覆盖旧验收报告；JSON不写NaN/Infinity。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stage1_acceptance.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    log_files = (("trajectories.csv", "trajectories"), ("sensor_records.csv", "sensor_records"))
    for filename, key in log_files:
        rows = [row for case in report["cases"] for row in case.get(key, [])]
        fields = list(dict.fromkeys(field for row in rows for field in row))
        with (output_dir / filename).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "events.json").open("x", encoding="utf-8") as stream:
        json.dump([case["event"] for case in report["cases"] if "event" in case],
                  stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> int:
    """命令入口只运行非学习验收，命令、真实时间与失败堆栈均可核对。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stage0.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "command.txt").open("x", encoding="utf-8") as stream:
        stream.write(f"cwd={Path.cwd()}\ncommand={sys.executable} {' '.join(sys.argv)}\n")
    try:
        report = run_acceptance(load_project_config(args.config))
        report["created_at_utc"] = datetime.now(UTC).isoformat()
        report["config_snapshot"] = load_project_config(args.config).to_dict()
        write_evidence(args.output_dir, report)
    except Exception:
        failure = dict(status="FAIL", created_at_utc=datetime.now(UTC).isoformat(),
                       traceback=traceback.format_exc(), scientific_rl_training_steps=0)
        with (args.output_dir / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump(failure, stream, ensure_ascii=False, indent=2)
        traceback.print_exc()
        return 1
    print(f"LOCAL STAGE 1 FIXED-SCENARIO ACCEPTANCE: PASS ({len(report['cases'])} cases)")
    print(f"NONLEARNING WORLD TRANSITIONS: {report['nonlearning_world_transitions']}")
    print("SCIENTIFIC RL TRAINING STEPS: 0; SCIENTIFIC GRADIENT UPDATES: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
