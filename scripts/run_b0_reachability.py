"""LOCAL §25.3 的六个预登记非学习可达性见证；不证明整个训练分布可达。

§7、§13、§18、附录B规定共同动力学、任务reward、当前真值B0及无执行过滤。
六个固定案例、LOS反馈律及一次尝试上限属于本轮工程选择，运行前已写入规格。
此脚本不创建学习器、Replay或优化器，不作为新的论文算法基线。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from auv_risk_rl.config import ProjectConfig, load_project_config
from auv_risk_rl.env.b0_navigation import B0NavigationEnv
from auv_risk_rl.env.local_task import command_to_normalized
from auv_risk_rl.types import AUVState, ControlCommand, GroundTruthObstacleState

ROOT = Path(__file__).resolve().parents[1]
MAX_TRANSITIONS_PER_CASE = 1000
SURGE_COMMAND_MPS = 1.2
LOS_GAIN_PER_SECOND = 1.0
GOAL_DISTANCE_TOLERANCE_M = 1e-8
LIMIT_ROUNDOFF_TOLERANCE = 1e-10


@dataclass(frozen=True)
class ReachabilityCase:
    """预登记初态；深度为NED Down米，障碍速度为NED地速m/s。"""

    case_id: str
    seed: int
    start_down_m: float
    goal_down_m: float
    obstacle_position_ned_m: tuple[float, float, float] | None = None
    obstacle_velocity_ned_mps: tuple[float, float, float] | None = None
    obstacle_radius_m: float = 0.6

    def initial_state(self) -> AUVState:
        """将初始姿态对准固定目标，最低前进速度和零角率不改变响应模型。"""

        return AUVState(
            np.array([20.0, 50.0, self.start_down_m], dtype=np.float64),
            0.0,
            -math.atan2(self.goal_down_m - self.start_down_m, 60.0),
            0.3,
            0.0,
            0.0,
        )

    def goal(self) -> np.ndarray:
        """返回固定NED目标，单位m。"""

        return np.array([80.0, 50.0, self.goal_down_m], dtype=np.float64)

    def obstacles(self) -> tuple[GroundTruthObstacleState, ...]:
        """每次构造独立真值对象；空场景仅用于可达性诊断。"""

        if self.obstacle_position_ned_m is None:
            return ()
        if self.obstacle_velocity_ned_mps is None:
            raise ValueError("有障碍位置时必须明确登记CV地速。")
        return (GroundTruthObstacleState(
            1,
            np.asarray(self.obstacle_position_ned_m, dtype=np.float64),
            np.asarray(self.obstacle_velocity_ned_mps, dtype=np.float64),
            self.obstacle_radius_m,
        ),)


def fixed_cases() -> tuple[ReachabilityCase, ...]:
    """返回运行前冻结的六个案例，先无障碍，后简单CV交会；不筛选随机训练样本。"""

    return (
        ReachabilityCase("R01_horizontal_empty", 810001, 20.0, 20.0),
        ReachabilityCase("R02_deeper_empty", 810002, 12.0, 28.0),
        ReachabilityCase("R03_shallower_empty", 810003, 28.0, 12.0),
        ReachabilityCase("R04_head_on_lateral_offset", 810004, 20.0, 20.0,
                         (60.0, 53.0, 20.0), (-0.4, 0.0, 0.0)),
        ReachabilityCase("R05_crossing", 810005, 20.0, 20.0,
                         (50.0, 36.0, 20.0), (0.0, 0.4, 0.0)),
        ReachabilityCase("R06_head_on_vertical_offset", 810006, 20.0, 20.0,
                         (60.0, 50.0, 24.0), (-0.4, 0.0, 0.0)),
    )


def los_command(state: AUVState, goal_ned_m: np.ndarray,
                config: ProjectConfig) -> ControlCommand:
    """有界目标跟随反馈；只读取自身和任务目标，不读取障碍、未来、风险或TTC。"""

    delta = np.asarray(goal_ned_m, dtype=np.float64) - state.position_ned_m
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("目标相对位置必须是有限(3,)米制向量。")
    horizontal_m = float(np.linalg.norm(delta[:2]))
    desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
    if horizontal_m == 0:
        desired_yaw = state.yaw_rad
    desired_pitch = -math.atan2(float(delta[2]), horizontal_m)
    dynamics = config.dynamics
    # 限制反馈律的目标/指令，不修改真实状态，不替代物理操作边界事件。
    desired_pitch = float(np.clip(desired_pitch, -dynamics.max_pitch_rad,
                                  dynamics.max_pitch_rad))
    yaw_error = math.atan2(math.sin(desired_yaw - state.yaw_rad),
                           math.cos(desired_yaw - state.yaw_rad))
    yaw_rate = float(np.clip(LOS_GAIN_PER_SECOND * yaw_error,
                             -dynamics.max_yaw_rate_rad_s, dynamics.max_yaw_rate_rad_s))
    pitch_rate = float(np.clip(LOS_GAIN_PER_SECOND * (desired_pitch - state.pitch_rad),
                               -dynamics.max_pitch_rate_rad_s,
                               dynamics.max_pitch_rate_rad_s))
    return ControlCommand(SURGE_COMMAND_MPS, yaw_rate, pitch_rate)


def _state_record(state: AUVState) -> dict[str, Any]:
    """只向离线证据保存完整自身状态，单位为m、rad、m/s、rad/s。"""

    return dict(position_ned_m=state.position_ned_m.tolist(), yaw_rad=state.yaw_rad,
                pitch_rad=state.pitch_rad, surge_speed_mps=state.surge_speed_mps,
                yaw_rate_rad_s=state.yaw_rate_rad_s, pitch_rate_rad_s=state.pitch_rate_rad_s)


def _obstacle_records(obstacles: tuple[GroundTruthObstacleState, ...]) -> list[dict[str, Any]]:
    """离线保存当前真值；控制器不读取此记录，也不查询真未来。"""

    return [dict(obstacle_id=o.obstacle_id, position_ned_m=o.position_ned_m.tolist(),
                 velocity_ned_mps=o.velocity_ned_mps.tolist(), radius_m=o.radius_m)
            for o in obstacles]


def state_is_legal(state: AUVState, config: ProjectConfig) -> bool:
    """按已有完整AUV球包络及响应状态界限检查，不检查障碍中心是否在操作盒内。"""

    tolerance = LIMIT_ROUNDOFF_TOLERANCE
    dynamics = config.dynamics
    lower = np.asarray(config.environment.position_lower_bound_ned_m) + config.risk.auv_radius_m
    upper = np.asarray(config.environment.position_upper_bound_ned_m) - config.risk.auv_radius_m
    values = np.array([*state.position_ned_m, state.yaw_rad, state.pitch_rad,
                       state.surge_speed_mps, state.yaw_rate_rad_s, state.pitch_rate_rad_s])
    return bool(
        np.all(np.isfinite(values))
        and np.all(state.position_ned_m >= lower - tolerance)
        and np.all(state.position_ned_m <= upper + tolerance)
        and abs(state.pitch_rad) <= dynamics.max_pitch_rad + tolerance
        and dynamics.min_surge_speed_mps - tolerance <= state.surge_speed_mps
        <= dynamics.max_surge_speed_mps + tolerance
        and abs(state.yaw_rate_rad_s) <= dynamics.max_yaw_rate_rad_s + tolerance
        and abs(state.pitch_rate_rad_s) <= dynamics.max_pitch_rate_rad_s + tolerance
    )


def _node(case_id: str, env: B0NavigationEnv, action: np.ndarray | None,
          reward: float | None, info: dict[str, Any] | None) -> dict[str, Any]:
    """记录控制节点与实际指令；终止周期只记录真实已执行前缀。"""

    state = env.world.auv_state
    row = dict(case_id=case_id, control_step=env.world.control_step_index,
               timestamp_s=env.world.timestamp_s, north_m=float(state.position_ned_m[0]),
               east_m=float(state.position_ned_m[1]), down_m=float(state.position_ned_m[2]),
               yaw_rad=state.yaw_rad, pitch_rad=state.pitch_rad,
               surge_speed_mps=state.surge_speed_mps, yaw_rate_rad_s=state.yaw_rate_rad_s,
               pitch_rate_rad_s=state.pitch_rate_rad_s,
               nominal_a0=None, nominal_a1=None, nominal_a2=None,
               executed_a0=None, executed_a1=None, executed_a2=None,
               surge_command_mps=None, yaw_rate_command_rad_s=None,
               pitch_rate_command_rad_s=None, reward=reward, elapsed_s=None,
               minimum_clearance_m=None, failure_type=None,
               reward_progress=None, reward_goal=None, reward_time=None, reward_smoothness=None)
    if action is not None and info is not None:
        command = env.previous_command
        row.update({f"nominal_a{i}": float(action[i]) for i in range(3)})
        row.update({f"executed_a{i}": float(info["executed_action_normalized"][i])
                    for i in range(3)})
        clearance = float(info["minimum_clearance"])
        row.update(surge_command_mps=command.surge_speed_command_mps,
                   yaw_rate_command_rad_s=command.yaw_rate_command_rad_s,
                   pitch_rate_command_rad_s=command.pitch_rate_command_rad_s,
                   elapsed_s=info["elapsed_s"],
                   minimum_clearance_m=clearance if np.isfinite(clearance) else None,
                   failure_type=info["failure_type"])
        row.update({f"reward_{key}": value for key, value in info["reward_components"].items()})
    return row


def run_case(case: ReachabilityCase, config: ProjectConfig) -> dict[str, Any]:
    """运行一次固定案例至真实结束或1000步；失败保留证据，不重试、不改控制律。"""

    initial = case.initial_state()
    env = B0NavigationEnv(config, initial, case.obstacles(), case.goal(),
                          f"engineering-reachability-{case.case_id}")
    result: dict[str, Any] = dict(
        case_id=case.case_id, seed=case.seed, run_kind="engineering_reachability",
        task_profile="fixed_reachability_diagnostic", split="engineering_reachability",
        attempt_count=1, maximum_attempts=1, maximum_transitions=MAX_TRANSITIONS_PER_CASE,
        initial_state=_state_record(initial), initial_obstacles=_obstacle_records(case.obstacles()),
        goal_position_ned_m=case.goal().tolist(), trajectories=[], obstacle_trajectories=[],
        actual_transitions=0, nonlearning_warmup_world_transitions=0,
        scientific_training_steps=0, scientific_training_updates=0, complete=False,
        status="REACHABILITY_NOT_ESTABLISHED",
    )
    try:
        observation, reset_info = env.reset(seed=case.seed)
        if not np.all(np.isfinite(observation)):
            raise AssertionError("合法warm-up后B0观察必须有限。")
        result["warmup_info"] = reset_info
        result["formal_initial_state"] = _state_record(env.world.auv_state)
        result["formal_initial_obstacles"] = _obstacle_records(env.world.obstacle_states)
        result["nonlearning_warmup_world_transitions"] = round(
            reset_info["warmup_duration_s"] / config.dynamics.control_dt_s)
        result["trajectories"].append(_node(case.case_id, env, None, None, None))
        legal = state_is_legal(env.world.auv_state, config)
        minimum_clearance_m = float("inf")
        final_info: dict[str, Any] = {}
        for _ in range(MAX_TRANSITIONS_PER_CASE):
            command = los_command(env.world.auv_state, case.goal(), config)
            action = command_to_normalized(command, config.dynamics)
            if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1):
                raise AssertionError("控制律生成非法指令；不得裁剪后继续。")
            observation, reward, terminated, truncated, final_info = env.step(action)
            result["actual_transitions"] += 1
            if not np.all(np.isfinite(observation)) or not np.isfinite(reward):
                raise AssertionError("真实B0转移产生非有限观察或reward。")
            if not np.array_equal(action, final_info["executed_action_normalized"]):
                raise AssertionError("B0诊断发现名义/执行指令不一致。")
            legal &= state_is_legal(env.world.auv_state, config)
            minimum_clearance_m = min(minimum_clearance_m,
                                      float(final_info["minimum_clearance"]))
            result["trajectories"].append(_node(case.case_id, env, action, reward, final_info))
            result["obstacle_trajectories"].extend(
                dict(case_id=case.case_id, control_step=env.world.control_step_index,
                     timestamp_s=env.world.timestamp_s, **item)
                for item in _obstacle_records(env.world.obstacle_states))
            if terminated or truncated:
                result["complete"] = bool(terminated and not truncated)
                break
        distance_m = float(np.linalg.norm(case.goal() - env.world.auv_state.position_ned_m))
        success = final_info.get("failure_type") == "goal_success"
        clearance_legal = minimum_clearance_m >= -LIMIT_ROUNDOFF_TOLERANCE
        witness = (success and result["complete"] and legal and clearance_legal
                   and distance_m <= config.environment.goal_radius_m + GOAL_DISTANCE_TOLERANCE_M)
        result.update(
            status="WITNESS_FOUND" if witness else "REACHABILITY_NOT_ESTABLISHED",
            witness_found=bool(witness), operation_limits_satisfied=bool(legal),
            final_event=dict(failure_type=final_info.get("failure_type"),
                             timestamp_s=env.world.timestamp_s),
            final_goal_distance_m=distance_m, actual_duration_s=env.world.timestamp_s - 1.0,
            minimum_clearance_m=(minimum_clearance_m if np.isfinite(minimum_clearance_m) else None),
            clearance_applicable=bool(case.obstacles()),
        )
    except Exception:
        result.update(status="ERROR", witness_found=False, error=traceback.format_exc())
    return result


def write_evidence(output_dir: Path, report: dict[str, Any]) -> None:
    """只保存必要JSON和控制节点CSV；不覆盖既有验收结果。"""

    names = ("reachability.json", "trajectories.csv", "obstacles.csv")
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("可达性结果目录已有同名证据，禁止覆盖历史。")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = [row for case in report["cases"] for row in case["trajectories"]]
    obstacles = [row for case in report["cases"] for row in case["obstacle_trajectories"]]
    summary = dict(report)
    summary["cases"] = [dict(case, trajectories_file="trajectories.csv",
                             obstacle_trajectories_file="obstacles.csv")
                        for case in report["cases"]]
    for case in summary["cases"]:
        case.pop("trajectories")
        case.pop("obstacle_trajectories")
    (output_dir / names[0]).write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                allow_nan=False) + "\n", encoding="utf-8")
    if trajectories:
        with (output_dir / names[1]).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trajectories[0]))
            writer.writeheader()
            writer.writerows(trajectories)
    with (output_dir / names[2]).open("w", newline="", encoding="utf-8") as stream:
        fields = ["case_id", "control_step", "timestamp_s", "obstacle_id",
                  "position_ned_m", "velocity_ned_mps", "radius_m"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(obstacles)


def main() -> int:
    """显式输出目录和代码版本入口；只执行预登记非学习可达性案例。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stage0.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    if (args.output_dir / "reachability.json").exists():
        raise FileExistsError("输出目录已有证据，禁止重复尝试并覆盖。")
    config = load_project_config(args.config)
    cases = [run_case(case, config) for case in fixed_cases()]
    confirmed = all(case["status"] == "WITNESS_FOUND" for case in cases)
    report = dict(
        created_at=datetime.now(UTC).isoformat(), code_version=args.code_version,
        run_kind="engineering_reachability", config_path=str(args.config.resolve()),
        config_snapshot=config.to_dict(), cases=cases,
        status="REACHABILITY_CONFIRMED" if confirmed else "REACHABILITY_NOT_ESTABLISHED",
        nonlearning_env_transitions=sum(case["actual_transitions"] for case in cases),
        nonlearning_warmup_world_transitions=sum(
            case["nonlearning_warmup_world_transitions"] for case in cases),
        scientific_training_steps=0, scientific_training_updates=0,
        scientific_stage2="NOT RUN", attempts_per_case=1,
        statement="仅证明六个固定案例存在合法可执行解，不证明随机训练分布全部可达。",
    )
    write_evidence(args.output_dir, report)
    print(report["status"])
    print(f"Nonlearning B0 transitions: {report['nonlearning_env_transitions']}")
    print("Scientific training steps/updates: 0/0; LOCAL Stage2: NOT RUN")
    return 0 if confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
