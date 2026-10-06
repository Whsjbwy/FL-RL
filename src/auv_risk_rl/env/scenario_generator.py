"""TRAIN_SCENARIO_V1 的可复现训练场景生成器与环境工厂。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.local_navigation import LocalNavigationEnv
from auv_risk_rl.env.local_task import LocalTaskConfig
from auv_risk_rl.frames import rotation_body_to_ned
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.types import AUVState, GroundTruthObstacleState

TRAIN_SCENARIO_VERSION = "TRAIN_SCENARIO_V1"


def _pair(values: list[float], field: str) -> tuple[float, float]:
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{field} 必须是两个数值组成的列表。")
    result = (float(values[0]), float(values[1]))
    if not all(math.isfinite(value) for value in result) or result[0] >= result[1]:
        raise ValueError(f"{field} 必须是有限且严格递增的范围。")
    return result


@dataclass(frozen=True)
class TrainingScenarioConfig:
    """TRAIN_SCENARIO_V1 单一分布配置；加载时拒绝 OOD 设置。"""

    distribution_version: str
    start_N_range: tuple[float, float]
    start_E_range: tuple[float, float]
    start_D_range: tuple[float, float]
    goal_N_range: tuple[float, float]
    goal_E_range: tuple[float, float]
    goal_D_range: tuple[float, float]
    forced_vertical_separation_m: float
    vertical_stratum_rule: str
    obstacle_count_values: tuple[int, ...]
    obstacle_count_probabilities: tuple[float, ...]
    obstacle_distance_range_m: tuple[float, float]
    obstacle_horizontal_angle_deg: tuple[float, float]
    obstacle_vertical_angle_deg: tuple[float, float]
    obstacle_radius_range_m: tuple[float, float]
    obstacle_speed_range_mps: tuple[float, float]
    velocity_direction: str
    motion_model: str
    current: str
    auv_initial_orientation: str
    auv_initial_surge: str
    auv_initial_rates: str

    def __post_init__(self) -> None:
        ranges = (
            self.start_N_range, self.start_E_range, self.start_D_range,
            self.goal_N_range, self.goal_E_range, self.goal_D_range,
            self.obstacle_distance_range_m, self.obstacle_horizontal_angle_deg,
            self.obstacle_vertical_angle_deg, self.obstacle_radius_range_m,
            self.obstacle_speed_range_mps,
        )
        if any(pair[0] >= pair[1] or not all(map(math.isfinite, pair)) for pair in ranges):
            raise ValueError("所有场景范围必须有限且严格递增。")
        if self.distribution_version != TRAIN_SCENARIO_VERSION:
            raise ValueError("只接受 TRAIN_SCENARIO_V1。")
        if self.start_N_range != (15.0, 25.0) or self.start_E_range != (20.0, 80.0):
            raise ValueError("起点 N/E 支持必须为 [15,25]/[20,80]。")
        if self.start_D_range != (8.0, 32.0):
            raise ValueError("起点 D 支持必须为 [8,32]。")
        if self.goal_N_range != (75.0, 85.0) or self.goal_E_range != (20.0, 80.0):
            raise ValueError("目标 N/E 支持必须为 [75,85]/[20,80]。")
        if self.goal_D_range != (8.0, 32.0):
            raise ValueError("目标 D 支持必须为 [8,32]。")
        if self.forced_vertical_separation_m != 4.0:
            raise ValueError("强制垂向分层必须为 4 m。")
        if self.vertical_stratum_rule != "even_index_conditional_uniform":
            raise ValueError("垂向分层规则必须是偶数索引条件均匀分布。")
        if self.obstacle_count_values != (1, 2, 3, 4):
            raise ValueError("训练障碍数量必须严格为 1–4。")
        probabilities = self.obstacle_count_probabilities
        if len(probabilities) != 4 or any(p < 0.0 or not math.isfinite(p) for p in probabilities):
            raise ValueError("障碍数量概率必须是四个有限非负值。")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("障碍数量概率之和必须为 1。")
        if any(not math.isclose(p, 0.25, rel_tol=0.0, abs_tol=1.0e-12)
               for p in probabilities):
            raise ValueError("TRAIN_SCENARIO_V1 数量律必须为离散均匀分布。")
        if self.obstacle_distance_range_m != (8.0, 23.0):
            raise ValueError("障碍相对距离支持必须为 [8,23] m。")
        if self.obstacle_horizontal_angle_deg != (-50.0, 50.0):
            raise ValueError("障碍水平角支持必须为 [-50,50] 度。")
        if self.obstacle_vertical_angle_deg != (-25.0, 25.0):
            raise ValueError("障碍垂直角支持必须为 [-25,25] 度。")
        if self.obstacle_radius_range_m != (0.5, 1.0):
            raise ValueError("障碍半径支持必须为 [0.5,1.0] m。")
        if self.obstacle_speed_range_mps != (0.2, 0.8):
            raise ValueError("障碍速度支持必须为 [0.2,0.8] m/s。")
        if self.velocity_direction != "isotropic_s2":
            raise ValueError("速度方向必须为 isotropic_s2。")
        if self.motion_model != "CV":
            raise ValueError("主训练运动模型必须为 CV。")
        if self.current != "zero":
            raise ValueError("主训练海流必须为 zero。")
        if self.auv_initial_orientation != "goal_line_of_sight":
            raise ValueError("AUV 初始姿态必须对准目标视线。")
        if self.auv_initial_surge != "min_surge" or self.auv_initial_rates != "zero":
            raise ValueError("AUV 初始速度必须为 min_surge，初始角率必须为 zero。")


def load_training_scenario_config(path: str | Path) -> TrainingScenarioConfig:
    """严格加载场景 YAML；未知、缺失或 OOD 字段均失败。"""

    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("场景配置必须是 YAML mapping。")
    expected = set(TrainingScenarioConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError(
            f"场景配置字段不匹配：missing={expected-set(raw)}, extra={set(raw)-expected}"
        )
    pair_fields = (
        "start_N_range", "start_E_range", "start_D_range", "goal_N_range",
        "goal_E_range", "goal_D_range", "obstacle_distance_range_m",
        "obstacle_horizontal_angle_deg", "obstacle_vertical_angle_deg",
        "obstacle_radius_range_m", "obstacle_speed_range_mps",
    )
    values: dict[str, Any] = dict(raw)
    for field in pair_fields:
        values[field] = _pair(raw[field], field)
    values["obstacle_count_values"] = tuple(int(value) for value in raw["obstacle_count_values"])
    values["obstacle_count_probabilities"] = tuple(
        float(value) for value in raw["obstacle_count_probabilities"]
    )
    values["forced_vertical_separation_m"] = float(raw["forced_vertical_separation_m"])
    return TrainingScenarioConfig(**values)


@dataclass(frozen=True)
class ObstacleGenerationMetadata:
    obstacle_id: int
    distance_m: float
    horizontal_angle_deg: float
    vertical_angle_deg: float
    radius_m: float
    speed_mps: float
    velocity_unit_direction_ned: tuple[float, float, float]


@dataclass(frozen=True)
class ScenarioGenerationMetadata:
    sampled_start_ned_m: tuple[float, float, float]
    sampled_goal_ned_m: tuple[float, float, float]
    vertical_separation_stratum: str
    obstacle_count: int
    obstacles: tuple[ObstacleGenerationMetadata, ...]
    motion_model: str
    current: str
    rng_namespace: str
    count_draw_count: int
    goal_depth_retry_count: int
    velocity_direction_retry_count: int
    generation_retry_count: int
    initial_obstacle_overlap_count: int
    scenario_generation_version: str


def _readonly_vector(values: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("场景向量必须是有限 shape=(3,) 数组。")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TrainingScenario:
    """不含策略结果的只读训练场景值对象。"""

    scenario_id: str
    distribution_version: str
    root_seed: int
    scenario_index: int
    initial_auv_state: AUVState
    initial_obstacle_states: tuple[GroundTruthObstacleState, ...]
    goal_position_ned_m: np.ndarray
    metadata: ScenarioGenerationMetadata

    def __post_init__(self) -> None:
        auv = self.initial_auv_state
        object.__setattr__(self, "initial_auv_state", AUVState(
            position_ned_m=_readonly_vector(auv.position_ned_m), yaw_rad=float(auv.yaw_rad),
            pitch_rad=float(auv.pitch_rad), surge_speed_mps=float(auv.surge_speed_mps),
            yaw_rate_rad_s=float(auv.yaw_rate_rad_s),
            pitch_rate_rad_s=float(auv.pitch_rate_rad_s),
        ))
        obstacles = tuple(GroundTruthObstacleState(
            obstacle_id=int(state.obstacle_id),
            position_ned_m=_readonly_vector(state.position_ned_m),
            velocity_ned_mps=_readonly_vector(state.velocity_ned_mps),
            radius_m=float(state.radius_m),
        ) for state in self.initial_obstacle_states)
        object.__setattr__(self, "initial_obstacle_states", obstacles)
        object.__setattr__(self, "goal_position_ned_m", _readonly_vector(self.goal_position_ned_m))

    def to_dict(self) -> dict[str, Any]:
        """返回跨进程稳定、可 JSON 序列化的等价表示。"""

        auv = self.initial_auv_state
        return {
            "scenario_id": self.scenario_id,
            "distribution_version": self.distribution_version,
            "root_seed": self.root_seed,
            "scenario_index": self.scenario_index,
            "initial_auv_state": {
                "position_ned_m": auv.position_ned_m.tolist(),
                "yaw_rad": auv.yaw_rad,
                "pitch_rad": auv.pitch_rad,
                "surge_speed_mps": auv.surge_speed_mps,
                "yaw_rate_rad_s": auv.yaw_rate_rad_s,
                "pitch_rate_rad_s": auv.pitch_rate_rad_s,
            },
            "initial_obstacle_states": [{
                "obstacle_id": state.obstacle_id,
                "position_ned_m": state.position_ned_m.tolist(),
                "velocity_ned_mps": state.velocity_ned_mps.tolist(),
                "radius_m": state.radius_m,
            } for state in self.initial_obstacle_states],
            "goal_position_ned_m": self.goal_position_ned_m.tolist(),
            "metadata": asdict(self.metadata),
        }


class LocalTrainingScenarioGenerator:
    """只消费局部 scenario 命名空间的索引可寻址生成器。"""

    def __init__(self, config: TrainingScenarioConfig, project_config: ProjectConfig) -> None:
        self.config = config
        self.project_config = project_config
        environment = project_config.environment
        if environment.position_lower_bound_ned_m != (0.0, 0.0, 2.0):
            raise ValueError("TRAIN_SCENARIO_V1 要求 N/E/D 下界为 0/0/2 m。")
        if environment.position_upper_bound_ned_m != (100.0, 100.0, 40.0):
            raise ValueError("TRAIN_SCENARIO_V1 要求 N/E/D 上界为 100/100/40 m。")
        if environment.goal_radius_m != 2.0 or environment.max_episode_control_steps != 1000:
            raise ValueError("TRAIN_SCENARIO_V1 要求成功半径 2 m、时域 1000 步。")
        if project_config.dynamics.min_surge_speed_mps != 0.3:
            raise ValueError("TRAIN_SCENARIO_V1 要求初始 surge 为 0.3 m/s。")

    @staticmethod
    def _rng(root_seed: int, scenario_index: int) -> np.random.Generator:
        if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
            raise ValueError("root_seed 必须是非负整数。")
        if isinstance(scenario_index, bool) or not isinstance(scenario_index, int) \
                or scenario_index < 0:
            raise ValueError("scenario_index 必须是非负整数。")
        index_seed = np.random.SeedSequence([root_seed, scenario_index]).generate_state(
            1, dtype=np.uint64
        )[0]
        return SeedManager(int(index_seed)).get_rng("scenario")

    @staticmethod
    def _body_direction(horizontal_rad: float, vertical_rad: float) -> np.ndarray:
        return np.array([
            math.cos(vertical_rad) * math.cos(horizontal_rad),
            math.cos(vertical_rad) * math.sin(horizontal_rad),
            -math.sin(vertical_rad),
        ], dtype=np.float64)

    def generate(self, root_seed: int, scenario_index: int) -> TrainingScenario:
        """按已冻结支持生成一项，无任务难度或策略条件拒绝。"""

        rng = self._rng(root_seed, scenario_index)
        cfg = self.config
        start = np.array([
            rng.uniform(*cfg.start_N_range), rng.uniform(*cfg.start_E_range),
            rng.uniform(*cfg.start_D_range),
        ], dtype=np.float64)
        goal_n = rng.uniform(*cfg.goal_N_range)
        goal_e = rng.uniform(*cfg.goal_E_range)
        goal_depth_retry_count = 0
        goal_d = rng.uniform(*cfg.goal_D_range)
        stratum = "forced_vertical_separation" if scenario_index % 2 == 0 else "ordinary"
        if scenario_index % 2 == 0:
            while abs(goal_d - start[2]) < cfg.forced_vertical_separation_m:
                goal_depth_retry_count += 1
                goal_d = rng.uniform(*cfg.goal_D_range)
        goal = np.array([goal_n, goal_e, goal_d], dtype=np.float64)
        delta = goal - start
        yaw = math.atan2(float(delta[1]), float(delta[0]))
        pitch = -math.atan2(float(delta[2]), math.hypot(float(delta[0]), float(delta[1])))
        auv = AUVState(
            position_ned_m=start, yaw_rad=yaw, pitch_rad=pitch,
            surge_speed_mps=self.project_config.dynamics.min_surge_speed_mps,
            yaw_rate_rad_s=0.0, pitch_rate_rad_s=0.0,
        )
        obstacle_count = int(rng.choice(
            cfg.obstacle_count_values, p=cfg.obstacle_count_probabilities
        ))
        rotation = rotation_body_to_ned(yaw, pitch)
        obstacles: list[GroundTruthObstacleState] = []
        obstacle_metadata: list[ObstacleGenerationMetadata] = []
        velocity_direction_retry_count = 0
        for obstacle_id in range(obstacle_count):
            distance = float(rng.uniform(*cfg.obstacle_distance_range_m))
            horizontal_deg = float(rng.uniform(*cfg.obstacle_horizontal_angle_deg))
            vertical_deg = float(rng.uniform(*cfg.obstacle_vertical_angle_deg))
            relative_body = distance * self._body_direction(
                math.radians(horizontal_deg), math.radians(vertical_deg)
            )
            position = start + rotation @ relative_body
            radius = float(rng.uniform(*cfg.obstacle_radius_range_m))
            speed = float(rng.uniform(*cfg.obstacle_speed_range_mps))
            direction = rng.normal(0.0, 1.0, 3)
            norm = float(np.linalg.norm(direction))
            while norm == 0.0:
                velocity_direction_retry_count += 1
                direction = rng.normal(0.0, 1.0, 3)
                norm = float(np.linalg.norm(direction))
            direction /= norm
            velocity = speed * direction
            obstacles.append(GroundTruthObstacleState(
                obstacle_id=obstacle_id, position_ned_m=position,
                velocity_ned_mps=velocity, radius_m=radius,
            ))
            obstacle_metadata.append(ObstacleGenerationMetadata(
                obstacle_id=obstacle_id, distance_m=distance,
                horizontal_angle_deg=horizontal_deg, vertical_angle_deg=vertical_deg,
                radius_m=radius, speed_mps=speed,
                velocity_unit_direction_ned=tuple(float(value) for value in direction),
            ))
        overlap_count = sum(
            float(np.linalg.norm(left.position_ned_m-right.position_ned_m))
            < left.radius_m+right.radius_m
            for index, left in enumerate(obstacles) for right in obstacles[index+1:]
        )
        metadata = ScenarioGenerationMetadata(
            sampled_start_ned_m=tuple(float(value) for value in start),
            sampled_goal_ned_m=tuple(float(value) for value in goal),
            vertical_separation_stratum=stratum, obstacle_count=obstacle_count,
            obstacles=tuple(obstacle_metadata), motion_model=cfg.motion_model,
            current=cfg.current, rng_namespace="scenario", count_draw_count=1,
            goal_depth_retry_count=goal_depth_retry_count,
            velocity_direction_retry_count=velocity_direction_retry_count,
            generation_retry_count=goal_depth_retry_count+velocity_direction_retry_count,
            initial_obstacle_overlap_count=overlap_count,
            scenario_generation_version=cfg.distribution_version,
        )
        scenario_id = f"train-v1-seed-{root_seed}-idx-{scenario_index}"
        return TrainingScenario(
            scenario_id=scenario_id, distribution_version=cfg.distribution_version,
            root_seed=root_seed, scenario_index=scenario_index, initial_auv_state=auv,
            initial_obstacle_states=tuple(obstacles), goal_position_ned_m=goal,
            metadata=metadata,
        )


def make_local_navigation_env(
    project_config: ProjectConfig,
    scenario: TrainingScenario,
    task_config: LocalTaskConfig | None = None,
) -> LocalNavigationEnv:
    """将场景逐值交给固定环境；不重采样、不改 reset 语义。"""

    return LocalNavigationEnv(
        project_config, scenario.initial_auv_state, scenario.initial_obstacle_states,
        scenario.goal_position_ned_m, scenario.scenario_id, task_config,
    )
