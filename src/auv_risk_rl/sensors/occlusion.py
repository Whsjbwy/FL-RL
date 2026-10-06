"""LOCAL目标包络遮挡前端；没有声传播、风险膨胀或速度信息。"""

import numpy as np

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.sensors.rays import sphere_ray_distance
from auv_risk_rl.sensors.sonar import generate_sonar_detections
from auv_risk_rl.types import AUVState, GroundTruthObstacleState, SensorDetection


def unoccluded_centers(auv_state: AUVState,
                       obstacles: tuple[GroundTruthObstacleState, ...],
                       ) -> tuple[GroundTruthObstacleState, ...]:
    """SIMULATION IMPLEMENTATION DETAIL：中心视线被更近物理包络截断则缺测。"""

    visible = []
    for target in obstacles:
        relative_ned_m = target.position_ned_m - auv_state.position_ned_m
        distance_m = float(np.linalg.norm(relative_ned_m))
        if distance_m == 0:
            visible.append(target)
            continue
        direction_ned = relative_ned_m / distance_m
        target_surface_m = max(0.0, distance_m - target.radius_m)
        blocked = any(
            other.obstacle_id != target.obstacle_id
            and sphere_ray_distance(other.position_ned_m - auv_state.position_ned_m,
                                    other.radius_m, direction_ned) < target_surface_m
            for other in obstacles
        )
        if not blocked:
            visible.append(target)
    return tuple(visible)


def capture_center_detections(auv_state: AUVState,
                              obstacles: tuple[GroundTruthObstacleState, ...],
                              timestamp_s: float, tick: int, config: ProjectConfig,
                              noise_rng: np.random.Generator,
                              dropout_rng: np.random.Generator,
                              ) -> tuple[SensorDetection, ...]:
    """遮挡过滤后复用冻结中心传感器；发生时刻与整数到达tick保持分离。"""

    return generate_sonar_detections(
        auv_state, unoccluded_centers(auv_state, obstacles), timestamp_s,
        config.sensor, config.dynamics, noise_rng,
        measurement_control_tick=tick, dropout_rng=dropout_rng,
    )
