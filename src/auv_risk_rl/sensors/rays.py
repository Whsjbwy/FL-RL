"""LOCAL 第9章物理包络射线；只读取当前位置/半径，不读取运动或风险。"""

from dataclasses import dataclass

import numpy as np

from auv_risk_rl.config import SensorConfig
from auv_risk_rl.frames import rotation_body_to_ned
from auv_risk_rl.types import AUVState, GroundTruthObstacleState


@dataclass(frozen=True)
class RayFrame:
    """同一时刻45条Body射线的float64米制距离和0/1有效mask。"""

    ranges_m: np.ndarray
    valid_masks: np.ndarray


def ray_directions_body(sensor: SensorConfig) -> np.ndarray:
    """返回(45,3)单位向量；SIMULATION IMPLEMENTATION DETAIL：垂直外层扇区中心。"""

    bearings = (np.arange(9) + 0.5) * np.deg2rad(sensor.horizontal_fov_deg) / 9
    bearings -= np.deg2rad(sensor.horizontal_fov_deg) / 2
    elevations = (np.arange(5) + 0.5) * np.deg2rad(sensor.vertical_fov_deg) / 5
    elevations -= np.deg2rad(sensor.vertical_fov_deg) / 2
    return np.array([
        [np.cos(e) * np.cos(b), np.cos(e) * np.sin(b), -np.sin(e)]
        for e in elevations for b in bearings
    ], dtype=np.float64)


def sphere_ray_distance(center_body_m: np.ndarray, radius_m: float,
                        direction_body: np.ndarray) -> float:
    """从Body原点沿单位射线到物理球的最近非负交距；无交点为inf，内部为0。"""

    if radius_m <= 0 or not np.isfinite(radius_m):
        raise ValueError("物理半径必须正且有限。")
    squared_distance = float(center_body_m @ center_body_m)
    if squared_distance <= radius_m**2:
        return 0.0
    projection = float(center_body_m @ direction_body)
    perpendicular_sq = max(0.0, squared_distance - projection**2)
    discriminant = radius_m**2 - perpendicular_sq
    if projection < 0 or discriminant < 0:
        return float("inf")
    return max(0.0, projection - float(np.sqrt(discriminant)))


def cast_sonar_rays(auv_state: AUVState,
                    obstacles: tuple[GroundTruthObstacleState, ...],
                    sensor: SensorConfig) -> RayFrame:
    """按LOCAL物理表面求最近交距，边界<=量程；无回波量程/mask0，不膨胀半径。"""

    rotation = rotation_body_to_ned(auv_state.yaw_rad, auv_state.pitch_rad).T
    centers = [(rotation @ (o.position_ned_m - auv_state.position_ned_m), o.radius_m)
               for o in obstacles]
    ranges = np.full(45, sensor.range_m, dtype=np.float64)
    masks = np.zeros(45, dtype=np.float64)
    for index, direction in enumerate(ray_directions_body(sensor)):
        nearest = min((sphere_ray_distance(c, r, direction) for c, r in centers),
                      default=float("inf"))
        if nearest <= sensor.range_m:
            ranges[index] = nearest
            masks[index] = 1.0
    return RayFrame(ranges, masks)
