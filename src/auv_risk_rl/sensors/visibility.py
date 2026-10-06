"""
简化前视声呐的几何可见性模块。

功能：
1. 将障碍相对位置转换到当前 Body 系；
2. 检查量程、水平 FOV 与垂直 FOV；
3. 将“几何可见”与随机 dropout 分开，便于审计两类缺测来源。

说明：
本模块不实现遮挡。复杂遮挡属于 Stage 1 基础环境验证的后续脚本项，不能在 Stage 0
未验证时假装已经覆盖。
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from auv_risk_rl.config import SensorConfig
from auv_risk_rl.frames import relative_position_body
from auv_risk_rl.types import AUVState, GroundTruthObstacleState

FloatVector = NDArray[np.float64]


def body_bearing_elevation_rad(relative_position_body_m: FloatVector) -> tuple[float, float]:
    """
    从 Body 前/右/下相对位置计算水平方位角和正向上仰角。

    对应技术协议：
        Eq. (18)

    参数：
        relative_position_body_m:
            shape=(3,)，单位 m，Body 前/右/下。

    返回：
        horizontal_bearing_rad:
            atan2(right, forward)，单位 rad。
        elevation_rad:
            向上为正；由于 Body z 轴向下，使用 atan2(-down, horizontal_range)，单位 rad。

    关键假设：
        A1；传感器与 Body 同轴。

    重要限制：
        这只是几何角，不包含真实声呐波束响应或声学传播。
    """

    if relative_position_body_m.shape != (3,) or not np.all(
        np.isfinite(relative_position_body_m)
    ):
        raise ValueError("relative_position_body_m 必须为有限 shape=(3,) Body 向量。")
    forward_m, right_m, down_m = relative_position_body_m
    horizontal_range_m = math.hypot(float(forward_m), float(right_m))
    horizontal_bearing_rad = math.atan2(float(right_m), float(forward_m))
    elevation_rad = math.atan2(-float(down_m), horizontal_range_m)
    return horizontal_bearing_rad, elevation_rad


def is_obstacle_geometrically_visible(
    auv_state: AUVState,
    obstacle_state: GroundTruthObstacleState,
    sensor_config: SensorConfig,
) -> tuple[bool, FloatVector]:
    """
    检查障碍中心是否位于简化声呐量程/FOV 内。

    对应技术协议：
        Eq. (16)、Eq. (18) 与第 9 章量程/FOV 规则。

    参数：
        auv_state:
            测量发生时刻 AUV 真值位姿；位置 NED。
        obstacle_state:
            同一时刻障碍真值；位置 NED。
        sensor_config:
            量程与 FOV 配置。

    返回：
        is_visible:
            是否满足纯几何可见条件。
        relative_position_body_m:
            障碍相对中心位置，shape=(3,)，单位 m，Body。

    关键假设：
        A1、A5；MVP 正确关联。

    重要限制：
        这里尚不实现目标间遮挡、海洋声学衰减和检测概率随距离变化。
    """

    relative_position_body_m = relative_position_body(
        auv_position_ned_m=auv_state.position_ned_m,
        obstacle_position_ned_m=obstacle_state.position_ned_m,
        yaw_rad=auv_state.yaw_rad,
        pitch_rad=auv_state.pitch_rad,
    )
    range_m = float(np.linalg.norm(relative_position_body_m))
    horizontal_bearing_rad, elevation_rad = body_bearing_elevation_rad(
        relative_position_body_m
    )
    half_horizontal_fov_rad = math.radians(sensor_config.horizontal_fov_deg) / 2.0
    half_vertical_fov_rad = math.radians(sensor_config.vertical_fov_deg) / 2.0
    is_visible = bool(
        range_m <= sensor_config.range_m
        and abs(horizontal_bearing_rad) <= half_horizontal_fov_rad
        and abs(elevation_rad) <= half_vertical_fov_rad
    )
    return is_visible, relative_position_body_m
