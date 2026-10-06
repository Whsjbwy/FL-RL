"""
简化声呐笛卡尔测量转换模块。

功能：
1. 将 Body 系相对位置测量转换为 NED 绝对位置测量；
2. 同步旋转测量协方差；
3. 保留测量发生时间戳，避免延迟测量被误当作当前量。
"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.frames import rotate_covariance_body_to_ned, rotation_body_to_ned
from auv_risk_rl.types import SensorDetection


def detection_to_ned_measurement(
    detection: SensorDetection,
    auv_position_ned_m: np.ndarray,
    auv_yaw_rad: float,
    auv_pitch_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将简化声呐相对位置测量转换为 NED 绝对位置测量。

    对应技术协议：
        Eq. (16)–(17)

    参数：
        detection:
            Body 系相对测量；位置 shape=(3,)，单位 m；协方差 shape=(3,3)，单位 m^2。
        auv_position_ned_m:
            测量发生时刻的 AUV 位置，shape=(3,)，单位 m，NED。
        auv_yaw_rad:
            测量发生时刻航向角，单位 rad。
        auv_pitch_rad:
            测量发生时刻俯仰角，单位 rad。

    返回：
        measurement_position_ned_m:
            障碍绝对位置测量，shape=(3,)，单位 m，NED。
        measurement_covariance_ned_m2:
            测量协方差，shape=(3,3)，单位 m^2，NED。

    关键假设：
        A1、A5；自身位姿在主实验中准确已知。

    重要限制：
        必须使用测量发生时刻而不是到达时刻的 AUV 位姿，否则延迟实验会引入系统时间错位。
    """

    rotation_matrix = rotation_body_to_ned(auv_yaw_rad, auv_pitch_rad)
    measurement_position_ned_m = (
        auv_position_ned_m + rotation_matrix @ detection.relative_position_body_m
    )
    measurement_covariance_ned_m2 = rotate_covariance_body_to_ned(
        detection.measurement_covariance_body_m2,
        auv_yaw_rad,
        auv_pitch_rad,
    )
    return measurement_position_ned_m, measurement_covariance_ned_m2
