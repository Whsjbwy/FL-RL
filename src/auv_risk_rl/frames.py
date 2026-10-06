"""
AUV 坐标系与旋转变换模块。

功能：
1. 构造 Body→NED 旋转矩阵；
2. 完成相对位置、相对速度与协方差坐标变换；
3. 提供坐标一致性所需的基础数学函数。

说明：
本模块严格遵循技术协议 v2.0 第 6 章与 Eq. (1)–(4)。主模型 roll 恒为零，
但不能据此错误地把机体系 roll-rate 分量也强制设为零。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .exceptions import CoordinateFrameError

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


def rotation_body_to_ned(yaw_rad: float, pitch_rad: float) -> FloatMatrix:
    """
    构造 roll=0 条件下的 Body→NED 旋转矩阵。

    对应技术协议：
        Eq. (1)

    数学模型：
        ZYX Euler 旋转，在主模型中 roll=0。

    参数：
        yaw_rad:
            航向角，单位 rad。
        pitch_rad:
            俯仰角，单位 rad；正俯仰表示抬头。

    返回：
        rotation_body_to_ned_matrix:
            旋转矩阵，shape=(3,3)，无量纲，将 Body 向量变换到 NED。

    关键假设：
        A1、A2。

    重要限制：
        使用 NED 的 Down 正方向，因此正 pitch 时 Body 前向轴在 NED 中具有负 Down 分量。
    """

    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    cos_pitch = float(np.cos(pitch_rad))
    sin_pitch = float(np.sin(pitch_rad))

    return np.array(
        [
            [cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch],
            [sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch],
            [-sin_pitch, 0.0, cos_pitch],
        ],
        dtype=np.float64,
    )


def relative_position_body(
    auv_position_ned_m: FloatVector,
    obstacle_position_ned_m: FloatVector,
    yaw_rad: float,
    pitch_rad: float,
) -> FloatVector:
    """
    将 NED 中的位置差转换为当前冻结 Body 系相对位置。

    对应技术协议：
        Eq. (2)

    参数：
        auv_position_ned_m:
            AUV 位置，shape=(3,)，单位 m，NED。
        obstacle_position_ned_m:
            障碍位置，shape=(3,)，单位 m，NED。
        yaw_rad:
            AUV 航向角，单位 rad。
        pitch_rad:
            AUV 俯仰角，单位 rad。

    返回：
        relative_position_body_m:
            相对位置，shape=(3,)，单位 m，Body。

    关键假设：
        A1、A2。

    重要限制：
        必须先在同一 NED 坐标系做位置差，再旋转到 Body；禁止跨坐标系直接相减。
    """

    rotation_matrix = rotation_body_to_ned(yaw_rad, pitch_rad)
    relative_position_ned_m = obstacle_position_ned_m - auv_position_ned_m
    return rotation_matrix.T @ relative_position_ned_m


def angular_velocity_body_from_euler_rates(
    yaw_rate_rad_s: float,
    pitch_rate_rad_s: float,
    pitch_rad: float,
) -> FloatVector:
    """
    将 roll=0 的 Euler yaw/pitch 变化率转换为机体系角速度。

    对应技术协议：
        Eq. (4)

    参数：
        yaw_rate_rad_s:
            Euler yaw 变化率，单位 rad/s。
        pitch_rate_rad_s:
            Euler pitch 变化率，单位 rad/s。
        pitch_rad:
            当前俯仰角，单位 rad。

    返回：
        angular_velocity_body_rad_s:
            机体系角速度 [p,q,r]，shape=(3,)，单位 rad/s，Body。

    关键假设：
        A1、A2，roll=0 且远离 Euler 奇异点。

    重要限制：
        即使 roll 角恒为零，yaw 变化在 pitch 非零时仍会投影出 Body x 轴角速度分量，
        因而不能简单返回 [0, pitch_rate, yaw_rate]。
    """

    return np.array(
        [
            -yaw_rate_rad_s * np.sin(pitch_rad),
            pitch_rate_rad_s,
            yaw_rate_rad_s * np.cos(pitch_rad),
        ],
        dtype=np.float64,
    )


def relative_velocity_body(
    relative_position_body_m: FloatVector,
    auv_velocity_ned_mps: FloatVector,
    obstacle_velocity_ned_mps: FloatVector,
    yaw_rad: float,
    pitch_rad: float,
    yaw_rate_rad_s: float,
    pitch_rate_rad_s: float,
) -> FloatVector:
    """
    计算旋转 Body 坐标系下相对位置的一阶导数。

    对应技术协议：
        Eq. (3)–(4)

    参数：
        relative_position_body_m:
            当前相对位置，shape=(3,)，单位 m，Body。
        auv_velocity_ned_mps:
            AUV 地速，shape=(3,)，单位 m/s，NED。
        obstacle_velocity_ned_mps:
            障碍地速，shape=(3,)，单位 m/s，NED。
        yaw_rad, pitch_rad:
            AUV 姿态角，单位 rad。
        yaw_rate_rad_s, pitch_rate_rad_s:
            Euler 角变化率，单位 rad/s。

    返回：
        relative_velocity_body_mps:
            相对位置在旋转 Body 系中的导数，shape=(3,)，单位 m/s，Body。

    关键假设：
        A1、A2。

    重要限制：
        这里必须包含 -ω×r 的旋转系项；只旋转 NED 相对地速会漏掉坐标系自身转动。
    """

    rotation_matrix = rotation_body_to_ned(yaw_rad, pitch_rad)
    relative_velocity_ned_mps = obstacle_velocity_ned_mps - auv_velocity_ned_mps
    angular_velocity_body_rad_s = angular_velocity_body_from_euler_rates(
        yaw_rate_rad_s=yaw_rate_rad_s,
        pitch_rate_rad_s=pitch_rate_rad_s,
        pitch_rad=pitch_rad,
    )
    return (
        rotation_matrix.T @ relative_velocity_ned_mps
        - np.cross(angular_velocity_body_rad_s, relative_position_body_m)
    )


def rotate_covariance_body_to_ned(
    covariance_body_m2: FloatMatrix,
    yaw_rad: float,
    pitch_rad: float,
) -> FloatMatrix:
    """
    将三维位置协方差从 Body 旋转到 NED。

    对应技术协议：
        Eq. (4)、Eq. (17)

    参数：
        covariance_body_m2:
            协方差，shape=(3,3)，单位 m^2，Body。
        yaw_rad, pitch_rad:
            AUV 姿态角，单位 rad。

    返回：
        covariance_ned_m2:
            协方差，shape=(3,3)，单位 m^2，NED。

    关键假设：
        A1、A2、A5。

    重要限制：
        均值和协方差必须使用同一旋转；不能只变换测量均值而保留 Body 协方差。
    """

    if covariance_body_m2.shape != (3, 3):
        raise CoordinateFrameError(
            f"协方差 shape 应为 (3,3)，实际为 {covariance_body_m2.shape}。"
        )
    rotation_matrix = rotation_body_to_ned(yaw_rad, pitch_rad)
    covariance_ned_m2 = rotation_matrix @ covariance_body_m2 @ rotation_matrix.T
    # 这里只修正浮点乘法带来的舍入级非对称，不改变协方差的特征值结构。
    return 0.5 * (covariance_ned_m2 + covariance_ned_m2.T)
