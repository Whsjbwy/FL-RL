"""
三维 Constant-Velocity 连续白加速度模型。

功能：
1. 构造任意时间间隔的状态转移矩阵 F(T)；
2. 对连续白加速度谱密度精确离散得到 Q(T)；
3. 为 KF 与未来预测共享同一数学实现。

说明：
本模块对应技术协议 v2.0 Eq. (11)–(15)。Q 的单位必须按 pp/pv/vv 分块理解，
不能用简单 qI 替代连续噪声积分结果。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]


def cv_state_transition(delta_t_s: float) -> FloatMatrix:
    """
    构造三维 CV 状态转移矩阵。

    对应技术协议：
        Eq. (12)

    参数：
        delta_t_s:
            时间间隔，单位 s。

    返回：
        state_transition:
            shape=(6,6)，坐标系 NED；状态顺序为 [position, velocity]。
            分块量纲中右上块的时间因子负责将 m/s 转为 m。

    关键假设：
        A3。

    重要限制：
        F 的不同分块具有不同物理量纲，不能把整个矩阵称为无量纲矩阵。
    """

    if delta_t_s < 0.0:
        raise ValueError(f"delta_t_s 不能为负，actual={delta_t_s}。")

    identity_3 = np.eye(3, dtype=np.float64)
    state_transition = np.eye(6, dtype=np.float64)
    state_transition[0:3, 3:6] = delta_t_s * identity_3
    return state_transition


def cv_process_noise_covariance(
    delta_t_s: float,
    acceleration_spectral_density_ned_m2_s3: FloatMatrix,
) -> FloatMatrix:
    """
    精确离散三维 CV 连续白加速度过程噪声协方差。

    对应技术协议：
        Eq. (13)–(15)

    参数：
        delta_t_s:
            离散时间间隔，单位 s。
        acceleration_spectral_density_ned_m2_s3:
            三轴白加速度谱密度，shape=(3,3)，单位 m^2/s^3，NED。

    返回：
        process_noise_covariance:
            shape=(6,6)，坐标系 NED；pp/pv/vv 分块单位分别为 m^2、m^2/s、m^2/s^2。

    关键假设：
        A3、A4。

    重要限制：
        在假设 A3–A4 成立时条件精确。不能用 qI 忽略时间积分和分块单位。
    """

    if delta_t_s < 0.0:
        raise ValueError(f"delta_t_s 不能为负，actual={delta_t_s}。")
    if acceleration_spectral_density_ned_m2_s3.shape != (3, 3):
        raise ValueError(
            "acceleration_spectral_density_ned_m2_s3 shape 必须为 (3,3)，"
            f"actual={acceleration_spectral_density_ned_m2_s3.shape}。"
        )

    delta_t_2_s2 = delta_t_s * delta_t_s
    delta_t_3_s3 = delta_t_2_s2 * delta_t_s

    process_noise_covariance = np.block(
        [
            [
                (delta_t_3_s3 / 3.0) * acceleration_spectral_density_ned_m2_s3,
                (delta_t_2_s2 / 2.0) * acceleration_spectral_density_ned_m2_s3,
            ],
            [
                (delta_t_2_s2 / 2.0) * acceleration_spectral_density_ned_m2_s3,
                delta_t_s * acceleration_spectral_density_ned_m2_s3,
            ],
        ]
    )
    # 解析式理论上严格对称；这里只去除浮点装配造成的舍入级非对称。
    return 0.5 * (process_noise_covariance + process_noise_covariance.T)
