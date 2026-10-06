"""
动态障碍真值推进模块。

功能：
1. 按主实验严格 CV 真值推进障碍位置；
2. 保持障碍速度定义为 NED 地速；
3. 不在真值推进中偷偷加入海流或滤波过程噪声。
"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.exceptions import InvalidEnvironmentStateError
from auv_risk_rl.types import GroundTruthObstacleState


def propagate_cv_obstacle_truth(
    obstacle_state: GroundTruthObstacleState,
    delta_t_s: float,
) -> GroundTruthObstacleState:
    """
    按 CV 模型推进一个障碍真值状态。

    对应技术协议：
        Eq. (11)–(12)

    参数：
        obstacle_state:
            当前障碍真值；位置 shape=(3,) m、速度 shape=(3,) m/s，均为 NED。
        delta_t_s:
            真值推进时长，单位 s。

    返回：
        下一时刻 GroundTruthObstacleState；速度保持不变。

    关键假设：
        A3；Stage 0/主 ID 环境使用严格 CV 真值。

    重要限制：
        本函数不采样 KF 的过程噪声；滤波器非零 Q 是估计模型假设，不等于主真值随机驱动。
    """

    if delta_t_s < 0.0 or not np.isfinite(delta_t_s):
        raise InvalidEnvironmentStateError(f"delta_t_s 必须为非负有限数，actual={delta_t_s}。")
    if obstacle_state.position_ned_m.shape != (3,) or obstacle_state.velocity_ned_mps.shape != (3,):
        raise InvalidEnvironmentStateError("障碍位置与速度必须为 shape=(3,) NED 向量。")
    if not np.all(np.isfinite(obstacle_state.position_ned_m)) or not np.all(
        np.isfinite(obstacle_state.velocity_ned_mps)
    ):
        raise InvalidEnvironmentStateError("障碍位置与速度不得包含 NaN/Inf。")
    if obstacle_state.radius_m <= 0.0 or not np.isfinite(obstacle_state.radius_m):
        raise InvalidEnvironmentStateError(
            f"障碍物理半径必须为正有限数，actual={obstacle_state.radius_m}。"
        )

    next_position_ned_m = (
        obstacle_state.position_ned_m + delta_t_s * obstacle_state.velocity_ned_mps
    )
    return GroundTruthObstacleState(
        obstacle_id=obstacle_state.obstacle_id,
        position_ned_m=next_position_ned_m.astype(np.float64, copy=True),
        velocity_ned_mps=obstacle_state.velocity_ned_mps.astype(np.float64, copy=True),
        radius_m=float(obstacle_state.radius_m),
    )
