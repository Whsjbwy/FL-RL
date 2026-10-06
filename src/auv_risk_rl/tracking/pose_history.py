"""
AUV 测量发生时刻位姿历史模块。

功能：
1. 保存控制时刻的 AUV 状态快照；
2. 按精确时间戳提供历史位姿；
3. 防止延迟检测使用到达时刻位姿做 Body→NED 转换。
"""

from __future__ import annotations

import math

import numpy as np

from auv_risk_rl.exceptions import TimestampOrderError
from auv_risk_rl.types import AUVState


class AUVPoseHistory:
    """
    保存离散控制时刻 AUV 位姿历史。

    对应技术协议：
        Eq. (17) 与第 9 章 Delay 规则。

    输入/输出：
        输入 AUVState 与世界时间；按测量发生时间返回同一时刻 AUVState。

    shape/单位/坐标系：
        AUV 位置 shape=(3,) m，NED；姿态 rad；时间 s。

    关键假设：
        传感器在已登记控制时刻采样；主实验自身位姿准确。

    重要限制：
        当前 reference implementation 只支持精确登记时间戳，不对任意时间做姿态插值。
    """

    def __init__(self, timestamp_tolerance_s: float = 1.0e-10) -> None:
        """创建空历史；时间容差只用于识别浮点表示相同的控制时刻。"""

        if timestamp_tolerance_s <= 0.0 or not math.isfinite(timestamp_tolerance_s):
            raise ValueError("timestamp_tolerance_s 必须为正有限数。")
        self._timestamp_tolerance_s = timestamp_tolerance_s
        self._samples: list[tuple[float, AUVState]] = []

    def add(self, timestamp_s: float, auv_state: AUVState) -> None:
        """
        保存一个 AUV 状态快照。

        参数：
            timestamp_s:
                世界时间，单位 s。
            auv_state:
                同一时刻 AUVState。

        返回：
            无。

        关键假设：
            新增时间戳不早于最后历史时刻。

        重要限制：
            不允许无声覆盖同一时间戳下不同状态。
        """

        if not math.isfinite(timestamp_s):
            raise TimestampOrderError("timestamp_s 必须有限。")
        if self._samples and timestamp_s < self._samples[-1][0] - self._timestamp_tolerance_s:
            raise TimestampOrderError(
                "AUV 位姿历史必须按时间单调写入："
                f"last={self._samples[-1][0]}, new={timestamp_s}。"
            )
        if self._samples and abs(timestamp_s - self._samples[-1][0]) <= self._timestamp_tolerance_s:
            previous_state = self._samples[-1][1]
            is_same_state = bool(
                np.array_equal(previous_state.position_ned_m, auv_state.position_ned_m)
                and previous_state.yaw_rad == auv_state.yaw_rad
                and previous_state.pitch_rad == auv_state.pitch_rad
                and previous_state.surge_speed_mps == auv_state.surge_speed_mps
                and previous_state.yaw_rate_rad_s == auv_state.yaw_rate_rad_s
                and previous_state.pitch_rate_rad_s == auv_state.pitch_rate_rad_s
            )
            if not is_same_state:
                raise TimestampOrderError("同一测量时刻出现不一致的 AUV 位姿快照。")
            return
        self._samples.append((float(timestamp_s), auv_state))

    def get_exact(self, timestamp_s: float) -> AUVState:
        """
        取得与测量发生时刻匹配的 AUV 状态。

        参数：
            timestamp_s:
                测量发生时刻，单位 s。

        返回：
            AUVState。

        关键假设：
            该时刻已通过 add 登记。

        重要限制：
            找不到历史时显式报错，禁止退化为使用当前位姿。
        """

        for sample_timestamp_s, auv_state in reversed(self._samples):
            if abs(sample_timestamp_s - timestamp_s) <= self._timestamp_tolerance_s:
                return auv_state
        raise TimestampOrderError(
            "缺少测量发生时刻的 AUV 位姿历史，禁止用到达时刻位姿代替："
            f"measurement_timestamp_s={timestamp_s}。"
        )
