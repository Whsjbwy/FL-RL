"""
延迟测量按发生时间戳重放的参考实现。

功能：
1. 保存测量发生时间与对应 NED 位置测量；
2. 对迟到测量按时间戳重新排序；
3. 从锚点后验重新执行 predict/update，再传播到当前时刻。

说明：
Stage 0 优先正确性与可审计性，因此这里保留 reference implementation，
不为速度实现复杂的局部回滚缓存。后续如优化，必须与本实现交叉验证。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from auv_risk_rl.exceptions import TimestampOrderError
from auv_risk_rl.tracking.kalman_filter import predict_cv_state, update_position_measurement
from auv_risk_rl.types import KFTrackState


@dataclass(frozen=True)
class TimestampedPositionMeasurement:
    """
    延迟重放使用的 NED 位置测量记录。

    对应技术协议：
        Eq. (17)、Eq. (19)–(22) 与第 9 章 Delay 规则。

    参数：
        measurement_timestamp_s:
            测量发生时刻，单位 s。
        position_ned_m:
            位置测量，shape=(3,)，单位 m，NED。
        covariance_ned_m2:
            测量协方差，shape=(3,3)，单位 m^2，NED。

    关键假设：
        A3–A6。

    重要限制：
        本结构只承载已经完成 Body→NED 转换的测量，不允许混入到达时刻位姿转换结果。
    """

    measurement_timestamp_s: float
    position_ned_m: np.ndarray
    covariance_ned_m2: np.ndarray


class DelayedMeasurementReplayer:
    """
    单目标 KF 延迟测量 reference replay。

    对应技术协议：
        Eq. (19)–(22)；第 9、10 章延迟测量按发生时间回溯更新再重放。

    输入/输出：
        输入为一个锚点 KFTrackState 和后续 NED 位置测量；输出为指定当前时刻的 KFTrackState。

    shape/单位/坐标系：
        KF 状态 shape=(6,)，协方差 shape=(6,6)，NED；时间单位 s。

    关键假设：
        A3–A6；同一目标正确关联。

    重要限制：
        这是 Stage 0 的清晰参考实现，会从锚点重新播放全部测量，复杂度随历史长度线性增长。
        后续高性能实现不得改变时间戳语义，并必须与本实现交叉验证。
    """

    def __init__(
        self,
        anchor_track_state: KFTrackState,
        acceleration_spectral_density_ned_m2_s3: np.ndarray,
        symmetry_tolerance: float,
        psd_tolerance: float,
    ) -> None:
        self._anchor_track_state = anchor_track_state
        self._acceleration_spectral_density_ned_m2_s3 = (
            acceleration_spectral_density_ned_m2_s3.copy()
        )
        self._symmetry_tolerance = symmetry_tolerance
        self._psd_tolerance = psd_tolerance
        self._measurements: list[TimestampedPositionMeasurement] = []

    def add_measurement(self, measurement: TimestampedPositionMeasurement) -> None:
        """
        加入一条测量，并按发生时间戳排序。

        对应技术协议：
            第 9 章 Delay；核心滤波仍对应 Eq. (19)–(22)。

        参数：
            measurement:
                NED 位置测量；位置 shape=(3,)，单位 m；协方差 shape=(3,3)，单位 m^2。

        返回：
            无。

        关键假设：
            测量发生时刻不得早于锚点后验时刻。

        重要限制：
            不允许通过把迟到测量时间改成当前时间来简化实现。
        """

        if measurement.measurement_timestamp_s < self._anchor_track_state.state_timestamp_s:
            raise TimestampOrderError(
                "测量发生时刻早于重放锚点："
                f"measurement={measurement.measurement_timestamp_s}, "
                f"anchor={self._anchor_track_state.state_timestamp_s}。"
            )
        self._measurements.append(measurement)
        self._measurements.sort(key=lambda item: item.measurement_timestamp_s)

    def replay_to(self, current_timestamp_s: float) -> KFTrackState:
        """
        将锚点后验、全部已到达测量重放到指定当前时刻。

        对应技术协议：
            Eq. (19)–(22)

        参数：
            current_timestamp_s:
                目标当前时刻，单位 s。

        返回：
            KFTrackState：当前时刻后验/预测状态，NED。

        关键假设：
            A3–A6；输入测量均已实际到达系统。

        重要限制：
            未来时刻测量不会被提前使用；若 measurement_timestamp_s > current_timestamp_s，
            本函数显式拒绝，以防未来信息泄漏。
    """

        if current_timestamp_s < self._anchor_track_state.state_timestamp_s:
            raise TimestampOrderError(
                "current_timestamp_s 早于锚点："
                f"current={current_timestamp_s}, "
                f"anchor={self._anchor_track_state.state_timestamp_s}。"
            )

        future_measurements = [
            measurement
            for measurement in self._measurements
            if measurement.measurement_timestamp_s > current_timestamp_s
        ]
        if future_measurements:
            raise TimestampOrderError(
                "重放集合包含当前时刻之后的测量，禁止提前使用未来信息："
                f"first_future={future_measurements[0].measurement_timestamp_s}, "
                f"current={current_timestamp_s}。"
            )

        state_mean_ned = self._anchor_track_state.state_mean_ned.copy()
        state_covariance_ned = self._anchor_track_state.state_covariance_ned.copy()
        state_timestamp_s = self._anchor_track_state.state_timestamp_s
        last_measurement_timestamp_s = self._anchor_track_state.last_measurement_timestamp_s

        for measurement in self._measurements:
            delta_t_s = measurement.measurement_timestamp_s - state_timestamp_s
            state_mean_ned, state_covariance_ned = predict_cv_state(
                state_mean_ned=state_mean_ned,
                state_covariance_ned=state_covariance_ned,
                delta_t_s=delta_t_s,
                acceleration_spectral_density_ned_m2_s3=(
                    self._acceleration_spectral_density_ned_m2_s3
                ),
                symmetry_tolerance=self._symmetry_tolerance,
                psd_tolerance=self._psd_tolerance,
            )
            (
                state_mean_ned,
                state_covariance_ned,
                _,
                _,
            ) = update_position_measurement(
                predicted_state_mean_ned=state_mean_ned,
                predicted_state_covariance_ned=state_covariance_ned,
                measurement_position_ned_m=measurement.position_ned_m,
                measurement_covariance_ned_m2=measurement.covariance_ned_m2,
                symmetry_tolerance=self._symmetry_tolerance,
                psd_tolerance=self._psd_tolerance,
            )
            state_timestamp_s = measurement.measurement_timestamp_s
            last_measurement_timestamp_s = measurement.measurement_timestamp_s

        remaining_delta_t_s = current_timestamp_s - state_timestamp_s
        state_mean_ned, state_covariance_ned = predict_cv_state(
            state_mean_ned=state_mean_ned,
            state_covariance_ned=state_covariance_ned,
            delta_t_s=remaining_delta_t_s,
            acceleration_spectral_density_ned_m2_s3=(
                self._acceleration_spectral_density_ned_m2_s3
            ),
            symmetry_tolerance=self._symmetry_tolerance,
            psd_tolerance=self._psd_tolerance,
        )
        return KFTrackState(
            obstacle_id=self._anchor_track_state.obstacle_id,
            state_mean_ned=state_mean_ned,
            state_covariance_ned=state_covariance_ned,
            state_timestamp_s=current_timestamp_s,
            last_measurement_timestamp_s=last_measurement_timestamp_s,
        )
