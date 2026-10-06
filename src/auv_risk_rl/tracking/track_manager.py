"""
多目标 CV-KF Stage 0 reference 管理器。

功能：
1. 将已到达 SensorDetection 按其测量发生时刻的 AUV 位姿转换到 NED；
2. 首次检测按协议工程先验初始化位置/零速度；
3. 保存每个目标全部已到达位置测量并按发生时间重放；
4. 缺测时只传播，不删除目标。

说明：
MVP 默认正确关联，因此 obstacle_id 可直接作为 track key。这里不解决数据关联问题。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.sensors.measurement import detection_to_ned_measurement
from auv_risk_rl.tracking.delay_replay import (
    DelayedMeasurementReplayer,
    TimestampedPositionMeasurement,
)
from auv_risk_rl.tracking.pose_history import AUVPoseHistory
from auv_risk_rl.types import KFTrackState, SensorDetection


@dataclass
class _TrackMeasurementHistory:
    """单目标已到达 NED 位置测量历史；仅 track manager 内部使用。"""

    measurements: list[TimestampedPositionMeasurement]


class MultiTargetCVTracker:
    """
    Stage 0 多目标 CV-KF reference tracker。

    对应技术协议：
        Eq. (17)、Eq. (19)–(25) 与第 9–10 章初始化/缺测/延迟规则。

    输入/输出：
        输入已到达 SensorDetection 与 AUVPoseHistory；输出当前时刻 KFTrackState 元组。

    shape/单位/坐标系：
        检测输入为 Body m/m²；内部测量和 KF 状态统一 NED；时间单位 s。

    关键假设：
        A1、A3–A6；正确关联，自身位姿准确。

    重要限制：
        这是优先正确性的重放 reference implementation；每次查询会从最早测量重建目标轨迹。
    """

    def __init__(self, config: ProjectConfig) -> None:
        """保存冻结配置并创建空目标历史。"""

        self._config = config
        self._history_by_obstacle_id: dict[int, _TrackMeasurementHistory] = {}

    def _convert_detection(
        self,
        detection: SensorDetection,
        auv_pose_history: AUVPoseHistory,
    ) -> TimestampedPositionMeasurement:
        """
        使用测量发生时刻 AUV 位姿把 Body 检测转换为 NED 测量。

        这是延迟链中最关键的数据时序边界；不能使用 detection.arrival_timestamp_s 对应位姿。
        """

        measurement_auv_state = auv_pose_history.get_exact(
            detection.measurement_timestamp_s
        )
        position_ned_m, covariance_ned_m2 = detection_to_ned_measurement(
            detection=detection,
            auv_position_ned_m=measurement_auv_state.position_ned_m,
            auv_yaw_rad=measurement_auv_state.yaw_rad,
            auv_pitch_rad=measurement_auv_state.pitch_rad,
        )
        return TimestampedPositionMeasurement(
            measurement_timestamp_s=detection.measurement_timestamp_s,
            position_ned_m=position_ned_m,
            covariance_ned_m2=covariance_ned_m2,
        )

    def add_arrived_detections(
        self,
        arrived_detections: tuple[SensorDetection, ...],
        auv_pose_history: AUVPoseHistory,
    ) -> None:
        """
        把当前已经到达的检测加入对应目标历史。

        对应技术协议：
            Eq. (17) 与第 9 章 Delay 规则。

        参数：
            arrived_detections:
                已由延迟队列释放的 SensorDetection 元组。
            auv_pose_history:
                AUV 历史位姿，用 measurement_timestamp_s 查找转换位姿。

        返回：
            无。

        关键假设：
            正确关联；同一目标同一测量时刻最多一条检测。

        重要限制：
            只接收“已经到达”的检测；未来 arrival 不应提前传入本函数。
        """

        for detection in arrived_detections:
            converted_measurement = self._convert_detection(detection, auv_pose_history)
            history = self._history_by_obstacle_id.setdefault(
                detection.obstacle_id,
                _TrackMeasurementHistory(measurements=[]),
            )
            duplicate_timestamp = any(
                abs(
                    existing.measurement_timestamp_s
                    - converted_measurement.measurement_timestamp_s
                )
                <= self._config.kf.covariance_symmetry_tolerance
                for existing in history.measurements
            )
            if duplicate_timestamp:
                raise ValueError(
                    "同一目标同一测量时刻出现重复检测："
                    f"obstacle_id={detection.obstacle_id}, "
                    f"timestamp={detection.measurement_timestamp_s}。"
                )
            history.measurements.append(converted_measurement)
            history.measurements.sort(key=lambda item: item.measurement_timestamp_s)

    def _build_track_at(
        self,
        obstacle_id: int,
        measurement_history: _TrackMeasurementHistory,
        current_timestamp_s: float,
    ) -> KFTrackState:
        """
        从最早已到达测量按协议初始化并重放到当前时刻。

        首检测位置均值等于测量、速度均值为零；位置协方差取测量协方差，速度标准差来自配置。
        该初始化是工程先验，不宣称等于真实 Bayes 后验。
        """

        measurements = measurement_history.measurements
        if not measurements:
            raise ValueError("空测量历史不能构造 KF track。")
        first_measurement = measurements[0]
        if first_measurement.measurement_timestamp_s > current_timestamp_s:
            raise ValueError("不能使用当前时刻之后的测量初始化 track。")

        initial_state_mean_ned = np.concatenate(
            (
                first_measurement.position_ned_m.astype(np.float64, copy=True),
                np.zeros(3, dtype=np.float64),
            )
        )
        initial_state_covariance_ned = np.zeros((6, 6), dtype=np.float64)
        initial_state_covariance_ned[0:3, 0:3] = first_measurement.covariance_ned_m2
        velocity_variance_m2_s2 = self._config.kf.initial_velocity_std_mps**2
        initial_state_covariance_ned[3:6, 3:6] = (
            velocity_variance_m2_s2 * np.eye(3, dtype=np.float64)
        )
        anchor_track = KFTrackState(
            obstacle_id=obstacle_id,
            state_mean_ned=initial_state_mean_ned,
            state_covariance_ned=initial_state_covariance_ned,
            state_timestamp_s=first_measurement.measurement_timestamp_s,
            last_measurement_timestamp_s=first_measurement.measurement_timestamp_s,
        )
        spectral_density_ned_m2_s3 = np.diag(
            np.asarray(
                self._config.kf.acceleration_spectral_density_m2_s3,
                dtype=np.float64,
            )
        )
        replayer = DelayedMeasurementReplayer(
            anchor_track_state=anchor_track,
            acceleration_spectral_density_ned_m2_s3=spectral_density_ned_m2_s3,
            symmetry_tolerance=self._config.kf.covariance_symmetry_tolerance,
            psd_tolerance=self._config.kf.covariance_psd_tolerance,
        )
        for measurement in measurements[1:]:
            if measurement.measurement_timestamp_s <= current_timestamp_s:
                replayer.add_measurement(measurement)
        return replayer.replay_to(current_timestamp_s)

    def get_track_states(self, current_timestamp_s: float) -> tuple[KFTrackState, ...]:
        """
        返回全部已维护目标传播到当前时刻的 KF 状态。

        对应技术协议：
            Eq. (19)–(25) 与“缺测仅 predict、不立即删除目标”的规则。

        参数：
            current_timestamp_s:
                当前世界时间，单位 s。

        返回：
            以 obstacle_id 排序的 KFTrackState 元组。

        关键假设：
            所有历史测量均已经实际到达系统。

        重要限制：
            当前 Stage 0 不实现超长未见目标删除；保守冻结问题留给后续机制实验记录。
        """

        return tuple(
            self._build_track_at(obstacle_id, history, current_timestamp_s)
            for obstacle_id, history in sorted(self._history_by_obstacle_id.items())
        )
