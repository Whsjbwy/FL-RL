"""验证延迟测量必须按发生时间戳回溯更新再重放。"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.tracking.delay_replay import (
    DelayedMeasurementReplayer,
    TimestampedPositionMeasurement,
)
from auv_risk_rl.types import KFTrackState


def test_delayed_measurement_replay_matches_chronological_processing(project_config) -> None:
    """
    验证相同测量集合无论到达顺序如何，按发生时间重放后得到相同当前后验。

    若失败，说明时间戳语义或 replay 顺序错误，不能把迟到测量直接当当前量使用。
    """

    initial_track = KFTrackState(
        obstacle_id=7,
        state_mean_ned=np.zeros(6, dtype=np.float64),
        state_covariance_ned=np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).astype(np.float64),
        state_timestamp_s=0.0,
        last_measurement_timestamp_s=0.0,
    )
    spectral_density = np.diag(
        np.asarray(project_config.kf.acceleration_spectral_density_m2_s3, dtype=np.float64)
    )
    first_measurement = TimestampedPositionMeasurement(
        measurement_timestamp_s=0.2,
        position_ned_m=np.array([0.1, 0.0, 0.0], dtype=np.float64),
        covariance_ned_m2=np.diag([0.04, 0.04, 0.09]).astype(np.float64),
    )
    second_measurement = TimestampedPositionMeasurement(
        measurement_timestamp_s=0.4,
        position_ned_m=np.array([0.25, 0.0, 0.0], dtype=np.float64),
        covariance_ned_m2=np.diag([0.04, 0.04, 0.09]).astype(np.float64),
    )

    chronological = DelayedMeasurementReplayer(
        initial_track,
        spectral_density,
        project_config.kf.covariance_symmetry_tolerance,
        project_config.kf.covariance_psd_tolerance,
    )
    chronological.add_measurement(first_measurement)
    chronological.add_measurement(second_measurement)

    delayed = DelayedMeasurementReplayer(
        initial_track,
        spectral_density,
        project_config.kf.covariance_symmetry_tolerance,
        project_config.kf.covariance_psd_tolerance,
    )
    delayed.add_measurement(second_measurement)
    delayed.add_measurement(first_measurement)

    chronological_state = chronological.replay_to(current_timestamp_s=0.6)
    delayed_state = delayed.replay_to(current_timestamp_s=0.6)
    mean_error = float(
        np.max(np.abs(chronological_state.state_mean_ned - delayed_state.state_mean_ned))
    )
    covariance_error = float(
        np.max(
            np.abs(
                chronological_state.state_covariance_ned
                - delayed_state.state_covariance_ned
            )
        )
    )
    tolerance = 1.0e-12
    assert mean_error <= tolerance, f"expected <= {tolerance}, actual={mean_error}"
    assert covariance_error <= tolerance, (
        f"expected <= {tolerance}, actual={covariance_error}"
    )
