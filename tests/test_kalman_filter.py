"""验证 CV-KF Eq. (19)–(25) 的数值与协方差性质。"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.prediction.predictor import predict_position_distribution
from auv_risk_rl.tracking.kalman_filter import predict_cv_state, update_position_measurement
from auv_risk_rl.types import KFTrackState


def _spectral_density_from_config(project_config) -> np.ndarray:
    """把配置中的三轴谱密度转换为 NED 对角矩阵。"""

    return np.diag(
        np.asarray(project_config.kf.acceleration_spectral_density_m2_s3, dtype=np.float64)
    )


def test_kf_covariance_remains_psd_after_joseph_update(project_config) -> None:
    """
    验证 Joseph form 更新后协方差保持对称 PSD。

    失败时应检查增益求解、H 矩阵和 Joseph 展开，而不是对负特征值强制 clip。
    """

    state_mean_ned = np.zeros(6, dtype=np.float64)
    state_covariance_ned = np.diag([2.0, 2.0, 2.0, 1.0, 1.0, 1.0]).astype(np.float64)
    predicted_mean, predicted_covariance = predict_cv_state(
        state_mean_ned=state_mean_ned,
        state_covariance_ned=state_covariance_ned,
        delta_t_s=0.2,
        acceleration_spectral_density_ned_m2_s3=_spectral_density_from_config(project_config),
        symmetry_tolerance=project_config.kf.covariance_symmetry_tolerance,
        psd_tolerance=project_config.kf.covariance_psd_tolerance,
    )
    measurement_position_ned_m = np.array([1.0, -0.5, 0.2], dtype=np.float64)
    measurement_covariance_ned_m2 = np.diag([0.04, 0.04, 0.09]).astype(np.float64)
    _, posterior_covariance, _, _ = update_position_measurement(
        predicted_state_mean_ned=predicted_mean,
        predicted_state_covariance_ned=predicted_covariance,
        measurement_position_ned_m=measurement_position_ned_m,
        measurement_covariance_ned_m2=measurement_covariance_ned_m2,
        symmetry_tolerance=project_config.kf.covariance_symmetry_tolerance,
        psd_tolerance=project_config.kf.covariance_psd_tolerance,
    )

    asymmetry_error = float(np.max(np.abs(posterior_covariance - posterior_covariance.T)))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(posterior_covariance)))
    tolerance = 1.0e-10
    assert asymmetry_error <= tolerance, (
        f"expected symmetry error <= {tolerance}, actual={asymmetry_error}"
    )
    assert minimum_eigenvalue >= -tolerance, (
        f"expected min eigenvalue >= {-tolerance}, actual={minimum_eigenvalue}"
    )


def test_arbitrary_horizon_prediction_matches_direct_cv_blocks(project_config) -> None:
    """验证 Eq. (24)–(25) 的未来位置均值和协方差与直接状态传播一致。"""

    state_mean_ned = np.array([1.0, 2.0, 3.0, 0.5, -0.2, 0.1], dtype=np.float64)
    state_covariance_ned = np.array(
        [
            [1.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 1.2, 0.0, 0.0, -0.05, 0.0],
            [0.0, 0.0, 1.5, 0.0, 0.0, 0.03],
            [0.1, 0.0, 0.0, 0.4, 0.0, 0.0],
            [0.0, -0.05, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.03, 0.0, 0.0, 0.6],
        ],
        dtype=np.float64,
    )
    track_state = KFTrackState(
        obstacle_id=1,
        state_mean_ned=state_mean_ned,
        state_covariance_ned=state_covariance_ned,
        state_timestamp_s=0.0,
        last_measurement_timestamp_s=0.0,
    )
    horizon_s = 2.3
    prediction = predict_position_distribution(
        track_state=track_state,
        prediction_horizon_s=horizon_s,
        acceleration_spectral_density_ned_m2_s3=_spectral_density_from_config(project_config),
        symmetry_tolerance=project_config.kf.covariance_symmetry_tolerance,
        psd_tolerance=project_config.kf.covariance_psd_tolerance,
    )
    direct_mean, direct_covariance = predict_cv_state(
        state_mean_ned=state_mean_ned,
        state_covariance_ned=state_covariance_ned,
        delta_t_s=horizon_s,
        acceleration_spectral_density_ned_m2_s3=_spectral_density_from_config(project_config),
        symmetry_tolerance=project_config.kf.covariance_symmetry_tolerance,
        psd_tolerance=project_config.kf.covariance_psd_tolerance,
    )
    mean_error = float(np.max(np.abs(prediction.position_mean_ned_m - direct_mean[0:3])))
    covariance_error = float(
        np.max(np.abs(prediction.position_covariance_ned_m2 - direct_covariance[0:3, 0:3]))
    )
    tolerance = 1.0e-12
    assert mean_error <= tolerance, f"expected <= {tolerance}, actual={mean_error}"
    assert covariance_error <= tolerance, (
        f"expected <= {tolerance}, actual={covariance_error}"
    )
