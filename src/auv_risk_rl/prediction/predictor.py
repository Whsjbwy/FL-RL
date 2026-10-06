"""
CV-KF 任意时域位置分布预测模块。

说明：
未来均值只是当前估计位置/速度在 CV 下的确定性展开，不应被描述成新增随机信息。
预测协方差保留位置—速度交叉协方差贡献。
"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.tracking.kalman_filter import predict_cv_state
from auv_risk_rl.types import KFTrackState, PredictionResult


def predict_position_distribution(
    track_state: KFTrackState,
    prediction_horizon_s: float,
    acceleration_spectral_density_ned_m2_s3: np.ndarray,
    symmetry_tolerance: float,
    psd_tolerance: float,
) -> PredictionResult:
    """
    预测指定未来时刻的障碍物位置分布。

    对应技术协议：
        Eq. (23)–(25)

    数学模型：
        CV-KF arbitrary-horizon prediction。

    参数：
        track_state:
            当前 KF 后验；state_mean shape=(6,)，state_covariance shape=(6,6)，坐标系 NED。
        prediction_horizon_s:
            预测时长，单位 s。
        acceleration_spectral_density_ned_m2_s3:
            白加速度谱密度，shape=(3,3)，单位 m^2/s^3，NED。
        symmetry_tolerance, psd_tolerance:
            协方差数值审计容差。

    返回：
        PredictionResult，其中：
            position_mean_ned_m shape=(3,)，单位 m，NED；
            position_covariance_ned_m2 shape=(3,3)，单位 m^2，NED。

    关键假设：
        A3–A5。

    重要限制：
        在假设 A3–A5 成立时条件精确。输出不包含未建模机动、错误关联、
        自身定位误差、未知海流等额外不确定性。
    """

    predicted_state_mean_ned, predicted_state_covariance_ned = predict_cv_state(
        state_mean_ned=track_state.state_mean_ned,
        state_covariance_ned=track_state.state_covariance_ned,
        delta_t_s=prediction_horizon_s,
        acceleration_spectral_density_ned_m2_s3=acceleration_spectral_density_ned_m2_s3,
        symmetry_tolerance=symmetry_tolerance,
        psd_tolerance=psd_tolerance,
    )
    position_mean_ned_m = predicted_state_mean_ned[0:3].copy()
    position_covariance_ned_m2 = predicted_state_covariance_ned[0:3, 0:3].copy()
    return PredictionResult(
        obstacle_id=track_state.obstacle_id,
        prediction_horizon_s=prediction_horizon_s,
        position_mean_ned_m=position_mean_ned_m,
        position_covariance_ned_m2=position_covariance_ned_m2,
    )
