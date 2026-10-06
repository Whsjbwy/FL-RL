"""
三维 CV-KF 预测与 Joseph 形式更新模块。

功能：
1. 执行 Eq. (19) 的模型预测；
2. 执行 Eq. (20)–(22) 的线性测量更新；
3. 对协方差进行显式有限性、对称性与 PSD 审计。

数值原则：
Kalman 增益通过 solve 求解，不显式构造 innovation covariance 的逆；后验协方差
使用 Joseph form，以降低有限精度下对称性和半正定结构的破坏。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from auv_risk_rl.exceptions import InvalidCovarianceError
from auv_risk_rl.tracking.cv_model import cv_process_noise_covariance, cv_state_transition

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]
POSITION_OBSERVATION_MATRIX = np.hstack(
    (np.eye(3, dtype=np.float64), np.zeros((3, 3), dtype=np.float64))
)


def validate_covariance_matrix(
    covariance: FloatMatrix,
    expected_shape: tuple[int, int],
    symmetry_tolerance: float,
    psd_tolerance: float,
    matrix_name: str,
) -> FloatMatrix:
    """
    检查并仅修正舍入级非对称，不掩盖真实 PSD 错误。

    对应技术协议：
        Eq. (22) 的数值稳定性要求，以及第 12.3、17 章数值异常规则。

    参数：
        covariance:
            待检查协方差，单位由调用方语义决定，坐标系由调用方保证。
        expected_shape:
            期望矩阵 shape。
        symmetry_tolerance:
            允许的最大反对称元素绝对值。
        psd_tolerance:
            允许的最小特征值负向舍入容差。
        matrix_name:
            错误消息中的矩阵名称。

    返回：
        symmetric_covariance:
            只做 0.5*(P+P.T) 对称化后的矩阵，shape 与输入一致。

    关键假设：
        调用方提供的 tolerance 来自配置而不是为了通过测试临时放宽。

    重要限制：
        不对明显负特征值做 clip；若 min eigenvalue < -tolerance，直接抛出异常。
    """

    if covariance.shape != expected_shape:
        raise InvalidCovarianceError(
            f"{matrix_name} shape 错误：expected={expected_shape}, actual={covariance.shape}。"
        )
    if not np.all(np.isfinite(covariance)):
        raise InvalidCovarianceError(f"{matrix_name} 包含 NaN 或 Inf。")

    asymmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    if asymmetry_error > symmetry_tolerance:
        raise InvalidCovarianceError(
            f"{matrix_name} 非对称超过容差：actual={asymmetry_error}, "
            f"tolerance={symmetry_tolerance}。"
        )

    symmetric_covariance = 0.5 * (covariance + covariance.T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric_covariance)))
    if minimum_eigenvalue < -psd_tolerance:
        raise InvalidCovarianceError(
            f"{matrix_name} 不是 PSD：min_eigenvalue={minimum_eigenvalue}, "
            f"tolerance={psd_tolerance}。"
        )
    return symmetric_covariance


def predict_cv_state(
    state_mean_ned: FloatVector,
    state_covariance_ned: FloatMatrix,
    delta_t_s: float,
    acceleration_spectral_density_ned_m2_s3: FloatMatrix,
    symmetry_tolerance: float,
    psd_tolerance: float,
) -> tuple[FloatVector, FloatMatrix]:
    """
    根据三维 CV-KF 模型预测未来状态均值和协方差。

    对应技术协议：
        Eq. (19)

    参数：
        state_mean_ned:
            当前后验状态均值，shape=(6,)，单位 [m,m,m,m/s,m/s,m/s]，NED。
        state_covariance_ned:
            当前后验协方差，shape=(6,6)，分块单位，NED。
        delta_t_s:
            预测间隔，单位 s。
        acceleration_spectral_density_ned_m2_s3:
            白加速度谱密度，shape=(3,3)，单位 m^2/s^3，NED。
        symmetry_tolerance, psd_tolerance:
            协方差数值审计容差，由配置统一提供。

    返回：
        predicted_state_mean_ned:
            shape=(6,)，混合单位，NED。
        predicted_state_covariance_ned:
            shape=(6,6)，分块单位，NED。

    关键假设：
        A3–A5。

    重要限制：
        在声明的线性高斯模型下条件精确；不覆盖错误关联、模型切换和未知系统偏置。
    """

    if state_mean_ned.shape != (6,):
        raise ValueError(f"state_mean_ned shape 应为 (6,)，actual={state_mean_ned.shape}。")
    state_covariance_ned = validate_covariance_matrix(
        covariance=state_covariance_ned,
        expected_shape=(6, 6),
        symmetry_tolerance=symmetry_tolerance,
        psd_tolerance=psd_tolerance,
        matrix_name="state_covariance_ned",
    )

    state_transition = cv_state_transition(delta_t_s)
    process_noise_covariance = cv_process_noise_covariance(
        delta_t_s,
        acceleration_spectral_density_ned_m2_s3,
    )
    predicted_state_mean_ned = state_transition @ state_mean_ned
    predicted_state_covariance_ned = (
        state_transition @ state_covariance_ned @ state_transition.T
        + process_noise_covariance
    )
    predicted_state_covariance_ned = validate_covariance_matrix(
        covariance=predicted_state_covariance_ned,
        expected_shape=(6, 6),
        symmetry_tolerance=symmetry_tolerance,
        psd_tolerance=psd_tolerance,
        matrix_name="predicted_state_covariance_ned",
    )
    return predicted_state_mean_ned, predicted_state_covariance_ned


def update_position_measurement(
    predicted_state_mean_ned: FloatVector,
    predicted_state_covariance_ned: FloatMatrix,
    measurement_position_ned_m: FloatVector,
    measurement_covariance_ned_m2: FloatMatrix,
    symmetry_tolerance: float,
    psd_tolerance: float,
) -> tuple[FloatVector, FloatMatrix, FloatVector, FloatMatrix]:
    """
    使用三维位置测量执行线性 KF 更新。

    对应技术协议：
        Eq. (20)–(22)

    参数：
        predicted_state_mean_ned:
            先验均值，shape=(6,)，混合单位，NED。
        predicted_state_covariance_ned:
            先验协方差，shape=(6,6)，分块单位，NED。
        measurement_position_ned_m:
            位置测量，shape=(3,)，单位 m，NED。
        measurement_covariance_ned_m2:
            测量协方差，shape=(3,3)，单位 m^2，NED。
        symmetry_tolerance, psd_tolerance:
            数值审计容差。

    返回：
        posterior_state_mean_ned:
            shape=(6,)，混合单位，NED。
        posterior_state_covariance_ned:
            shape=(6,6)，分块单位，NED。
        innovation_ned_m:
            shape=(3,)，单位 m，NED。
        innovation_covariance_ned_m2:
            shape=(3,3)，单位 m^2，NED。

    关键假设：
        A3–A6。

    重要限制：
        在这些假设下条件精确。Kalman 增益通过线性方程求解，禁止显式求逆；
        Joseph 形式用于提高有限精度下协方差数值稳定性，而不是改变理论后验。
    """

    predicted_state_covariance_ned = validate_covariance_matrix(
        covariance=predicted_state_covariance_ned,
        expected_shape=(6, 6),
        symmetry_tolerance=symmetry_tolerance,
        psd_tolerance=psd_tolerance,
        matrix_name="predicted_state_covariance_ned",
    )
    measurement_covariance_ned_m2 = validate_covariance_matrix(
        covariance=measurement_covariance_ned_m2,
        expected_shape=(3, 3),
        symmetry_tolerance=symmetry_tolerance,
        psd_tolerance=psd_tolerance,
        matrix_name="measurement_covariance_ned_m2",
    )

    observation_matrix = POSITION_OBSERVATION_MATRIX
    innovation_ned_m = measurement_position_ned_m - observation_matrix @ predicted_state_mean_ned
    innovation_covariance_ned_m2 = (
        observation_matrix @ predicted_state_covariance_ned @ observation_matrix.T
        + measurement_covariance_ned_m2
    )
    innovation_covariance_ned_m2 = validate_covariance_matrix(
        covariance=innovation_covariance_ned_m2,
        expected_shape=(3, 3),
        symmetry_tolerance=symmetry_tolerance,
        psd_tolerance=psd_tolerance,
        matrix_name="innovation_covariance_ned_m2",
    )

    cross_covariance = predicted_state_covariance_ned @ observation_matrix.T
    # solve(S, C^T)^T 与 C @ inv(S) 数学等价，但避免显式构造逆矩阵造成额外数值误差。
    kalman_gain = np.linalg.solve(innovation_covariance_ned_m2, cross_covariance.T).T
    posterior_state_mean_ned = predicted_state_mean_ned + kalman_gain @ innovation_ned_m

    identity_state = np.eye(6, dtype=np.float64)
    joseph_left = identity_state - kalman_gain @ observation_matrix
    # Joseph 形式把先验误差和测量噪声两部分都显式保留，更有利于有限精度下维持 PSD 结构。
    posterior_state_covariance_ned = (
        joseph_left @ predicted_state_covariance_ned @ joseph_left.T
        + kalman_gain @ measurement_covariance_ned_m2 @ kalman_gain.T
    )
    posterior_state_covariance_ned = validate_covariance_matrix(
        covariance=posterior_state_covariance_ned,
        expected_shape=(6, 6),
        symmetry_tolerance=symmetry_tolerance,
        psd_tolerance=psd_tolerance,
        matrix_name="posterior_state_covariance_ned",
    )
    return (
        posterior_state_mean_ned,
        posterior_state_covariance_ned,
        innovation_ned_m,
        innovation_covariance_ned_m2,
    )
