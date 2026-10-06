"""验证三维 CV 状态转移与连续白加速度离散过程噪声 Eq. (12)–(15)。"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad_vec

from auv_risk_rl.tracking.cv_model import cv_process_noise_covariance, cv_state_transition


def test_cv_state_transition_has_expected_position_velocity_block() -> None:
    """验证 F(T) 右上分块为 T*I，确保位置按当前速度外推。"""

    horizon_s = 1.7
    state_transition = cv_state_transition(horizon_s)
    expected_block = horizon_s * np.eye(3)
    error = float(np.max(np.abs(state_transition[0:3, 3:6] - expected_block)))
    tolerance = 1.0e-14
    assert error <= tolerance, f"expected <= {tolerance}, actual={error}"


def test_cv_process_noise_matches_independent_numerical_integral() -> None:
    """
    用独立数值积分验证 Eq. (14)–(15) 的 Q(T)。

    若失败，优先检查连续噪声模型、单位和积分核，禁止把解析容差从 1e-10 放宽到大数值。
    """

    horizon_s = 1.3
    spectral_density = np.diag(np.array([0.001, 0.002, 0.003], dtype=np.float64))
    analytical_q = cv_process_noise_covariance(horizon_s, spectral_density)

    def integrand(time_s: float) -> np.ndarray:
        state_transition = cv_state_transition(time_s)
        noise_input = np.vstack(
            (np.zeros((3, 3), dtype=np.float64), np.eye(3, dtype=np.float64))
        )
        propagated_noise = state_transition @ noise_input
        return propagated_noise @ spectral_density @ propagated_noise.T

    numerical_q, _ = quad_vec(integrand, 0.0, horizon_s)
    error = float(np.max(np.abs(analytical_q - numerical_q)))
    tolerance = 1.0e-10
    assert error <= tolerance, f"expected <= {tolerance}, actual={error}"


def test_cv_q_semigroup_consistency() -> None:
    """
    验证 CV 白加速度噪声离散协方差满足半群一致性。

    若该测试失败，说明 Q(T) 实现或 F(T) 组合存在数学错误，不允许通过调大容差绕过。
    """

    first_interval_s = 0.7
    second_interval_s = 1.1
    spectral_density = np.diag(np.array([0.001, 0.001, 0.001], dtype=np.float64))

    q_total = cv_process_noise_covariance(first_interval_s + second_interval_s, spectral_density)
    f_second = cv_state_transition(second_interval_s)
    q_first = cv_process_noise_covariance(first_interval_s, spectral_density)
    q_second = cv_process_noise_covariance(second_interval_s, spectral_density)
    composed_q = f_second @ q_first @ f_second.T + q_second

    error = float(np.max(np.abs(q_total - composed_q)))
    tolerance = 1.0e-12
    assert error <= tolerance, f"expected <= {tolerance}, actual={error}"
