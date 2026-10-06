"""验证 NED/Body 坐标与旋转公式 Eq. (1)–(4)。"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.frames import (
    angular_velocity_body_from_euler_rates,
    relative_position_body,
    rotation_body_to_ned,
)


def test_rotation_matrix_is_orthonormal() -> None:
    """
    验证 Body→NED 旋转矩阵保持正交且 det=+1。

    若该测试失败，说明 Eq. (1) 的旋转顺序或符号存在错误，不能通过放宽容差绕过。
    """

    rotation_matrix = rotation_body_to_ned(yaw_rad=0.7, pitch_rad=-0.3)
    identity_error = np.max(np.abs(rotation_matrix.T @ rotation_matrix - np.eye(3)))
    determinant_error = abs(float(np.linalg.det(rotation_matrix)) - 1.0)
    tolerance = 1.0e-12
    assert identity_error <= tolerance, (
        f"expected orthogonality error <= {tolerance}, actual={identity_error}"
    )
    assert determinant_error <= tolerance, (
        f"expected determinant error <= {tolerance}, actual={determinant_error}"
    )


def test_positive_pitch_moves_forward_axis_up_in_ned_convention() -> None:
    """
    验证 NED 约定下正 pitch 使 Body 前向轴产生负 Down 分量。

    这是协议对 Eq. (1) 的关键符号约束；若失败会直接反转三维垂向机动语义。
    """

    positive_pitch_rad = np.deg2rad(20.0)
    rotation_matrix = rotation_body_to_ned(yaw_rad=0.0, pitch_rad=positive_pitch_rad)
    forward_axis_ned = rotation_matrix @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    assert forward_axis_ned[2] < 0.0, (
        f"expected negative Down component, actual={forward_axis_ned[2]}"
    )


def test_relative_position_roundtrip_preserves_vector() -> None:
    """
    验证 Eq. (2) 的先 NED 相减、再旋转到 Body 的往返一致性。
    """

    auv_position_ned_m = np.array([10.0, 20.0, 15.0], dtype=np.float64)
    obstacle_position_ned_m = np.array([13.0, 18.0, 12.0], dtype=np.float64)
    yaw_rad = 0.4
    pitch_rad = -0.2
    relative_body_m = relative_position_body(
        auv_position_ned_m,
        obstacle_position_ned_m,
        yaw_rad,
        pitch_rad,
    )
    rotation_matrix = rotation_body_to_ned(yaw_rad, pitch_rad)
    reconstructed_ned_m = auv_position_ned_m + rotation_matrix @ relative_body_m
    error_m = float(np.max(np.abs(reconstructed_ned_m - obstacle_position_ned_m)))
    tolerance_m = 1.0e-12
    assert error_m <= tolerance_m, f"expected <= {tolerance_m}, actual={error_m}"


def test_euler_yaw_rate_can_create_body_roll_rate_component() -> None:
    """
    验证 roll=0 不代表 Body p=0，防止错误简化 Eq. (4)。
    """

    angular_velocity_body_rad_s = angular_velocity_body_from_euler_rates(
        yaw_rate_rad_s=0.2,
        pitch_rate_rad_s=0.0,
        pitch_rad=np.deg2rad(30.0),
    )
    assert angular_velocity_body_rad_s[0] < 0.0, (
        "expected nonzero negative body x angular velocity component, "
        f"actual={angular_velocity_body_rad_s[0]}"
    )
