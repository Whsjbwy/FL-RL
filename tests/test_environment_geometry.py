"""验证 Stage 0 环境扫掠球、完整包络边界与事件时刻几何。"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.env.geometry import (
    first_moving_sphere_collision_fraction,
    first_sphere_boundary_violation_fraction,
    moving_sphere_minimum_clearance,
)


def test_moving_sphere_collision_detects_between_nodes() -> None:
    """
    验证两端节点均未重叠但线性相对轨迹穿过物理碰撞球时能定位碰撞。

    该测试是环境真值几何，不使用风险 extra margin；若失败说明节点碰撞检查会漏检穿越事件。
    """

    auv_start_ned_m = np.array([-2.0, 0.0, 0.0], dtype=np.float64)
    auv_end_ned_m = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    obstacle_start_ned_m = np.zeros(3, dtype=np.float64)
    obstacle_end_ned_m = np.zeros(3, dtype=np.float64)
    combined_radius_m = 1.0

    collision_fraction = first_moving_sphere_collision_fraction(
        auv_start_ned_m,
        auv_end_ned_m,
        obstacle_start_ned_m,
        obstacle_end_ned_m,
        combined_radius_m,
    )
    minimum_clearance_m, minimizing_fraction = moving_sphere_minimum_clearance(
        auv_start_ned_m,
        auv_end_ned_m,
        obstacle_start_ned_m,
        obstacle_end_ned_m,
        combined_radius_m,
    )

    expected_collision_fraction = 0.25
    tolerance = 1.0e-12
    assert collision_fraction is not None
    assert abs(collision_fraction - expected_collision_fraction) <= tolerance, (
        f"expected={expected_collision_fraction}, actual={collision_fraction}, "
        f"tolerance={tolerance}"
    )
    assert abs(minimum_clearance_m + combined_radius_m) <= tolerance, (
        f"expected minimum clearance=-{combined_radius_m}, actual={minimum_clearance_m}, "
        f"tolerance={tolerance}"
    )
    assert abs(minimizing_fraction - 0.5) <= tolerance, (
        f"expected minimizing fraction=0.5, actual={minimizing_fraction}, tolerance={tolerance}"
    )


def test_boundary_check_uses_complete_auv_sphere(project_config) -> None:
    """
    验证环境边界检查使用 AUV 完整物理球包络，而不是只检查中心点。

    中心点仍在 [0,100] 内，但 North=99.5 m 配合 0.75 m 半径已经越界，应在 fraction=0 失败。
    """

    start_position_ned_m = np.array([99.5, 50.0, 20.0], dtype=np.float64)
    lower_bound_ned_m = np.asarray(
        project_config.environment.position_lower_bound_ned_m,
        dtype=np.float64,
    )
    upper_bound_ned_m = np.asarray(
        project_config.environment.position_upper_bound_ned_m,
        dtype=np.float64,
    )
    fraction = first_sphere_boundary_violation_fraction(
        start_position_ned_m,
        start_position_ned_m,
        lower_bound_ned_m,
        upper_bound_ned_m,
        project_config.risk.auv_radius_m,
    )
    assert fraction == 0.0, f"expected boundary violation at start, actual={fraction}"
