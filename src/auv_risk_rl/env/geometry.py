"""
环境分段线性事件几何模块。

功能：
1. 计算移动 AUV 球与移动障碍球在一个积分小步内的最小物理净间距；
2. 求线性相对运动下首次球碰撞的归一化时刻；
3. 求 AUV 完整球包络首次越出 NED 空间边界的时刻；
4. 为环境终止逻辑提供独立、可单元测试的参考几何。

说明：
这里处理的是仿真真值几何，不使用 KF 协方差，也不计算概率风险。
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from auv_risk_rl.exceptions import InvalidEnvironmentStateError

FloatVector = NDArray[np.float64]


def _validate_vector3(vector_ned_m: FloatVector, variable_name: str) -> None:
    """检查 NED 三维向量的 shape 与有限性，避免几何函数静默传播 NaN。"""

    if vector_ned_m.shape != (3,) or not np.all(np.isfinite(vector_ned_m)):
        raise InvalidEnvironmentStateError(
            f"{variable_name} 必须是有限的 shape=(3,) NED 向量，actual={vector_ned_m}。"
        )


def moving_sphere_minimum_clearance(
    auv_start_position_ned_m: FloatVector,
    auv_end_position_ned_m: FloatVector,
    obstacle_start_position_ned_m: FloatVector,
    obstacle_end_position_ned_m: FloatVector,
    combined_physical_radius_m: float,
) -> tuple[float, float]:
    """
    计算一个积分小步内两移动球的最小物理净间距。

    对应技术协议：
        Eq. (27) 与第 7、21 章扫掠球真实几何。

    数学模型：
        AUV 与障碍中心在积分节点间均按线性插值；转化为相对位置线段到原点的最短距离。

    参数：
        auv_start_position_ned_m / auv_end_position_ned_m:
            AUV 小步起止中心位置，shape=(3,)，单位 m，NED。
        obstacle_start_position_ned_m / obstacle_end_position_ned_m:
            障碍小步起止中心位置，shape=(3,)，单位 m，NED。
        combined_physical_radius_m:
            AUV 物理半径与障碍物理半径之和，单位 m；不包含额外验证裕量。

    返回：
        minimum_clearance_m:
            最小中心距离减物理半径和，单位 m；负值表示包络重叠。
        minimizing_fraction:
            最小距离点在小步内的归一化位置，范围 [0,1]。

    关键假设：
        A8。

    重要限制：
        这是实际仿真碰撞几何，不是 Eq. (34) 的概率风险上界。
    """

    for variable_name, vector_ned_m in (
        ("auv_start_position_ned_m", auv_start_position_ned_m),
        ("auv_end_position_ned_m", auv_end_position_ned_m),
        ("obstacle_start_position_ned_m", obstacle_start_position_ned_m),
        ("obstacle_end_position_ned_m", obstacle_end_position_ned_m),
    ):
        _validate_vector3(vector_ned_m, variable_name)
    if combined_physical_radius_m < 0.0 or not math.isfinite(combined_physical_radius_m):
        raise InvalidEnvironmentStateError(
            "combined_physical_radius_m 必须是非负有限数，"
            f"actual={combined_physical_radius_m}。"
        )

    relative_start_ned_m = obstacle_start_position_ned_m - auv_start_position_ned_m
    relative_end_ned_m = obstacle_end_position_ned_m - auv_end_position_ned_m
    relative_delta_ned_m = relative_end_ned_m - relative_start_ned_m
    denominator_m2 = float(relative_delta_ned_m @ relative_delta_ned_m)

    if denominator_m2 == 0.0:
        minimizing_fraction = 0.0
    else:
        minimizing_fraction = float(
            np.clip(
                -float(relative_start_ned_m @ relative_delta_ned_m) / denominator_m2,
                0.0,
                1.0,
            )
        )
    closest_relative_position_ned_m = (
        relative_start_ned_m + minimizing_fraction * relative_delta_ned_m
    )
    center_distance_m = float(np.linalg.norm(closest_relative_position_ned_m))
    return center_distance_m - combined_physical_radius_m, minimizing_fraction


def first_moving_sphere_collision_fraction(
    auv_start_position_ned_m: FloatVector,
    auv_end_position_ned_m: FloatVector,
    obstacle_start_position_ned_m: FloatVector,
    obstacle_end_position_ned_m: FloatVector,
    combined_physical_radius_m: float,
    discriminant_tolerance_m4: float = 1.0e-12,
) -> float | None:
    """
    求一个积分小步内首次物理球碰撞的归一化时刻。

    对应技术协议：
        Eq. (27) 与第 7 章扫掠球事件定义。

    参数：
        auv_start_position_ned_m / auv_end_position_ned_m:
            AUV 起止中心，shape=(3,)，单位 m，NED。
        obstacle_start_position_ned_m / obstacle_end_position_ned_m:
            障碍起止中心，shape=(3,)，单位 m，NED。
        combined_physical_radius_m:
            两个物理包络半径之和，单位 m，不包含 extra_margin_m。
        discriminant_tolerance_m4:
            二次方程判别式只允许修正舍入级微小负值，单位 m^4。

    返回：
        首次碰撞 fraction∈[0,1]；若整段无碰撞返回 None。

    关键假设：
        A8。

    重要限制：
        不允许使用额外安全裕量替代物理碰撞半径，否则会把 near-miss 误记为 collision。
    """

    minimum_clearance_m, _ = moving_sphere_minimum_clearance(
        auv_start_position_ned_m,
        auv_end_position_ned_m,
        obstacle_start_position_ned_m,
        obstacle_end_position_ned_m,
        combined_physical_radius_m,
    )
    if minimum_clearance_m > 0.0:
        return None

    relative_start_ned_m = obstacle_start_position_ned_m - auv_start_position_ned_m
    relative_end_ned_m = obstacle_end_position_ned_m - auv_end_position_ned_m
    relative_delta_ned_m = relative_end_ned_m - relative_start_ned_m
    radius_m = combined_physical_radius_m
    quadratic_a_m2 = float(relative_delta_ned_m @ relative_delta_ned_m)
    quadratic_b_m2 = 2.0 * float(relative_start_ned_m @ relative_delta_ned_m)
    quadratic_c_m2 = float(relative_start_ned_m @ relative_start_ned_m) - radius_m**2

    if quadratic_c_m2 <= 0.0:
        return 0.0
    if quadratic_a_m2 == 0.0:
        return None

    discriminant_m4 = quadratic_b_m2**2 - 4.0 * quadratic_a_m2 * quadratic_c_m2
    if discriminant_m4 < -discriminant_tolerance_m4:
        return None
    discriminant_m4 = max(discriminant_m4, 0.0)
    sqrt_discriminant_m2 = math.sqrt(discriminant_m4)
    denominator_m2 = 2.0 * quadratic_a_m2
    first_root = (-quadratic_b_m2 - sqrt_discriminant_m2) / denominator_m2
    second_root = (-quadratic_b_m2 + sqrt_discriminant_m2) / denominator_m2
    valid_roots = [root for root in (first_root, second_root) if 0.0 <= root <= 1.0]
    if not valid_roots:
        return None
    return float(min(valid_roots))


def first_sphere_boundary_violation_fraction(
    start_position_ned_m: FloatVector,
    end_position_ned_m: FloatVector,
    lower_bound_ned_m: FloatVector,
    upper_bound_ned_m: FloatVector,
    sphere_radius_m: float,
) -> float | None:
    """
    求 AUV 完整球包络首次越出矩形 NED 操作空间的归一化时刻。

    对应技术协议：
        Eq. (57) 的确定性操作边界要求。

    参数：
        start_position_ned_m / end_position_ned_m:
            小步起止 AUV 中心，shape=(3,)，单位 m，NED。
        lower_bound_ned_m / upper_bound_ned_m:
            环境物理边界，shape=(3,)，单位 m，NED。
        sphere_radius_m:
            AUV 物理包络半径，单位 m。

    返回：
        首次违规 fraction∈[0,1]；整段完整包络均在边界内时返回 None。

    关键假设：
        A8；位置在积分节点间线性插值。

    重要限制：
        这是事件定位，不是把越界状态 clip 回合法区域；环境在事件时刻终止。
    """

    for variable_name, vector_ned_m in (
        ("start_position_ned_m", start_position_ned_m),
        ("end_position_ned_m", end_position_ned_m),
        ("lower_bound_ned_m", lower_bound_ned_m),
        ("upper_bound_ned_m", upper_bound_ned_m),
    ):
        _validate_vector3(vector_ned_m, variable_name)
    if sphere_radius_m < 0.0 or not math.isfinite(sphere_radius_m):
        raise InvalidEnvironmentStateError(
            f"sphere_radius_m 必须为非负有限数，actual={sphere_radius_m}。"
        )

    center_lower_bound_ned_m = lower_bound_ned_m + sphere_radius_m
    center_upper_bound_ned_m = upper_bound_ned_m - sphere_radius_m
    if np.any(center_lower_bound_ned_m > center_upper_bound_ned_m):
        raise InvalidEnvironmentStateError("AUV 物理半径大于操作空间至少一个轴向半宽。")

    is_start_valid = bool(
        np.all(start_position_ned_m >= center_lower_bound_ned_m)
        and np.all(start_position_ned_m <= center_upper_bound_ned_m)
    )
    if not is_start_valid:
        return 0.0

    candidate_fractions: list[float] = []
    position_delta_ned_m = end_position_ned_m - start_position_ned_m
    for axis_index in range(3):
        axis_delta_m = float(position_delta_ned_m[axis_index])
        if (
            axis_delta_m < 0.0
            and end_position_ned_m[axis_index] < center_lower_bound_ned_m[axis_index]
        ):
            fraction = (
                center_lower_bound_ned_m[axis_index] - start_position_ned_m[axis_index]
            ) / axis_delta_m
            candidate_fractions.append(float(fraction))
        elif (
            axis_delta_m > 0.0
            and end_position_ned_m[axis_index] > center_upper_bound_ned_m[axis_index]
        ):
            fraction = (
                center_upper_bound_ned_m[axis_index] - start_position_ned_m[axis_index]
            ) / axis_delta_m
            candidate_fractions.append(float(fraction))

    valid_fractions = [fraction for fraction in candidate_fractions if 0.0 <= fraction <= 1.0]
    return min(valid_fractions) if valid_fractions else None


def first_scalar_limit_violation_fraction(
    start_value: float,
    end_value: float,
    lower_limit: float,
    upper_limit: float,
) -> float | None:
    """
    求线性插值标量首次越出闭区间的归一化时刻。

    对应技术协议：
        Eq. (57) 的 pitch/速度/角率确定性操作约束；无独立几何公式编号。

    参数：
        start_value / end_value:
            小步起止标量，单位由调用方保持一致。
        lower_limit / upper_limit:
            同单位合法闭区间。

    返回：
        首次违规 fraction∈[0,1]；若整段合法返回 None。

    关键假设：
        小步内用线性插值定位事件，仅用于确定性操作边界。

    重要限制：
        对非线性真实连续状态只是一种与积分节点事件定义一致的局部定位。
    """

    values = (start_value, end_value, lower_limit, upper_limit)
    if not all(math.isfinite(value) for value in values):
        raise InvalidEnvironmentStateError(f"标量边界输入必须有限，actual={values}。")
    if lower_limit > upper_limit:
        raise InvalidEnvironmentStateError(
            f"lower_limit 必须 <= upper_limit，actual=({lower_limit}, {upper_limit})。"
        )
    if start_value < lower_limit or start_value > upper_limit:
        return 0.0
    if lower_limit <= end_value <= upper_limit:
        return None

    delta = end_value - start_value
    if delta == 0.0:
        return 0.0
    crossing_limit = lower_limit if end_value < lower_limit else upper_limit
    fraction = (crossing_limit - start_value) / delta
    return float(np.clip(fraction, 0.0, 1.0))
