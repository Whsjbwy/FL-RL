"""
高斯半空间碰撞概率上界与线段并集界。

功能：
1. 实现 Eq. (33)–(35) 的单点半空间概率上界；
2. 实现 Eq. (36)–(38) 的分段线性扫掠上界；
3. 实现 Eq. (39) 的时间/多目标 Boole 聚合。

术语约束：
返回值称“高斯预测模型下的碰撞概率上界”，不是一般真实 collision probability。
Monte Carlo 或特殊各向同性情形才可用于精确概率核验。
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from auv_risk_rl.exceptions import NumericalRiskError
from auv_risk_rl.tracking.kalman_filter import validate_covariance_matrix
from auv_risk_rl.types import RiskResult


def _normal_cdf(value: float) -> float:
    """使用稳定的标准正态 CDF 实现，返回 Python float。"""

    return float(ndtr(value))


def _validated_projected_variance(variance_m2: float, negative_tolerance_m2: float) -> float:
    """仅校正负向舍入误差；任何严格正方差都保留给 Gaussian CDF。"""

    if not np.isfinite(negative_tolerance_m2) or negative_tolerance_m2 < 0.0:
        raise NumericalRiskError("负向方差容差必须是有限非负数。")
    if not np.isfinite(variance_m2) or variance_m2 < -negative_tolerance_m2:
        raise NumericalRiskError(
            f"投影方差非有限或明显为负：value={variance_m2}, "
            f"negative_tolerance={negative_tolerance_m2}。"
        )
    if variance_m2 < 0.0:
        # 仅在配置的微小负区间 [-tol,0) 内作数值校正，不抬高或归零任何正方差。
        return 0.0
    return variance_m2


def point_collision_upper_bound(
    relative_mean_ned_m: np.ndarray,
    position_covariance_ned_m2: np.ndarray,
    safe_radius_m: float,
    mean_norm_tolerance_m: float,
    variance_tolerance_m2: float,
    covariance_symmetry_tolerance: float,
    covariance_psd_tolerance: float,
) -> float:
    """
    计算单时刻球碰撞事件的高斯半空间概率上界。

    对应技术协议：
        Eq. (33)–(35)

    参数：
        relative_mean_ned_m:
            AUV 候选位置减障碍位置的相对均值，shape=(3,)，单位 m，NED。
        position_covariance_ned_m2:
            相对位置协方差；当前主模型只含障碍协方差，shape=(3,3)，单位 m^2，NED。
        safe_radius_m:
            AUV 物理半径 + 障碍物理半径 + 额外验证裕量，单位 m。
        mean_norm_tolerance_m:
            判定均值方向退化的数值容差，单位 m。
        variance_tolerance_m2:
            仅用于微小负投影方差的舍入校正上限，单位 m^2；正值不归零。
        covariance_symmetry_tolerance, covariance_psd_tolerance:
            协方差审计容差。

    返回：
        risk_upper_bound:
            [0,1] 内标量，无量纲。

    关键假设：
        A3–A7；候选 AUV 位置按确定值处理。

    重要限制：
        这是球事件的保守上界，不是精确球碰撞概率。均值近零时没有稳定的均值方向，
        协议要求保守返回 1，而不是用任意微小方向制造虚假安全。
    """

    if relative_mean_ned_m.shape != (3,):
        raise NumericalRiskError(
            f"relative_mean_ned_m shape 应为 (3,)，actual={relative_mean_ned_m.shape}。"
        )
    if not np.all(np.isfinite(relative_mean_ned_m)):
        raise NumericalRiskError("relative_mean_ned_m 必须有限。")
    if not np.isfinite(safe_radius_m) or safe_radius_m < 0.0:
        raise NumericalRiskError(f"safe_radius_m 必须有限非负，actual={safe_radius_m}。")

    covariance = validate_covariance_matrix(
        covariance=position_covariance_ned_m2,
        expected_shape=(3, 3),
        symmetry_tolerance=covariance_symmetry_tolerance,
        psd_tolerance=covariance_psd_tolerance,
        matrix_name="position_covariance_ned_m2",
    )
    mean_norm_m = float(np.linalg.norm(relative_mean_ned_m))
    if mean_norm_m <= mean_norm_tolerance_m:
        return 1.0

    projection_direction_ned = relative_mean_ned_m / mean_norm_m
    projected_mean_m = float(projection_direction_ned @ relative_mean_ned_m)
    projected_variance_m2 = float(
        projection_direction_ned @ covariance @ projection_direction_ned
    )
    projected_variance_m2 = _validated_projected_variance(
        projected_variance_m2, variance_tolerance_m2
    )
    if projected_variance_m2 == 0.0:
        # 精确零或明确的负向舍入校正才属于确定性分支。
        return 1.0 if projected_mean_m <= safe_radius_m else 0.0

    projected_std_m = float(np.sqrt(projected_variance_m2))
    standardized_threshold = (safe_radius_m - projected_mean_m) / projected_std_m
    return _normal_cdf(standardized_threshold)


def segment_projection_direction(
    relative_mean_start_ned_m: np.ndarray,
    relative_mean_end_ned_m: np.ndarray,
    mean_norm_tolerance_m: float,
) -> np.ndarray:
    """
    为一个相对均值线段选择固定投影方向。

    对应技术协议：
        Eq. (38)

    参数：
        relative_mean_start_ned_m, relative_mean_end_ned_m:
            线段两端相对均值，shape=(3,)，单位 m，NED。
        mean_norm_tolerance_m:
            退化方向判定容差，单位 m。

    返回：
        projection_direction_ned:
            shape=(3,)，无量纲，NED 单位向量。

    关键假设：
        方向只依赖当前已知均值与候选动作，不依赖未来随机误差实现。

    重要限制：
        该模块属于协议定义的工程启发式机制，不是最紧上界的严格理论推导。
        严格使用两端均值之和；和恰为零时使用固定 NED North 轴。
        保留容差参数以兼容原接口，但不再把非零和向量改成固定轴。
    """

    for mean in (relative_mean_start_ned_m, relative_mean_end_ned_m):
        if mean.shape != (3,) or not np.all(np.isfinite(mean)):
            raise NumericalRiskError("线段均值必须为有限的 shape=(3,) NED 向量。")
    if not np.isfinite(mean_norm_tolerance_m) or mean_norm_tolerance_m < 0.0:
        raise NumericalRiskError("方向接口容差必须为有限非负值。")
    mean_sum_ned_m = relative_mean_start_ned_m + relative_mean_end_ned_m
    if not np.all(np.isfinite(mean_sum_ned_m)):
        raise NumericalRiskError("线段两端均值之和超出有限数值范围。")
    scale_m = float(np.max(np.abs(mean_sum_ned_m)))
    if scale_m == 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # 等价缩放防止非零小向量的范数下溢，不改变 LOCAL 式(38)的方向。
    scaled_sum = mean_sum_ned_m / scale_m
    return scaled_sum / np.linalg.norm(scaled_sum)


def _endpoint_halfspace_probability(
    relative_mean_ned_m: np.ndarray,
    position_covariance_ned_m2: np.ndarray,
    projection_direction_ned: np.ndarray,
    safe_radius_m: float,
    variance_tolerance_m2: float,
    covariance_symmetry_tolerance: float,
    covariance_psd_tolerance: float,
) -> float:
    """计算固定投影方向下一个端点的半空间事件概率。"""

    covariance = validate_covariance_matrix(
        covariance=position_covariance_ned_m2,
        expected_shape=(3, 3),
        symmetry_tolerance=covariance_symmetry_tolerance,
        psd_tolerance=covariance_psd_tolerance,
        matrix_name="segment_endpoint_covariance_ned_m2",
    )
    projected_mean_m = float(projection_direction_ned @ relative_mean_ned_m)
    projected_variance_m2 = float(
        projection_direction_ned @ covariance @ projection_direction_ned
    )
    projected_variance_m2 = _validated_projected_variance(
        projected_variance_m2, variance_tolerance_m2
    )
    if projected_variance_m2 == 0.0:
        return 1.0 if projected_mean_m <= safe_radius_m else 0.0
    projected_std_m = float(np.sqrt(projected_variance_m2))
    return _normal_cdf((safe_radius_m - projected_mean_m) / projected_std_m)


def segment_collision_upper_bound(
    relative_mean_start_ned_m: np.ndarray,
    relative_mean_end_ned_m: np.ndarray,
    position_covariance_start_ned_m2: np.ndarray,
    position_covariance_end_ned_m2: np.ndarray,
    safe_radius_m: float,
    mean_norm_tolerance_m: float,
    variance_tolerance_m2: float,
    covariance_symmetry_tolerance: float,
    covariance_psd_tolerance: float,
) -> float:
    """
    计算指定分段线性几何上的模型条件碰撞概率上界。

    对应技术协议：
        Eq. (36)–(38)

    参数：
        relative_mean_start_ned_m, relative_mean_end_ned_m:
            相对均值线段端点，shape=(3,)，单位 m，NED。
        position_covariance_start_ned_m2, position_covariance_end_ned_m2:
            两端位置协方差，shape=(3,3)，单位 m^2，NED。
        safe_radius_m:
            验证中心距离，单位 m。
        mean_norm_tolerance_m, variance_tolerance_m2:
            退化分支数值容差。
        covariance_symmetry_tolerance, covariance_psd_tolerance:
            协方差审计容差。

    返回：
        segment_upper_bound:
            单线段的 Boole 上界，截断到 [0,1]，无量纲。

    关键假设：
        A3–A8；同一线段使用同一固定投影方向。

    重要限制：
        对协议定义的积分节点间线性插值严格，但不覆盖任意连续随机桥或真实未建模中间偏移。
    """

    if not np.isfinite(safe_radius_m) or safe_radius_m < 0.0:
        raise NumericalRiskError("线段验证半径必须是有限非负数。")
    projection_direction_ned = segment_projection_direction(
        relative_mean_start_ned_m,
        relative_mean_end_ned_m,
        mean_norm_tolerance_m,
    )
    start_probability = _endpoint_halfspace_probability(
        relative_mean_start_ned_m,
        position_covariance_start_ned_m2,
        projection_direction_ned,
        safe_radius_m,
        variance_tolerance_m2,
        covariance_symmetry_tolerance,
        covariance_psd_tolerance,
    )
    end_probability = _endpoint_halfspace_probability(
        relative_mean_end_ned_m,
        position_covariance_end_ned_m2,
        projection_direction_ned,
        safe_radius_m,
        variance_tolerance_m2,
        covariance_symmetry_tolerance,
        covariance_psd_tolerance,
    )
    return min(1.0, start_probability + end_probability)


def aggregate_union_bounds(component_upper_bounds: list[float]) -> RiskResult:
    """
    使用 Boole 不等式聚合时间段或多障碍风险上界。

    对应技术协议：
        Eq. (39)

    参数：
        component_upper_bounds:
            各分项有效上界，无量纲；可来自不同时间段或不同已维护目标。

    返回：
        RiskResult：同时保留截断上界与未截断 U。

    shape/单位/坐标系：
        输入输出均为风险标量，无量纲，不涉及坐标系。

    关键假设：
        每个分项本身是相应事件的有效上界；不需要事件相互独立。

    重要限制：
        未截断 U 可能大于 1，且随着时间段/目标数量增加而变保守；它用于排序和诊断，
        不能被解释为真实概率。
    """

    if any((not np.isfinite(value)) or value < 0.0 for value in component_upper_bounds):
        raise NumericalRiskError(f"风险分项必须为有限非负数：{component_upper_bounds}。")
    untruncated_union_bound = float(sum(component_upper_bounds))
    return RiskResult(
        risk_upper_bound=min(1.0, untruncated_union_bound),
        untruncated_union_bound=untruncated_union_bound,
        is_numerically_valid=True,
        detail="boole_union_bound",
    )
