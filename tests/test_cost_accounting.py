"""验证 Eq. (45)–(46)、Eq. (51) 的有限时域归一化与失败吸收尾项。"""

from __future__ import annotations

from auv_risk_rl.costs.finite_horizon import (
    absorbing_failure_tail_cost,
    finite_horizon_discount_normalizer,
    normalized_finite_horizon_cost,
)


def test_normalized_cost_constant_one_equals_one(project_config) -> None:
    """
    验证完整计划时域所有成本均为 1 时 J_C 精确等于 1。

    若失败，说明 Eq. (46) 归一化系数实现错误，不能通过修改预算 d_C 掩盖。
    """

    horizon_steps = project_config.cost.planned_horizon_control_steps
    costs = [1.0] * horizon_steps
    normalized_cost = normalized_finite_horizon_cost(
        costs,
        project_config.cost.discount_gamma,
        horizon_steps,
    )
    tolerance = 1.0e-12
    assert abs(normalized_cost - 1.0) <= tolerance, (
        f"expected=1.0, actual={normalized_cost}, tolerance={tolerance}"
    )


def test_failure_tail_matches_explicit_absorbing_sequence(project_config) -> None:
    """
    验证安全失败尾项与显式补全剩余单位成本序列得到完全相同 J_C。

    该测试直接约束后续 cost critic Eq. (51) 的失败尾项语义。
    """

    gamma = project_config.cost.discount_gamma
    horizon_steps = project_config.cost.planned_horizon_control_steps
    failure_step_index = 37
    realized_costs = [0.0] * failure_step_index + [1.0]
    reference_costs = [0.0] * failure_step_index + [1.0] * (horizon_steps - failure_step_index)

    direct = normalized_finite_horizon_cost(reference_costs, gamma, horizon_steps)
    with_tail = normalized_finite_horizon_cost(
        realized_costs,
        gamma,
        horizon_steps,
        safety_failure_step_index=failure_step_index,
    )
    tail_only = absorbing_failure_tail_cost(failure_step_index, gamma, horizon_steps)
    tolerance = 1.0e-12
    assert abs(direct - with_tail) <= tolerance, (
        f"expected direct={direct}, actual={with_tail}, error={abs(direct-with_tail)}, "
        f"tolerance={tolerance}"
    )
    assert abs(tail_only - direct) <= tolerance, (
        f"expected tail={direct}, actual={tail_only}, error={abs(tail_only-direct)}, "
        f"tolerance={tolerance}"
    )


def test_discount_normalizer_matches_closed_geometric_sum(project_config) -> None:
    """验证 β_N 与独立几何级数直接求和结果一致。"""

    gamma = project_config.cost.discount_gamma
    horizon_steps = project_config.cost.planned_horizon_control_steps
    beta_n = finite_horizon_discount_normalizer(gamma, horizon_steps)
    direct_beta_n = 1.0 / sum(gamma**step_index for step_index in range(horizon_steps))
    tolerance = 1.0e-15
    assert abs(beta_n - direct_beta_n) <= tolerance, (
        f"expected={direct_beta_n}, actual={beta_n}, tolerance={tolerance}"
    )
