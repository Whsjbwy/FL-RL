"""
有限任务时域归一化成本与失败吸收尾项。

功能：
1. 计算 Eq. (46) 的有限时域折扣归一化系数；
2. 计算安全失败后的单位成本吸收尾项；
3. 提供与后续 cost critic 目标 Eq. (51) 一致的 reference 数学实现。

说明：
Stage 0 只冻结数学语义，不实现神经网络 cost critic、actor 或拉格朗日乘子更新。
"""

from __future__ import annotations

import math


def finite_horizon_discount_normalizer(discount_gamma: float, horizon_steps: int) -> float:
    """
    计算有限时域折扣权重归一化系数 β_N。

    对应技术协议：
        Eq. (46)

    数学模型：
        β_N = 1 / sum_{t=0}^{N-1} γ^t，使全时域成本恒为 1 时归一化 J_C=1。

    参数：
        discount_gamma:
            折扣因子，无量纲，范围 (0,1]。
        horizon_steps:
            完整计划时域控制步数，无量纲正整数。

    返回：
        β_N，无量纲。

    关键假设：
        固定有限计划时域 N。

    重要限制：
        该归一化只赋予“折扣加权不合规/失败占用比例”解释，不是任务碰撞概率。
    """

    if not math.isfinite(discount_gamma) or not 0.0 < discount_gamma <= 1.0:
        raise ValueError(f"discount_gamma 必须位于 (0,1]，actual={discount_gamma}。")
    if horizon_steps <= 0:
        raise ValueError(f"horizon_steps 必须为正整数，actual={horizon_steps}。")
    if discount_gamma == 1.0:
        return 1.0 / horizon_steps
    geometric_sum = (1.0 - discount_gamma**horizon_steps) / (1.0 - discount_gamma)
    return 1.0 / geometric_sum


def absorbing_failure_tail_cost(
    failure_step_index: int,
    discount_gamma: float,
    horizon_steps: int,
) -> float:
    """
    计算安全失败后剩余计划时段单位成本的归一化折扣尾项。

    对应技术协议：
        Eq. (45)–(46)、Eq. (51)

    参数：
        failure_step_index:
            发生安全失败的控制步索引，0 表示第一时段已经失败。
        discount_gamma:
            折扣因子，无量纲。
        horizon_steps:
            完整计划时域控制步数 N。

    返回：
        从 failure_step_index 到 N-1 全部记为 1 后，对 J_C 的归一化贡献，无量纲。

    关键假设：
        安全失败后的剩余计划时段按 1 计账，但不继续模拟虚假轨迹。

    重要限制：
        这是操作性成本账本，不是未来每一步真实都发生碰撞的概率陈述。
    """

    if failure_step_index < 0 or failure_step_index >= horizon_steps:
        raise ValueError(
            "failure_step_index 必须位于完整计划时域内："
            f"index={failure_step_index}, horizon_steps={horizon_steps}。"
        )
    beta_n = finite_horizon_discount_normalizer(discount_gamma, horizon_steps)
    if discount_gamma == 1.0:
        return beta_n * (horizon_steps - failure_step_index)
    remaining_sum = (
        discount_gamma**failure_step_index
        * (1.0 - discount_gamma ** (horizon_steps - failure_step_index))
        / (1.0 - discount_gamma)
    )
    return beta_n * remaining_sum


def normalized_finite_horizon_cost(
    realized_costs: list[float],
    discount_gamma: float,
    horizon_steps: int,
    safety_failure_step_index: int | None = None,
) -> float:
    """
    计算带安全失败吸收尾项的有限时域归一化成本。

    对应技术协议：
        Eq. (45)–(46)

    参数：
        realized_costs:
            已真实经历时段的成本序列，每项应位于 [0,1]；列表索引即控制时段索引。
        discount_gamma:
            折扣因子，无量纲。
        horizon_steps:
            完整计划控制步数 N。
        safety_failure_step_index:
            若发生碰撞/操作边界安全失败，给出对应时段索引；从该时段起剩余成本均按 1 计账。
            成功或超时则为 None，不添加吸收尾项。

    返回：
        J_C∈[0,1] 的 reference 数学值。

    关键假设：
        realized_costs 与安全失败索引使用同一控制时段定义。

    重要限制：
        本函数不判断某事件是否应被定义为成本 1；该语义由上层任务逻辑固定。
    """

    if len(realized_costs) > horizon_steps:
        raise ValueError(
            f"realized_costs 长度不能超过计划时域，actual={len(realized_costs)}, "
            f"horizon={horizon_steps}。"
        )
    if any((not math.isfinite(cost)) or cost < 0.0 or cost > 1.0 for cost in realized_costs):
        raise ValueError("realized_costs 每项必须为 [0,1] 内有限数。")

    beta_n = finite_horizon_discount_normalizer(discount_gamma, horizon_steps)
    if safety_failure_step_index is None:
        weighted_sum = sum(
            (discount_gamma**step_index) * cost
            for step_index, cost in enumerate(realized_costs)
        )
        return beta_n * weighted_sum

    if safety_failure_step_index >= len(realized_costs):
        raise ValueError(
            "安全失败索引必须落在已记录真实时段中，"
            f"failure={safety_failure_step_index}, realized_length={len(realized_costs)}。"
        )
    prefix_sum = sum(
        (discount_gamma**step_index) * realized_costs[step_index]
        for step_index in range(safety_failure_step_index)
    )
    return beta_n * prefix_sum + absorbing_failure_tail_cost(
        failure_step_index=safety_failure_step_index,
        discount_gamma=discount_gamma,
        horizon_steps=horizon_steps,
    )
