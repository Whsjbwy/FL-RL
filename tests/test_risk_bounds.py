"""验证 Eq. (33)–(39) 的高斯上界、退化分支与扫掠几何。"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.risk.gaussian_bounds import (
    aggregate_union_bounds,
    point_collision_upper_bound,
    segment_collision_upper_bound,
)
from auv_risk_rl.seeding import SeedManager


def test_risk_upper_bound_dominates_mc_estimate(project_config) -> None:
    """
    用独立 Monte Carlo 检查单点球碰撞概率不超过 Eq. (34) 半空间上界。

    Monte Carlo 只用于离线核验，不把样本比例当在线模块实现。
    """

    relative_mean_ned_m = np.array([4.0, 0.3, -0.2], dtype=np.float64)
    covariance_ned_m2 = np.array(
        [[0.36, 0.05, 0.0], [0.05, 0.25, 0.0], [0.0, 0.0, 0.16]],
        dtype=np.float64,
    )
    safe_radius_m = 1.8
    upper_bound = point_collision_upper_bound(
        relative_mean_ned_m=relative_mean_ned_m,
        position_covariance_ned_m2=covariance_ned_m2,
        safe_radius_m=safe_radius_m,
        mean_norm_tolerance_m=project_config.risk.mean_norm_tolerance_m,
        variance_tolerance_m2=project_config.risk.variance_tolerance_m2,
        covariance_symmetry_tolerance=project_config.kf.covariance_symmetry_tolerance,
        covariance_psd_tolerance=project_config.kf.covariance_psd_tolerance,
    )

    rng = SeedManager(root_seed=20260917).get_rng("evaluation")
    sample_count = 40000
    samples_ned_m = rng.multivariate_normal(
        mean=relative_mean_ned_m,
        cov=covariance_ned_m2,
        size=sample_count,
    )
    collision_fraction = float(
        np.mean(np.linalg.norm(samples_ned_m, axis=1) <= safe_radius_m)
    )
    sampling_slack = 0.01
    assert collision_fraction <= upper_bound + sampling_slack, (
        "expected MC collision estimate <= analytical upper bound + sampling slack, "
        f"actual={collision_fraction}, upper_bound={upper_bound}, slack={sampling_slack}"
    )


def test_segment_bound_detects_endpoint_only_counterexample(project_config) -> None:
    """
    构造两端都在安全球外、线段穿过球心的反例，验证不能只做节点球距离检查。
    """

    start_mean_ned_m = np.array([-2.0, 0.0, 0.0], dtype=np.float64)
    end_mean_ned_m = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    zero_covariance_ned_m2 = np.zeros((3, 3), dtype=np.float64)
    safe_radius_m = 1.0
    segment_bound = segment_collision_upper_bound(
        relative_mean_start_ned_m=start_mean_ned_m,
        relative_mean_end_ned_m=end_mean_ned_m,
        position_covariance_start_ned_m2=zero_covariance_ned_m2,
        position_covariance_end_ned_m2=zero_covariance_ned_m2,
        safe_radius_m=safe_radius_m,
        mean_norm_tolerance_m=project_config.risk.mean_norm_tolerance_m,
        variance_tolerance_m2=project_config.risk.variance_tolerance_m2,
        covariance_symmetry_tolerance=project_config.kf.covariance_symmetry_tolerance,
        covariance_psd_tolerance=project_config.kf.covariance_psd_tolerance,
    )
    assert segment_bound == 1.0, (
        "expected deterministic swept collision bound 1, "
        f"actual={segment_bound}"
    )


def test_union_bound_preserves_untruncated_sum() -> None:
    """验证 Eq. (39) 同时保存截断风险和未截断 U，避免排序诊断信息丢失。"""

    result = aggregate_union_bounds([0.4, 0.7, 0.2])
    tolerance = 1.0e-15
    assert abs(result.risk_upper_bound - 1.0) <= tolerance, (
        f"expected clipped bound 1, actual={result.risk_upper_bound}"
    )
    assert abs(result.untruncated_union_bound - 1.3) <= tolerance, (
        f"expected U=1.3, actual={result.untruncated_union_bound}"
    )
