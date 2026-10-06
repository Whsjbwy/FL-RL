"""验证模型匹配三维高斯位置边缘的 95% 椭球覆盖参考性质。"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from auv_risk_rl.seeding import SeedManager


def test_gaussian_95pct_ellipsoid_coverage_within_sampling_tolerance() -> None:
    """
    用独立 Monte Carlo 验证三维高斯 95% 置信椭球覆盖率接近 0.95。

    这是模型内概率对象检查，不是 AUV 导航碰撞率。固定样本数下采用预先给定采样容差，
    失败时应检查白化与自由度，不允许通过事后放宽容差绕过。
    """

    covariance_ned_m2 = np.array(
        [[1.0, 0.2, 0.1], [0.2, 0.8, -0.05], [0.1, -0.05, 1.2]],
        dtype=np.float64,
    )
    mean_ned_m = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    sample_count = 30000
    rng = SeedManager(20260917).get_rng("evaluation")
    samples_ned_m = rng.multivariate_normal(mean_ned_m, covariance_ned_m2, size=sample_count)
    centered = samples_ned_m - mean_ned_m
    solved = np.linalg.solve(covariance_ned_m2, centered.T).T
    mahalanobis_squared = np.einsum("ni,ni->n", centered, solved)
    threshold = float(chi2.ppf(0.95, df=3))
    coverage = float(np.mean(mahalanobis_squared <= threshold))
    expected_coverage = 0.95
    sampling_tolerance = 0.01
    assert abs(coverage - expected_coverage) <= sampling_tolerance, (
        f"expected={expected_coverage}±{sampling_tolerance}, actual={coverage}, "
        f"sample_count={sample_count}"
    )
