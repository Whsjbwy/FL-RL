"""CM02/CM03：冻结方向、统一符号与极小正方差的正式回归。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from auv_risk_rl.risk.gaussian_bounds import (
    segment_collision_upper_bound,
    segment_projection_direction,
)


def test_segment_direction_matches_frozen_spec(project_config) -> None:
    """PA01：和向量方向及审计反例的单段概率都对应原式(38)。"""

    start = np.array([2.0, 0.0, 0.0])
    end = np.array([2.0, 2.0, 0.0])
    direction = (start + end) / np.linalg.norm(start + end)
    actual = segment_projection_direction(start, end, project_config.risk.mean_norm_tolerance_m)
    np.testing.assert_allclose(actual, direction, rtol=0.0, atol=1e-15)
    covariance = 0.04 * np.eye(3)
    expected = float(ndtr((1.5 - direction @ start) / 0.2)
                     + ndtr((1.5 - direction @ end) / 0.2))
    bound = segment_collision_upper_bound(
        start, end, covariance, covariance, 1.5,
        project_config.risk.mean_norm_tolerance_m,
        project_config.risk.variance_tolerance_m2,
        project_config.kf.covariance_symmetry_tolerance,
        project_config.kf.covariance_psd_tolerance,
    )
    assert bound == pytest.approx(expected, abs=1e-15)
    assert bound > 0.05


def test_zero_sum_segment_direction_uses_fixed_north_axis(project_config) -> None:
    """两端和恰为零时，退回固定 North 轴而不是未知未来方向。"""

    start = np.array([0.0, -2.0, 0.0])
    actual = segment_projection_direction(start, -start, project_config.risk.mean_norm_tolerance_m)
    np.testing.assert_array_equal(actual, np.array([1.0, 0.0, 0.0]))


def test_nonzero_tiny_direction_is_not_replaced_by_axis(project_config) -> None:
    """非零和很小时依然使用原式方向，不由正容差静默替换。"""

    start = np.array([0.0, 1e-200, 0.0])
    actual = segment_projection_direction(start, start, project_config.risk.mean_norm_tolerance_m)
    np.testing.assert_array_equal(actual, np.array([0.0, 1.0, 0.0]))


def test_positive_tiny_projection_variance_not_deterministic(project_config) -> None:
    """PA02：合法秩一高斯反例不能被误判成零风险确定事件。"""

    from auv_risk_rl.risk.gaussian_bounds import point_collision_upper_bound

    std = 1e-7
    mean = np.array([1.0 + 0.5 * std, 0.0, 0.0])
    covariance = np.diag([1e-14, 0.0, 0.0])
    expected = float(ndtr((1.0 - mean[0]) / std))
    exact_ball = expected - float(ndtr((-1.0 - mean[0]) / std))
    actual = point_collision_upper_bound(
        mean, covariance, 1.0, project_config.risk.mean_norm_tolerance_m,
        project_config.risk.variance_tolerance_m2,
        project_config.kf.covariance_symmetry_tolerance,
        project_config.kf.covariance_psd_tolerance,
    )
    assert actual == pytest.approx(expected, abs=1e-14)
    assert actual >= exact_ball - 1e-14
    assert actual > 0.30
    segment = segment_collision_upper_bound(
        mean, mean, covariance, covariance, 1.0,
        project_config.risk.mean_norm_tolerance_m,
        project_config.risk.variance_tolerance_m2,
        project_config.kf.covariance_symmetry_tolerance,
        project_config.kf.covariance_psd_tolerance,
    )
    assert segment == pytest.approx(min(1.0, 2.0 * expected), abs=1e-14)


@pytest.mark.parametrize("variance", [1e-20, 1e-16, 1e-14])
def test_every_positive_variance_uses_gaussian_boundary_probability(
    project_config, variance
) -> None:
    """均值恰在半空间边界时，任意正方差均为0.5而不是确定事件1。"""

    from auv_risk_rl.risk.gaussian_bounds import point_collision_upper_bound

    actual = point_collision_upper_bound(
        np.array([1.0, 0.0, 0.0]), np.diag([variance, 0.0, 0.0]), 1.0,
        project_config.risk.mean_norm_tolerance_m,
        project_config.risk.variance_tolerance_m2,
        project_config.kf.covariance_symmetry_tolerance,
        project_config.kf.covariance_psd_tolerance,
    )
    assert actual == 0.5


def test_negative_projection_correction_is_one_sided(project_config) -> None:
    """仅微小负值可校正为零，超容差负值必须抛具体风险异常。"""

    from auv_risk_rl.exceptions import NumericalRiskError
    from auv_risk_rl.risk.gaussian_bounds import _validated_projected_variance

    tolerance = project_config.risk.variance_tolerance_m2
    assert _validated_projected_variance(-0.5 * tolerance, tolerance) == 0.0
    assert _validated_projected_variance(0.5 * tolerance, tolerance) == 0.5 * tolerance
    assert _validated_projected_variance(0.0, tolerance) == 0.0
    with pytest.raises(NumericalRiskError):
        _validated_projected_variance(-2.0 * tolerance, tolerance)
    with pytest.raises(NumericalRiskError):
        _validated_projected_variance(float("nan"), tolerance)


def test_risk_rotation_and_isotropic_exact_reference(project_config) -> None:
    """补齐旋转与各向同性精确CDF回归；原34项未含独立的全部特殊情形。"""

    from scipy.stats import ncx2

    from auv_risk_rl.frames import rotation_body_to_ned
    from auv_risk_rl.risk.gaussian_bounds import point_collision_upper_bound

    args = (
        project_config.risk.mean_norm_tolerance_m,
        project_config.risk.variance_tolerance_m2,
        project_config.kf.covariance_symmetry_tolerance,
        project_config.kf.covariance_psd_tolerance,
    )
    mean = np.array([2.0, 0.5, -0.1])
    covariance = np.diag([0.1, 0.2, 0.3])
    rotation = rotation_body_to_ned(0.7, -0.2)
    reference = point_collision_upper_bound(mean, covariance, 1.5, *args)
    rotated = point_collision_upper_bound(rotation @ mean, rotation @ covariance @ rotation.T,
                                          1.5, *args)
    assert rotated == pytest.approx(reference, abs=1e-14)
    end = mean + np.array([0.3, 0.1, -0.1])
    segment = segment_collision_upper_bound(mean, end, covariance, covariance, 1.5, *args)
    rotated_segment = segment_collision_upper_bound(
        rotation @ mean, rotation @ end, rotation @ covariance @ rotation.T,
        rotation @ covariance @ rotation.T, 1.5, *args,
    )
    assert segment == pytest.approx(rotated_segment, abs=1e-14)
    for distance in (0.5, 1.5, 3.0):
        mu = np.array([distance, 0.0, 0.0])
        exact = float(ncx2.cdf((1.5 / 0.2) ** 2, 3, (distance / 0.2) ** 2))
        bound = point_collision_upper_bound(mu, 0.04 * np.eye(3), 1.5, *args)
        assert bound + 1e-14 >= exact
    assert point_collision_upper_bound(np.array([2.0, 0.0, 0.0]), np.zeros((3, 3)),
                                       1.5, *args) == 0.0
    assert point_collision_upper_bound(np.array([1.0, 0.0, 0.0]), np.zeros((3, 3)),
                                       1.5, *args) == 1.0


def test_validator_risk_uses_auv_minus_obstacle_sign(project_config, monkeypatch) -> None:
    """原式(28)定义AUV减障碍；固定轴退化时该约定不能含糊。"""

    from auv_risk_rl.safety import validator
    from auv_risk_rl.types import AUVState, KFTrackState, TrackedObstacle

    recorded = []

    def record_segment(**kwargs):
        """只记录真实风险接口的相对均值，不替代主概率回归测试。"""
        recorded.append(kwargs["relative_mean_start_ned_m"].copy())
        return 0.0

    monkeypatch.setattr(validator, "segment_collision_upper_bound", record_segment)
    start = AUVState(np.array([50.0, 50.0, 20.0]), 0.0, 0.0, 0.8, 0.0, 0.0)
    end = AUVState(np.array([50.04, 50.0, 20.0]), 0.0, 0.0, 0.8, 0.0, 0.0)
    track = KFTrackState(1, np.array([70.0, 50.0, 20.0, 0.2, 0.0, 0.0]), np.eye(6), 0.0, 0.0)
    validator._calculate_rollout_risk(
        [start, end], 0.0, [TrackedObstacle(track, 0.5)], project_config
    )
    np.testing.assert_array_equal(recorded[0], np.array([-20.0, 0.0, 0.0]))
