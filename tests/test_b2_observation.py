"""表13-1逐维、独立矩阵参考和所有目标风险反例。"""

import numpy as np
import pytest

from auv_risk_rl.env.observation import ObservationBuilder
from auv_risk_rl.frames import rotation_body_to_ned
from auv_risk_rl.safety.validator import evaluate_candidate_action
from auv_risk_rl.sensors.rays import RayFrame
from auv_risk_rl.types import AUVState, ControlCommand, KFTrackState, TrackedObstacle


def state():
    """导航夹具。"""
    return AUVState(np.array([20., 50., 20.]), .4, -.2, .9, .07, -.05)


def tracked(identity, relative, velocity=None, covariance=None):
    """直接合成合法后验而非真值。"""
    mean = np.r_[state().position_ned_m + relative,
                 np.zeros(3) if velocity is None else velocity]
    matrix = np.eye(6)*.01 if covariance is None else covariance
    return TrackedObstacle(KFTrackState(identity, mean, matrix, 1., .4), .5)


def build(config, tracks=()):
    """统一调用唯一编码入口。"""
    return ObservationBuilder(config).build(
        state(), np.array([80., 60., 25.]), np.array([-.4, .2, -.1]), 731,
        RayFrame(np.linspace(0, 25, 45), np.arange(45) % 2), tuple(tracks), 1.)


@pytest.mark.parametrize('count', [0, 1, 4, 6, 8])
def test_obs_01_02_06_07_08_padding(project_config, count):
    targets = [tracked(i, [i+1., 0., 0.]) for i in range(count)]
    output = build(project_config, list(reversed(targets)))
    assert output.shape == (234,) and output.dtype == np.float32
    assert np.all(np.isfinite(output))
    slots = output[108:].reshape(6, 21)
    assert np.sum(slots[:, 20]) == min(count, 6)
    np.testing.assert_array_equal(slots[min(count, 6):], 0)
    np.testing.assert_array_equal(build(project_config)[108:], 0)


def test_obs_03_04_05_exact_indices(project_config):
    output = build(project_config)
    rotation = rotation_body_to_ned(.4, -.2).T
    reference = np.r_[rotation @ np.array([60., 10., 5.])/100,
                       2*(np.array([20., 50., 20.])-[0, 0, 2])/[100, 100, 38]-1,
                       0., .2, -.2, np.sin(.4), np.cos(.4), np.sin(-.2), np.cos(-.2),
                       -.4, .2, -.1, .731, .25]
    np.testing.assert_allclose(output[:18], reference, atol=3e-8)
    np.testing.assert_allclose(output[18:63], np.linspace(0, 1, 45), atol=3e-8)
    np.testing.assert_array_equal(output[63:108], np.arange(45) % 2)


def test_obs_10_11_independent_matrix(project_config):
    matrix = np.arange(36).reshape(6, 6)/100
    covariance = matrix @ matrix.T + np.eye(6)*.02
    targets = [tracked(2, [3., 0., 0.], [100., 0., 0.], covariance),
               tracked(1, [-3., 0., 0.], [.1, -.2, .3], covariance)]
    output = build(project_config, targets)
    slot = output[108:129]
    rotation = rotation_body_to_ned(.4, -.2).T
    np.testing.assert_allclose(slot[:3], rotation @ [-3., 0., 0.]/25, atol=1e-8)
    np.testing.assert_allclose(slot[3:6], rotation @ [-2.5, -1., 1.5]/25, atol=1e-8)
    for seconds, offset in ((1., 6), (5., 12)):
        projection = np.c_[np.eye(3), seconds*np.eye(3)]
        expected = projection @ covariance @ projection.T + np.eye(3)*.001*seconds**3/3
        expected = rotation @ expected @ rotation.T
        np.testing.assert_allclose(slot[offset:offset+6],
                                   np.arcsinh(expected[np.triu_indices(3)]), rtol=1e-7)
    np.testing.assert_allclose(slot[18:], [.5, np.arcsinh(.6), 1.], rtol=1e-7)
    np.testing.assert_array_equal(targets[0].track_state.state_covariance_ned, covariance)
    np.testing.assert_array_equal(output, build(project_config, targets[::-1]))


def test_obs_09_eighth_target_changes_real_validator(project_config):
    own = AUVState(state().position_ned_m.copy(), 0., 0., .9, 0., 0.)
    near = [tracked(i, [0., 6.+i*.1, 0.], covariance=np.eye(6)*1e-6) for i in range(6)]
    danger = tracked(7, [10., 0., 0.], [-2., 0., 0.], np.eye(6)*1e-6)
    eighth = tracked(8, [0., 15., 0.], covariance=np.eye(6)*1e-6)
    command = ControlCommand(.9, 0., 0.)
    six = evaluate_candidate_action(own, 1., command, 0, near, project_config)
    eight = evaluate_candidate_action(own, 1., command, 0, [*near, danger, eighth], project_config)
    assert six.passes_validator
    assert not eight.passes_validator
    np.testing.assert_array_equal(build(project_config, near),
                                  build(project_config, [*near, danger, eighth]))
