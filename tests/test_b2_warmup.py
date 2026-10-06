"""B2合法历史与延迟发生时刻测试。"""

from dataclasses import replace

import numpy as np
import pytest

from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.runtime.local_perception import PerceptionSession, legal_warmup
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.types import AUVState, ControlCommand, GroundTruthObstacleState


def setup_history(config, delay=0, dropout=0.):
    """使用固定可见目标；噪声仍为协议真实非零噪声。"""
    config = replace(config, sensor=replace(config.sensor,
                     measurement_delay_control_steps=delay, dropout_probability=dropout))
    state = AUVState(np.array([20., 50., 20.]), 0., 0., .3, 0., 0.)
    target = GroundTruthObstacleState(1, np.array([30., 50., 20.]), np.array([.2, 0., 0.]), 1.)
    world = AUVWorld(config, state, (target,), np.array([80., 50., 20.]))
    session = PerceptionSession(config, SeedManager(19), {1: 1.})
    return config, world, session


def test_warm_01_exact_span(project_config):
    config, world, session = setup_history(project_config)
    legal_warmup(world, session, config)
    np.testing.assert_allclose(session.capture_times_s, [0, .2, .4, .6, .8, 1.])
    assert session.capture_times_s[-1] - session.capture_times_s[0] == 1.
    assert len(session.generated) == 6
    assert world.timestamp_s == pytest.approx(1.)
    assert len(session.tracks(1.)) == 1


def test_warm_02_first_noisy_zero_velocity(project_config):
    _, world, session = setup_history(project_config)
    session.capture(world, 0)
    track = session.tracks(0)[0].track_state
    measured = session.generated[0]
    np.testing.assert_array_equal(track.state_mean_ned[3:], 0)
    np.testing.assert_allclose(track.state_mean_ned[:3],
                               world.auv_state.position_ned_m + measured.relative_position_body_m)
    assert not np.array_equal(track.state_mean_ned[:3], world.obstacle_states[0].position_ned_m)
    np.testing.assert_array_equal(track.state_covariance_ned[3:, 3:], np.eye(3))
    np.testing.assert_array_equal(track.state_covariance_ned[:3, 3:], 0)
    np.testing.assert_array_equal(track.state_covariance_ned[:3, :3],
                                  measured.measurement_covariance_body_m2)


def test_warm_03_unseen(project_config):
    config, world, session = setup_history(project_config, dropout=1.)
    legal_warmup(world, session, config)
    assert session.tracks(1.) == ()
    assert session.generated == []


@pytest.mark.parametrize('delay', [1, 2])
def test_warm_04_delay(project_config, delay):
    config, world, session = setup_history(project_config, delay)
    legal_warmup(world, session, config)
    assert len(session.generated) == 6
    assert len(session.arrived) == 6-delay
    for detection in session.arrived:
        assert detection.arrival_control_tick == detection.measurement_control_tick + delay
        assert detection.arrival_timestamp_s == pytest.approx(
            detection.measurement_timestamp_s + .2*delay)
    assert session.tracks(1.)[0].track_state.last_measurement_timestamp_s == pytest.approx(
        (5-delay)*.2)
    restarted = AUVWorld(config, world.auv_state, world.obstacle_states,
                         np.array([80., 50., 20.]), initial_timestamp_s=1.)
    restarted.step(ControlCommand(.3, 0., 0.))
    session.capture(restarted, 6)
    assert len(session.arrived) == 7-delay


def test_occlusion_predict_only(project_config):
    config, world, session = setup_history(project_config)
    session.capture(world, 0)
    world.step(ControlCommand(.3, 0., 0.))
    blocker = GroundTruthObstacleState(2, np.array([25., 50., 20.]), np.zeros(3), 1.)
    occluded = AUVWorld(config, world.auv_state, (*world.obstacle_states, blocker),
                        np.array([80., 50., 20.]), initial_timestamp_s=.2)
    session.radius_by_id_m[2] = 1.
    session.capture(occluded, 1)
    original = next(t.track_state for t in session.tracks(.2) if t.track_state.obstacle_id == 1)
    assert original.last_measurement_timestamp_s == 0.
    assert original.state_timestamp_s == .2
    assert [d.obstacle_id for d in session.generated] == [1, 2]


def test_fov_missing_predict_only(project_config):
    config, world, session = setup_history(project_config)
    session.capture(world, 0)
    world.step(ControlCommand(.3, 0., 0.))
    turned = replace(world.auv_state, yaw_rad=np.pi)
    hidden = AUVWorld(config, turned, world.obstacle_states,
                      np.array([80., 50., 20.]), initial_timestamp_s=.2)
    session.capture(hidden, 1)
    posterior = session.tracks(.2)[0].track_state
    assert posterior.last_measurement_timestamp_s == 0.
    assert posterior.state_timestamp_s == .2 and len(session.generated) == 1


def test_warmup_rejects_shortened_history(project_config):
    config, world, session = setup_history(project_config)
    near_goal = AUVWorld(config, world.auv_state, (), np.array([22.1, 50., 20.]))
    with pytest.raises(ValueError, match='warm-up'):
        legal_warmup(near_goal, session, config)
