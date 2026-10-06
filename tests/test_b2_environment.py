"""B2环境薄层：真实世界转移及明确标记的验证器接口替身。"""

from dataclasses import replace

import numpy as np
import pytest

from auv_risk_rl.env.local_navigation import LocalNavigationEnv
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.types import AUVState, ControlCommand, GroundTruthObstacleState, ValidationDecision


def make_env(config, goal=None, obstacles=(), state=None):
    """默认空障碍场景；所有reset仍执行完整真实warm-up。"""
    own = state or AUVState(np.array([20., 50., 20.]), 0., 0., .3, 0., 0.)
    return LocalNavigationEnv(config, own, obstacles,
                              np.array([80., 50., 20.]) if goal is None else goal, 'b2-test')


def fixed_validator(monkeypatch, command, kind='modified'):
    """只替换决策接口以隔离执行语义；世界与reward使用真实代码。"""
    def decide(auv_state, current_timestamp_s, nominal_action, previous_executed_action,
               tracked_obstacles, config):
        return ValidationDecision(nominal_action, command, kind, 1., 2., 'test_unverified')
    monkeypatch.setattr('auv_risk_rl.env.local_navigation.validate_nominal_action', decide)


def test_env_01_02_03_executed_pipeline(project_config, monkeypatch):
    rewards = []
    for command in (ControlCommand(.3, 0., 0.), ControlCommand(.9, .175, 0.)):
        env = make_env(project_config)
        initial, _ = env.reset(seed=9)
        reference = AUVWorld(project_config, env.world.auv_state, (), env.goal_position_ned_m, 1.)
        expected = reference.step(command)
        fixed_validator(monkeypatch, command)
        observation, reward, terminated, truncated, info = env.step(np.zeros(3))
        np.testing.assert_array_equal(env.world.auv_state.position_ned_m,
                                      expected.auv_state.position_ned_m)
        np.testing.assert_allclose(
            observation[13:16], info['executed_action_normalized'], atol=1e-7)
        difference = info['executed_action_normalized']-initial[13:16]
        assert info['reward_components']['smoothness'] == pytest.approx(
            -.02*(difference@difference))
        assert reward == pytest.approx(sum(info['reward_components'].values()))
        assert info['cost'] == 1 and info['cost_components']['c_real'] == 0
        assert not terminated and not truncated
        assert info['policy_version'] is None
        rewards.append(info['reward_components']['smoothness'])
    assert rewards[0] != rewards[1]


@pytest.mark.parametrize('kind', ['fallback', 'modified'])
def test_env_08_11_reject_is_not_terminal(project_config, monkeypatch, kind):
    env = make_env(project_config)
    env.reset()
    fixed_validator(monkeypatch, ControlCommand(.3, 0., 0.), kind)
    _, _, terminated, truncated, info = env.step(np.zeros(3))
    assert not terminated and not truncated
    assert info['failure_type'] == 'none' and info['intervention']
    assert info['fallback'] == (kind == 'fallback')


def test_env_09_reset_and_external_cutoff(project_config):
    env = make_env(project_config)
    initial, info = env.reset(seed=71, options={'external_max_steps': 1})
    _, _, terminated, truncated, step_info = env.step(np.zeros(3))
    assert not terminated and truncated
    assert step_info['failure_type'] == 'external_truncation'
    with pytest.raises(RuntimeError):
        env.step(np.zeros(3))
    again, again_info = env.reset(seed=71)
    np.testing.assert_array_equal(initial, again)
    assert info['warmup_measurement_times_s'] == again_info['warmup_measurement_times_s']
    assert env.world.control_step_index == 0
    assert again[16] == 1 and env.world.timestamp_s == 1.


@pytest.mark.parametrize('event', ['success', 'collision', 'boundary'])
def test_env_05_06_07_actual_earliest_events(project_config, monkeypatch, event):
    config = replace(project_config, sensor=replace(project_config.sensor, dropout_probability=1.))
    goal = np.array([22.34, 50., 20.]) if event == 'success' else None
    obstacles = ()
    own = None
    if event == 'collision':
        obstacles = (GroundTruthObstacleState(1, np.array([22.09, 50., 20.]), np.zeros(3), 1.),)
    if event == 'boundary':
        own = AUVState(np.array([98.91, 50., 20.]), 0., 0., .3, 0., 0.)
    env = make_env(config, goal, obstacles, own)
    env.reset()
    fixed_validator(monkeypatch, ControlCommand(.3, 0., 0.), 'fallback')
    _, _, terminated, truncated, info = env.step(np.array([-1., 0., 0.]))
    assert terminated and not truncated
    names = {'success': 'goal_success', 'collision': 'collision',
             'boundary': 'operational_boundary_failure'}
    assert info['failure_type'] == names[event]
    assert 0 < info['elapsed_s'] < .2
    assert info['reward_components']['time'] == pytest.approx(-.01*info['elapsed_s']/.2)
    assert info['cost_components']['f_t'] == int(event != 'success')


def test_env_10_intrinsic_horizon(project_config, monkeypatch):
    env = make_env(project_config, goal=np.array([80., 80., 20.]))
    env.reset()
    fixed_validator(monkeypatch, ControlCommand(.3, 0., 0.), 'nominal')
    for _ in range(999):
        _, _, terminated, truncated, _ = env.step(np.array([-1., 0., 0.]))
        assert not terminated and not truncated
    observation, _, terminated, truncated, info = env.step(np.array([-1., 0., 0.]))
    assert terminated and not truncated and info['failure_type'] == 'task_horizon'
    assert observation[16] == 0 and info['task_control_step'] == 1000


def test_env_real_validator_no_obstacles(project_config):
    env = make_env(project_config)
    env.reset(seed=11)
    observation, _, terminated, _, info = env.step(np.zeros(3))
    assert observation.shape == (234,) and not terminated
    assert info['cost'] == 0 and not info['intervention']
    assert info['unclipped_risk_U'] == 0


def test_env_all_maintained_tracks_reach_original_validator(project_config, monkeypatch):
    from auv_risk_rl.safety.validator import validate_nominal_action
    from auv_risk_rl.types import KFTrackState, TrackedObstacle

    env = make_env(project_config)
    env.reset()
    origin = env.world.auv_state.position_ned_m.copy()
    targets = []
    for index in range(8):
        relative = [0., 6.+index*.1, 0.] if index < 6 else [10.+index-6, 0., 0.]
        velocity = [0., 0., 0.] if index < 6 else [-2., 0., 0.]
        targets.append(TrackedObstacle(KFTrackState(index, np.r_[origin+relative, velocity],
                                                    np.eye(6)*1e-6, 1., 1.), .5))
    def current_tracks(timestamp):
        return tuple(replace(t, track_state=replace(t.track_state, state_timestamp_s=timestamp))
                     for t in targets)
    env.perception.tracks = current_tracks
    captured = []
    def checked(*args):
        captured.append(len(args[4]))
        return validate_nominal_action(*args)
    monkeypatch.setattr('auv_risk_rl.env.local_navigation.validate_nominal_action', checked)
    before = env._observation()
    assert np.sum(before[128::21]) == 6
    _, _, _, _, info = env.step(np.zeros(3))
    assert captured == [8] and info['cost_components']['c_risk'] == 1


def test_env_numerical_fallback_not_terminal(project_config, monkeypatch):
    from auv_risk_rl.safety.validator import CandidateEvaluation
    from auv_risk_rl.types import RiskResult

    env = make_env(project_config)
    env.reset()
    def invalid_evaluation(auv_state, current_timestamp_s, action, candidate_id,
                           tracked_obstacles, config):
        return CandidateEvaluation(action, candidate_id, True,
                                   RiskResult(1., float('inf'), False, 'sentinel_numeric'), False)
    monkeypatch.setattr(
        'auv_risk_rl.safety.validator.evaluate_candidate_action', invalid_evaluation)
    observation, _, terminated, truncated, info = env.step(np.zeros(3))
    assert info['fallback'] and info['cost'] == 1
    assert info['fallback_type'] == 'no_valid_evaluation_unverified_fallback'
    assert not terminated and not truncated and np.all(np.isfinite(observation))


def test_reset_clears_previous_episode_tracks(project_config):
    config = replace(project_config, sensor=replace(project_config.sensor, dropout_probability=0.))
    obstacle = GroundTruthObstacleState(1, np.array([30., 50., 20.]), np.zeros(3), 1.)
    env = make_env(config, obstacles=(obstacle,))
    first, _ = env.reset(seed=37)
    env.step(np.array([-1., 0., 0.]))
    second, _ = env.reset(seed=37)
    np.testing.assert_array_equal(first, second)
    assert len(env.perception.generated) == 6
    np.testing.assert_array_equal(second[129:], 0)
