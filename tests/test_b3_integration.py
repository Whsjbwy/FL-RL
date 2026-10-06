"""普通SAC Replay、工程编排与完整恢复；不作科学性能结论。"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from auv_risk_rl.env.local_navigation import LocalNavigationEnv
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.rl.replay import ReplayBuffer
from auv_risk_rl.rl.runner import OrdinarySACRunner
from auv_risk_rl.types import AUVState, ControlCommand, ValidationDecision


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def transition(value=0, **overrides):
    result = dict(obs=np.full(234, value, np.float32), nominal_action=np.full(3, .2, np.float32),
                  executed_action=np.full(3, -.4, np.float32), reward=float(value), cost=1.,
                  next_obs=np.full(234, value+1, np.float32), terminated=False, truncated=False,
                  failure_type='none', episode_id=0, task_step=value+1)
    result.update(overrides)
    return result


def make_agent(**overrides):
    config = replace(SACConfig(batch_size=4, replay_capacity=16, learning_starts=4), **overrides)
    return OrdinarySACAgent(config, source_fingerprint='synthetic-test')


def make_env(config):
    return LocalNavigationEnv(config, AUVState(np.array([20., 50., 20.]), 0., 0., .3, 0., 0.),
                              (), np.array([80., 50., 20.]), 'b3-engineering-only')


def batch():
    replay = ReplayBuffer(16)
    for i in range(4):
        replay.add(**transition(i))
    return replay.at(np.arange(4))


def assert_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree(a[key], b[key])
    elif isinstance(a, list | tuple):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_tree(left, right)
    else:
        assert a == b


def test_sac_r01_overwrite():
    replay = ReplayBuffer(3)
    for i in range(5):
        replay.add(**transition(i))
    assert len(replay) == 3 and replay.cursor == 2
    np.testing.assert_array_equal(replay.at(np.arange(3))['reward'], [3., 4., 2.])


@pytest.mark.parametrize('field,shape,dtype', [
    ('obs', (4, 234), np.float32), ('nominal_action', (4, 3), np.float32),
    ('executed_action', (4, 3), np.float32), ('reward', (4,), np.float32),
    ('cost', (4,), np.float32), ('terminated', (4,), np.bool_),
    ('truncated', (4,), np.bool_), ('failure_type', (4,), np.uint8),
    ('episode_id', (4,), np.int64), ('task_step', (4,), np.int64)],
    ids=['R02_obs', 'R03_nominal', 'R04_executed', 'R05_reward', 'R05_cost',
         'R06_terminated', 'R06_truncated', 'failure', 'episode', 'step'])
def test_sac_replay_shapes(field, shape, dtype):
    values = batch()[field]
    assert values.shape == shape and values.dtype == dtype


def test_sac_r04_r05_dual_fields():
    data = batch()
    assert np.all(data['nominal_action'] == np.float32(.2))
    assert np.all(data['executed_action'] == np.float32(-.4))
    np.testing.assert_array_equal(data['reward'], np.arange(4))
    assert np.all(data['cost'] == 1)


def test_sac_r06_flags():
    replay = ReplayBuffer(3)
    replay.add(**transition(terminated=True, truncated=False))
    replay.add(**transition(terminated=False, truncated=True))
    data = replay.at(np.arange(2))
    np.testing.assert_array_equal(data['terminated'], [True, False])
    np.testing.assert_array_equal(data['truncated'], [False, True])


def test_sac_r07_full_batch():
    replay = ReplayBuffer()
    assert replay.capacity == 500000 and not replay.chunks
    for i in range(256):
        replay.add(**transition(i))
    assert replay.sample(256)['obs'].shape == (256, 234)


def test_sac_r08_cpu_and_copy():
    replay = ReplayBuffer(4)
    for i in range(4):
        replay.add(**transition(i))
    before = deepcopy(replay.chunks)
    sample = replay.sample(4)
    for value in sample.values():
        assert isinstance(value, np.ndarray)
        value.fill(0)
    assert_tree(before, replay.chunks)


def test_sac_replay_validation_atomic():
    replay = ReplayBuffer(4)
    with pytest.raises(ValueError):
        replay.add(**transition(next_obs=np.full(234, np.nan)))
    assert len(replay) == 0 and not replay.chunks
    replay.add(**transition(next_obs=np.full(234, np.nan), terminated=True))
    assert len(replay) == 1


def test_sac_o01_before_starts(project_config):
    agent = make_agent(learning_starts=10000)
    runner = OrdinarySACRunner(make_env(project_config), agent)
    for _ in range(2):
        assert not runner.step()['metrics']
    assert agent.counters['environment_steps'] == 2
    assert agent.counters['gradient_updates'] == 0


def test_sac_o02_o03_exact_boundary(project_config):
    agent = make_agent(learning_starts=10000)
    for i in range(4):
        agent.store_transition(**transition(i))
    # 显式计数边界夹具；不是运行9998个科学transition。
    agent.counters['environment_steps'] = 9998
    runner = OrdinarySACRunner(make_env(project_config), agent)
    assert not runner.step()['metrics']
    assert runner.step()['metrics']
    assert agent.counters['environment_steps'] == 10000
    assert agent.counters['gradient_updates'] == 1
    runner.step()
    assert agent.counters['gradient_updates'] == 2
    assert all(agent.counters[k] == 2 for k in ('q1', 'q2', 'actor', 'alpha'))
    assert agent.config.utd == 1


def test_sac_o_batch_required(project_config):
    agent = make_agent(learning_starts=0)
    runner = OrdinarySACRunner(make_env(project_config), agent)
    runner.step()
    assert not agent.eligible() and agent.counters['gradient_updates'] == 0


def test_sac_o04_o05_warmup_substeps(project_config):
    runner = OrdinarySACRunner(make_env(project_config), make_agent())
    runner.step()
    assert runner.env.world.timestamp_s == pytest.approx(1.2)
    assert runner.env.world.control_step_index == 1
    assert len(runner.agent.replay) == 1
    assert runner.agent.counters['environment_steps'] == 1
    assert runner.agent.replay.at(np.array([0]))['task_step'][0] == 1


def test_sac_o06_o07_wrapper_once_execution(project_config, monkeypatch):
    command = ControlCommand(.3, .1, 0.)
    calls = []

    def decide(state, time, nominal, previous, tracks, config):
        calls.append(nominal)
        return ValidationDecision(nominal, command, 'modified', 1., 2., 'engineering-test')

    monkeypatch.setattr('auv_risk_rl.env.local_navigation.validate_nominal_action', decide)
    env = make_env(project_config)
    env.reset(seed=0)
    reference = AUVWorld(project_config, deepcopy(env.world.auv_state), (),
                         env.goal_position_ned_m, initial_timestamp_s=1.)
    expected = reference.step(command)
    runner = OrdinarySACRunner(env, make_agent())
    runner.step()
    assert len(calls) == 1
    np.testing.assert_array_equal(env.world.auv_state.position_ned_m,
                                  expected.auv_state.position_ned_m)
    data = runner.agent.replay.at(np.array([0]))
    assert not np.array_equal(data['nominal_action'], data['executed_action'])


def test_sac_r09_external_next_before_reset(project_config):
    runner = OrdinarySACRunner(make_env(project_config), make_agent(),
                               reset_options={'external_max_steps': 1})
    result = runner.step()
    assert result['truncated'] and not result['terminated']
    actual_next = runner.obs.copy()
    runner.step()
    data = runner.agent.replay.at(np.array([0, 1]))
    np.testing.assert_array_equal(data['next_obs'][0], actual_next)
    assert not np.array_equal(data['next_obs'][0], data['obs'][1])
    assert runner.agent.counters['episodes'] == 2


def test_sac_o08_o09_cost_executed_invariant():
    left, right = make_agent(), make_agent()
    first, second = batch(), batch()
    second['executed_action'][:] = -.8
    second['cost'][:] = np.nan  # 损失不能读取成本，即使测试哨兵无效。
    metrics_left, metrics_right = left.update(first), right.update(second)
    assert metrics_left == metrics_right
    assert_tree(left.state_dict(), right.state_dict())


@pytest.fixture
def trained_pair(tmp_path):
    left = make_agent()
    for i in range(4):
        left.store_transition(**transition(i))
    left.update()
    left.counters['episodes'] = 2
    saved_rng = left.generator.get_state().clone()
    path = tmp_path/'full.pt'
    left.save_checkpoint(path)
    assert torch.equal(saved_rng, left.generator.get_state())
    right = make_agent()
    right.load_checkpoint(path)
    return left, right


@pytest.mark.parametrize('component', ['actor', 'q1', 'q2', 'target_q1', 'target_q2'],
                         ids=['C01_actor', 'C02_q1', 'C02_q2', 'C02_target1', 'C02_target2'])
def test_sac_c01_c02_models(trained_pair, component):
    left, right = trained_pair
    assert_tree(getattr(left, component).state_dict(), getattr(right, component).state_dict())


def test_sac_c03_adam(trained_pair):
    left, right = trained_pair
    for name in left.optimizers:
        assert left.optimizers[name].state_dict()['state']
        assert_tree(left.optimizers[name].state_dict(), right.optimizers[name].state_dict())


def test_sac_c04_alpha(trained_pair):
    left, right = trained_pair
    assert torch.equal(left.log_alpha, right.log_alpha)


def test_sac_c05_counters(trained_pair):
    left, right = trained_pair
    assert left.counters == right.counters
    assert left.counters['environment_steps'] == 4 and left.counters['gradient_updates'] == 1


def test_sac_c06_full_replay_rng_next_update(trained_pair):
    left, right = trained_pair
    assert_tree(left.replay.state_dict(), right.replay.state_dict())
    assert left.state_dict()['format'] == 'ordinary-sac-full-resume-v1'
    assert left.update() == right.update()
    assert_tree(left.state_dict(), right.state_dict())


def test_sac_runner_mid_episode_resume(project_config, tmp_path):
    left = OrdinarySACRunner(make_env(project_config), make_agent(batch_size=2, learning_starts=2))
    left.step()
    left.step()
    path = tmp_path/'runner.pt'
    left.save_checkpoint(path)
    right = OrdinarySACRunner(make_env(project_config), make_agent(batch_size=2, learning_starts=2))
    right.load_checkpoint(path)
    assert_tree(left.step(), right.step())
    np.testing.assert_array_equal(left.obs, right.obs)
    assert_tree(left.agent.state_dict(), right.agent.state_dict())


def test_sac_checkpoint_source_mismatch(trained_pair):
    left, right = trained_pair
    state = left.state_dict()
    state['source_fingerprint'] = 'wrong-source'
    with pytest.raises(ValueError, match='源码'):
        right.load_state_dict(state)


def test_sac_numeric_fail_fast():
    agent = make_agent()
    values = batch()
    values['obs'][0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        agent.update(values)
    assert agent.last_failure_metadata['field'] == 'obs'
    assert agent.last_failure_metadata['batch_size'] == 4
    assert agent.last_failure_metadata['task_step'] == [1, 2, 3, 4]
    assert agent.counters['gradient_updates'] == 0


def test_sac_defaults():
    config = SACConfig()
    assert (config.gamma, config.batch_size, config.replay_capacity, config.learning_starts,
            config.utd) == (.999, 256, 500000, 10000, 1)


def test_sac_target_inf_not_hidden_by_min():
    agent = make_agent()
    with torch.no_grad():
        agent.target_q1.net[-1].bias.fill_(float('inf'))
    with pytest.raises(FloatingPointError, match='target_q1'):
        agent.update(batch())
    assert agent.last_failure_metadata['batch_size'] == 4
    assert agent.counters['q1'] == 0


def test_sac_terminal_nan_full_update():
    agent = make_agent()
    values = batch()
    values['terminated'][:] = True
    values['next_obs'][:] = np.nan
    metrics = agent.update(values)
    assert all(np.isfinite(v) for v in metrics.values())


def test_sac_replay_chunk_boundary():
    replay = ReplayBuffer(4097)
    for i in range(4100):
        replay.add(**transition(i))
    values = replay.at(np.array([0, 1, 2, 3, 4096]))
    np.testing.assert_array_equal(values['reward'], [4097, 4098, 4099, 3, 4096])
    restored = ReplayBuffer(4097)
    restored.load_state_dict(replay.state_dict())
    assert_tree(replay.sample(256), restored.sample(256))
