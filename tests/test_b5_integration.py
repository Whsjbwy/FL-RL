"""B5编排、双动作、历史数学保持及checkpoint；仅工程测试。"""

import inspect
from copy import deepcopy

import numpy as np
import pytest
import torch

from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.constrained_runner import ConstrainedSACRunner
from auv_risk_rl.rl.cost_agent import CostLearningAgent
from auv_risk_rl.rl.cost_runner import CostLearningRunner
from auv_risk_rl.rl.episode_cost import EpisodeCostLedger
from auv_risk_rl.rl.losses import actor_loss
from auv_risk_rl.rl.runner import OrdinarySACRunner
from test_b3_integration import assert_tree, batch, make_env, transition
from test_b4_learning import complete_fixture
from test_b5_actor import make_agent


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def populated():
    agent = make_agent()
    for i in range(4):
        agent.store_transition(**transition(i))
    agent.update()
    for i in range(19):
        agent.multiplier.submit(complete_fixture(i, .1))
    return agent


@pytest.mark.parametrize('field', [
    'models', 'optimizers', 'log_alpha', 'actor_rng', 'replay', 'counters',
    'cost_q', 'cost_target', 'cost_optimizer', 'cost_rng', 'multiplier', 'actor_components',
])
def test_checkpoint_parts(tmp_path, field):
    left = populated()
    path = tmp_path/'b5.pt'
    left.save_checkpoint(path)
    right = make_agent()
    right.load_checkpoint(path)
    a, b = left.state_dict(), right.state_dict()
    if field in a['b4_state']['ordinary']:
        assert_tree(a['b4_state']['ordinary'][field], b['b4_state']['ordinary'][field])
    elif field in a['b4_state']:
        assert_tree(a['b4_state'][field], b['b4_state'][field])
    else:
        assert_tree(a[field], b[field])
    assert_tree(a, b)


def test_checkpoint_next_update_and_tenth_episode(tmp_path):
    left = populated()
    left.save_checkpoint(tmp_path/'resume.pt')
    right = make_agent()
    right.load_checkpoint(tmp_path/'resume.pt')
    for agent in (left, right):
        agent.multiplier.submit(complete_fixture(19, .2))
    assert_tree(left.update(), right.update())
    assert_tree(left.state_dict(), right.state_dict())
    assert left.multiplier.lambda_update_count == 2


def test_b4_migration_and_old_checkpoint_unchanged(tmp_path):
    new = make_agent()
    old = CostLearningAgent(new.config, source_fingerprint='b4-source')
    for i in range(4):
        old.store_transition(**transition(i))
    old.update()
    old.multiplier.submit(complete_fixture(0, .1))
    path = tmp_path/'old-b4.pt'
    old.save_checkpoint(path)
    raw = path.read_bytes()
    source = old.state_dict()
    metadata = new.migrate_from_b4(source, expected_source='b4-source')
    assert metadata['status'] == 'MIGRATED_FROM_B4' and not metadata['exact_B5_resume']
    expected = deepcopy(source)
    expected['ordinary']['source_fingerprint'] = 'b5-test'
    expected['migration'] = metadata
    assert_tree(expected, new.state_dict()['b4_state'])
    assert_tree(source, old.state_dict())
    metrics = new.update(batch())
    assert metrics['actor_cost_q_term'] > 0
    restored = CostLearningAgent(new.config, source_fingerprint='b4-source')
    restored.load_checkpoint(path)
    assert_tree(restored.state_dict(), source)
    assert path.read_bytes() == raw


@pytest.mark.parametrize('kind', ['source', 'format', 'used'])
def test_migration_rejects_incompatible(kind):
    new = make_agent()
    old = CostLearningAgent(new.config, source_fingerprint='b4')
    state = old.state_dict()
    if kind == 'source':
        state['ordinary']['source_fingerprint'] = 'wrong'
    elif kind == 'format':
        state['format'] = 'ordinary-sac-full-resume-v1'
    else:
        new.update(batch())
    with pytest.raises(ValueError):
        new.migrate_from_b4(state, expected_source='b4')


def test_ordinary_independent_no_monkeypatch(tmp_path, project_config):
    method = OrdinarySACAgent.update_actor
    new = make_agent()
    ordinary = OrdinarySACAgent(new.config, source_fingerprint='ordinary')
    assert type(ordinary) is OrdinarySACAgent
    assert OrdinarySACAgent.update_actor is method
    assert 'cost' not in inspect.signature(actor_loss).parameters
    assert 'lambda' not in inspect.getsource(OrdinarySACAgent.update_actor)
    runner = OrdinarySACRunner(make_env(project_config), ordinary)
    for _ in range(4):
        result = runner.step()
    assert result['metrics'] and 'actor_cost_q_term' not in result['metrics']
    ordinary.save_checkpoint(tmp_path/'ordinary.pt')
    other = OrdinarySACAgent(new.config, source_fingerprint='ordinary')
    other.load_checkpoint(tmp_path/'ordinary.pt')
    assert_tree(ordinary.update(batch()), other.update(batch()))
    assert_tree(ordinary.state_dict(), other.state_dict())


def test_b4_update_order_reused(monkeypatch):
    agent = make_agent()
    calls = []
    for name, optimizer in agent.optimizers.items():
        original = optimizer.step

        def observed(*args, label=name, function=original, **kwargs):
            calls.append(label)
            return function(*args, **kwargs)

        monkeypatch.setattr(optimizer, 'step', observed)
    original = agent.cost_optimizer.step

    def cost_step(*args, **kwargs):
        calls.append('cost')
        return original(*args, **kwargs)

    monkeypatch.setattr(agent.cost_optimizer, 'step', cost_step)
    agent.update(batch())
    assert calls == ['q1', 'q2', 'actor', 'alpha', 'cost']
    assert type(agent.ordinary).update is OrdinarySACAgent.update
    assert ConstrainedSACRunner.step is CostLearningRunner.step


def test_executed_action_does_not_enter_learning():
    left, right = make_agent(), make_agent()
    a, b = batch(), batch()
    b['executed_action'][:] = .99
    assert_tree(left.update(a), right.update(b))
    assert_tree(left.state_dict(), right.state_dict())


def test_reward_and_temperature_cost_independence():
    left, right = make_agent(), make_agent()
    a, b = batch(), batch()
    b['cost'][:] = 0
    right.multiplier.value = 10.
    x, y = left.update(a), right.update(b)
    # 本步Actor不同，但reward目标/先行Q更新及同样本logpi的温度更新不读成本。
    for key in ('q1', 'q2', 'target_q1', 'target_q2'):
        assert_tree(getattr(left, key).state_dict(), getattr(right, key).state_dict())
    assert_tree(left.optimizers['alpha'].state_dict(), right.optimizers['alpha'].state_dict())
    assert torch.equal(left.log_alpha, right.log_alpha)
    assert x['actor_cost_q_term'] != y['actor_cost_q_term']


def test_runner_full_resume_and_b4_migration(tmp_path, project_config):
    left = ConstrainedSACRunner(make_env(project_config), make_agent(),
                                reset_options={'external_max_steps':5})
    for _ in range(4):
        left.step()
    left.save_checkpoint(tmp_path/'runner.pt')
    right = ConstrainedSACRunner(make_env(project_config), make_agent())
    right.load_checkpoint(tmp_path/'runner.pt')
    assert_tree(left.step(), right.step())
    assert_tree(left.agent.state_dict(), right.agent.state_dict())
    assert_tree(vars(left.ledger), vars(right.ledger))
    np.testing.assert_array_equal(left.obs, right.obs)
    assert left.agent.multiplier.incomplete_count == 1
    assert left.agent.multiplier.lambda_update_count == 0
    old = CostLearningRunner(make_env(project_config),
                            CostLearningAgent(make_agent().config, source_fingerprint='b4'))
    old.step()
    old.save_checkpoint(tmp_path/'b4-runner.pt')
    migrated = ConstrainedSACRunner(make_env(project_config), make_agent())
    meta = migrated.migrate_from_b4_checkpoint(tmp_path/'b4-runner.pt', expected_source='b4')
    assert meta['status'] == 'MIGRATED_FROM_B4'
    assert_tree(vars(old.ledger), vars(migrated.ledger))
    np.testing.assert_array_equal(old.obs, migrated.obs)
    migrated.step()
    assert len(migrated.ledger.instant_costs) == 2


def test_lambda_lifecycle_is_original():
    agent = make_agent()
    before = agent.multiplier.state_dict()
    agent.update(batch())
    assert_tree(before, agent.multiplier.state_dict())
    incomplete = EpisodeCostLedger(100).append(1., 1, False, True, 'external_truncation')
    agent.multiplier.submit(incomplete)
    for i in range(9):
        agent.multiplier.submit(complete_fixture(i, .2))
    assert agent.multiplier.value == 1.
    agent.multiplier.submit(complete_fixture(9, .2))
    assert agent.multiplier.value == pytest.approx(1.15)
    assert agent.multiplier.incomplete_count == 1


def test_lambda_zero_full_b4_update_identity():
    new = make_agent()
    old = CostLearningAgent(new.config, source_fingerprint='b5-test')
    old.update(batch())
    old.multiplier.value = 0.
    CostLearningAgent.load_state_dict(new, old.state_dict())
    old.update(batch())
    new.update(batch())
    assert_tree(old.state_dict(), new.state_dict()['b4_state'])


def test_b5_wrapper_executes_and_replay_dual_action(monkeypatch, project_config):
    from auv_risk_rl.env.world import AUVWorld
    from auv_risk_rl.types import ControlCommand, ValidationDecision

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
    runner = ConstrainedSACRunner(env, make_agent())
    runner.step()
    assert len(calls) == 1
    np.testing.assert_array_equal(env.world.auv_state.position_ned_m,
                                  expected.auv_state.position_ned_m)
    data = runner.agent.replay.at(np.array([0]))
    assert not np.array_equal(data['nominal_action'], data['executed_action'])
    assert len(runner.ledger.instant_costs) == 1
