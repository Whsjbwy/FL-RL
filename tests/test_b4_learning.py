"""B4乘子、B3精确不干扰、完整checkpoint与工程生命周期。"""

import ast
import inspect
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.rl.cost_agent import CostLearningAgent
from auv_risk_rl.rl.cost_math import BETA, local_cost_tail
from auv_risk_rl.rl.cost_runner import CostLearningRunner
from auv_risk_rl.rl.episode_cost import EpisodeCostLedger, EpisodeMultiplier
from auv_risk_rl.rl.losses import actor_loss, reward_target
from test_b3_integration import assert_tree, batch, make_env, transition


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_cost_agent(**kwargs):
    config = replace(SACConfig(batch_size=4, replay_capacity=16, learning_starts=4), **kwargs)
    return CostLearningAgent(config, source_fingerprint='b4-test')


def complete_fixture(episode_id, value):
    """合成完整episode汇总，仅测试十个新episode的统计，不冒称真实导航。"""
    return dict(episode_id=episode_id, G_C=value, terminal_type='goal_success',
                eligible_for_lambda=True, actual_task_steps=1)


def test_lam01_initial_scalar():
    multiplier = EpisodeMultiplier()
    assert multiplier.value == 1 and multiplier.eta == 1 and multiplier.budget == .05
    assert isinstance(multiplier.value, float)


@pytest.mark.parametrize('count,updates,pending', [(9, 0, 9), (10, 1, 0), (20, 2, 0)],
                         ids=['LAM02_nine', 'LAM03_ten', 'LAM04_twenty'])
def test_lam_frequency(count, updates, pending):
    multiplier = EpisodeMultiplier()
    for i in range(count):
        multiplier.submit(complete_fixture(i, .1))
    assert multiplier.lambda_update_count == updates
    assert multiplier.completed_episodes_since_last_lambda_update == pending
    assert multiplier.value == pytest.approx(1+updates*.05)
    if count == 20:
        assert multiplier.last_batch['episode_ids'] == list(range(10, 20))


def test_lam05_no_reuse():
    multiplier = EpisodeMultiplier()
    for i in range(10):
        multiplier.submit(complete_fixture(i, .1))
    with pytest.raises(ValueError, match='重复'):
        multiplier.submit(complete_fixture(0, .1))
    assert multiplier.complete_count == 10 and multiplier.lambda_update_count == 1


@pytest.mark.parametrize('initial,cost,expected', [
    (1., .3, 1.25), (1., .01, .96), (.01, 0., 0.), (0., 0., 0.), (1., .05, 1.)],
    ids=['LAM06_over', 'LAM07_under', 'LAM08_projection', 'LAM08_zero', 'LAM09_equal'])
def test_lam_direction(initial, cost, expected):
    multiplier = EpisodeMultiplier()
    multiplier.value = initial
    for i in range(10):
        multiplier.submit(complete_fixture(i, cost))
    assert multiplier.value == pytest.approx(expected, abs=1e-15)


def test_lam10_external_excluded():
    ledger = EpisodeCostLedger(3)
    report = ledger.append(1, 1, False, True, 'external_truncation')
    multiplier = EpisodeMultiplier()
    multiplier.submit(report)
    assert report['G_C'] is None and not report['eligible_for_lambda']
    assert multiplier.incomplete_count == 1 and multiplier.complete_count == 0
    assert multiplier.pending_complete_episode_costs == []


@pytest.mark.parametrize('kind', ['goal_success', 'collision',
                                  'operational_boundary_failure', 'task_horizon'])
def test_lam11_complete_terminal_types(kind):
    ledger = EpisodeCostLedger(2)
    final = 1000 if kind == 'task_horizon' else 1
    for step in range(1, final):
        ledger.append(0, step, False, False, 'none')
    report = ledger.append(1, final, True, False, kind)
    multiplier = EpisodeMultiplier()
    multiplier.submit(report)
    assert multiplier.complete_count == 1
    assert multiplier.pending_complete_episode_costs == [report['G_C']]


def test_incomplete_cannot_forge_complete():
    multiplier = EpisodeMultiplier()
    report = complete_fixture(0, .5)
    report['terminal_type'] = 'external_truncation'
    with pytest.raises(ValueError):
        multiplier.submit(report)


def test_lam12_actor_static_boundary():
    tree = ast.parse(inspect.getsource(actor_loss))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {'cost_q', 'multiplier', 'lambda_value', 'cost'}
    agent = make_cost_agent()
    assert agent.update_actor.__func__ is OrdinarySACAgent.update_actor


@pytest.mark.parametrize('component', ['actor', 'q1', 'q2'],
                         ids=['NI01_actor', 'NI02_q1', 'NI02_q2'])
def test_b4_initialization_non_interference(component):
    four = make_cost_agent()
    before = torch.random.get_rng_state().clone()
    three = OrdinarySACAgent(four.config, source_fingerprint='b4-test')
    assert torch.equal(before, torch.random.get_rng_state())
    assert_tree(getattr(three, component).state_dict(), getattr(four, component).state_dict())


def test_cost_q05_q06_q07_independence():
    agent = make_cost_agent()
    assert_tree(agent.cost_q.state_dict(), agent.cost_target_q.state_dict())
    cost_ids = {p.data_ptr() for p in agent.cost_q.parameters()}
    target_ids = {p.data_ptr() for p in agent.cost_target_q.parameters()}
    ordinary_ids = {p.data_ptr() for model in (agent.actor, agent.q1, agent.q2)
                    for p in model.parameters()}
    assert not cost_ids & (target_ids | ordinary_ids)
    assert all(not p.requires_grad for p in agent.cost_target_q.parameters())
    optimizer_ids = {p.data_ptr() for g in agent.cost_optimizer.param_groups for p in g['params']}
    assert optimizer_ids == cost_ids and not target_ids & optimizer_ids


@pytest.mark.parametrize('multiplier', [0., 1000.], ids=['NI04_lambda_zero', 'NI04_lambda1000'])
def test_b4_ni03_through_ni10_full_update(multiplier):
    four = make_cost_agent()
    three = OrdinarySACAgent(four.config, source_fingerprint='b4-test')
    for i in range(4):
        three.store_transition(**transition(i))
    three.update()  # 从含Adam moments的同一B3 checkpoint开始。
    four.ordinary.load_state_dict(three.state_dict())
    four.multiplier.value = multiplier
    values = batch()
    old_cost = deepcopy(four.cost_q.state_dict())
    before_rng = torch.random.get_rng_state().clone()
    a = three.update(values)
    b = four.update(values)
    assert {k: b[k] for k in a} == a
    # 模型、目标、Adam、alpha与RNG全部精确一致。
    assert_tree(three.state_dict(), four.ordinary.state_dict())
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert any(not torch.equal(v, four.cost_q.state_dict()[k]) for k, v in old_cost.items())
    for agent in (three, four):
        result = reward_target(torch.tensor([1.]*4), torch.zeros(4, 234), torch.zeros(4),
                                agent.actor, agent.target_q1, agent.target_q2, agent.alpha,
                                agent.config.gamma, agent.generator)
        if agent is three:
            reference = result
        else:
            assert torch.equal(reference, result)


def test_b4_lambda_change_no_actor_effect():
    left, right = make_cost_agent(), make_cost_agent()
    right.multiplier.value = 1000
    a, b = left.update(batch()), right.update(batch())
    assert a['actor_loss'] == b['actor_loss']
    assert_tree(left.ordinary.state_dict(), right.ordinary.state_dict())


def test_b4_single_replay_draw():
    four = make_cost_agent()
    three = OrdinarySACAgent(four.config, source_fingerprint='b4-test')
    for i in range(4):
        four.store_transition(**transition(i))
        three.store_transition(**transition(i))
    three.update()
    four.update()
    assert_tree(three.state_dict(), four.ordinary.state_dict())


def test_cost_bound_diagnostic_only():
    agent = make_cost_agent()
    values = batch()
    values['task_step'][:] = 1000
    values['terminated'][:] = True
    values['failure_type'][:] = 4
    values['next_obs'][:] = np.nan
    metrics = agent.update(values)
    assert metrics['q_above_remaining_theoretical_bound_count'] == 4
    assert metrics['cost_target_max'] == pytest.approx(BETA)
    assert metrics['cost_q_min'] > local_cost_tail(1)


@pytest.fixture
def restored(tmp_path):
    original = make_cost_agent()
    for i in range(4):
        original.store_transition(**transition(i))
    original.update()
    for i in range(19):
        original.multiplier.submit(complete_fixture(i, .2))
    path = tmp_path/'b4.pt'
    original.save_checkpoint(path)
    loaded = make_cost_agent()
    loaded.load_checkpoint(path)
    return original, loaded


@pytest.mark.parametrize('part', ['cost_q', 'cost_target', 'cost_optimizer', 'multiplier',
                                 'cost_rng', 'diagnostics', 'ordinary'],
                         ids=['C01_online', 'C01_target', 'C02_adam', 'C03_C04_C05_lambda',
                              'C_rng', 'C_diagnostics', 'C_B3_all'])
def test_b4_checkpoint_parts(restored, part):
    left, right = restored
    assert_tree(left.state_dict()[part], right.state_dict()[part])


def test_b4_c06_next_lambda_and_optimizer_update(restored):
    left, right = restored
    for agent in (left, right):
        assert agent.multiplier.completed_episodes_since_last_lambda_update == 9
        agent.multiplier.submit(complete_fixture(19, .8))
        assert agent.multiplier.lambda_update_count == 2
    assert left.update() == right.update()
    assert_tree(left.state_dict(), right.state_dict())


def test_b4_c07_migration_marked():
    four = make_cost_agent()
    three = OrdinarySACAgent(four.config, source_fingerprint='b3-old-source')
    for i in range(4):
        three.store_transition(**transition(i))
    three.update()
    cost_before = deepcopy(four.cost_q.state_dict())
    marker = four.migrate_from_b3(three.state_dict(), expected_source='b3-old-source')
    assert marker['status'] == 'MIGRATED_FROM_B3' and not marker['exact_B4_resume']
    assert four.cost_update_count == 0 and four.multiplier.value == 1
    assert_tree(cost_before, four.cost_q.state_dict())
    expected = three.state_dict()
    expected['source_fingerprint'] = 'b4-test'
    assert_tree(expected, four.ordinary.state_dict())


def test_b4_migration_wrong_source():
    four = make_cost_agent()
    with pytest.raises(ValueError):
        four.migrate_from_b3(four.ordinary.state_dict(), expected_source='incorrect')


def test_b4_real_runner_truncated_ledger(project_config):
    runner = CostLearningRunner(make_env(project_config), make_cost_agent(batch_size=2),
                                reset_options={'external_max_steps': 2})
    runner.step()
    result = runner.step()
    assert result['episode_cost']['actual_task_steps'] == 2
    assert result['episode_cost']['G_C'] is None
    assert result['lambda_update_count'] == 0 and result['pending_complete_episodes'] == 0
    assert len(runner.agent.replay) == 2


def test_b4_runner_mid_episode_checkpoint(project_config, tmp_path):
    left = CostLearningRunner(make_env(project_config),
                              make_cost_agent(batch_size=2, learning_starts=2))
    left.step()
    left.step()
    path = tmp_path/'runner.pt'
    left.save_checkpoint(path)
    right = CostLearningRunner(make_env(project_config),
                               make_cost_agent(batch_size=2, learning_starts=2))
    right.load_checkpoint(path)
    assert_tree(vars(left.ledger), vars(right.ledger))
    assert_tree(left.step(), right.step())
    assert_tree(left.agent.state_dict(), right.agent.state_dict())


def test_b4_nonfinite_fail_metadata():
    agent = make_cost_agent()
    values = batch()
    values['next_obs'][0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        agent.update(values)
    assert agent.diagnostics['nonfinite_count'] == 1
    assert agent.last_failure_metadata['task_steps'] == [1, 2, 3, 4]
    assert agent.counters['gradient_updates'] == 0


class OneStepEpisodeFixture:
    """单步终止接口替身；仅验证账本编排，非物理模拟或科学结果。"""

    def __init__(self, config, terminal_type):
        self.config = config
        self.terminal_type = terminal_type

    def reset(self, *, seed, options):
        return np.zeros(234, np.float32), {}

    def step(self, action):
        return np.zeros(234, np.float32), 0., True, False, dict(
            cost=1., executed_action_normalized=action.copy(),
            failure_type=self.terminal_type, task_control_step=1)


@pytest.mark.parametrize('kind', ['goal_success', 'collision', 'operational_boundary_failure'])
def test_complete_runner_ten_episodes(project_config, kind):
    runner = CostLearningRunner(OneStepEpisodeFixture(project_config, kind),
                                make_cost_agent(learning_starts=10000))
    for _ in range(10):
        result = runner.step()
    assert len(runner.agent.replay) == 10  # 失败尾项绝不生成额外Replay。
    assert result['lambda_update_count'] == 1
    expected_cost = BETA if kind == 'goal_success' else 1.
    assert result['lambda_value'] == pytest.approx(1+expected_cost-.05, abs=3e-14)
    assert runner.agent.counters['gradient_updates'] == 0


def test_cost_diagnostics_above_one():
    agent = make_cost_agent()
    with torch.no_grad():
        for p in agent.cost_target_q.parameters():
            p.zero_()
        agent.cost_target_q.net[-2].bias.fill_(float(np.log(.9999/.0001)))
    before = deepcopy(agent.cost_target_q.state_dict())
    result = agent.update(batch())
    assert result['cost_target_min'] > 1 and result['target_above_1_count'] == 4
    assert result['target_below_0_count'] == 0 and result['nonfinite_count'] == 0
    for key, new in agent.cost_target_q.state_dict().items():
        expected = .995*before[key]+.005*agent.cost_q.state_dict()[key]
        torch.testing.assert_close(new, expected, rtol=1e-7, atol=1e-8)


def test_abandoned_episode_excluded(project_config):
    runner = CostLearningRunner(make_env(project_config), make_cost_agent())
    runner.step()
    runner.needs_reset = True  # 显式提前reset，旧片段不得成为完整episode。
    runner.step()
    assert runner.agent.multiplier.incomplete_count == 1
    assert runner.agent.multiplier.complete_count == 0
    assert runner.last_episode['G_C'] is None
