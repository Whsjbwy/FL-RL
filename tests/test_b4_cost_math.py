"""LOCAL成本独立高精度参考与近似边界，禁止把目标裁剪成合法范围。"""

import inspect
from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch
from torch import nn

from auv_risk_rl.rl.cost_math import (
    BETA,
    GAMMA,
    HORIZON,
    CostCritic,
    cost_loss,
    cost_target,
    local_cost_tail,
    validate_cost_transition,
)
from auv_risk_rl.rl.episode_cost import EpisodeCostLedger
from auv_risk_rl.rl.replay import FAILURE_TYPES
from test_b3_integration import batch


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reference_sum(length):
    with localcontext() as context:
        context.prec = 60
        gamma = Decimal('0.999')
        weights = [gamma**i for i in range(1000)]
        return float(sum(weights[:length])/sum(weights))


def test_cost_m01_beta_high_precision():
    assert BETA == pytest.approx(reference_sum(1), rel=2e-14)


@pytest.mark.parametrize('cost', [0., 1.], ids=['M02_all_zero', 'M03_all_one'])
def test_cost_complete_horizon(cost):
    ledger = EpisodeCostLedger(7)
    for step in range(1, 1001):
        report = ledger.append(cost, step, step == 1000, False,
                               'task_horizon' if step == 1000 else 'none')
    assert report['G_C'] == pytest.approx(cost, abs=2e-14)
    assert report['actual_task_steps'] == 1000 and report['eligible_for_lambda']


@pytest.mark.parametrize('kind', ['goal_success', 'collision', 'operational_boundary_failure'],
                         ids=['M04_success', 'M05_collision', 'M05_boundary'])
def test_cost_early_tail(kind):
    ledger = EpisodeCostLedger(1)
    ledger.append(0., 1, False, False, 'none')
    report = ledger.append(1., 2, True, False, kind)
    expected = BETA*GAMMA if kind == 'goal_success' else 1-BETA
    assert report['G_C'] == pytest.approx(expected, abs=3e-14)
    assert len(report['instant_costs']) == 2


def test_cost_m06_zero_tail():
    assert local_cost_tail(0) == 0


class StubActor:
    def __init__(self):
        self.rows = 0

    def sample(self, obs, generator):
        assert torch.isfinite(obs).all()
        self.rows += len(obs)
        return torch.zeros(len(obs), 3), torch.full((len(obs), 1), float('nan'))


class StubQ:
    def __init__(self, value):
        self.value = value

    def __call__(self, obs, nominal):
        return torch.full((len(obs), 1), self.value)


@pytest.mark.parametrize('kind,step,terminal,truncated', [
    ('task_horizon', 1000, True, False), ('collision', 1, True, False),
    ('operational_boundary_failure', 998, True, False), ('goal_success', 50, True, False),
    ('none', 1, False, False), ('external_truncation', 32, False, True)],
    ids=['M07_M11_last_step', 'M08_failure_gamma', 'M08_boundary_gamma', 'M12_terminal_nan',
         'M09_rejection_continues', 'M10_external_bootstrap'])
def test_cost_target_exact(kind, step, terminal, truncated):
    values = batch()
    values['cost'][:] = 1
    values['task_step'][:] = step
    values['terminated'][:] = terminal
    values['truncated'][:] = truncated
    values['failure_type'][:] = FAILURE_TYPES.index(kind)
    if terminal:
        values['next_obs'][:] = np.nan
    actor = StubActor()
    actual = cost_target(values, actor, StubQ(.4), torch.Generator(), torch.device('cpu'))
    expected = BETA
    if kind in ('collision', 'operational_boundary_failure'):
        expected += GAMMA*reference_sum(HORIZON-step)
    elif not terminal:
        expected += GAMMA*.4
    torch.testing.assert_close(actual, torch.full((4, 1), expected), rtol=1e-6, atol=1e-8)
    assert actor.rows == (0 if terminal else 4)
    assert not actual.requires_grad  # Q08/Q09: NaN entropy unused; target detached.


@pytest.mark.parametrize('case', ['failure_cost_zero', 'failure_nonterminal', 'early_horizon'])
def test_failure_schema_rejects_invalid(case):
    args = dict(cost=1., task_step=1, terminated=True, truncated=False, failure_type='collision')
    if case == 'failure_cost_zero':
        args['cost'] = 0.
    elif case == 'failure_nonterminal':
        args['terminated'] = False
    else:
        args['failure_type'] = 'task_horizon'
    with pytest.raises(ValueError):
        validate_cost_transition(**args)


def test_cost_q01_q02_architecture():
    model = CostCritic()
    assert [m.in_features for m in model.net if isinstance(m, nn.Linear)] == [237, 256, 256]
    assert [m.out_features for m in model.net if isinstance(m, nn.Linear)] == [256, 256, 1]
    assert isinstance(model.net[-1], nn.Sigmoid)
    assert list(inspect.signature(model.forward).parameters) == ['obs', 'nominal_action']


def test_cost_q03_executed_not_input():
    model = CostCritic()
    values = batch()
    obs, nominal = torch.from_numpy(values['obs']), torch.from_numpy(values['nominal_action'])
    before = model(obs, nominal)
    values['executed_action'][:] = -.99
    assert torch.equal(before, model(obs, nominal))


def test_cost_q04_sigmoid():
    result = CostCritic()(torch.zeros(20, 234), torch.zeros(20, 3))
    assert result.shape == (20, 1)
    assert torch.isfinite(result).all() and torch.all((result > 0) & (result < 1))


def test_cost_q10_half_mse():
    prediction = torch.tensor([[1.], [5.]], requires_grad=True)
    target = torch.tensor([[3.], [2.]], requires_grad=True)
    loss = cost_loss(prediction, target)
    assert loss.item() == 3.25
    loss.backward()
    assert target.grad is None
    torch.testing.assert_close(prediction.grad, torch.tensor([[-1.], [1.5]]))


def test_cost_b01_b04_approximation_above_one():
    values = batch()
    target_model = CostCritic()
    with torch.no_grad():
        for p in target_model.parameters():
            p.zero_()
        target_model.net[-2].bias.fill_(float(np.log(.9999/.0001)))
    prediction = target_model(torch.zeros(4, 234), torch.zeros(4, 3))
    assert torch.all(prediction < 1)
    target = cost_target(values, StubActor(), target_model, torch.Generator(), torch.device('cpu'))
    assert torch.all(target > 1)
    torch.testing.assert_close(target, BETA+GAMMA*prediction.detach(), rtol=0, atol=0)
    assert torch.isfinite(cost_loss(prediction, target))
    assert target[0].item() == pytest.approx(1.0004816163, abs=2e-7)


@pytest.mark.parametrize('remaining', [0, 1, 2, 15, 500, 999, 1000])
def test_cost_b05_b06_analytical_bound(remaining):
    bound = local_cost_tail(remaining)
    assert bound == pytest.approx(reference_sum(remaining), abs=3e-14)
    assert 0 <= bound <= 1+2e-14
    if remaining:
        assert BETA+GAMMA*local_cost_tail(remaining-1) == pytest.approx(bound, abs=3e-14)


def test_ledger_no_duplicate_finish():
    ledger = EpisodeCostLedger(0)
    ledger.append(0, 1, True, False, 'goal_success')
    with pytest.raises(ValueError):
        ledger.finish('goal_success', complete=True)
