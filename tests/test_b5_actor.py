"""LOCAL52公式、符号、动作梯度与λ=0精确退化；不作性能实验。"""

import inspect
from copy import deepcopy

import pytest
import torch
from torch import nn

from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.rl.constrained_agent import ConstrainedSACAgent, constrained_actor_loss
from auv_risk_rl.rl.losses import actor_loss
from test_b3_integration import assert_tree, batch


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_agent():
    return ConstrainedSACAgent(SACConfig(batch_size=4, replay_capacity=16, learning_starts=4),
                               source_fingerprint='b5-test')


def test_sign_01_loss_formula():
    logpi = torch.tensor([[-2.], [-1.]], dtype=torch.float64)
    q1, q2 = logpi+5, logpi+6
    cost = torch.tensor([[.2], [.6]], dtype=torch.float64)
    alpha = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    result = constrained_actor_loss(logpi, q1, q2, cost, alpha, 3.)
    expected = sum(.2*lp-min(a, b)+3*c for lp, a, b, c in
                   [(-2., 3., 4., .2), (-1., 4., 5., .6)])/2
    assert result.item() == pytest.approx(expected, rel=1e-9, abs=1e-10)
    assert result.item() == pytest.approx(actor_loss(logpi, q1, q2, alpha).item()+1.2)


def test_sign_02_loss_gradient():
    theta = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    zero = torch.zeros((), dtype=torch.float64)
    loss = constrained_actor_loss(zero, zero, zero, .5*theta.square(), zero, 3.)
    loss.backward()
    assert theta.grad.item() == 6.


def test_sign_03_descent_direction():
    theta = nn.Parameter(torch.tensor(2., dtype=torch.float64))
    optimizer = torch.optim.SGD([theta], lr=.1)
    zero = torch.zeros((), dtype=torch.float64)
    constrained_actor_loss(zero, zero, zero, .5*theta.square(), zero, 3.).backward()
    assert theta.grad.item() == 6.
    optimizer.step()
    assert theta.item() == pytest.approx(1.4, rel=1e-9, abs=1e-10)


@pytest.mark.parametrize('multiplier', [0., 1., 10., 1000.])
def test_lambda_analytic_gradient_and_finite_difference(multiplier):
    theta = torch.tensor(.4, dtype=torch.float64, requires_grad=True)

    def objective(x):
        # fresh nominal reparameterized fixture a=theta; known entropy/reward/cost derivatives.
        return constrained_actor_loss(x.square(), 2*x, 3*x, .5*x.square(),
                                      torch.tensor(.2, dtype=torch.float64), multiplier)

    loss = objective(theta)
    gradient, = torch.autograd.grad(loss, theta)
    expected = .4*.4-2+multiplier*.4
    assert gradient.item() == pytest.approx(expected, rel=1e-9, abs=1e-10)
    epsilon = 1e-5
    fd = (objective(theta.detach()+epsilon)-objective(theta.detach()-epsilon))/(2*epsilon)
    assert gradient.item() == pytest.approx(fd.item(), rel=1e-6, abs=1e-7)
    assert torch.isfinite(loss) and torch.isfinite(gradient)


@pytest.mark.parametrize('left,right', [(0., 1.), (1., 10.), (0., 1000.)])
def test_fixed_sample_lambda_difference(left, right):
    x = torch.tensor([[.2], [.7]], dtype=torch.float64)
    args = (x, x+1, x+2, x.square(), torch.tensor(.2))
    delta = constrained_actor_loss(*args, right)-constrained_actor_loss(*args, left)
    assert delta.item() == pytest.approx((right-left)*x.square().mean().item(),
                                       rel=1e-9, abs=1e-10)


def test_constant_cost_offset_no_gradient():
    x = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    zero = torch.zeros_like(x)
    args = (x.square(), x, 2*x, zero+.7, zero+.2)
    a, b = constrained_actor_loss(*args, 0.), constrained_actor_loss(*args, 10.)
    ga, = torch.autograd.grad(a, x, retain_graph=True)
    gb, = torch.autograd.grad(b, x)
    assert torch.equal(ga, gb)
    assert (b-a).item() == pytest.approx(7.)


@pytest.mark.parametrize('forbidden', ['beta', 'remaining', 'risk', 'executed', 'task_step'])
def test_actor_objective_no_other_inputs(forbidden):
    assert forbidden not in inspect.getsource(constrained_actor_loss)
    assert list(inspect.signature(constrained_actor_loss).parameters) == [
        'log_pi', 'q1', 'q2', 'cost_q', 'alpha', 'multiplier']


def test_same_fresh_action_online_critics_and_frozen_params():
    agent = make_agent()
    obs = torch.from_numpy(batch()['obs'])
    generator = torch.Generator().set_state(agent.generator.get_state())
    expected_action, expected_log = agent.actor.sample(obs, generator)
    before = [deepcopy(m.state_dict()) for m in (agent.q1, agent.q2, agent.cost_q)]
    seen, handles = [], []
    for model in (agent.q1, agent.q2, agent.cost_q):
        handles.append(model.register_forward_pre_hook(
            lambda m, args: seen.append((args[1], [p.requires_grad for p in m.parameters()]))))
    logpi, value, norm = agent.update_actor(obs)
    for h in handles:
        h.remove()
    assert len(seen) == 3 and seen[0][0] is seen[1][0] is seen[2][0]
    assert torch.equal(seen[0][0], expected_action) and torch.equal(logpi, expected_log.detach())
    assert seen[0][0].requires_grad and all(not any(flags) for _, flags in seen)
    for model, state in zip((agent.q1, agent.q2, agent.cost_q), before, strict=True):
        assert_tree(model.state_dict(), state)
        assert all(p.grad is None and p.requires_grad for p in model.parameters())
    assert torch.equal(agent.generator.get_state(), generator.get_state())
    assert norm > 0 and torch.isfinite(torch.tensor(value))
    terms = agent.actor_components
    assert terms['actor_cost_q_term'] == terms['lambda_value']*terms['mean_cost_q']
    assert value == pytest.approx(sum(terms[k] for k in (
        'actor_entropy_term', 'actor_reward_q_term', 'actor_cost_q_term')), rel=1e-6, abs=1e-7)


def test_lambda_zero_exact_nonempty_adam_degeneration():
    constrained = make_agent()
    ordinary = OrdinarySACAgent(constrained.config, source_fingerprint='b5-test')
    ordinary.update(batch())  # Adam moments nonempty, not initialization-only equality.
    constrained.ordinary.load_state_dict(ordinary.state_dict())
    constrained.multiplier.value = 0.
    obs = torch.from_numpy(batch()['obs'])
    a = ordinary.update_actor(obs)
    b = constrained.update_actor(obs)
    assert_tree(a, b)
    for x, y in zip(ordinary.actor.parameters(), constrained.actor.parameters(), strict=True):
        assert torch.equal(x.grad, y.grad)
    assert_tree(ordinary.state_dict(), constrained.ordinary.state_dict())
    assert constrained.multiplier.value == 0. and constrained.multiplier.lambda_update_count == 0


class AnalyticQ(nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(slope))

    def forward(self, obs, action):
        return self.slope*action[:, :1]


def test_actual_actor_cost_and_reward_gradient_reference():
    agent = make_agent()
    agent.ordinary.q1 = AnalyticQ(2.)
    agent.ordinary.q2 = AnalyticQ(2.)
    agent.cost_q = AnalyticQ(.7)
    agent.multiplier.value = 3.
    reference = deepcopy(agent.actor)
    rng = torch.Generator().set_state(agent.generator.get_state())
    obs = torch.zeros(4, 234)
    action, logpi = reference.sample(obs, rng)
    independent = (agent.alpha.detach()*logpi-2*action[:, :1]+2.1*action[:, :1]).mean()
    independent.backward()
    agent.update_actor(obs)
    for actual, expected in zip(agent.actor.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)
    assert all(m.slope.grad is None for m in (agent.q1, agent.q2, agent.cost_q))


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1.])
def test_lambda_fail_fast_restores_flags(invalid):
    agent = make_agent()
    agent.multiplier.value = invalid
    before = deepcopy(agent.actor.state_dict())
    with pytest.raises(FloatingPointError):
        agent.update(batch())
    assert_tree(before, agent.actor.state_dict())
    assert agent.last_actor_failure_metadata['episode_id'] == [0, 0, 0, 0]
    assert all(p.requires_grad for p in agent.cost_q.parameters())


def test_autograd_lambda_rejected():
    zero = torch.zeros(1)
    multiplier = torch.tensor(1., requires_grad=True)
    with pytest.raises(TypeError):
        constrained_actor_loss(zero, zero, zero, zero, zero, multiplier)
    assert multiplier.grad is None


@pytest.mark.parametrize('field', ['q1', 'q2', 'cost_q', 'log_pi', 'gradient', 'parameter'])
def test_nonfinite_actor_fail_fast(field, monkeypatch):
    agent = make_agent()
    agent.ordinary._batch_metadata = dict(episode_id=[7], task_step=[3])
    if field in ('q1', 'q2', 'cost_q'):
        getattr(agent, field).register_forward_hook(
            lambda module, args, output: output*float('nan'))
    elif field == 'log_pi':
        sample = agent.actor.sample

        def bad_sample(*args):
            action, logpi = sample(*args)
            return action, logpi*float('nan')

        monkeypatch.setattr(agent.actor, 'sample', bad_sample)
    elif field == 'gradient':
        next(agent.actor.parameters()).register_hook(lambda grad: grad*float('nan'))
    else:
        step = agent.optimizers['actor'].step

        def bad_step():
            step()
            with torch.no_grad():
                next(agent.actor.parameters()).fill_(float('inf'))

        monkeypatch.setattr(agent.optimizers['actor'], 'step', bad_step)
    with pytest.raises(FloatingPointError):
        agent.update_actor(torch.zeros(1, 234))
    assert agent.last_actor_failure_metadata['episode_id'] == [7]
    assert agent.last_actor_failure_metadata['task_step'] == [3]
    for model in (agent.q1, agent.q2, agent.cost_q):
        assert all(p.requires_grad for p in model.parameters())
