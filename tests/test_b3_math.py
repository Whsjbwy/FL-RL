"""B3独立数学与梯度参考测试；不运行科学训练。"""

import inspect
import math
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.rl.losses import (
    actor_loss,
    critic_loss,
    polyak_update,
    reward_target,
    temperature_loss,
)
from auv_risk_rl.rl.networks import Actor, RewardCritic, squashed_log_prob


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def agent():
    return OrdinarySACAgent(SACConfig(batch_size=4, replay_capacity=16),
                            source_fingerprint='synthetic-test')


def test_sac_a01_architecture(agent):
    mean, scale = agent.actor(torch.zeros(7, 234))
    assert mean.shape == scale.shape == (7, 3)
    assert [m.out_features for m in agent.actor.net if isinstance(m, nn.Linear)] == [256, 256, 6]
    assert sum(p.numel() for p in agent.actor.parameters()) == 127494


def test_sac_a02_bounded_samples(agent):
    action, log_pi = agent.actor.sample(torch.zeros(256, 234), agent.generator)
    assert action.shape == (256, 3) and log_pi.shape == (256, 1)
    assert torch.all(action.abs() < 1)


def test_sac_a03_deterministic_no_rng(agent):
    rng = agent.generator.get_state().clone()
    obs = np.zeros(234, np.float32)
    np.testing.assert_array_equal(agent.deterministic_action(obs),
                                  agent.actor(torch.from_numpy(obs))[0].tanh().detach().numpy())
    assert torch.equal(rng, agent.generator.get_state())


def test_sac_a04_reparameterization(agent):
    action, _ = agent.actor.sample(torch.ones(4, 234), agent.generator)
    action.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in agent.actor.parameters())
    assert sum(p.grad.abs().sum() for p in agent.actor.parameters()) > 0


def test_sac_a05_independent_float64_reference():
    rng = np.random.default_rng(581)
    mean = rng.normal(size=(128, 3))
    log_std = rng.uniform(-1, .5, size=(128, 3))
    z = rng.uniform(-3, 3, size=(128, 3))
    reference = (-.5 * ((z-mean)/np.exp(log_std))**2 - log_std
                 - .5*np.log(2*np.pi) - np.log(1-np.tanh(z)**2)).sum(1, keepdims=True)
    actual = squashed_log_prob(*(torch.from_numpy(x) for x in (z, mean, log_std)))
    np.testing.assert_allclose(actual.numpy(), reference, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize('magnitude', [20., 100., 1000.])
def test_sac_a06_large_z(magnitude):
    z = torch.tensor([[magnitude, -magnitude, 0.]])
    result = squashed_log_prob(z, torch.zeros_like(z), torch.zeros_like(z))
    assert torch.isfinite(result).all()


def test_sac_a07_actor_interface():
    assert list(inspect.signature(Actor.forward).parameters) == ['self', 'obs']
    assert list(inspect.signature(Actor.sample).parameters) == ['self', 'obs', 'generator']


def test_sac_a08_initialization_isolated(agent):
    before = torch.random.get_rng_state().clone()
    other = OrdinarySACAgent(agent.config, source_fingerprint='synthetic-test')
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(
        agent.actor.parameters(), other.actor.parameters(), strict=True))


def test_sac_q01_independent(agent):
    assert not {p.data_ptr() for p in agent.q1.parameters()} & {
        p.data_ptr() for p in agent.q2.parameters()}
    assert any(not torch.equal(a, b) for a, b in zip(
        agent.q1.parameters(), agent.q2.parameters(), strict=True))


def test_sac_q02_architecture(agent):
    assert agent.q1.net[0].in_features == 237
    assert [m.out_features for m in agent.q1.net if isinstance(m, nn.Linear)] == [256, 256, 1]
    assert sum(p.numel() for p in agent.q1.parameters()) == 126977


def test_sac_q03_nominal_interface():
    assert list(inspect.signature(RewardCritic.forward).parameters) == [
        'self', 'obs', 'nominal_action']


def test_sac_q04_executed_invariance(agent):
    batch = dict(obs=torch.zeros(4, 234), nominal=torch.ones(4, 3)*.2, executed=torch.zeros(4, 3))
    seen = []
    hook = agent.q1.net[0].register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
    first = agent.q1(batch['obs'], batch['nominal'])
    batch['executed'].fill_(-.9)
    second = agent.q1(batch['obs'], batch['nominal'])
    hook.remove()
    assert torch.equal(first, second) and torch.equal(seen[0], seen[1])
    assert torch.equal(seen[0][:, 234:], batch['nominal'])


def test_sac_q05_target_hard_copy(agent):
    for online, target in ((agent.q1, agent.target_q1), (agent.q2, agent.target_q2)):
        assert all(torch.equal(a, b) and a.data_ptr() != b.data_ptr()
                   for a, b in zip(online.parameters(), target.parameters(), strict=True))


def test_sac_q06_no_target_optimizer(agent):
    params = {id(p) for opt in agent.optimizers.values() for group in opt.param_groups
              for p in group['params']}
    for p in (*agent.target_q1.parameters(), *agent.target_q2.parameters()):
        assert not p.requires_grad and id(p) not in params


class FixedActor:
    def __init__(self):
        self.rows = 0

    def sample(self, obs, generator):
        assert torch.isfinite(obs).all()
        self.rows += len(obs)
        return torch.zeros(len(obs), 3), torch.full((len(obs), 1), -2.)


class FixedQ:
    def __init__(self, value):
        self.value = value

    def __call__(self, obs, action):
        assert torch.isfinite(obs).all()
        return torch.full((len(obs), 1), self.value)


@pytest.mark.parametrize('kind,terminal', [
    ('T01_nonterminal', False), ('T02_goal', True), ('T03_collision', True),
    ('T04_boundary', True), ('T05_horizon', True), ('T06_external_truncation', False),
    ('T07_terminal_nan', True), ('T08_minimum', False), ('T09_entropy_sign', False)])
def test_sac_target_reference(kind, terminal):
    actor = FixedActor()
    obs = torch.full((3, 234), float('nan') if terminal else 0.)
    rewards = torch.tensor([1., -2., 3.])
    actual = reward_target(rewards, obs, torch.full((3,), terminal), actor,
                            FixedQ(4.), FixedQ(7.), torch.tensor(.2), .999, torch.Generator())
    expected = rewards if terminal else rewards + .999*(4 + .4)
    torch.testing.assert_close(actual[:, 0], expected)
    assert actor.rows == (0 if terminal else 3)


def test_sac_target_mixed_rows():
    actor = FixedActor()
    obs = torch.zeros(3, 234)
    obs[1] = float('nan')
    value = reward_target(torch.ones(3), obs, torch.tensor([False, True, False]), actor,
                           FixedQ(4.), FixedQ(7.), torch.tensor(.2), .999, torch.Generator())
    assert actor.rows == 2 and value[1].item() == 1
    assert torch.isfinite(value).all()


@pytest.mark.parametrize('critic_name', ['L01_q1', 'L02_q2'])
def test_sac_l01_l02_exact_mse(critic_name):
    q = torch.tensor([[1.], [5.]], requires_grad=True)
    target = torch.tensor([[3.], [2.]], requires_grad=True)
    loss = critic_loss(q, target)
    assert loss.item() == 6.5
    loss.backward()
    assert target.grad is None
    torch.testing.assert_close(q.grad, torch.tensor([[-2.], [3.]]))


def test_sac_l03_ordinary_actor_formula():
    alpha = torch.tensor(.2, requires_grad=True)
    loss = actor_loss(torch.tensor([[-2.], [-1.]]), torch.tensor([[3.], [4.]]),
                      torch.tensor([[2.], [6.]]), alpha)
    assert loss.item() == pytest.approx(-3.3)
    assert not loss.requires_grad


def test_sac_l04_l05_actor_only_changes(agent):
    old_actor = deepcopy(agent.actor.state_dict())
    old_q1, old_q2 = deepcopy(agent.q1.state_dict()), deepcopy(agent.q2.state_dict())
    _, loss, norm = agent.update_actor(torch.ones(4, 234))
    assert math.isfinite(loss) and 0 < norm < float('inf')
    assert any(not torch.equal(v, agent.actor.state_dict()[k]) for k, v in old_actor.items())
    assert all(torch.equal(v, agent.q1.state_dict()[k]) for k, v in old_q1.items())
    assert all(torch.equal(v, agent.q2.state_dict()[k]) for k, v in old_q2.items())


def test_sac_l06_dq_da(agent):
    obs = torch.ones(4, 234)
    action, log_pi = agent.actor.sample(obs, agent.generator)
    for q in (agent.q1, agent.q2):
        q.requires_grad_(False)
    loss = actor_loss(log_pi, agent.q1(obs, action), agent.q2(obs, action), torch.tensor(0.))
    gradients = torch.autograd.grad(loss, list(agent.actor.parameters()))
    assert sum(g.abs().sum() for g in gradients) > 0


def test_sac_l07_target_detached(agent):
    result = reward_target(torch.ones(4), torch.ones(4, 234), torch.zeros(4), agent.actor,
                            agent.target_q1, agent.target_q2, agent.alpha, .999, agent.generator)
    assert not result.requires_grad
    assert all(p.grad is None for p in (*agent.actor.parameters(), *agent.target_q1.parameters()))


def test_sac_alpha01_initial(agent):
    assert agent.alpha.item() == pytest.approx(.2)


def test_sac_alpha02_target(agent):
    assert agent.config.target_entropy == -3


def test_sac_alpha03_positive():
    for initial in (-20., 0., 10.):
        assert torch.tensor(initial).exp().item() > 0


def test_sac_alpha04_logpi_detached():
    log_pi = torch.ones(4, requires_grad=True)
    log_alpha = torch.tensor(math.log(.2), requires_grad=True)
    temperature_loss(log_alpha, log_pi).backward()
    assert log_pi.grad is None and log_alpha.grad is not None


@pytest.mark.parametrize('log_pi,increase', [(5., True), (1., False)],
                         ids=['ALPHA05_low_entropy', 'ALPHA06_high_entropy'])
def test_sac_alpha_direction(log_pi, increase):
    log_alpha = nn.Parameter(torch.tensor(math.log(.2)))
    optimizer = torch.optim.Adam([log_alpha], lr=3e-4)
    before = log_alpha.exp().item()
    temperature_loss(log_alpha, torch.tensor([log_pi])).backward()
    optimizer.step()
    assert (log_alpha.exp().item() > before) == increase


def test_sac_alpha07_exact_original():
    value = torch.tensor(math.log(.2), dtype=torch.float64, requires_grad=True)
    loss = temperature_loss(value, torch.tensor([-2., 1.], dtype=torch.float64))
    assert loss.item() == pytest.approx(.7)
    loss.backward()
    assert value.grad.item() == pytest.approx(.7)  # 代理式梯度为3.5，必须区分。


def test_sac_p01_default():
    assert SACConfig().tau == .005


def test_sac_p02_direction():
    target, online = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        target.weight.zero_()
        online.weight.fill_(1)
    polyak_update(target, online)
    assert target.weight.item() == pytest.approx(.005)


def test_sac_p03_independent_targets(agent):
    with torch.no_grad():
        for p in agent.q1.parameters():
            p.fill_(1)
        for p in agent.q2.parameters():
            p.fill_(2)
        for p in (*agent.target_q1.parameters(), *agent.target_q2.parameters()):
            p.zero_()
    polyak_update(agent.target_q1, agent.q1)
    polyak_update(agent.target_q2, agent.q2)
    assert all(torch.allclose(p, torch.full_like(p, .005)) for p in agent.target_q1.parameters())
    assert all(torch.allclose(p, torch.full_like(p, .01)) for p in agent.target_q2.parameters())
