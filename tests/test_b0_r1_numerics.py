"""R1独立标量/梯度反例核验；不训练导航策略，不改变冻结数学或旧容差。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.b0_navigation import B0ObservationBuilder
from auv_risk_rl.env.local_task import LocalTaskConfig, normalized_to_command, task_reward
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.exceptions import InvalidEnvironmentStateError
from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.rl.losses import (
    actor_loss,
    critic_loss,
    polyak_update,
    reward_target,
    temperature_loss,
)
from auv_risk_rl.rl.networks import Actor, squashed_log_prob
from auv_risk_rl.sensors.rays import RayFrame
from auv_risk_rl.types import AUVState, ControlCommand


def _scalar_log_probability(z: float, mean: float, log_std: float) -> float:
    """标量高斯密度与tanh导数直接计算，不调用生产log-prob函数。"""
    sigma = math.exp(log_std)
    gaussian = -0.5 * ((z - mean) / sigma) ** 2 - log_std
    gaussian -= 0.5 * math.log(2.0 * math.pi)
    return gaussian - math.log(1.0 - math.tanh(z) ** 2)


@pytest.mark.parametrize('z', [-1.3, -0.2, 0.7, 1.6])
def test_r1_eq48_value_and_independent_derivative(z: float) -> None:
    """float64解析atol1e-10；中心差分h1e-3、atol1e-3为运行前登记。"""
    mean, log_std, step = 0.3, -0.4, 1.0e-3
    tensor = torch.tensor([[z]], dtype=torch.float64, requires_grad=True)
    actual = squashed_log_prob(tensor, torch.tensor([[mean]], dtype=torch.float64),
                               torch.tensor([[log_std]], dtype=torch.float64))
    expected = _scalar_log_probability(z, mean, log_std)
    assert actual.item() == pytest.approx(expected, rel=0, abs=1.0e-10)
    derivative = torch.autograd.grad(actual, tensor)[0].item()
    analytic = -(z - mean) / math.exp(2.0 * log_std) + 2.0 * math.tanh(z)
    finite_difference = (_scalar_log_probability(z + step, mean, log_std)
                         - _scalar_log_probability(z - step, mean, log_std)) / (2.0 * step)
    assert derivative == pytest.approx(analytic, rel=0, abs=1.0e-10)
    assert derivative == pytest.approx(finite_difference, rel=0, abs=1.0e-3)


def test_r1_reparameterized_sample_analytic_gradient() -> None:
    """受控网络末层给定均值/尺度，以独立epsilon核验da/dmean和da/dlogstd。"""
    actor = Actor()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.net[-1].bias.copy_(torch.tensor([0.2, -0.3, 0.4, -0.4, 0.1, -0.2]))
    generator = torch.Generator().manual_seed(918410)
    reference_generator = torch.Generator().set_state(generator.get_state().clone())
    epsilon = torch.randn((1, 3), generator=reference_generator)
    mean = np.array([0.2, -0.3, 0.4])
    sigma = np.exp(np.array([-0.4, 0.1, -0.2]))
    z = mean + sigma * epsilon.numpy()[0]
    expected_action = np.tanh(z)
    action, _ = actor.sample(torch.zeros(1, 234), generator)
    action.sum().backward()
    derivative_mean = 1.0 - expected_action ** 2
    derivative_log_std = derivative_mean * sigma * epsilon.numpy()[0]
    np.testing.assert_allclose(action.detach().numpy()[0], expected_action,
                               rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(actor.net[-1].bias.grad.numpy(),
                               np.r_[derivative_mean, derivative_log_std],
                               rtol=1.0e-5, atol=1.0e-5)
    assert torch.equal(generator.get_state(), reference_generator.get_state())


class _LinearActionCritic(nn.Module):
    """可手算的动作线性Q，只用于证明生产Actor更新的梯度路径。"""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.4, -0.7, 0.9]))
        self.offset = offset

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """观察不影响夹具Q；两个Q仅有固定偏移，min分支无歧义。"""
        return (action * self.weight).sum(dim=-1, keepdim=True) + self.offset


def test_r1_production_actor_step_keeps_action_gradient_and_freezes_q() -> None:
    """仅一次合成Actor Adam步骤；解析梯度不是另一调用生产loss生成的参考。"""
    agent = OrdinarySACAgent(SACConfig(), source_fingerprint='r1-scalar-fixture')
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
    agent.q1, agent.q2 = _LinearActionCritic(0.0), _LinearActionCritic(10.0)
    weights_before = [critic.weight.detach().clone() for critic in (agent.q1, agent.q2)]
    reference_generator = torch.Generator().set_state(agent.generator.get_state().clone())
    epsilon = torch.randn((4, 3), generator=reference_generator).numpy()
    action = np.tanh(epsilon)
    weight, alpha = np.array([0.4, -0.7, 0.9]), 0.2
    expected_mean_gradient = (2.0 * alpha * action - weight * (1.0 - action ** 2)).mean(0)
    expected_log_std_gradient = (
        alpha * (-1.0 + 2.0 * action * epsilon)
        - weight * (1.0 - action ** 2) * epsilon).mean(0)
    _, value, norm = agent.update_actor(torch.zeros(4, 234))
    np.testing.assert_allclose(agent.actor.net[-1].bias.grad.numpy(),
                               np.r_[expected_mean_gradient, expected_log_std_gradient],
                               rtol=1.0e-5, atol=1.0e-5)
    assert math.isfinite(value) and math.isfinite(norm) and norm > 0
    for critic, before in zip((agent.q1, agent.q2), weights_before, strict=True):
        assert torch.equal(critic.weight, before)
        assert critic.weight.grad is None and critic.weight.requires_grad
    assert agent.log_alpha.grad is None
    assert agent.counters['actor'] == 1 and agent.counters['gradient_updates'] == 0
    assert agent.counters['q1'] == agent.counters['q2'] == agent.counters['alpha'] == 0


class _FixedTargetActor:
    """显式固定名义动作/logpi；断言terminal NaN观察不会传入续接。"""

    def __init__(self) -> None:
        self.rows = 0

    def sample(self, obs: torch.Tensor, generator: torch.Generator,
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """两行可续接夹具；日志密度逐行固定且不消耗随机流。"""
        assert torch.isfinite(obs).all()
        self.rows += len(obs)
        return torch.zeros(len(obs), 3, dtype=obs.dtype), obs[:, :1]


class _FixedTargetCritic:
    """目标Q仅取预先固定的观察标量，便于逐行手算。"""

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def __call__(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """固定第三个观察值就是夹具目标Q，不访问风险或成本。"""
        return obs[:, 2:3] + self.offset


def test_r1_eq49_terminal_mask_shape_and_external_bootstrap() -> None:
    """两种真终止不续接；外部截断行以terminated=False正常续接。"""
    observation = torch.zeros(4, 234, dtype=torch.float64)
    observation[:, 0] = torch.tensor([-0.5, float('nan'), 0.2, float('nan')], dtype=torch.float64)
    observation[:, 2] = torch.tensor([2.0, float('nan'), -1.0, float('nan')], dtype=torch.float64)
    reward = torch.tensor([0.3, 100.0, -0.4, -0.01], dtype=torch.float64)
    terminated = torch.tensor([False, True, False, True])
    actor = _FixedTargetActor()
    actual = reward_target(reward, observation, terminated, actor,
                           _FixedTargetCritic(0.0), _FixedTargetCritic(3.0),
                           torch.tensor(0.2, dtype=torch.float64), 0.999, torch.Generator())
    expected = [0.3 + 0.999 * (2.0 + 0.1), 100.0,
                -0.4 + 0.999 * (-1.0 - 0.04), -0.01]
    assert actual.shape == (4, 1) and actor.rows == 2 and not actual.requires_grad
    np.testing.assert_allclose(actual.numpy()[:, 0], expected, rtol=0, atol=1.0e-10)


def test_r1_critic_scalar_mse_gradient_and_detached_target() -> None:
    """两个样本手算MSE与2(q-target)/B，不出现B×B广播目标。"""
    q = torch.tensor([[2.0], [-1.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.5], [3.0]], dtype=torch.float64, requires_grad=True)
    value = critic_loss(q, target)
    value.backward()
    assert value.item() == pytest.approx((1.5 ** 2 + (-4.0) ** 2) / 2.0, abs=1.0e-10)
    np.testing.assert_allclose(q.grad.numpy()[:, 0], [1.5, -4.0], rtol=0, atol=1.0e-10)
    assert target.grad is None


@pytest.mark.parametrize('log_pi', [-2.0, 1.0, 3.0, 5.0])
def test_r1_native_eq53_exp_temperature_gradient(log_pi: float) -> None:
    """原生式53嵌套括号：-exp(ell)*(logpi-3)，绝不换log-alpha代理。"""
    ell = torch.tensor(math.log(0.2), dtype=torch.float64, requires_grad=True)
    log_density = torch.tensor([log_pi], dtype=torch.float64, requires_grad=True)
    value = temperature_loss(ell, log_density, -3.0)
    value.backward()
    expected = -0.2 * (log_pi - 3.0)
    assert value.item() == pytest.approx(expected, rel=0, abs=1.0e-10)
    assert ell.grad.item() == pytest.approx(expected, rel=0, abs=1.0e-10)
    assert log_density.grad is None
    step = 1.0e-3
    def reference(x: float) -> float:
        """式53独立标量参考，保持exp而不是log代理。"""
        return -math.exp(x) * (log_pi - 3.0)

    derivative = (reference(ell.item() + step) - reference(ell.item() - step)) / (2.0 * step)
    assert ell.grad.item() == pytest.approx(derivative, rel=0, abs=1.0e-3)
    # 纯标量梯度下降方向检查，不额外执行训练或优化器step。
    next_alpha = math.exp(ell.item() - 0.1 * ell.grad.item())
    assert (next_alpha > 0.2) == (log_pi > 3.0)


def test_r1_eq54_scalar_polyak_reference() -> None:
    """独立旧目标×.995+在线×.005逐参数参考。"""
    target, online = nn.Linear(2, 1).double(), nn.Linear(2, 1).double()
    with torch.no_grad():
        target.weight.copy_(torch.tensor([[2.0, -3.0]], dtype=torch.float64))
        target.bias.fill_(0.5)
        online.weight.copy_(torch.tensor([[-1.0, 5.0]], dtype=torch.float64))
        online.bias.fill_(-0.7)
    polyak_update(target, online, 0.005)
    np.testing.assert_allclose(target.weight.detach().numpy(), [[1.985, -2.96]],
                               rtol=0, atol=1.0e-10)
    assert target.bias.item() == pytest.approx(0.494, rel=0, abs=1.0e-10)


def test_r1_body_goal_and_physical_action_independent_reference(
    project_config: ProjectConfig,
) -> None:
    """yaw90/pitch30时独立分量；近目标特征小不等于编码错误。"""
    state = AUVState(np.array([20.0, 50.0, 20.0]), math.pi / 2, math.pi / 6,
                     0.9, 0.07, -0.05)
    goal = np.array([24.0, 56.0, 22.0])
    rays = RayFrame(np.full(45, 25.0), np.zeros(45))
    observation = B0ObservationBuilder(project_config).build(
        state, goal, np.array([-0.2, 0.3, -0.4]), 700, rays, (), 12.0)
    expected_body = np.array([6.0 * math.sqrt(3) / 2 - 1.0,
                              -4.0, 3.0 + math.sqrt(3)])
    np.testing.assert_allclose(observation[:3], expected_body / 100.0,
                               rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(observation[7:9], [0.2, -0.2],
                               rtol=1.0e-5, atol=1.0e-5)
    command = normalized_to_command(np.array([-0.4, 0.6, -0.8]), project_config.dynamics)
    assert command.surge_speed_command_mps == pytest.approx(0.66, rel=0, abs=1.0e-10)
    assert command.yaw_rate_command_rad_s == pytest.approx(0.21, rel=0, abs=1.0e-10)
    assert command.pitch_rate_command_rad_s == pytest.approx(-0.2, rel=0, abs=1.0e-10)
    assert observation.shape == (234,) and np.all(observation[108:] == 0)


def test_r1_goal_entry_before_control_end_and_no_extra_advance(
    project_config: ProjectConfig,
) -> None:
    """恒定稳态1.5m/s，距目标2.06m；t=.04s首次入2m球且立即停止。"""
    world = AUVWorld(project_config, AUVState(np.array([20.0, 50.0, 20.0]),
                                             0.0, 0.0, 1.5, 0.0, 0.0), (),
                     np.array([22.06, 50.0, 20.0]))
    result = world.step(ControlCommand(1.5, 0.0, 0.0))
    assert result.event.reason == 'success' and result.is_terminated
    assert result.timestamp_s == pytest.approx(0.04, rel=0, abs=1.0e-10)
    assert result.event.event_fraction_of_integration_step == pytest.approx(
        0.8, rel=0, abs=1.0e-8)
    assert result.auv_state.position_ned_m[0] == pytest.approx(20.06, rel=0, abs=1.0e-10)
    with pytest.raises(InvalidEnvironmentStateError, match='已经终止'):
        world.step(ControlCommand(1.5, 0.0, 0.0))
    assert world.timestamp_s == result.timestamp_s


def test_r1_pitch_limit_is_first_event_interpolation_not_clip(
    project_config: ProjectConfig,
) -> None:
    """稳态pitchrate=.2rad/s，离界.006rad；未提交小步越界，t=.03s定位。"""
    limit = project_config.dynamics.max_pitch_rad
    state = AUVState(np.array([20.0, 50.0, 20.0]), 0.0, limit - 0.006, 0.8, 0.0, 0.2)
    world = AUVWorld(project_config, state, (), np.array([80.0, 50.0, 20.0]))
    result = world.step(ControlCommand(0.8, 0.0, 0.2))
    assert result.event.reason == 'boundary'
    assert result.timestamp_s == pytest.approx(0.03, rel=0, abs=1.0e-10)
    assert result.event.event_fraction_of_integration_step == pytest.approx(
        0.6, rel=0, abs=1.0e-8)
    assert result.auv_state.pitch_rad == pytest.approx(limit, rel=0, abs=1.0e-10)
    expected_north = 20.0 + 0.6 * 0.05 * 0.8 * math.cos(limit - 0.001)
    expected_down = 20.0 - 0.6 * 0.05 * 0.8 * math.sin(limit - 0.001)
    np.testing.assert_allclose(result.auv_state.position_ned_m,
                               [expected_north, 50.0, expected_down], rtol=0, atol=1.0e-10)
    assert result.auv_state.pitch_rate_rad_s == pytest.approx(0.2, rel=0, abs=1.0e-10)


def test_r1_reward_four_terms_and_discounted_progress_not_telescope() -> None:
    """距离60→52→54→50：未折扣进展10m；折扣项另算，不冒称无限刷回报。"""
    distances = [60.0, 52.0, 54.0, 50.0]
    actions = [np.array([0.0, 0.0, 0.0]), np.array([0.2, -0.1, 0.0]),
               np.array([0.1, -0.1, 0.3]), np.array([0.2, 0.0, 0.1])]
    progress = []
    for index in range(3):
        total, parts = task_reward(distances[index], distances[index + 1], False,
                                   0.2, 0.2, actions[index], actions[index + 1], LocalTaskConfig())
        delta = actions[index + 1] - actions[index]
        reference_smoothness = -0.02 * sum(float(value) ** 2 for value in delta)
        reference_progress = distances[index] - distances[index + 1]
        assert parts == pytest.approx(dict(progress=reference_progress, goal=0.0,
                                           time=-0.01, smoothness=reference_smoothness),
                                     rel=0, abs=1.0e-10)
        assert total == pytest.approx(reference_progress - 0.01 + reference_smoothness,
                                      rel=0, abs=1.0e-10)
        progress.append(parts['progress'])
    assert sum(progress) == 10.0
    assert sum(0.999 ** index * value for index, value in enumerate(progress)) == pytest.approx(
        8.0 - 2.0 * 0.999 + 4.0 * 0.999 ** 2, rel=0, abs=1.0e-10)
    assert sum(0.999 ** index * value for index, value in enumerate(progress)) != 10.0


def test_r1_ordinary_actor_gradient_scalar_finite_difference() -> None:
    """已知minQ动作导数与熵导数中心差分；不读成本字段且alpha不获Actor梯度。"""
    theta, alpha = 0.4, torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    action = parameter.tanh()
    # 固定epsilon=0的重参数化logpi对均值的导数来自tanh Jacobian。
    log_pi = -0.5 * math.log(2.0 * math.pi) - torch.log(1.0 - action.square())
    value = actor_loss(log_pi, 0.7 * action, 0.7 * action + 2.0, alpha)
    value.backward()
    derivative = 0.4 * math.tanh(theta) - 0.7 * (1.0 - math.tanh(theta) ** 2)
    assert parameter.grad.item() == pytest.approx(derivative, rel=0, abs=1.0e-10)
    assert alpha.grad is None
    def reference(x: float) -> float:
        """固定epsilon的标量Actor目标，不使用生产loss计算参考。"""
        return (0.2 * (-0.5 * math.log(2.0 * math.pi)
                       - math.log(1.0 - math.tanh(x) ** 2)) - 0.7 * math.tanh(x))

    step = 1.0e-3
    numerical = (reference(theta + step) - reference(theta - step)) / (2.0 * step)
    assert derivative == pytest.approx(numerical, rel=0, abs=1.0e-3)
