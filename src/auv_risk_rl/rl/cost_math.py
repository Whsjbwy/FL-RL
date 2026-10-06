"""LOCAL45/46/50/51固定完整时域成本；无剩余时域重新归一化。"""

import math

import numpy as np
import torch
from torch import nn

from auv_risk_rl.costs.finite_horizon import finite_horizon_discount_normalizer
from auv_risk_rl.rl.networks import Actor
from auv_risk_rl.rl.replay import FAILURE_TYPES

GAMMA = 0.999
HORIZON = 1000
BETA = finite_horizon_discount_normalizer(GAMMA, HORIZON)
FAILURES = ('collision', 'operational_boundary_failure')
COMPLETED = (*FAILURES, 'goal_success', 'task_horizon')


def local_cost_tail(length: int) -> float:
    """式51局部指数尾项，不含外部gamma，也不含episode绝对时间折扣。"""
    if not isinstance(length, int | np.integer) or not 0 <= length <= HORIZON:
        raise ValueError("尾项长度必须为0..1000整数。")
    return BETA * (-math.expm1(length * math.log(GAMMA))) / (1 - GAMMA)


def validate_cost_transition(cost: float, task_step: int, terminated: bool,
                             truncated: bool, failure_type: str) -> bool:
    """复用B2类型语义，严格区分拒绝成本与真实安全失败。"""
    if cost not in (0, 1) or not 1 <= task_step <= HORIZON or int(task_step) != task_step:
        raise ValueError("成本必须为0/1；任务控制步必须为1..1000整数。")
    if failure_type not in FAILURE_TYPES:
        raise ValueError("未知终止类型。")
    if bool(terminated) != (failure_type in COMPLETED):
        raise ValueError("真实终止标志与类型不一致。")
    if not terminated and (bool(truncated) != (failure_type == 'external_truncation')):
        raise ValueError("外部截断标志与类型不一致。")
    if failure_type == 'task_horizon' and task_step != HORIZON:
        raise ValueError("内在时域结束必须为第1000步。")
    if task_step == HORIZON and not terminated:
        raise ValueError("第1000步必须真实终止。")
    failure = failure_type in FAILURES
    if failure and cost != 1:
        raise ValueError("安全失败必须有c_train=1。")
    return failure


class CostCritic(nn.Module):
    """独立237→256→256→1 sigmoid，输入只为观察及名义归一化动作。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(237, 256), nn.ReLU(), nn.Linear(256, 256),
                                 nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, obs: torch.Tensor, nominal_action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, nominal_action), dim=-1))


@torch.no_grad()
def cost_target(batch: dict[str, np.ndarray], actor: Actor, target: CostCritic,
                generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """原生式51；不含熵项、不裁剪；终止行绝不访问Actor/目标网络。"""
    count = len(batch['cost'])
    failure = []
    for i in range(count):
        code = int(batch['failure_type'][i])
        if not 0 <= code < len(FAILURE_TYPES):
            raise ValueError("Replay终止类型编码无效。")
        failure.append(validate_cost_transition(
            float(batch['cost'][i]), int(batch['task_step'][i]), bool(batch['terminated'][i]),
            bool(batch['truncated'][i]), FAILURE_TYPES[code]))
    value = torch.as_tensor(batch['cost'], device=device, dtype=torch.float32).reshape(-1, 1)*BETA
    tails = [local_cost_tail(HORIZON-int(batch['task_step'][i])) if failure[i] else 0.
             for i in range(count)]
    value += GAMMA * torch.tensor(tails, device=device, dtype=torch.float32).reshape(-1, 1)
    active = ~torch.as_tensor(batch['terminated'], device=device, dtype=torch.bool)
    if active.any():
        obs = torch.as_tensor(batch['next_obs'], device=device, dtype=torch.float32)[active]
        if not torch.isfinite(obs).all():
            raise FloatingPointError("cost_next_obs非有限。")
        nominal, _ = actor.sample(obs, generator)
        prediction = target(obs, nominal)
        if not torch.isfinite(nominal).all() or not torch.isfinite(prediction).all():
            raise FloatingPointError("cost_next_action/target非有限。")
        value[active] += GAMMA * prediction
    return value


def cost_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unified Math IC01：1/2均方误差，目标停止梯度；不改变原式51目标。"""
    return .5 * (prediction-target.detach()).square().mean()
