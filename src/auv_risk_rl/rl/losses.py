"""普通SAC损失，逐式对应LOCAL49、52普通部分、53、54。"""

import torch
from torch import nn

from auv_risk_rl.rl.networks import Actor, RewardCritic


@torch.no_grad()
def reward_target(reward: torch.Tensor, next_obs: torch.Tensor, terminated: torch.Tensor,
                  actor: Actor, target_q1: RewardCritic, target_q2: RewardCritic,
                  alpha: torch.Tensor, gamma: float,
                  generator: torch.Generator) -> torch.Tensor:
    """式49只求值可续接行；外部truncated不终止Bellman续接。"""
    result = reward.reshape(-1, 1).clone()
    active = ~terminated.reshape(-1).bool()
    if active.any():
        obs = next_obs[active]
        nominal, log_pi = actor.sample(obs, generator)
        q1, q2 = target_q1(obs, nominal), target_q2(obs, nominal)
        for name, value in (('next_obs', obs), ('next_action', nominal),
                            ('next_log_pi', log_pi), ('target_q1', q1), ('target_q2', q2)):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"{name}包含非有限值。")
        q = torch.minimum(q1, q2)
        result[active] += gamma * (q - alpha * log_pi)
    return result


def critic_loss(q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """式49均方误差，无额外1/2系数。"""
    return (q - target.detach()).square().mean()


def actor_loss(log_pi: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor,
               alpha: torch.Tensor) -> torch.Tensor:
    """普通SAC，不读取安全成本。"""
    return (alpha.detach() * log_pi - torch.minimum(q1, q2)).mean()


def temperature_loss(log_alpha: torch.Tensor, log_pi: torch.Tensor,
                     target_entropy: float = -3.0) -> torch.Tensor:
    """原生式53：exp(log_alpha)，不是常见的log_alpha代理损失。"""
    return (-log_alpha.exp() * (log_pi.detach() + target_entropy)).mean()


@torch.no_grad()
def polyak_update(target: nn.Module, online: nn.Module, tau: float = 0.005) -> None:
    """式54；目标向在线参数移动tau。"""
    for dst, src in zip(target.parameters(), online.parameters(), strict=True):
        dst.mul_(1 - tau).add_(src, alpha=tau)
