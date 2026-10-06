"""LOCAL式47/48的名义策略及独立237维reward Q。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def squashed_log_prob(z: torch.Tensor, mean: torch.Tensor,
                      log_std: torch.Tensor) -> torch.Tensor:
    """由pre-tanh量计算式48；饱和时不反算atanh或log(1-a²)。"""
    normal = -0.5 * ((z - mean) * torch.exp(-log_std)).square()
    normal = normal - log_std - 0.5 * math.log(2 * math.pi)
    jacobian = 2 * (math.log(2) - z - F.softplus(-2 * z))
    return (normal - jacobian).sum(dim=-1, keepdim=True)


class Actor(nn.Module):
    """234→256→256→6；六输出为三均值和三log尺度，不是六维动作。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(234, 256), nn.ReLU(), nn.Linear(256, 256),
                                 nn.ReLU(), nn.Linear(256, 6))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.net(obs)
        if not torch.isfinite(output).all():
            raise FloatingPointError("actor_raw_output包含非有限值。")
        mean, raw_log_std = output.chunk(2, dim=-1)
        return mean, raw_log_std.clamp(-20, 2)

    def sample(self, obs: torch.Tensor, generator: torch.Generator,
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """使用独立训练随机流重参数化采样归一化名义动作。"""
        mean, log_std = self(obs)
        epsilon = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device,
                              generator=generator)
        z = mean + log_std.exp() * epsilon
        return z.tanh(), squashed_log_prob(z, mean, log_std)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """仅供调试的tanh均值；不消耗Actor随机流。"""
        return self(obs)[0].tanh()


class RewardCritic(nn.Module):
    """独立237→256→256→1网络，接口仅接收观察与名义动作。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(237, 256), nn.ReLU(), nn.Linear(256, 256),
                                 nn.ReLU(), nn.Linear(256, 1))

    def forward(self, obs: torch.Tensor, nominal_action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, nominal_action), dim=-1))
