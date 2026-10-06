"""只处理普通SAC模型、更新、Replay与完整状态；不实现环境科学计算。"""

import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.rl.losses import (
    actor_loss,
    critic_loss,
    polyak_update,
    reward_target,
    temperature_loss,
)
from auv_risk_rl.rl.networks import Actor, RewardCritic
from auv_risk_rl.rl.replay import ReplayBuffer


class OrdinarySACAgent:
    """独立初始化/策略/Replay随机流，CPU或显式CUDA，神经路径float32。"""

    def __init__(self, config: SACConfig | None = None, *, source_fingerprint: str) -> None:
        self.config = config or SACConfig()
        self.source_fingerprint = source_fingerprint
        self.device = torch.device(self.config.device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.config.initialization_seed)
            self.actor = Actor().to(self.device)
            self.q1 = RewardCritic().to(self.device)
            self.q2 = RewardCritic().to(self.device)
        self.target_q1 = deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = deepcopy(self.q2).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(self.config.initial_alpha),
                                                   dtype=torch.float32, device=self.device))
        self.optimizers = {
            'actor': torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate),
            'q1': torch.optim.Adam(self.q1.parameters(), lr=self.config.learning_rate),
            'q2': torch.optim.Adam(self.q2.parameters(), lr=self.config.learning_rate),
            'alpha': torch.optim.Adam([self.log_alpha], lr=self.config.learning_rate),
        }
        self.generator = torch.Generator(device=self.device).manual_seed(self.config.actor_seed)
        self.replay = ReplayBuffer(self.config.replay_capacity, self.config.replay_seed)
        self.counters = dict(environment_steps=0, gradient_updates=0, episodes=0,
                             actor=0, q1=0, q2=0, alpha=0)
        self.last_failure_metadata: dict[str, Any] | None = None
        self._batch_metadata: dict[str, Any] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _finite(self, name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            self.last_failure_metadata = dict(field=name, counters=self.counters.copy(),
                                              **self._batch_metadata)
            raise FloatingPointError(str(self.last_failure_metadata))

    def _positive_alpha(self) -> None:
        self._finite('alpha', self.alpha)
        if self.alpha.item() <= 0:
            self.last_failure_metadata = dict(field='alpha_nonpositive',
                                              counters=self.counters.copy(), **self._batch_metadata)
            raise FloatingPointError(str(self.last_failure_metadata))

    @torch.no_grad()
    def sample_action(self, obs: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self._finite('observation', tensor)
        action, log_pi = self.actor.sample(tensor, self.generator)
        self._finite('action', action)
        self._finite('log_pi', log_pi)
        return action.cpu().numpy().copy()

    @torch.no_grad()
    def deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        result = self.actor.deterministic(tensor)
        self._finite('deterministic_action', result)
        return result.cpu().numpy().copy()

    def store_transition(self, **transition: Any) -> None:
        self.replay.add(**transition)
        self.counters['environment_steps'] += 1

    def eligible(self) -> bool:
        return (self.counters['environment_steps'] >= self.config.learning_starts
                and len(self.replay) >= self.config.batch_size)

    def _step(self, name: str, loss: torch.Tensor,
              parameters: list[nn.Parameter]) -> float:
        self._finite(name + '_loss', loss)
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            raise RuntimeError(f"{name}梯度路径断开。")
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        self._finite(name + '_gradient_norm', norm)
        optimizer.step()
        for p in parameters:
            self._finite(name + '_parameter', p)
        self.counters[name] += 1
        return float(norm.detach())

    def update_actor(self, obs: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """冻结Q参数但保留dQ/da；返回同一采样的log_pi供式53使用。"""
        critics = [*self.q1.parameters(), *self.q2.parameters()]
        previous = [p.requires_grad for p in critics]
        try:
            for p in critics:
                p.requires_grad_(False)
            action, log_pi = self.actor.sample(obs, self.generator)
            self._finite('actor_action', action)
            self._finite('actor_log_pi', log_pi)
            q1, q2 = self.q1(obs, action), self.q2(obs, action)
            self._finite('actor_q1', q1)
            self._finite('actor_q2', q2)
            loss = actor_loss(log_pi, q1, q2, self.alpha)
            norm = self._step('actor', loss, list(self.actor.parameters()))
            return log_pi.detach(), float(loss.detach()), norm
        except FloatingPointError as error:
            self.last_failure_metadata = dict(field=str(error), counters=self.counters.copy(),
                                              **self._batch_metadata)
            raise FloatingPointError(str(self.last_failure_metadata)) from error
        finally:
            for p, flag in zip(critics, previous, strict=True):
                p.requires_grad_(flag)

    def update(self, batch: dict[str, np.ndarray] | None = None) -> dict[str, float]:
        """一次完整更新；显式batch供合成工程测试，真实环境起步条件由runner控制。"""
        batch = self.replay.sample(self.config.batch_size) if batch is None else batch
        self._batch_metadata = dict(batch_size=len(batch['obs']),
                                    episode_id=batch.get('episode_id', []).tolist()
                                    if 'episode_id' in batch else [],
                                    task_step=batch.get('task_step', []).tolist()
                                    if 'task_step' in batch else [])
        # 严格白名单：executed/cost仅存储，不传入任何普通SAC损失。
        values = {k: torch.as_tensor(batch[k], device=self.device, dtype=torch.float32)
                  for k in ('obs', 'nominal_action', 'reward', 'next_obs', 'terminated')}
        for k in ('obs', 'nominal_action', 'reward', 'terminated'):
            self._finite(k, values[k])
        self._positive_alpha()
        try:
            target = reward_target(values['reward'], values['next_obs'], values['terminated'],
                                   self.actor, self.target_q1, self.target_q2, self.alpha,
                                   self.config.gamma, self.generator)
        except FloatingPointError as error:
            self.last_failure_metadata = dict(field=str(error), counters=self.counters.copy(),
                                              **self._batch_metadata)
            raise FloatingPointError(str(self.last_failure_metadata)) from error
        self._finite('reward_target', target)
        metrics: dict[str, float] = {}
        for name, q in (('q1', self.q1), ('q2', self.q2)):
            prediction = q(values['obs'], values['nominal_action'])
            self._finite(name, prediction)
            loss = critic_loss(prediction, target)
            metrics[name + '_gradient_norm'] = self._step(name, loss, list(q.parameters()))
            metrics[name + '_loss'] = float(loss.detach())
        log_pi, actor_value, actor_norm = self.update_actor(values['obs'])
        metrics.update(actor_loss=actor_value, actor_gradient_norm=actor_norm)
        loss = temperature_loss(self.log_alpha, log_pi, self.config.target_entropy)
        metrics['alpha_gradient_norm'] = self._step('alpha', loss, [self.log_alpha])
        metrics['alpha_loss'] = float(loss.detach())
        self._positive_alpha()
        polyak_update(self.target_q1, self.q1, self.config.tau)
        polyak_update(self.target_q2, self.q2, self.config.tau)
        self.counters['gradient_updates'] += 1
        metrics['alpha'] = float(self.alpha.detach())
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """完整训练恢复：模型、Adam、温度、计数器、CPU Replay及训练随机流。"""
        return deepcopy(dict(
            format='ordinary-sac-full-resume-v1', config=asdict(self.config),
            source_fingerprint=self.source_fingerprint,
            models={k: getattr(self, k).state_dict()
                    for k in ('actor', 'q1', 'q2', 'target_q1', 'target_q2')},
            optimizers={k: v.state_dict() for k, v in self.optimizers.items()},
            log_alpha=self.log_alpha.detach().clone(), counters=self.counters,
            actor_rng=self.generator.get_state(), replay=self.replay.state_dict()))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state['format'] != 'ordinary-sac-full-resume-v1'
                or state['config'] != asdict(self.config)
                or state['source_fingerprint'] != self.source_fingerprint):
            raise ValueError("完整恢复要求格式、配置、源码标识一致。")
        for k, value in state['models'].items():
            getattr(self, k).load_state_dict(value)
        for k, value in state['optimizers'].items():
            self.optimizers[k].load_state_dict(value)
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'])
        self.counters = state['counters'].copy()
        self.generator.set_state(state['actor_rng'].cpu())
        self.replay.load_state_dict(state['replay'])

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str | Path) -> None:
        """仅加载可信本地文件；完整Replay需要Python对象反序列化。"""
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
