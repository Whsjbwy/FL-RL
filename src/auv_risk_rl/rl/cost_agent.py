"""B4组合成本学习器：B3普通SAC路径逐字保留，成本不反馈Actor。"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.rl.cost_math import (
    GAMMA,
    HORIZON,
    CostCritic,
    cost_loss,
    cost_target,
    local_cost_tail,
)
from auv_risk_rl.rl.episode_cost import EpisodeMultiplier
from auv_risk_rl.rl.losses import polyak_update


class CostLearningAgent:
    """只增加成本网络和episode乘子；普通Actor及其Adam不读取这些状态。"""

    def __init__(self, config: SACConfig | None = None, *, source_fingerprint: str,
                 cost_initialization_seed: int = 31001, cost_action_seed: int = 31002) -> None:
        self.ordinary = OrdinarySACAgent(config, source_fingerprint=source_fingerprint)
        if self.config.gamma != GAMMA:
            raise ValueError("成本与收益必须使用冻结gamma=.999。")
        self.cost_initialization_seed = cost_initialization_seed
        self.cost_action_seed = cost_action_seed
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cost_initialization_seed)
            self.cost_q = CostCritic().to(self.device)
        self.cost_target_q = deepcopy(self.cost_q).requires_grad_(False)
        self.cost_optimizer = torch.optim.Adam(self.cost_q.parameters(),
                                                lr=self.config.learning_rate)
        self.cost_generator = torch.Generator(device=self.device).manual_seed(cost_action_seed)
        self.multiplier = EpisodeMultiplier()
        self.cost_update_count = 0
        self.diagnostics: dict[str, Any] = dict(nonfinite_count=0, target_above_1_count=0,
                                               target_below_0_count=0,
                                               q_above_remaining_theoretical_bound_count=0)
        self.last_failure_metadata: dict[str, Any] | None = None
        self.migration: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        """转发只属于B3的动作、Replay和计数器接口，不复制实现。"""
        return getattr(object.__getattribute__(self, 'ordinary'), name)

    def _finite(self, name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(name)

    def update(self, batch: dict[str, np.ndarray] | None = None) -> dict[str, float]:
        """独立成本target→原B3整步→Cost Adam→Cost Polyak；Replay只采样一次。"""
        batch = self.replay.sample(self.config.batch_size) if batch is None else batch
        try:
            target = cost_target(batch, self.actor, self.cost_target_q,
                                 self.cost_generator, self.device)
            self._finite('cost_target', target)
            metrics = self.ordinary.update(batch)
            obs = torch.as_tensor(batch['obs'], device=self.device, dtype=torch.float32)
            action = torch.as_tensor(batch['nominal_action'], device=self.device,
                                     dtype=torch.float32)
            prediction = self.cost_q(obs, action)
            self._finite('cost_q', prediction)
            loss = cost_loss(prediction, target)
            self._finite('cost_loss', loss)
            self.cost_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(p.grad) for p in self.cost_q.parameters()]))
            self._finite('cost_gradient_norm', norm)
            self.cost_optimizer.step()
            for p in self.cost_q.parameters():
                self._finite('cost_parameter', p)
            polyak_update(self.cost_target_q, self.cost_q, self.config.tau)
            self.cost_update_count += 1
            bounds = torch.tensor([local_cost_tail(HORIZON-int(s)+1) for s in batch['task_step']],
                                  device=self.device).reshape(-1, 1)
            for key, value in (
                ('target_above_1_count', (target > 1).sum()),
                ('target_below_0_count', (target < 0).sum()),
                ('q_above_remaining_theoretical_bound_count', (prediction > bounds).sum()),
            ):
                self.diagnostics[key] += int(value)
            self.diagnostics.update(cost_q_min=float(prediction.detach().min()),
                                    cost_q_max=float(prediction.detach().max()),
                                    cost_target_min=float(target.min()),
                                    cost_target_max=float(target.max()))
            metrics.update(cost_loss=float(loss.detach()), cost_gradient_norm=float(norm),
                           **self.diagnostics)
            return metrics
        except FloatingPointError as error:
            self.diagnostics['nonfinite_count'] += 1
            self.last_failure_metadata = dict(field=str(error), batch_size=len(batch['cost']),
                                              episode_ids=batch['episode_id'].tolist(),
                                              task_steps=batch['task_step'].tolist(),
                                              cost_update_count=self.cost_update_count)
            raise FloatingPointError(str(self.last_failure_metadata)) from error

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(dict(format='b4-cost-learning-full-resume-v1',
                             ordinary=self.ordinary.state_dict(), cost_q=self.cost_q.state_dict(),
                             cost_target=self.cost_target_q.state_dict(),
                             cost_optimizer=self.cost_optimizer.state_dict(),
                             cost_rng=self.cost_generator.get_state(),
                             cost_initialization_seed=self.cost_initialization_seed,
                             cost_action_seed=self.cost_action_seed,
                             multiplier=self.multiplier.state_dict(),
                             diagnostics=self.diagnostics, cost_update_count=self.cost_update_count,
                             migration=self.migration, replay_schema='unchanged-B3'))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state['format'] != 'b4-cost-learning-full-resume-v1'
                or state['cost_initialization_seed'] != self.cost_initialization_seed
                or state['cost_action_seed'] != self.cost_action_seed):
            raise ValueError("B4完整恢复格式或随机命名空间不一致。")
        self.ordinary.load_state_dict(state['ordinary'])
        self.cost_q.load_state_dict(state['cost_q'])
        self.cost_target_q.load_state_dict(state['cost_target'])
        self.cost_optimizer.load_state_dict(state['cost_optimizer'])
        self.cost_generator.set_state(state['cost_rng'].cpu())
        self.multiplier.load_state_dict(state['multiplier'])
        self.diagnostics = deepcopy(state['diagnostics'])
        self.cost_update_count = state['cost_update_count']
        self.migration = deepcopy(state['migration'])

    def migrate_from_b3(self, state: dict[str, Any], *, expected_source: str) -> dict[str, Any]:
        """显式阶段初始化；只加载B3状态，新成本状态保持独立初始化，不冒称恢复。"""
        if (state['format'] != 'ordinary-sac-full-resume-v1'
                or state['source_fingerprint'] != expected_source):
            raise ValueError("B3迁移来源格式或源码标识错误。")
        if (self.cost_update_count or self.counters['environment_steps']
                or self.multiplier.seen_episode_ids):
            raise ValueError("阶段迁移必须使用全新B4对象。")
        migrated = deepcopy(state)
        migrated['source_fingerprint'] = self.source_fingerprint
        self.ordinary.load_state_dict(migrated)
        self.migration = dict(status='MIGRATED_FROM_B3', source=expected_source,
                              destination=self.source_fingerprint, exact_B4_resume=False)
        return deepcopy(self.migration)

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str | Path) -> None:
        """只加载可信本地完整文件，保留B3对源码/配置/device的一致性检查。"""
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
