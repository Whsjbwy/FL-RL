"""B5仅接通LOCAL52正号成本惩罚；复用B3/B4更新与状态，不修改普通基线。"""

import math
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.rl.cost_agent import CostLearningAgent


def constrained_actor_loss(log_pi: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor,
                           cost_q: torch.Tensor, alpha: torch.Tensor,
                           multiplier: float) -> torch.Tensor:
    """最小化式52；标量λ不建图，单一公式在λ=0自然退化，无额外成本缩放。"""
    if not isinstance(multiplier, float | int):
        raise TypeError("lambda必须是非autograd的本地标量。")
    if not math.isfinite(multiplier) or multiplier < 0:
        raise FloatingPointError("lambda必须有限且非负。")
    return (alpha.detach()*log_pi - torch.minimum(q1, q2) + multiplier*cost_q).mean()


class _ConstrainedActorCore(OrdinarySACAgent):
    """仅覆盖Actor方法；继承原更新编排，不对OrdinarySACAgent做猴子补丁。"""

    def __init__(self, owner: "ConstrainedSACAgent") -> None:
        super().__init__(owner.ordinary.config,
                         source_fingerprint=owner.ordinary.source_fingerprint)
        self.owner = owner

    def update_actor(self, obs: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        critics = [*self.q1.parameters(), *self.q2.parameters(), *self.owner.cost_q.parameters()]
        flags = [p.requires_grad for p in critics]
        try:
            multiplier = self.owner.multiplier.value
            if not isinstance(multiplier, float | int):
                raise TypeError("lambda必须为本地标量。")
            if not math.isfinite(multiplier) or multiplier < 0:
                raise FloatingPointError("actor_lambda")
            for p in critics:
                p.requires_grad_(False)
            action, log_pi = self.actor.sample(obs, self.generator)
            q1, q2 = self.q1(obs, action), self.q2(obs, action)
            cost_q = self.owner.cost_q(obs, action)
            cost_term = multiplier*cost_q
            for name, value in (('action', action), ('log_pi', log_pi), ('q1', q1),
                                ('q2', q2), ('cost_q', cost_q), ('cost_term', cost_term)):
                self._finite('constrained_actor_'+name, value)
            loss = constrained_actor_loss(log_pi, q1, q2, cost_q, self.alpha, multiplier)
            norm = self._step('actor', loss, list(self.actor.parameters()))
            self.owner.actor_components = dict(
                actor_entropy_term=float((self.alpha.detach()*log_pi).mean().detach()),
                actor_reward_q_term=float((-torch.minimum(q1, q2)).mean().detach()),
                actor_cost_q_term=float(cost_term.mean().detach()), lambda_value=float(multiplier),
                mean_cost_q=float(cost_q.mean().detach()), total_actor_loss=float(loss.detach()))
            self.owner.constrained_actor_updates += 1
            return log_pi.detach(), float(loss.detach()), norm
        except FloatingPointError as error:
            self.last_failure_metadata = dict(field=str(error), counters=self.counters.copy(),
                                              **self._batch_metadata)
            self.owner.last_actor_failure_metadata = deepcopy(self.last_failure_metadata)
            raise FloatingPointError(str(self.last_failure_metadata)) from error
        finally:
            for p, flag in zip(critics, flags, strict=True):
                p.requires_grad_(flag)


class ConstrainedSACAgent(CostLearningAgent):
    """B4组合对象拥有独立Actor子类；成本链、Replay、温度及目标更新原样复用。"""

    OBJECTIVE = 'LOCAL52: mean(alpha*log_pi-min(Q_R1,Q_R2)+lambda*Q_C)'

    def __init__(self, config: SACConfig | None = None, *, source_fingerprint: str,
                 cost_initialization_seed: int = 31001, cost_action_seed: int = 31002) -> None:
        super().__init__(config, source_fingerprint=source_fingerprint,
                         cost_initialization_seed=cost_initialization_seed,
                         cost_action_seed=cost_action_seed)
        previous = self.ordinary.state_dict()
        core = _ConstrainedActorCore(self)
        core.load_state_dict(previous)
        self.ordinary = core
        self.actor_components: dict[str, float] = {}
        self.constrained_actor_updates = 0
        self.last_actor_failure_metadata: dict[str, Any] | None = None

    def update(self, batch: dict[str, np.ndarray] | None = None) -> dict[str, float]:
        metrics = super().update(batch)
        metrics.update(self.actor_components)
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(dict(format='b5-constrained-sac-full-resume-v1', algorithm_stage='B5',
                             objective=self.OBJECTIVE, source_fingerprint=self.source_fingerprint,
                             b4_state=super().state_dict(), actor_components=self.actor_components,
                             constrained_actor_updates=self.constrained_actor_updates,
                             last_actor_failure_metadata=self.last_actor_failure_metadata))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state['format'] != 'b5-constrained-sac-full-resume-v1'
                or state['algorithm_stage'] != 'B5' or state['objective'] != self.OBJECTIVE
                or state['source_fingerprint'] != self.source_fingerprint):
            raise ValueError("B5恢复要求算法版本、目标和源码一致。")
        super().load_state_dict(state['b4_state'])
        self.actor_components = deepcopy(state['actor_components'])
        self.constrained_actor_updates = state['constrained_actor_updates']
        self.last_actor_failure_metadata = deepcopy(state['last_actor_failure_metadata'])

    def migrate_from_b4(self, state: dict[str, Any], *, expected_source: str) -> dict[str, Any]:
        """算法目标改变，只称阶段迁移；源checkpoint只读，下一Actor步采用式52。"""
        if (state['format'] != 'b4-cost-learning-full-resume-v1'
                or state['ordinary']['source_fingerprint'] != expected_source):
            raise ValueError("B4迁移来源格式或源码不一致。")
        if (self.constrained_actor_updates or self.cost_update_count
                or any(self.counters.values()) or len(self.replay)
                or self.multiplier.seen_episode_ids):
            raise ValueError("迁移必须使用全新B5对象。")
        migrated = deepcopy(state)
        migrated['ordinary']['source_fingerprint'] = self.source_fingerprint
        super().load_state_dict(migrated)
        self.migration = dict(status='MIGRATED_FROM_B4', source=expected_source,
                              destination=self.source_fingerprint, exact_B5_resume=False,
                              prior_migration=deepcopy(state['migration']))
        return deepcopy(self.migration)

    def migrate_from_b3(self, state: dict[str, Any], *, expected_source: str) -> dict[str, Any]:
        raise ValueError("B5只接受显式B4迁移；B3旧接口仍由原类提供。")
