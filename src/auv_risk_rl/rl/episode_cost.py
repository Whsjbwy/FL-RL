"""真实episode生命周期与LOCAL成本账本、每十个新完整episode的乘子更新。"""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from auv_risk_rl.costs.finite_horizon import normalized_finite_horizon_cost
from auv_risk_rl.rl.cost_math import COMPLETED, FAILURES, GAMMA, HORIZON, validate_cost_transition


@dataclass
class EpisodeCostLedger:
    """保留真实时段成本；尾项由既有有限时域helper计账，不插入Replay。"""

    episode_id: int
    instant_costs: list[float] = field(default_factory=list)
    closed: bool = False

    def append(self, cost: float, task_step: int, terminated: bool,
               truncated: bool, failure_type: str) -> dict[str, Any] | None:
        if self.closed or task_step != len(self.instant_costs)+1:
            raise ValueError("账本已关闭或任务步不连续。")
        validate_cost_transition(cost, task_step, terminated, truncated, failure_type)
        self.instant_costs.append(float(cost))
        if not terminated and not truncated:
            return None
        return self.finish(failure_type, complete=terminated)

    def finish(self, terminal_type: str = 'abandoned', *, complete: bool = False,
               ) -> dict[str, Any]:
        """提前reset只能标为不完整；不得把部分和冒充完整G_C。"""
        if self.closed:
            raise ValueError("同一账本不能重复提交。")
        if complete:
            if not self.instant_costs:
                raise ValueError("没有真实转移，不能构造完整episode。")
            validate_cost_transition(self.instant_costs[-1], len(self.instant_costs),
                                     True, False, terminal_type)
        self.closed = True
        failure_index = len(self.instant_costs)-1 if terminal_type in FAILURES else None
        total = normalized_finite_horizon_cost(self.instant_costs, GAMMA, HORIZON,
                                               failure_index) if complete else None
        return dict(episode_id=self.episode_id, actual_task_steps=len(self.instant_costs),
                    terminal_type=terminal_type, instant_costs=self.instant_costs.copy(),
                    G_C=total, eligible_for_lambda=complete,
                    status='COMPLETE' if complete else 'INCOMPLETE / NOT ELIGIBLE FOR LAMBDA')


class EpisodeMultiplier:
    """式52投影标量；不是神经参数，无Adam，无滑窗重复使用。"""

    def __init__(self) -> None:
        self.value = 1.0
        self.eta = 1.0
        self.budget = .05
        self.pending_complete_episode_costs: list[float] = []
        self.pending_episode_ids: list[int] = []
        self.seen_episode_ids: set[int] = set()
        self.lambda_update_count = 0
        self.complete_count = 0
        self.incomplete_count = 0
        self.episode_cost_min: float | None = None
        self.episode_cost_max: float | None = None
        self.last_batch: dict[str, Any] | None = None

    @property
    def completed_episodes_since_last_lambda_update(self) -> int:
        return len(self.pending_complete_episode_costs)

    def submit(self, episode: dict[str, Any]) -> None:
        episode_id = episode['episode_id']
        if episode_id in self.seen_episode_ids:
            raise ValueError("episode不能重复消费。")
        if not episode['eligible_for_lambda']:
            if episode['G_C'] is not None:
                raise ValueError("不完整episode不能声明完整G_C。")
            self.seen_episode_ids.add(episode_id)
            self.incomplete_count += 1
            return
        value = float(episode['G_C'])
        if episode['terminal_type'] not in COMPLETED:
            raise ValueError("只有真实完整任务终止可提交lambda。")
        if not math.isfinite(value) or not 0 <= value <= 1+1e-12:
            raise ValueError("完整成本必须有限且符合固定完整时域范围。")
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("乘子必须有限非负。")
        self.seen_episode_ids.add(episode_id)
        self.complete_count += 1
        self.episode_cost_min = value if self.episode_cost_min is None else min(
            value, self.episode_cost_min)
        self.episode_cost_max = value if self.episode_cost_max is None else max(
            value, self.episode_cost_max)
        self.pending_complete_episode_costs.append(value)
        self.pending_episode_ids.append(episode_id)
        if self.completed_episodes_since_last_lambda_update == 10:
            mean = math.fsum(self.pending_complete_episode_costs)/10
            self.value = max(0., self.value+self.eta*(mean-self.budget))
            self.last_batch = dict(episode_ids=self.pending_episode_ids.copy(), mean_G_C=mean)
            self.lambda_update_count += 1
            self.pending_complete_episode_costs.clear()
            self.pending_episode_ids.clear()

    def state_dict(self) -> dict[str, Any]:
        result = deepcopy(vars(self))
        result['completed_episodes_since_last_lambda_update'] = (
            self.completed_episodes_since_last_lambda_update)
        return result

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved = deepcopy(state)
        count = saved.pop('completed_episodes_since_last_lambda_update')
        if count != len(saved['pending_complete_episode_costs']) or not 0 <= count < 10:
            raise ValueError("乘子待处理episode计数损坏。")
        self.__dict__.update(saved)
