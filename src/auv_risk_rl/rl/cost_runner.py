"""将B3运行器真实transition接入有限时域账本，不增添物理/感知/验证器。"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_risk_rl.env.local_navigation import LocalNavigationEnv
from auv_risk_rl.rl.cost_agent import CostLearningAgent
from auv_risk_rl.rl.episode_cost import EpisodeCostLedger
from auv_risk_rl.rl.replay import FAILURE_TYPES
from auv_risk_rl.rl.runner import OrdinarySACRunner


class CostLearningRunner(OrdinarySACRunner):
    """沿用普通step；只读取该真实Replay行完成账本和episode乘子提交。"""

    def __init__(self, env: LocalNavigationEnv, agent: CostLearningAgent, seed: int = 0,
                 reset_options: dict[str, Any] | None = None) -> None:
        super().__init__(env, agent, seed, reset_options)
        if env.config.environment.max_episode_control_steps != 1000:
            raise ValueError("B4冻结完整计划时域为1000。")
        self.ledger: EpisodeCostLedger | None = None
        self.last_episode: dict[str, Any] | None = None

    def step(self) -> dict[str, Any]:
        if self.needs_reset:
            if self.ledger is not None and not self.ledger.closed:
                self.last_episode = self.ledger.finish()
                self.agent.multiplier.submit(self.last_episode)
                self.episode_id += 1
            self.ledger = EpisodeCostLedger(self.episode_id)
        result = super().step()
        index = (self.agent.replay.cursor-1) % self.agent.replay.capacity
        row = self.agent.replay.at(np.array([index]))
        completed = self.ledger.append(
            float(row['cost'][0]), int(row['task_step'][0]), bool(row['terminated'][0]),
            bool(row['truncated'][0]), FAILURE_TYPES[int(row['failure_type'][0])])
        if completed is not None:
            self.last_episode = completed
            self.agent.multiplier.submit(completed)
        result.update(episode_cost=completed, lambda_value=self.agent.multiplier.value,
                      lambda_update_count=self.agent.multiplier.lambda_update_count,
                      pending_complete_episodes=(
                          self.agent.multiplier.completed_episodes_since_last_lambda_update))
        return result

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(dict(format='b4-cost-runner-full-resume-v1', agent=self.agent.state_dict(),
                        env=deepcopy(self.env), obs=self.obs, seed=self.seed,
                        reset_options=self.reset_options, episode_id=self.episode_id,
                        needs_reset=self.needs_reset, ledger=self.ledger,
                        last_episode=self.last_episode), path)

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.agent.device, weights_only=False)
        if state['format'] != 'b4-cost-runner-full-resume-v1':
            raise ValueError("不是完整B4运行器恢复格式。")
        self.agent.load_state_dict(state['agent'])
        for k in ('env', 'obs', 'seed', 'reset_options', 'episode_id', 'needs_reset',
                  'ledger', 'last_episode'):
            setattr(self, k, state[k])
