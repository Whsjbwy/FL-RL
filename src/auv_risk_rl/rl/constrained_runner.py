"""B5沿用B4真实环境与完整episode账本，只版本化checkpoint和阶段迁移。"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from auv_risk_rl.env.local_navigation import LocalNavigationEnv
from auv_risk_rl.rl.constrained_agent import ConstrainedSACAgent
from auv_risk_rl.rl.cost_runner import CostLearningRunner


class ConstrainedSACRunner(CostLearningRunner):
    """不覆盖step：所有wrapper、执行动作、Replay与lambda生命周期继承B4。"""

    FIELDS = ('env', 'obs', 'seed', 'reset_options', 'episode_id', 'needs_reset',
              'ledger', 'last_episode')

    def __init__(self, env: LocalNavigationEnv, agent: ConstrainedSACAgent, seed: int = 0,
                 reset_options: dict[str, Any] | None = None) -> None:
        super().__init__(env, agent, seed, reset_options)

    def save_checkpoint(self, path: str | Path) -> None:
        state = {k: deepcopy(getattr(self, k)) for k in self.FIELDS}
        state.update(format='b5-constrained-runner-full-resume-v1', agent=self.agent.state_dict())
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """仅可信本地文件；完整同设备恢复由嵌套agent检查源码与配置。"""
        state = torch.load(path, map_location=self.agent.device, weights_only=False)
        if state['format'] != 'b5-constrained-runner-full-resume-v1':
            raise ValueError("不是B5完整运行器恢复格式。")
        self.agent.load_state_dict(state['agent'])
        for k in self.FIELDS:
            setattr(self, k, state[k])

    def migrate_from_b4_checkpoint(self, path: str | Path, *,
                                   expected_source: str) -> dict[str, Any]:
        state = torch.load(path, map_location=self.agent.device, weights_only=False)
        if state['format'] != 'b4-cost-runner-full-resume-v1':
            raise ValueError("不是B4完整运行器迁移格式。")
        metadata = self.agent.migrate_from_b4(state['agent'], expected_source=expected_source)
        for k in self.FIELDS:
            setattr(self, k, state[k])
        return metadata
