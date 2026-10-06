"""普通SAC薄运行器；直接调用B2环境，不重复验证器或科学计算。"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_risk_rl.env.local_navigation import LocalNavigationEnv
from auv_risk_rl.rl.agent import OrdinarySACAgent


class OrdinarySACRunner:
    """每个env.step一次存储，满足起步条件后一次更新；重置前保存真实next_obs。"""

    def __init__(self, env: LocalNavigationEnv, agent: OrdinarySACAgent,
                 seed: int = 0, reset_options: dict[str, Any] | None = None) -> None:
        self.env, self.agent = env, agent
        self.seed = seed
        self.reset_options = deepcopy(reset_options)
        self.obs: np.ndarray | None = None
        self.episode_id = 0
        self.needs_reset = True

    def step(self) -> dict[str, Any]:
        if self.needs_reset:
            self.obs, _ = self.env.reset(seed=self.seed + self.episode_id,
                                          options=self.reset_options)
            self.needs_reset = False
        nominal = self.agent.sample_action(self.obs)
        next_obs, reward, terminated, truncated, info = self.env.step(nominal)
        self.agent.store_transition(
            obs=self.obs, nominal_action=nominal,
            executed_action=info['executed_action_normalized'], reward=reward,
            cost=info['cost'], next_obs=next_obs, terminated=terminated, truncated=truncated,
            failure_type=info['failure_type'], episode_id=self.episode_id,
            task_step=info['task_control_step'])
        self.obs = next_obs.copy()
        metrics = self.agent.update() if self.agent.eligible() else {}
        if terminated or truncated:
            self.agent.counters['episodes'] += 1
            self.episode_id += 1
            self.needs_reset = True
        return dict(reward=reward, cost=info['cost'], terminated=terminated,
                    truncated=truncated, metrics=metrics,
                    counters=self.agent.counters.copy(), replay_size=len(self.agent.replay))

    def save_checkpoint(self, path: str | Path) -> None:
        """保存完整环境/KF/随机流快照，支持控制步边界的精确续接。"""
        torch.save(dict(format='ordinary-sac-runner-full-resume-v1', agent=self.agent.state_dict(),
                        env=deepcopy(self.env), obs=self.obs, seed=self.seed,
                        reset_options=self.reset_options, episode_id=self.episode_id,
                        needs_reset=self.needs_reset), path)

    def load_checkpoint(self, path: str | Path) -> None:
        """只接受可信本地完整快照；设备及源码配置须与保存时一致。"""
        state = torch.load(path, map_location=self.agent.device, weights_only=False)
        if state['format'] != 'ordinary-sac-runner-full-resume-v1':
            raise ValueError("不是完整运行器恢复格式。")
        self.agent.load_state_dict(state['agent'])
        for k in ('env', 'obs', 'seed', 'reset_options', 'episode_id', 'needs_reset'):
            setattr(self, k, state[k])
