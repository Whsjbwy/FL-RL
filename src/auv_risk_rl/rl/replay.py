"""CPU分块环形Replay；统一采样，不实现优先经验回放。"""

from copy import deepcopy
from typing import Any

import numpy as np

FAILURE_TYPES = ('none', 'goal_success', 'collision', 'operational_boundary_failure',
                 'task_horizon', 'external_truncation')
FIELDS = {
    'obs': ((234,), np.float32), 'next_obs': ((234,), np.float32),
    'nominal_action': ((3,), np.float32), 'executed_action': ((3,), np.float32),
    'reward': ((), np.float32), 'cost': ((), np.float32),
    'terminated': ((), np.bool_), 'truncated': ((), np.bool_),
    'failure_type': ((), np.uint8), 'episode_id': ((), np.int64), 'task_step': ((), np.int64),
}


class ReplayBuffer:
    """惰性分配4096槽CPU块；采样返回独立副本，容量为精确transition数。"""

    def __init__(self, capacity: int = 500_000, seed: int = 2) -> None:
        if capacity < 1:
            raise ValueError("Replay容量必须为正。")
        self.capacity = capacity
        self.chunk_size = min(4096, capacity)
        self.chunks: dict[int, dict[str, np.ndarray]] = {}
        self.size = 0
        self.cursor = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, *, obs: np.ndarray, nominal_action: np.ndarray,
            executed_action: np.ndarray, reward: float, cost: float, next_obs: np.ndarray,
            terminated: bool, truncated: bool, failure_type: str = 'none',
            episode_id: int = 0, task_step: int = 0) -> None:
        """先完整校验再写入；真终止next_obs可含NaN哨兵，其余数据必须有限。"""
        if failure_type not in FAILURE_TYPES:
            raise ValueError(f"未知终止类型：{failure_type}")
        values = dict(obs=obs, nominal_action=nominal_action, executed_action=executed_action,
                      reward=reward, cost=cost, next_obs=next_obs, terminated=terminated,
                      truncated=truncated, failure_type=FAILURE_TYPES.index(failure_type),
                      episode_id=episode_id, task_step=task_step)
        for name, (shape, dtype) in FIELDS.items():
            arr = np.asarray(values[name], dtype=dtype)
            if arr.shape != shape:
                raise ValueError(f"{name}形状应为{shape}，实际{arr.shape}。")
            if not (name == 'next_obs' and terminated) and not np.isfinite(arr).all():
                raise ValueError(f"{name}包含非有限值。")
            if name in ('nominal_action', 'executed_action') and np.any(np.abs(arr) > 1):
                raise ValueError(f"{name}超出归一化范围。")
            values[name] = arr
        block, offset = divmod(self.cursor, self.chunk_size)
        if block not in self.chunks:
            count = min(self.chunk_size, self.capacity - block * self.chunk_size)
            self.chunks[block] = {k: np.zeros((count, *s), dtype=t)
                                  for k, (s, t) in FIELDS.items()}
        for name, value in values.items():
            self.chunks[block][name][offset] = value
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """均匀有放回采样；不改变存储内容。"""
        if batch_size < 1 or self.size < batch_size:
            raise ValueError("Replay不足完整batch。")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return self.at(indices)

    def at(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """读取物理槽副本，用于采样与审计。"""
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Replay槽超出有效范围。")
        return {k: np.stack([self.chunks[int(i) // self.chunk_size][k][int(i) % self.chunk_size]
                             for i in indices]) for k in FIELDS}

    def state_dict(self) -> dict[str, Any]:
        """包含全部已分配CPU块、环指针及独立采样随机状态。"""
        return deepcopy(dict(capacity=self.capacity, chunk_size=self.chunk_size,
                             chunks=self.chunks, size=self.size, cursor=self.cursor,
                             rng=self.rng.bit_generator.state))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """完整恢复同容量Replay。"""
        if state['capacity'] != self.capacity or state['chunk_size'] != self.chunk_size:
            raise ValueError("Replay配置不兼容。")
        self.chunks = deepcopy(state['chunks'])
        self.size, self.cursor = state['size'], state['cursor']
        self.rng.bit_generator.state = deepcopy(state['rng'])
