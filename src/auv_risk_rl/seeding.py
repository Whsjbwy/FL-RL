"""
统一随机种子管理模块。

功能：
1. 禁止模块内部调用 np.random.seed；
2. 为 scenario、sensor、process、RL、evaluation 提供彼此独立的随机流；
3. 保证相同根种子和命名空间可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeedManager:
    """
    通过 NumPy SeedSequence 管理独立随机命名空间。

    对应技术协议：
        第 22 章统计协议、第 30 章工程代码结构；无独立编号公式。

    参数：
        root_seed:
            根随机种子，整数，无单位和坐标系。

    返回：
        类实例按名称创建 numpy.random.Generator。

    shape/单位/坐标系：
        不适用。

    关键假设：
        同一命名空间在同一 SeedManager 中只创建一次。

    重要限制：
        不同进程若需要完全一致的命名空间映射，应由上层显式记录命名顺序和根种子。
    """

    root_seed: int
    _generators: dict[str, np.random.Generator] = field(default_factory=dict, init=False)
    _seed_sequences: dict[str, np.random.SeedSequence] = field(default_factory=dict, init=False)

    def get_rng(self, namespace: str) -> np.random.Generator:
        """
        获取指定命名空间的独立随机数生成器。

        对应技术协议：
            第 22 章统计协议；无独立编号公式。

        参数：
            namespace:
                随机源名称，例如 scenario、sensor、process、rl、evaluation。

        返回：
            numpy.random.Generator。

        shape/单位/坐标系：
            不适用。

        关键假设：
            namespace 为稳定英文标识。

        重要限制：
            为避免隐藏状态，同一生成器应显式传入需要随机性的函数或对象。
        """

        if namespace not in self._generators:
            namespace_entropy = [ord(character) for character in namespace]
            seed_sequence = np.random.SeedSequence([self.root_seed, *namespace_entropy])
            self._seed_sequences[namespace] = seed_sequence
            self._generators[namespace] = np.random.default_rng(seed_sequence)
        return self._generators[namespace]
