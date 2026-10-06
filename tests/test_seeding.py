"""验证统一 SeedManager 的可复现性和命名空间隔离。"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.seeding import SeedManager


def test_seed_manager_reproduces_named_streams() -> None:
    """相同 root seed 与 namespace 应产生完全相同的随机序列。"""

    first_manager = SeedManager(root_seed=42)
    second_manager = SeedManager(root_seed=42)
    first_values = first_manager.get_rng("sensor").normal(size=8)
    second_values = second_manager.get_rng("sensor").normal(size=8)
    assert np.array_equal(first_values, second_values), (
        "expected identical sequences, "
        f"actual max error={np.max(np.abs(first_values-second_values))}"
    )


def test_seed_manager_separates_random_namespaces() -> None:
    """同一 root seed 下 sensor 与 scenario 流不应复用同一随机序列。"""

    manager = SeedManager(root_seed=42)
    sensor_values = manager.get_rng("sensor").normal(size=8)
    scenario_values = manager.get_rng("scenario").normal(size=8)
    assert not np.array_equal(sensor_values, scenario_values), (
        "expected distinct random streams for sensor and scenario namespaces"
    )
