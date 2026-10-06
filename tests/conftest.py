"""Stage 0 测试共享夹具。"""

from __future__ import annotations

from pathlib import Path

import pytest

from auv_risk_rl.config import ProjectConfig, load_project_config


@pytest.fixture(scope="session")
def project_config() -> ProjectConfig:
    """
    加载仓库冻结的 Stage 0 配置。

    测试目的：
        确保所有测试使用同一配置来源，避免在测试文件中复制关键项目参数。
    """

    repository_root = Path(__file__).resolve().parents[1]
    return load_project_config(repository_root / "configs" / "stage0.yaml")
