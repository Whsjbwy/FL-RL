"""
科研运行元数据采集模块。

功能：
1. 保存 config snapshot、seed、代码 hash、运行时间；
2. 记录 Python、NumPy、PyTorch、Gymnasium、SB3 等版本；
3. 对未安装依赖明确记录 not_installed，而不是伪造版本。
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from auv_risk_rl.config import ProjectConfig


def _package_version(package_name: str) -> str:
    """读取已安装包版本；未安装时返回明确字符串，不静默伪造。"""

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def _git_revision(repository_root: Path) -> str:
    """读取当前 Git 提交；非 Git 环境时返回 unavailable。"""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def build_run_manifest(
    config: ProjectConfig,
    run_id: str,
    seed: int,
    scenario_id: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """
    构造一次正式运行所需的可追溯元数据。

    对应技术协议：
        第 30 章、附录 D.3；无独立编号公式。

    参数：
        config:
            当前完整配置快照。
        run_id:
            运行唯一标识。
        seed:
            根随机种子。
        scenario_id:
            场景标识。
        repository_root:
            Git 仓库根目录。

    返回：
        manifest:
            JSON 可序列化字典。

    shape/单位/坐标系：
        不适用。

    关键假设：
        调用方提供的 run_id 与 scenario_id 在实验命名空间内唯一/稳定。

    重要限制：
        代码 hash 为 unavailable 时必须在正式实验报告中视为复现缺口，而不是 PASS。
    """

    repository_path = Path(repository_root)
    return {
        "run_id": run_id,
        "stage_id": config.stage,
        "scenario_id": scenario_id,
        "seed": seed,
        "protocol_version": config.protocol_version,
        "config_version": config.config_version,
        "code_revision": _git_revision(repository_path),
        "run_time_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": _package_version("numpy"),
        "scipy_version": _package_version("scipy"),
        "pytorch_version": _package_version("torch"),
        "gymnasium_version": _package_version("gymnasium"),
        "stable_baselines3_version": _package_version("stable-baselines3"),
        "config_snapshot": config.to_dict(),
    }


def save_run_manifest(manifest: dict[str, Any], output_path: str | Path) -> None:
    """
    将运行元数据保存为 UTF-8 JSON。

    对应技术协议：
        附录 D.3 运行记录要求；无独立编号公式。

    参数：
        manifest:
            build_run_manifest 产生的字典。
        output_path:
            输出 JSON 路径。

    返回：
        无。

    shape/单位/坐标系：
        不适用。

    关键假设：
        上层已创建可写输出目录。

    重要限制：
        保存失败会直接抛出 I/O 异常，不允许静默跳过正式实验元数据。
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
