"""B0入口不会默认训练，工程产物和未来科研计划保持明确身份。"""

from pathlib import Path

import pytest
from scripts.run_b0_training import main, read_run_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_entry_preflight_only(capsys) -> None:
    """无参数只显示身份，不构造环境或执行任何更新。"""
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "PREFLIGHT_ONLY" in output and "NOT_RUN" in output


def test_scientific_and_engineering_configs_separate() -> None:
    """生产默认不因smoke而缩小，工程启动边界显式登记。"""
    production, raw = read_run_config(ROOT / "configs/stage2_b0.yaml")
    engineering, _ = read_run_config(ROOT / "configs/stage2_b0_engineering.yaml")
    assert production.sac.learning_starts == 10000
    assert production.sac.replay_capacity == 500000
    assert not raw["preparation"]["execution_authorized"]
    assert raw["preparation"]["mvp_seeds"] == [11, 22, 33]
    assert engineering.sac.learning_starts == 256
    assert engineering.sac.batch_size == production.sac.batch_size == 256


@pytest.mark.parametrize("arguments", [
    ["--execute"],
    ["--execute", "--run-kind", "scientific_training", "--transition-budget", "300000"],
    ["--execute", "--run-kind", "engineering_smoke", "--transition-budget", "264"],
])
def test_unregistered_or_mismatched_execution_rejected(arguments) -> None:
    """显式身份、预算或科学授权不满足即停止，不能静默长跑。"""
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
