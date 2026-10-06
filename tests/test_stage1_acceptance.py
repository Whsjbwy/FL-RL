"""LOCAL Stage 1 新补独立参考与恢复路径；全部是非学习固定验收场景。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage1_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("stage1_acceptance", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
acceptance = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(acceptance)


@pytest.mark.parametrize("case", acceptance.MOTION_CASES)
def test_independent_motion_reference(project_config, case) -> None:
    """恒速、稳态恒转、执行器响应分别使用预登记独立参考。"""
    result = acceptance.run_motion_case(project_config, case)
    assert result["status"] == "PASS" and len(result["trajectories"]) == 51


@pytest.mark.parametrize("case", acceptance.EVENT_CASES)
def test_first_event_and_no_advance_after_terminal(project_config, case) -> None:
    """头对头/交叉/边界/姿态/成功/超时均按最早事件停止。"""
    result = acceptance.run_event_case(project_config, case)
    assert result["event"]["no_advance_after_terminal"]


@pytest.mark.parametrize("clearance", [0.25, 0.5, 0.75])
def test_physical_clearance_and_strict_nearmiss_boundary(project_config, clearance) -> None:
    """物理碰撞、额外验证裕量和严格小于0.5m的near-miss分别记录。"""
    result = acceptance.run_nearmiss_case(project_config, clearance)
    assert result["near_miss"] == (clearance < 0.5)


@pytest.mark.parametrize("case", acceptance.SENSOR_CASES)
def test_sensor_missing_and_recovery_through_real_world(project_config, case) -> None:
    """range/FOV/遮挡/dropout/延迟与真实检测、独立预测参考一致。"""
    result = acceptance.run_sensor_case(project_config, case)
    assert result["status"] == "PASS" and result["sensor_records"]


def test_small_evidence_preserves_trajectories_events_and_sensor_records(
    project_config, tmp_path,
) -> None:
    """结果包含可核对日志而非单独PASS；重复写入不得覆盖历史。"""
    cases = [acceptance.run_event_case(project_config, "crossing"),
             acceptance.run_sensor_case(project_config, "occlusion_release")]
    report = dict(status="PASS", cases=cases)
    acceptance.write_evidence(tmp_path, report)
    data = json.loads((tmp_path / "stage1_acceptance.json").read_text(encoding="utf-8"))
    assert len(data["cases"]) == 2
    assert (tmp_path / "trajectories.csv").stat().st_size > 100
    assert (tmp_path / "sensor_records.csv").stat().st_size > 100
    events = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    assert events[0]["reason"] == "collision"
    with pytest.raises(FileExistsError):
        acceptance.write_evidence(tmp_path, report)
