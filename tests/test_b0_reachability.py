"""B0可达性登记及控制律单测；不重复执行六个完整非学习诊断。"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from auv_risk_rl.env.local_task import command_to_normalized
from auv_risk_rl.types import AUVState

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_b0_reachability.py"
_SPEC = importlib.util.spec_from_file_location("b0_reachability", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
reachability = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = reachability
_SPEC.loader.exec_module(reachability)


def _level_state() -> AUVState:
    """构造控制律解析测试状态，零姿态且0.3m/s；不创建真实世界。"""

    return AUVState(np.array([20.0, 50.0, 20.0]), 0.0, 0.0, 0.3, 0.0, 0.0)


def test_predeclared_inventory_and_attempt_budget(project_config) -> None:
    """六个固定案例先无障碍后CV，种子/上限/控制参数不由运行结果选择。"""

    cases = reachability.fixed_cases()
    assert [case.case_id for case in cases] == [
        "R01_horizontal_empty", "R02_deeper_empty", "R03_shallower_empty",
        "R04_head_on_lateral_offset", "R05_crossing", "R06_head_on_vertical_offset",
    ]
    assert [case.seed for case in cases] == list(range(810001, 810007))
    assert reachability.MAX_TRANSITIONS_PER_CASE == 1000
    assert reachability.SURGE_COMMAND_MPS == 1.2
    assert reachability.LOS_GAIN_PER_SECOND == 1.0
    assert [len(case.obstacles()) for case in cases] == [0, 0, 0, 1, 1, 1]
    for case in cases:
        assert reachability.state_is_legal(case.initial_state(), project_config)
        np.testing.assert_array_equal(case.goal()[:2], [80.0, 50.0])
        assert case.initial_state().surge_speed_mps == 0.3
        for obstacle in case.obstacles():
            assert np.linalg.norm(obstacle.velocity_ned_mps) == pytest.approx(0.4)
            assert obstacle.radius_m == 0.6


@pytest.mark.parametrize("down_delta, expected_sign", [(16.0, -1), (-16.0, 1)])
def test_ned_depth_command_uses_correct_pitch_sign(project_config, down_delta,
                                                  expected_sign) -> None:
    """NED更深目标产生负pitch-rate，更浅目标产生正pitch-rate。"""

    state = _level_state()
    goal = state.position_ned_m + np.array([60.0, 0.0, down_delta])
    command = reachability.los_command(state, goal, project_config)
    assert np.sign(command.pitch_rate_command_rad_s) == expected_sign
    assert command.yaw_rate_command_rad_s == 0
    assert command.surge_speed_command_mps == 1.2


def test_yaw_error_wraps_without_wrong_full_turn(project_config) -> None:
    """跨±pi的误差用atan2(sin,cos)，同一短旋转应为正0.1rad/s。"""

    yaw = math.pi - 0.05
    state = AUVState(np.array([50.0, 50.0, 20.0]), yaw, 0.0, 0.3, 0.0, 0.0)
    goal = state.position_ned_m + 10 * np.array(
        [math.cos(-math.pi + 0.05), math.sin(-math.pi + 0.05), 0.0])
    command = reachability.los_command(state, goal, project_config)
    assert command.yaw_rate_command_rad_s == pytest.approx(0.1, rel=0, abs=1e-14)


def test_controller_is_bounded_and_does_not_modify_state(project_config) -> None:
    """控制律限制合法指令，没有修改真实速度、位置或姿态的捷径。"""

    state = _level_state()
    before = state.position_ned_m.copy()
    command = reachability.los_command(state, np.array([-80.0, 50.0, 200.0]), project_config)
    action = command_to_normalized(command, project_config.dynamics)
    assert np.all(np.isfinite(action)) and np.all(np.abs(action) <= 1)
    np.testing.assert_array_equal(state.position_ned_m, before)
    assert state.surge_speed_mps == 0.3 and state.pitch_rad == 0


def test_invalid_goal_is_rejected(project_config) -> None:
    """非法输入直接失败，不生成虚假合法控制指令。"""

    with pytest.raises(ValueError, match="有限"):
        reachability.los_command(_level_state(), np.array([np.nan, 1.0, 2.0]), project_config)


def test_physical_envelope_is_checked_without_clipping(project_config) -> None:
    """操作限制检查完整AUV球包络；错误状态没有被clip回盒内。"""

    state = AUVState(np.array([0.5, 50.0, 20.0]), 0.0, 0.0, 0.3, 0.0, 0.0)
    assert not reachability.state_is_legal(state, project_config)
    assert state.position_ned_m[0] == 0.5


def test_evidence_keeps_null_clearance_and_refuses_overwrite(tmp_path) -> None:
    """无障碍不适用净间距保持null；此夹具仅测序列化，不伪称真实可达结果。"""

    case = dict(case_id="serialization_fixture", status="NOT RUN", witness_found=False,
                trajectories=[dict(case_id="serialization_fixture", timestamp_s=1.0)],
                obstacle_trajectories=[], minimum_clearance_m=None)
    report = dict(status="NOT RUN", cases=[case], scientific_training_steps=0)
    reachability.write_evidence(tmp_path, report)
    saved = json.loads((tmp_path / "reachability.json").read_text(encoding="utf-8"))
    assert saved["cases"][0]["minimum_clearance_m"] is None
    assert saved["cases"][0]["status"] == "NOT RUN"
    assert (tmp_path / "trajectories.csv").exists()
    assert (tmp_path / "obstacles.csv").exists()
    with pytest.raises(FileExistsError):
        reachability.write_evidence(tmp_path, report)
