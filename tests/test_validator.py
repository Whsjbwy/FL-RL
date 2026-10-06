"""验证 Eq. (55)–(58) 的候选构造和在线验证基本行为。"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.safety.candidates import build_candidate_actions
from auv_risk_rl.safety.validator import validate_nominal_action
from auv_risk_rl.types import AUVState, ControlCommand


def test_candidate_library_contains_45_grid_plus_dynamic_actions(project_config) -> None:
    """
    验证固定网格为 3×5×3=45，并在名义/上一动作与网格不重复时额外加入二者。
    """

    nominal_action = ControlCommand(0.77, 0.123, -0.111)
    previous_action = ControlCommand(0.88, -0.123, 0.111)
    candidates = build_candidate_actions(
        nominal_action,
        previous_action,
        project_config.validator,
    )
    expected_count = 47
    actual_count = len(candidates)
    assert actual_count == expected_count, (
        f"expected={expected_count}, actual={actual_count}, error={actual_count - expected_count}"
    )


def test_validator_preserves_safe_nominal_action(project_config) -> None:
    """
    验证无障碍且操作边界满足时，验证器不应无故修改名义动作。

    这是 Eq. (57)–(58) 的最基本行为约束，可防止候选选择器在安全场景中引入额外控制偏差。
    """

    auv_state = AUVState(
        position_ned_m=np.array([50.0, 50.0, 20.0], dtype=np.float64),
        yaw_rad=0.0,
        pitch_rad=0.0,
        surge_speed_mps=0.8,
        yaw_rate_rad_s=0.0,
        pitch_rate_rad_s=0.0,
    )
    nominal_action = ControlCommand(0.9, 0.0, 0.0)
    previous_action = ControlCommand(0.8, 0.0, 0.0)
    decision = validate_nominal_action(
        auv_state=auv_state,
        current_timestamp_s=0.0,
        nominal_action=nominal_action,
        previous_executed_action=previous_action,
        tracked_obstacles=[],
        config=project_config,
    )
    assert decision.decision_type == "nominal", (
        f"expected nominal, actual={decision.decision_type}, reason={decision.reason}"
    )
    assert decision.executed_action == nominal_action, (
        f"expected={nominal_action}, actual={decision.executed_action}"
    )
