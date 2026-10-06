"""CM04/CM05：数值异常边界与冻结 fallback 规则回归。"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from auv_risk_rl.safety import validator
from auv_risk_rl.types import AUVState, ControlCommand, KFTrackState, TrackedObstacle


def _state(position=(50.0, 50.0, 20.0), speed=0.8) -> AUVState:
    """构造诊断状态，不读取训练或测试场景池。"""

    return AUVState(np.array(position), 0.0, 0.0, speed, 0.0, 0.0)


def test_invalid_covariance_enters_unverified_fallback(project_config, caplog) -> None:
    """PA03：协方差异常标记无效，按公共入口回退并记录每个候选异常。"""

    track = KFTrackState(
        1, np.array([70.0, 50.0, 20.0, 0.2, 0.0, 0.0]),
        np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 0.0, 0.0,
    )
    action = ControlCommand(0.8, 0.0, 0.0)
    with caplog.at_level(logging.WARNING, logger=validator.__name__):
        evaluation = validator.evaluate_candidate_action(
            _state(), 0.0, action, 17, [TrackedObstacle(track, 0.5)], project_config,
        )
        result = validator.validate_nominal_action(
            _state(), 0.0, action, action, [TrackedObstacle(track, 0.5)], project_config,
        )
    assert evaluation.evaluation_status == "numerical_invalid"
    assert not evaluation.passes_validator
    assert result.decision_type == "fallback"
    assert "unverified" in result.reason
    assert result.selected_untruncated_union_bound == float("inf")
    record = next(r for r in caplog.records if r.candidate_id == 17)
    assert record.exception_type == "InvalidCovarianceError"
    assert record.reason
    assert record.component == "safety.validator"


@pytest.mark.parametrize("error_type", [ValueError, TypeError, RuntimeError])
def test_validator_does_not_swallow_programmer_errors(
    project_config, monkeypatch, error_type
) -> None:
    """非约定风险异常必须上抛，不允许用宽泛捕获隐藏程序错误。"""

    def broken_risk(*args, **kwargs):
        """注入明确的程序错误，不构造伪安全结果。"""
        raise error_type("programmer diagnostic")

    monkeypatch.setattr(validator, "_calculate_rollout_risk", broken_risk)
    action = ControlCommand(0.8, 0.0, 0.0)
    with pytest.raises(error_type, match="programmer diagnostic"):
        validator.validate_nominal_action(_state(), 0.0, action, action, [], project_config)


def test_backup_fallback_uses_lexicographic_rule(project_config) -> None:
    """PA04：重现临界边界反例，独立枚举原 backup 字典序。"""

    from auv_risk_rl.dynamics.auv_kinematics import rollout_constant_command
    from auv_risk_rl.env.world import _operation_boundary_fraction
    from auv_risk_rl.safety.candidates import build_candidate_actions

    initial = _state((98.8, 50.0, 20.0), 1.5)
    nominal = ControlCommand(1.5, 0.0, 0.0)
    candidates = build_candidate_actions(nominal, nominal, project_config.validator)
    ranking = []
    for candidate_id, action in enumerate(candidates):
        states = rollout_constant_command(
            initial, action, np.zeros(3), project_config.validator.validation_horizon_s,
            project_config.dynamics.integration_dt_s, project_config.dynamics,
        )
        violation_time = None
        for m, (start, end) in enumerate(zip(states, states[1:], strict=False)):
            fraction = _operation_boundary_fraction(start, end, project_config)
            if fraction is not None:
                violation_time = (m + fraction) * project_config.dynamics.integration_dt_s
                break
        assert violation_time is not None
        normalized = np.array([
            (action.surge_speed_command_mps - 0.9) / 0.6,
            action.yaw_rate_command_rad_s / 0.35,
            action.pitch_rate_command_rad_s / 0.25,
        ])
        distance = float(np.linalg.norm(normalized - np.array([1.0, 0.0, 0.0])))
        ranking.append(((-violation_time, 0.0, distance, candidate_id), action))
    expected = min(ranking, key=lambda item: item[0])[1]
    actual = validator.validate_nominal_action(initial, 0.0, nominal, nominal, [], project_config)
    assert actual.executed_action == expected == ControlCommand(0.9, -0.35, -0.25)
    assert actual.selected_untruncated_union_bound == 0.0
    assert actual.decision_type == "fallback"
    assert "backup" in actual.reason and "unverified" in actual.reason
    selected = validator.evaluate_candidate_action(
        initial, 0.0, expected, 1, [], project_config,
    )
    assert selected.evaluation_status == "deterministic_boundary_violating"
    assert selected.first_hard_constraint_violation_time_s > 0.0


def test_backup_order_uses_unclipped_risk_distance_and_id(project_config) -> None:
    """手算排序键覆盖U大于1、名义差、编号与数值无效剔除。"""

    from auv_risk_rl.types import RiskResult

    nominal = ControlCommand(0.9, 0.0, 0.0)

    def candidate(cid, time_s, upper_sum, yaw=0.0, valid=True):
        """构造选择函数的输入记录，不替代真实 rollout 集成测试。"""
        return validator.CandidateEvaluation(
            ControlCommand(0.9, yaw, 0.0), cid, False,
            RiskResult(min(1.0, upper_sum), upper_sum, valid, "unit"),
            False, time_s,
        )

    items = [candidate(0, 1.0, 0.0), candidate(1, 2.0, 4.0),
             candidate(2, 2.0, 3.0, 0.1), candidate(4, 2.0, 3.0),
             candidate(3, 2.0, 3.0), candidate(5, 5.0, 0.0, valid=False)]
    selected = validator._select_backup_fallback_evaluation(items, nominal, project_config)
    assert selected.candidate_id == 3


def test_primary_fallback_excludes_invalid_predictions(project_config) -> None:
    """确定性可行不等于数值有效，异常预测不能进入主fallback排序。"""

    from auv_risk_rl.types import RiskResult

    action = ControlCommand(0.9, 0.0, 0.0)
    bad = validator.CandidateEvaluation(action, 0, True, RiskResult(0.0, 0.0, False, "bad"), False)
    good = validator.CandidateEvaluation(action, 1, True, RiskResult(1.0, 2.0, True, "ok"), False)
    assert validator._select_primary_fallback_evaluation([bad], action, project_config) is None
    selected = validator._select_primary_fallback_evaluation([bad, good], action, project_config)
    assert selected.candidate_id == 1
    assert selected.evaluation_status == "risk_invalid"
