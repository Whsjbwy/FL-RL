"""CM06：整个实际执行区间的最小净间距，不含终止后的假想路径。"""

from __future__ import annotations

import numpy as np
import pytest

from auv_risk_rl.env.geometry import moving_sphere_minimum_clearance
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.types import AUVState, ControlCommand, GroundTruthObstacleState


def _initial() -> AUVState:
    """固定稳态直行初值，便于独立解析核对。"""

    return AUVState(np.array([50.0, 50.0, 20.0]), 0.0, 0.0, 0.3, 0.0, 0.0)


def test_world_reports_whole_interval_minimum_clearance(project_config) -> None:
    """PA05：最小值发生在早期小步，最后小步不能覆盖它。"""

    obstacle = GroundTruthObstacleState(
        1, np.array([50.03, 51.75, 20.0]), np.array([-0.8, 0.0, 0.0]), 0.5,
    )
    initial = _initial()
    result = AUVWorld(project_config, initial, (obstacle,), np.array([90.0, 90.0, 20.0])).step(
        ControlCommand(0.3, 0.0, 0.0)
    )
    reference, _ = moving_sphere_minimum_clearance(
        initial.position_ned_m, result.auv_state.position_ned_m,
        obstacle.position_ned_m, result.obstacle_states[0].position_ned_m, 1.25,
    )
    assert result.event.reason == "none"
    assert result.event.minimum_clearance_m == pytest.approx(0.5, abs=1e-12)
    assert result.event.minimum_clearance_m == pytest.approx(reference, abs=1e-12)


def test_collision_clearance_excludes_unexecuted_suffix(project_config) -> None:
    """PA06：首次接触处停止时，未执行的小步后缀不能产生负实际净间距。"""

    obstacle = GroundTruthObstacleState(
        3, np.array([51.3, 50.0, 20.0]), np.array([-0.8, 0.0, 0.0]), 0.5,
    )
    result = AUVWorld(project_config, _initial(), (obstacle,), np.array([90.0, 90.0, 20.0])).step(
        ControlCommand(0.3, 0.0, 0.0)
    )
    assert result.event.reason == "collision"
    assert result.timestamp_s == pytest.approx(0.05 / 1.1, abs=1e-12)
    assert result.event.minimum_clearance_m == pytest.approx(0.0, abs=1e-12)


def test_success_clearance_excludes_future_collision_suffix(project_config) -> None:
    """同一小步先到达目标时，之后假想碰撞的负净间距不能计入。"""

    obstacle = GroundTruthObstacleState(
        3, np.array([51.3, 50.0, 20.0]), np.array([-0.8, 0.0, 0.0]), 0.5,
    )
    result = AUVWorld(project_config, _initial(), (obstacle,), np.array([52.006, 50.0, 20.0])).step(
        ControlCommand(0.3, 0.0, 0.0)
    )
    assert result.event.reason == "success"
    assert result.timestamp_s == pytest.approx(0.02, abs=1e-11)
    assert result.event.minimum_clearance_m == pytest.approx(0.028, abs=1e-11)


def test_terminal_event_retains_earlier_substep_clearance(project_config) -> None:
    """后续小步到达目标时，仍须保留更早小步的实际最小值。"""

    obstacle = GroundTruthObstacleState(
        1, np.array([50.03, 51.75, 20.0]), np.array([-0.8, 0.0, 0.0]), 0.5,
    )
    result = AUVWorld(project_config, _initial(), (obstacle,), np.array([52.045, 50.0, 20.0])).step(
        ControlCommand(0.3, 0.0, 0.0)
    )
    assert result.event.reason == "success"
    assert result.timestamp_s > 0.1
    assert result.event.minimum_clearance_m == pytest.approx(0.5, abs=1e-12)
