"""R2只读状态诊断；真实短接口对照与合成选择夹具，零SAC更新。"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from auv_risk_rl.env.scenario_generator import load_training_scenario_config
from auv_risk_rl.training.fixed_validation import FixedValidationPool
from auv_risk_rl.training.harness import states_equal
from auv_risk_rl.training.r2_registration import APPROVED
from auv_risk_rl.training.repair_r2 import (
    _observe_r2_world_step,
    harness_config,
    new_r2_episode_diagnostics,
    r2_diagnostic_step,
    r2_diagnostic_validation,
)
from auv_risk_rl.training.scenarios import B0ScenarioSource
from auv_risk_rl.types import AUVState, ControlCommand


def test_r2_observer_preserves_real_physics_and_sensor_rng(project_config: Any) -> None:
    """同初态/动作的八个真实正式转移精确一致；诊断不能增加Actor调用或修改物理。"""
    path = Path(__file__).parents[1] / 'configs/train_scenario_v1.yaml'
    config = harness_config(APPROVED, 'G200', 11)
    source = B0ScenarioSource(config, project_config, load_training_scenario_config(path))
    scenario = source.scenario(0)
    plain, observed = source.make_env(scenario), source.make_env(scenario)
    for env in (plain, observed):
        env.reset(seed=source.environment_seed(0))
    diagnostic = new_r2_episode_diagnostics(observed)
    for _ in range(4):
        action = np.array([.1, .02, .01])
        reference = plain.step(action)
        actual = r2_diagnostic_step(observed, diagnostic, action, observed.step)
        assert states_equal(reference, actual)
        assert states_equal(vars(plain), vars(observed))
        assert '_propose_integration_step' not in vars(observed.world)
    row = diagnostic['_r2_trajectory'][-1]
    assert len(row['state_ned_8d']) == 8 and row['state_basis'] == 'ACTUAL_CONTROL_NODE'
    assert row['state_ned_8d'][3:5] == [observed.world.auv_state.yaw_rad,
                                      observed.world.auv_state.pitch_rad]
    np.testing.assert_array_equal(row['action'], action)
    assert row['physical_command']['yaw_rate_command_rad_s'] \
        == observed.previous_command.yaw_rate_command_rad_s


@pytest.mark.parametrize('fraction,minimum,first5,first3', [
    (.4, 6.0, None, None), (.8, 2.0, .025, .035),
])
def test_closest_state_and_radius_entries_use_only_executed_prefix(
    project_config: Any, fraction: float, minimum: float,
    first5: float | None, first3: float | None,
) -> None:
    """解析10m直线proposal夹具：事件后未执行的5m/3m进入不得被记录。"""
    start = AUVState(np.zeros(3), 0., 0., .3, 0., 0.)
    end = replace(start, position_ned_m=np.array([10., 0., 0.]), yaw_rad=.2)
    proposal = SimpleNamespace(auv_end_state=end, event=SimpleNamespace(fraction=fraction))
    command = ControlCommand(.3, 0., 0.)

    def original(*_: Any) -> Any:
        """返回原对象哨兵，无真实世界推进。"""
        return proposal

    world = SimpleNamespace(auv_state=start, timestamp_s=0., _propose_integration_step=original)
    env = SimpleNamespace(world=world, config=project_config,
                          goal_position_ned_m=np.array([10., 0., 0.]))
    diagnostic = new_r2_episode_diagnostics(env)
    with _observe_r2_world_step(env, diagnostic):
        assert world._propose_integration_step(start, (), command, 0.) is proposal
    assert world._propose_integration_step is original
    assert diagnostic['_r2_minimum_state']['goal_distance_m'] == minimum
    assert diagnostic['_r2_minimum_state']['state_ned_8d'][0] == 10*fraction
    assert diagnostic['_r2_minimum_state']['state_ned_8d'][3] == pytest.approx(.2*fraction)
    for key, expected in (('first_within_5m_time_s', first5),
                          ('first_within_3m_time_s', first3)):
        if expected is None:
            assert diagnostic[key] is None
        else:
            assert diagnostic[key] == pytest.approx(expected, rel=0., abs=1e-12)


class DiagnosticEnvFixture:
    """单节点合成episode，仅检查日志选择/方法恢复，不代表真实物理事件。"""

    def __init__(self, scenario: Any, project: Any) -> None:
        """同一接口使用明确的静态合成状态，不建立另一套环境模型。"""
        self.scenario, self.config = scenario, project
        self.goal_position_ned_m = np.array([10., 0., 0.])

    def reset(self) -> None:
        """仅初始化夹具，不执行真实合法warm-up或计为环境采样。"""
        state = AUVState(np.zeros(3), 0., 0., .3, 0., 0.)
        self.world = SimpleNamespace(auv_state=state, timestamp_s=1.)
        self.world._propose_integration_step = lambda *_: SimpleNamespace(
            auv_end_state=replace(state, position_ned_m=np.array([1., 0., 0.])), event=None)

    def step(self, action: np.ndarray) -> Any:
        """仅返回一条声明为mock的完成事件；不调用动力学或任何Agent。"""
        command = ControlCommand(.3, 0., 0.)
        proposal = self.world._propose_integration_step(
            self.world.auv_state, (), command, self.world.timestamp_s)
        self.world.auv_state = proposal.auv_end_state
        self.world.timestamp_s += .2
        event = 'goal_success' if self.scenario.scenario_index >= 3 else 'collision'
        return np.zeros(234, np.float32), .1, True, False, dict(
            executed_action_physical=command, task_control_step=1, elapsed_s=.2,
            minimum_clearance=float('inf'), reward_components={'progress': .1}, failure_type=event)


class DiagnosticPoolFixture:
    """五例纯mock池证明只选0/1/2和最早成功/失败；没有真实Val300。"""

    TRAJECTORY_INDICES = (0, 1, 2)
    _training_state = staticmethod(FixedValidationPool._training_state)
    _rng_state = staticmethod(FixedValidationPool._rng_state)

    def __init__(self, fail: bool) -> None:
        """冻结成功/异常路径，不按结果重复尝试。"""
        self.fail = fail

    def evaluate(self, harness: Any, *_: Any) -> dict[str, Any]:
        """每例一次动作哨兵，不额外采样Actor、写Replay或改变训练发行。"""
        episodes = []
        for index in range(5):
            scenario = SimpleNamespace(scenario_id=f'fixture-{index}', scenario_index=index)
            env = harness._make_env(scenario)
            env.reset()
            result = env.step(np.zeros(3))
            if self.fail:
                raise RuntimeError('fixture evaluation failure')
            episodes.append(dict(scenario_id=scenario.scenario_id, index=index,
                                 failure_type=result[4]['failure_type']))
        return dict(fixture_only=True, episodes=episodes)


@pytest.mark.parametrize('fail', [False, True])
def test_r2_evaluation_selected_states_and_finally_restores_all_methods(
    project_config: Any, fail: bool,
) -> None:
    """成功或异常均恢复工厂/reset/step；未选例只留标量，训练状态精确不变。"""
    environments = []
    harness = SimpleNamespace(source=SimpleNamespace(cursor=7), failure_metadata=None)
    harness.state_dict = lambda: deepcopy(dict(training_marker=np.array([2, 5]), transition=0))

    def factory(scenario: Any) -> DiagnosticEnvFixture:
        """只构造显式模拟环境，保存引用供方法恢复断言。"""
        env = DiagnosticEnvFixture(scenario, project_config)
        environments.append(env)
        return env

    harness._make_env = factory
    before = DiagnosticPoolFixture._training_state(harness)
    if fail:
        with pytest.raises(RuntimeError, match='fixture evaluation failure'):
            r2_diagnostic_validation(DiagnosticPoolFixture(True), harness, False)
    else:
        result = r2_diagnostic_validation(DiagnosticPoolFixture(False), harness, False)
        assert result['r2_diagnostic_state_unchanged']
        assert [row['index'] for row in result['episodes'] if 'trajectory' in row] == [0, 1, 2, 3]
        assert result['episodes'][3]['trajectory_retention_reasons'] \
            == ['earliest_success_by_index']
        assert 'minimum_goal_state' not in result['episodes'][4]
        assert all(len(row['trajectory'][0]['state_ned_8d']) == 8
                   for row in result['episodes'][:4])
    assert harness._make_env is factory
    assert states_equal(before, DiagnosticPoolFixture._training_state(harness))
    assert all('reset' not in vars(env) and 'step' not in vars(env) for env in environments)
