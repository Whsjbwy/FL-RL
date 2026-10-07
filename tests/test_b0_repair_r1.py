"""R1固定调度/诊断/恢复接口；大计数是显式Mock，不额外跑100k科研训练。"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from auv_risk_rl.env.scenario_generator import load_training_scenario_config
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.training.config import B0HarnessConfig, derived_sac_config
from auv_risk_rl.training.harness import states_equal
from auv_risk_rl.training.repair_r1 import (
    B0RepairHarness,
    _first_ball_entry,
    boundary_labels,
    diagnostic_step,
    diagnostic_validation,
    harness_config,
    new_episode_diagnostics,
    observe_world_step,
)
from auv_risk_rl.training.repair_registration import APPROVED, REGISTRATION_ID
from auv_risk_rl.training.scenarios import B0ScenarioSource
from test_b0_harness import FixtureAgent, FixtureEnv


@pytest.fixture
def scenario_config() -> Any:
    """读取原冻结训练采样配置，不为复测修改分布。"""
    path = Path(__file__).parents[1] / 'configs/train_scenario_v1.yaml'
    return load_training_scenario_config(path)


def mock_evaluation(current: B0RepairHarness, profile: str, full: bool) -> dict[str, Any]:
    """只验证回调身份/调度，不生成真实评价episode或学习结果。"""
    return dict(fixture_only=True, count=300 if full else 30, profile=profile,
                evaluation_env_transitions=0, warmup_control_transitions=0,
                at_transition=current.transitions, full=full)


def make_repair(project: Any, scenario: Any, **kwargs: Any) -> B0RepairHarness:
    """明确无网络梯度的普通调度夹具。"""
    config = replace(harness_config(APPROVED, 11), sac=replace(
        harness_config(APPROVED, 11).sac, device='cpu'))
    callback = kwargs.pop('evaluation_callback', mock_evaluation)
    return B0RepairHarness(config, project, scenario, code_version='unit-r1-fixture',
                           evaluation_callback=callback, agent=FixtureAgent(config.sac),
                           env_factory=FixtureEnv, **kwargs)


def move_mock_counter(harness: B0RepairHarness, transitions: int) -> None:
    """已声明合成计数及过去评价键；不是实际环境或科研执行证据。"""
    harness.transitions = transitions
    harness.agent.counters['environment_steps'] = transitions
    harness.next_validation_transition = ((transitions // 25000)+1)*25000
    harness.next_checkpoint_transition = ((transitions // 25000)+1)*25000
    for point in (0, 25000, 50000, 75000):
        if point <= transitions:
            harness.completed_validation_keys.add(f'obstacle_free:{point}:monitor')
    harness.validation_count = len(harness.completed_validation_keys)


@pytest.mark.parametrize('seed', [11, 22, 33])
def test_r1_changes_only_public_learning_rate_and_preserves_effective_seeds(seed: int) -> None:
    """run_id/登记不进入随机命名空间；生产默认lr仍3e-4。"""
    config = harness_config(APPROVED, seed)
    original = derived_sac_config(SACConfig(device='cuda'), training_seed=seed,
                                  run_kind='scientific_training')
    assert replace(config.sac, learning_rate=0.0003) == original
    assert SACConfig().learning_rate == 0.0003
    assert config.transition_budget == 100000 and config.task_profile == 'obstacle_free'
    assert config.sac.learning_starts == 10000 and config.sac.batch_size == 256


def test_r1_validation_schedule_idempotence(project_config: Any, scenario_config: Any) -> None:
    """固定0/25k/50k/75k/100k点，只用mock，不提前运行科研大预算。"""
    calls = []

    def callback(current: B0RepairHarness, profile: str, full: bool) -> dict[str, Any]:
        """只记录Mock评价调度，不推进环境。"""
        calls.append((current.transitions, profile, full))
        return mock_evaluation(current, profile, full)

    current = make_repair(project_config, scenario_config, evaluation_callback=callback)
    for point in (0, 25000, 50000, 75000, 100000):
        current.transitions = point
        current.ensure_validation()
        before = current.state_dict()
        assert current.ensure_validation()['already_completed']
        assert states_equal(before, current.state_dict())
    assert calls == [(point, 'obstacle_free', point == 100000)
                     for point in (0, 25000, 50000, 75000, 100000)]
    current.transitions = 100001
    with pytest.raises(ValueError, match='登记'):
        current.ensure_validation()


def test_r1_100k_budget_stops_without_false_termination(
    project_config: Any, scenario_config: Any,
) -> None:
    """两次Mock步恰至预算，片段不伪造success/timeout，不多跑到episode结束。"""
    current = make_repair(project_config, scenario_config)
    move_mock_counter(current, 99998)
    current.step()
    current.step()
    current.record_final_budget_stop()
    assert current.transitions == 100000 and len(current.agent.replay) == 2
    assert current.agent.counters['gradient_updates'] == 0
    assert all(not row['complete'] and row['budget_stop'] and row['failure_type'] == 'none'
               for row in current.episode_log)
    assert not current.agent.replay.at(np.array([0, 1]))['terminated'].any()
    assert not current.agent.replay.at(np.array([0, 1]))['truncated'].any()
    before = current.state_dict()
    current.record_final_budget_stop()
    assert states_equal(before, current.state_dict())
    with pytest.raises(RuntimeError, match='预算'):
        current.step()


def test_r1_mid_episode_full_checkpoint_exact_mock_resume(
    project_config: Any, scenario_config: Any, tmp_path: Path,
) -> None:
    """完整Agent/Replay/环境/RNG及新增诊断字段同设备精确恢复；无神经更新。"""
    left = make_repair(project_config, scenario_config)
    left.ensure_validation()
    for _ in range(6):
        left.step()
    path = tmp_path / 'trusted_r1_fixture.pt'
    left.save_checkpoint(path)
    right = make_repair(project_config, scenario_config)
    right.load_checkpoint(path, trusted_local=True)
    assert states_equal(left.state_dict(), right.state_dict())
    for _ in range(4):
        left.step()
        right.step()
    assert states_equal(left.state_dict(), right.state_dict())
    assert left.next_scenario_index == right.next_scenario_index == 2


@pytest.mark.parametrize('fault', ['future', 'duplicate', 'missing', 'rate', 'endpoint', 'budget'])
def test_r1_damaged_resume_rejected_before_state_mutation(
    project_config: Any, scenario_config: Any, fault: str,
) -> None:
    """错误验证位置/修复身份不会先改变Actor/Replay再发现不一致。"""
    current = make_repair(project_config, scenario_config)
    current.ensure_validation()
    current.step()
    before = current.state_dict()
    damaged = deepcopy(before)
    repair = damaged['repair']
    if fault == 'future':
        repair['completed_validation_keys'].append('obstacle_free:25000:monitor')
    elif fault == 'duplicate':
        repair['completed_validation_keys'].append('obstacle_free:0:monitor')
    elif fault == 'missing':
        repair['completed_validation_keys'] = []
    elif fault == 'rate':
        repair['learning_rate'] = 0.0003
    elif fault == 'endpoint':
        repair['endpoint'] = 300000
    else:
        repair['final_budget_logged'] = True
    with pytest.raises(ValueError, match='R1恢复'):
        current.load_state_dict(damaged)
    assert states_equal(before, current.state_dict())


@pytest.mark.parametrize('fault', ['rng', 'count', 'exception'])
def test_r1_validation_fault_preserves_failure_metadata(
    project_config: Any, scenario_config: Any, fault: str,
) -> None:
    """不恢复RNG掩盖误消耗，失败回调不得记成完成验证。"""
    def callback(current: B0RepairHarness, profile: str, full: bool) -> dict[str, Any]:
        """注入明确的评价隔离反例，不执行真实学习。"""
        row = mock_evaluation(current, profile, full)
        if fault == 'rng':
            current.agent.rng.uniform()
        elif fault == 'count':
            row['count'] = 29
        else:
            current.failure_metadata = dict(evaluation_env_transitions=7, checkpoint_safe=False)
            raise RuntimeError('fixture fault')
        return row

    current = make_repair(project_config, scenario_config, evaluation_callback=callback)
    with pytest.raises((RuntimeError, ValueError)):
        current.ensure_validation()
    assert not current.completed_validation_keys and current.failure_metadata is not None
    if fault == 'exception':
        assert current.failure_metadata['evaluation_env_transitions'] == 7
    with pytest.raises(RuntimeError, match='无失败'):
        current.state_dict()


def test_read_only_episode_observer_keeps_actual_environment_exact(
    project_config: Any, scenario_config: Any,
) -> None:
    """真实无学习16控制步对照：proposal对象原样返回，观察不改变转移/随机流。"""
    config = B0HarnessConfig(run_kind='scientific_training', transition_budget=100000,
                             research_registration=REGISTRATION_ID)
    source = B0ScenarioSource(config, project_config, scenario_config)
    scenario = source.scenario(0)
    plain, observed = source.make_env(scenario), source.make_env(scenario)
    for env in (plain, observed):
        env.reset(seed=source.environment_seed(0), options={'external_max_steps': None})
    diagnostic = new_episode_diagnostics(observed)
    for action in (np.array([0.2, 0.02, 0.01]),)*8:
        reference = plain.step(action)
        actual = diagnostic_step(observed, diagnostic, action, observed.step)
        assert states_equal(reference, actual)
        assert states_equal(vars(plain), vars(observed))
        assert '_propose_integration_step' not in vars(observed.world)
        if reference[2] or reference[3]:
            break
    assert diagnostic['minimum_goal_distance_m'] <= diagnostic['initial_distance_m']
    assert diagnostic['first_goal_entry_time_s'] is None


def test_independent_ball_entry_and_pitch_boundary_labels(project_config: Any) -> None:
    """可手算球进入1/6及线性pitch边界，不调用生产几何作参考。"""
    fraction = _first_ball_entry(np.array([-3.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]),
                                 np.zeros(3), 2.0)
    assert fraction == pytest.approx(1.0/6.0, abs=1.0e-10)
    from auv_risk_rl.types import AUVState

    start = AUVState(position_ned_m=np.array([50.0, 50.0, 20.0]), yaw_rad=0.0,
                     pitch_rad=project_config.dynamics.max_pitch_rad-0.01,
                     surge_speed_mps=1.0, yaw_rate_rad_s=0.0, pitch_rate_rad_s=0.0)
    end = replace(start, pitch_rad=project_config.dynamics.max_pitch_rad+0.01)
    assert boundary_labels(start, end, project_config) == ['pitch_upper']


def test_diagnostic_goal_entry_survives_event_prefix_roundoff(project_config: Any) -> None:
    """合成事件前缀略超球面；日志须用原提议根和已执行fraction，不改世界事件。"""
    start = SimpleNamespace(position_ned_m=np.array([3.0, 0.0, 0.0]))
    end = SimpleNamespace(position_ned_m=np.array([1.0, 0.0, 0.0]))
    fraction = 0.5-1.0e-12
    proposal = SimpleNamespace(auv_end_state=end, event=SimpleNamespace(
        fraction=fraction, event=SimpleNamespace(reason='success')))

    class WorldFixture:
        """只返回预登记标量提议，不执行物理环境或科研transition。"""
        def _propose_integration_step(self, *args: Any) -> Any:
            """返回固定的未提交提议对象。"""
            return proposal

    env = SimpleNamespace(world=WorldFixture(), config=project_config,
                          goal_position_ned_m=np.zeros(3))
    diagnostic = dict(episode_origin_timestamp_s=0.0, minimum_goal_distance_m=3.0,
                      minimum_goal_distance_time_s=0.0, first_within_10m_time_s=0.0,
                      first_goal_entry_time_s=None, boundary_constraints=[], boundary_subtype=None)
    prefix = start.position_ned_m+fraction*(end.position_ned_m-start.position_ned_m)
    assert _first_ball_entry(start.position_ned_m, prefix, np.zeros(3), 2.0) is None
    with observe_world_step(env, diagnostic):
        returned = env.world._propose_integration_step(start, (), None, 0.0)
    assert returned is proposal
    assert diagnostic['first_goal_entry_time_s'] == pytest.approx(
        0.5*project_config.dynamics.integration_dt_s, abs=1.0e-10)


def test_real_environment_diagnostics_restore_exactly_without_learning(
    project_config: Any, scenario_config: Any, tmp_path: Path,
) -> None:
    """Mock Agent配真实B0，共14个非学习step；保存恢复含最小距离/感知/RNG。"""
    config = replace(harness_config(APPROVED, 11), sac=replace(
        harness_config(APPROVED, 11).sac, device='cpu'))
    source = B0ScenarioSource(config, project_config, scenario_config)

    def make() -> B0RepairHarness:
        """相同有效流创建真实环境/无网络Mock Agent。"""
        return B0RepairHarness(config, project_config, scenario_config,
                               code_version='real-env-nonlearning-r1-fixture',
                               evaluation_callback=mock_evaluation,
                               agent=FixtureAgent(config.sac), env_factory=source.make_env)

    left = make()
    left.ensure_validation()
    for _ in range(6):
        left.step()
    path = tmp_path / 'trusted_real_env_no_learning.pt'
    left.save_checkpoint(path)
    right = make()
    right.load_checkpoint(path, trusted_local=True)
    assert states_equal(left.state_dict(), right.state_dict())
    for _ in range(4):
        assert states_equal(left.step(), right.step())
    assert states_equal(left.state_dict(), right.state_dict())
    assert left.agent.counters['gradient_updates'] == right.agent.counters['gradient_updates'] == 0
    for slot in left.slots:
        assert slot['repair_diagnostics']['minimum_goal_distance_m'] is not None
        assert 'step' not in vars(slot['env'])
        assert '_propose_integration_step' not in vars(slot['env'].world)


@pytest.mark.parametrize('fault', [False, True])
def test_diagnostic_validation_factory_restores_and_training_is_unchanged(
    project_config: Any, scenario_config: Any, fault: bool,
) -> None:
    """微型隔离夹具：正常仅两步非学习评价；异常仅暖机，绝非Val300科学结果。"""
    config = replace(harness_config(APPROVED, 11), sac=replace(
        harness_config(APPROVED, 11).sac, device='cpu'))
    source = B0ScenarioSource(config, project_config, scenario_config)
    current = B0RepairHarness(config, project_config, scenario_config, code_version='unit-r1',
                              evaluation_callback=mock_evaluation,
                              agent=FixtureAgent(config.sac), env_factory=source.make_env)
    before = current.state_dict()

    class SmallPoolFixture:
        """直接检查临时验证工厂，不伪造30/300个真实episode。"""
        def evaluate(self, harness: B0RepairHarness, profile: str, full: bool,
                     progress_callback: Any) -> dict[str, Any]:
            """返回一个明确的微型工程结果，异常分支只检查清理。"""
            scenario = source.scenario(0)
            env = harness._make_env(scenario)
            env.reset(seed=source.environment_seed(0), options={'external_max_steps': None})
            if fault:
                raise RuntimeError('fixture validation fault')
            for _ in range(2):
                env.step(np.array([0.2, 0.02, 0.01]))
            return dict(fixture_only=True, episodes=[dict(scenario_id=scenario.scenario_id)])

    if fault:
        with pytest.raises(RuntimeError, match='fixture'):
            diagnostic_validation(SmallPoolFixture(), current, False)
    else:
        result = diagnostic_validation(SmallPoolFixture(), current, False)
        assert result['fixture_only'] is True and len(result['episodes']) == 1
        assert result['episodes'][0]['minimum_goal_distance_m'] is not None
    assert '_make_env' not in vars(current)
    assert states_equal(before, current.state_dict())
    assert len(current.agent.replay) == 0 and current.next_scenario_index == 0
