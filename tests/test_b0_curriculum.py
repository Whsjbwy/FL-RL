"""固定MVP课程与验证边界的纯调度夹具；不执行真实环境或神经更新。"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from auv_risk_rl.env.scenario_generator import load_training_scenario_config
from auv_risk_rl.training.config import B0HarnessConfig
from auv_risk_rl.training.curriculum import B0CurriculumHarness
from auv_risk_rl.training.harness import states_equal
from test_b0_harness import FixtureAgent, FixtureEnv


@pytest.fixture
def scenario_config() -> Any:
    """读取原冻结采样律，课程不会重新设计生成器。"""
    path = Path(__file__).parents[1] / 'configs/train_scenario_v1.yaml'
    return load_training_scenario_config(path)


def mock_evaluation(current: B0CurriculumHarness, profile: str, full: bool) -> dict[str, Any]:
    """只有回调身份夹具，没有真实评估episode或环境推进。"""
    return dict(fixture_only=True, count=300 if full else 30, profile=profile,
                evaluation_env_transitions=0, warmup_control_transitions=0,
                at_transition=current.transitions, full=full)


def make_course(project: Any, scenario: Any, **kwargs: Any) -> B0CurriculumHarness:
    """明确Mock Agent/Env的课程调度，不运行科研SAC。"""
    config = B0HarnessConfig(run_kind='scientific_training', training_seed=11,
                             transition_budget=300000,
                             research_registration=B0CurriculumHarness.REGISTRATION_ID)
    callback = kwargs.pop('evaluation_callback', mock_evaluation)
    return B0CurriculumHarness(config, project, scenario, code_version='unit-course-fixture',
                               evaluation_callback=callback,
                               agent=FixtureAgent(config.sac), env_factory=FixtureEnv, **kwargs)


def move_synthetic_counter(course: B0CurriculumHarness, transitions: int) -> None:
    """设置已声明的合成计数夹具，不把它当成实际采样证据。"""
    course.transitions = transitions
    course.agent.counters['environment_steps'] = transitions
    course.next_validation_transition = ((transitions // 25000) + 1) * 25000
    course.next_checkpoint_transition = ((transitions // 25000) + 1) * 25000


def test_explicit_switch_keeps_agent_replay_and_flags(
    project_config: Any, scenario_config: Any,
) -> None:
    """100k只记录片段并丢弃槽，不追改Replay终止flags、奖励或训练状态。"""
    course = make_course(project_config, scenario_config)
    move_synthetic_counter(course, 99998)
    course.step()
    course.step()
    assert course.transitions == 100000 and course.current_profile == 'obstacle_free'
    assert course.phase_transition_counts == dict(obstacle_free=100000, cv_train_v1=0)
    before = course.agent.state_dict()
    indices = np.arange(len(course.agent.replay))
    replay_before = course.agent.replay.at(indices)
    with pytest.raises(RuntimeError, match='尚未显式切换'):
        course.step()
    course.switch_to_cv()
    assert states_equal(before, course.agent.state_dict())
    assert states_equal(replay_before, course.agent.replay.at(indices))
    assert all(not row['complete'] and row['phase_boundary'] and not row['task_timeout']
               and not row['external_truncation'] for row in course.episode_log)
    assert [row['at_transition'] for row in course.episode_log] == [100000, 100000]
    assert all(slot['needs_reset'] for slot in course.slots)
    with pytest.raises(RuntimeError, match='monitor'):
        course.step()
    course.ensure_validation('cv_train_v1', False)
    course.step()
    assert course.transitions == 100001 and course.next_scenario_index == 3
    assert course.slots[0]['task_profile'] == 'cv_train_v1'
    assert course.slots[0]['scenario_id'].startswith('train-v1')
    assert course.phase_transition_counts['cv_train_v1'] == 1


def test_switch_requires_exact_boundary_and_full_validation(
    project_config: Any, scenario_config: Any,
) -> None:
    """不会因计数接近或仅monitor记录就越过课程边界。"""
    course = make_course(project_config, scenario_config)
    with pytest.raises(RuntimeError, match='Val300'):
        course.switch_to_cv()
    move_synthetic_counter(course, 100000)
    with pytest.raises(RuntimeError, match='Val300'):
        course.switch_to_cv()
    course.ensure_validation(full=True)
    course.switch_to_cv()
    before = course.state_dict()
    assert course.switch_to_cv()['already_switched']
    assert states_equal(before, course.state_dict())


def test_validation_fixed_schedule_and_idempotence(
    project_config: Any, scenario_config: Any,
) -> None:
    """固定完成点不重复执行，非登记时间不能追加验证。"""
    calls = []

    def callback(current: B0CurriculumHarness, profile: str, full: bool) -> dict[str, Any]:
        """记录调用次数，仍不实际运行评估环境。"""
        calls.append((current.transitions, profile, full))
        return mock_evaluation(current, profile, full)

    course = make_course(project_config, scenario_config, evaluation_callback=callback)
    course.ensure_validation()
    assert course.ensure_validation()['already_completed']
    move_synthetic_counter(course, 25000)
    course.ensure_validation()
    move_synthetic_counter(course, 100000)
    course.ensure_validation()
    course.switch_to_cv()
    course.ensure_validation()
    assert calls == [(0, 'obstacle_free', False), (25000, 'obstacle_free', False),
                     (100000, 'obstacle_free', True), (100000, 'cv_train_v1', False)]
    move_synthetic_counter(course, 100001)
    with pytest.raises(ValueError, match='登记'):
        course.ensure_validation()


@pytest.mark.parametrize('position', ['before_switch', 'after_switch', 'after_first_cv'])
def test_curriculum_checkpoint_continuity(
    project_config: Any, scenario_config: Any, tmp_path: Path, position: str,
) -> None:
    """保存边界/首条CV后的恢复继续有相同场景、计数、RNG和验证完成键。"""
    left = make_course(project_config, scenario_config)
    move_synthetic_counter(left, 99998)
    left.step()
    left.step()
    if position != 'before_switch':
        left.switch_to_cv()
        left.ensure_validation()
    if position == 'after_first_cv':
        left.step()
    path = tmp_path / 'trusted_course.pt'
    left.save_checkpoint(path)
    right = make_course(project_config, scenario_config)
    right.load_checkpoint(path, trusted_local=True)
    for current in (left, right):
        if not current.switched:
            current.switch_to_cv()
            current.ensure_validation()
        current.step()
    assert states_equal(left.state_dict(), right.state_dict())


def test_final_endpoint_stops_exact_budget_and_keeps_partial_flags(
    project_config: Any, scenario_config: Any,
) -> None:
    """300k不会增加第300001条transition或伪造成功/timeout。"""
    course = make_course(project_config, scenario_config)
    move_synthetic_counter(course, 100000)
    course.ensure_validation()
    course.switch_to_cv()
    course.ensure_validation()
    move_synthetic_counter(course, 299999)
    before_updates = course.agent.counters['gradient_updates']
    course.run_until(300000)
    assert course.transitions == 300000
    assert course.agent.counters['gradient_updates'] == before_updates
    assert course.phase_transition_counts == dict(obstacle_free=100000, cv_train_v1=200000)
    assert course.budget_stop_records[0]['complete'] is False
    assert course.budget_stop_records[0]['failure_type'] == 'none'
    assert not bool(course.agent.replay.at(np.array([0]))['terminated'][0])
    assert not bool(course.agent.replay.at(np.array([0]))['truncated'][0])
    before = course.state_dict()
    course.record_final_budget_stop()
    assert states_equal(before, course.state_dict())
    with pytest.raises(RuntimeError, match='预算'):
        course.step()


@pytest.mark.parametrize('fault', ['rng_mutation', 'wrong_count', 'pool_failure'])
def test_validation_mutation_or_wrong_count_fails_closed(
    project_config: Any, scenario_config: Any,
    fault: str,
) -> None:
    """回调改变策略流、案例数不符或池故障均拒绝继续且保留反例。"""
    def bad_callback(current: B0CurriculumHarness, profile: str, full: bool) -> dict[str, Any]:
        """注入已预定的纯调度故障，不运行环境或梯度。"""
        record = mock_evaluation(current, profile, full)
        if fault == 'rng_mutation':
            current.agent.rng.uniform()
        elif fault == 'wrong_count':
            record['count'] = 29
        else:
            current.failure_metadata = dict(current_validation_episode=17,
                                             evaluation_env_transitions=123,
                                             checkpoint_safe=False, trajectory=[{'fixture': True}])
            raise RuntimeError('fixture pool failure')
        return record

    course = make_course(project_config, scenario_config, evaluation_callback=bad_callback)
    with pytest.raises((RuntimeError, ValueError)):
        course.ensure_validation()
    assert not course.completed_validation_keys and course.failure_metadata is not None
    if fault == 'pool_failure':
        assert course.failure_metadata['current_validation_episode'] == 17
        assert course.failure_metadata['evaluation_env_transitions'] == 123
        assert course.failure_metadata['trajectory'] == [{'fixture': True}]
        assert course.failure_metadata['checkpoint_safe'] is False
    with pytest.raises(RuntimeError, match='无失败'):
        course.state_dict()


def test_checkpoint_rejects_conflicting_course_identity(
    project_config: Any, scenario_config: Any,
) -> None:
    """不把空场景状态伪装为CV，也不允许课程分项与全局计数不一致。"""
    course = make_course(project_config, scenario_config)
    state = deepcopy(course.state_dict())
    state['curriculum']['phase'] = 'cv_train_v1'
    state['curriculum']['switched'] = True
    with pytest.raises(ValueError, match='不一致'):
        course.load_state_dict(state)


@pytest.mark.parametrize('damage', ['future_key', 'duplicate_key', 'counts', 'missing_parent_full'])
def test_checkpoint_rejects_damaged_idempotence_before_state_mutation(
    project_config: Any, scenario_config: Any, damage: str,
) -> None:
    """损坏的恢复键和课程分项不能跳过验证，也不能先改变Agent状态再失败。"""
    course = make_course(project_config, scenario_config)
    move_synthetic_counter(course, 100000)
    course.ensure_validation()
    course.switch_to_cv()
    course.ensure_validation()
    before = course.state_dict()
    state = deepcopy(before)
    saved = state['curriculum']
    if damage == 'future_key':
        saved['completed_validation_keys'].append('cv_train_v1:125000:monitor')
    elif damage == 'duplicate_key':
        saved['completed_validation_keys'].append(saved['completed_validation_keys'][0])
    elif damage == 'counts':
        saved['phase_transition_counts']['obstacle_free'] -= 1
    else:
        saved['completed_validation_keys'].remove('obstacle_free:100000:full')
    with pytest.raises(ValueError, match='不一致'):
        course.load_state_dict(state)
    assert states_equal(before, course.state_dict())


def test_course_switch_event_is_emitted_once(project_config: Any, scenario_config: Any) -> None:
    """切换事件可追溯且恢复/重复调用不会再写一遍。"""
    records = []
    course = make_course(project_config, scenario_config,
                         log_sink=lambda kind, record: records.append((kind, record)))
    move_synthetic_counter(course, 100000)
    course.ensure_validation()
    course.switch_to_cv()
    course.switch_to_cv()
    switches = [record for kind, record in records if kind == 'course_switch']
    assert len(switches) == 1
    assert switches[0]['at_transition'] == 100000
    assert switches[0]['retained_agent_optimizer_replay'] is True
