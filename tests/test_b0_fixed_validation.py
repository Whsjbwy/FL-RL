"""固定Val300与验证隔离的纯调度夹具；不运行真实AUV采样或梯度更新。"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.scenario_generator import TrainingScenario, load_training_scenario_config
from auv_risk_rl.runtime import local_perception
from auv_risk_rl.training.fixed_validation import (
    FixedValidationPool,
    paired_validation_sensor_streams,
)
from auv_risk_rl.training.harness import states_equal
from auv_risk_rl.types import AUVState, GroundTruthObstacleState


@pytest.fixture(scope='module')
def scenario_config() -> Any:
    """读取真实分布配置，不为验证池修改TRAIN_SCENARIO_V1支持。"""
    return load_training_scenario_config(Path(__file__).parents[1]
                                         / 'configs/train_scenario_v1.yaml')


@pytest.fixture(scope='module')
def pool(project_config: ProjectConfig, scenario_config: Any) -> FixedValidationPool:
    """只生成300条几何值对象，不reset或step真实环境。"""
    return FixedValidationPool(project_config, scenario_config)


class DeterministicAgentFixture:
    """无网络的确定动作；可显式注入错误消费供隔离反例测试。"""

    def __init__(self) -> None:
        """该随机流属于合成训练状态，验证不能消耗。"""
        self.rng = np.random.default_rng(732)
        self.error: str | None = None

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        """返回合成动作；不调用backward、不写Replay。"""
        if self.error == 'torch':
            torch.rand(1)
        if self.error == 'numpy':
            np.random.random()
        if self.error == 'local_rng':
            self.rng.random()
        if self.error == 'nonfinite':
            return np.array([np.nan, 0.0, 0.0])
        return np.array([0.1, 0.0, 0.0], dtype=np.float32)


class ValidationEnvFixture:
    """每例两个合成节点，不声称为真实物理可达性或学习结果。"""

    def __init__(self, scenario: TrainingScenario, identities: list[tuple[Any, ...]],
                 external: bool = False) -> None:
        """保留构造身份；测试两个profile和重复验证是否用相同固定发行。"""
        self.scenario, self.identities, self.external = scenario, identities, external

    def reset(self, *, seed: int, options: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """禁止把工程最大步数写入科研验证；仅模拟完成一秒warm-up。"""
        assert options == {'external_max_steps': None}
        self.identities.append((self.scenario.scenario_id, self.scenario.scenario_index, seed))
        self.world = SimpleNamespace(auv_state=SimpleNamespace(
            position_ned_m=self.scenario.initial_auv_state.position_ned_m.copy()))
        self.steps = 0
        return np.zeros(234, np.float32), {'warmup_duration_s': 1.0}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """索引5、6固定失败，用于验证只保留升序最早失败轨迹。"""
        self.steps += 1
        self.world.auv_state.position_ned_m += np.array([0.01, 0.0, 0.0])
        ended = self.steps == 2
        failure = 'collision' if self.scenario.scenario_index in (5, 6) else 'goal_success'
        if self.external:
            failure = 'external_truncation'
        return np.zeros(234, np.float32), 0.1, ended and not self.external, \
            ended and self.external, dict(
                task_control_step=self.steps, elapsed_s=0.2, reward_components={'progress': 0.1},
                minimum_clearance=(float('inf') if not self.scenario.initial_obstacle_states
                                   else 2.0), failure_type=failure if ended else 'none')


class ValidationHarnessFixture:
    """直接比较完整模拟训练状态；不创建Actor、优化器或真实环境。"""

    def __init__(self, project: ProjectConfig, training_seed: int = 11) -> None:
        """定义独立训练位置与随机状态；评估不应修改任一字段。"""
        self.project_config = project
        self.config = SimpleNamespace(training_seed=training_seed)
        self.source = SimpleNamespace(next_scenario_index=77)
        self.agent = DeterministicAgentFixture()
        self._inside_transition = False
        self.failure_metadata = None
        self.transitions = 25000
        self.next_slot = 1
        self.identities: list[tuple[Any, ...]] = []
        self.external = False
        self.mutate_source = False

    def state_dict(self) -> dict[str, Any]:
        """权重哨兵、Replay索引、调度触发位置与策略RNG都纳入直接比较。"""
        return deepcopy(dict(transitions=self.transitions, next_slot=self.next_slot,
                             next_validation_transition=50000, replay_indices=[4, 7, 9],
                             weights=torch.tensor([1.0, 2.0]),
                             actor_rng=self.agent.rng.bit_generator.state,
                             training_seed=self.config.training_seed))

    def _make_env(self, scenario: TrainingScenario) -> ValidationEnvFixture:
        """只返回模拟环境；错误注入不能静默消费训练发行位置。"""
        if self.mutate_source:
            self.source.next_scenario_index += 1
        return ValidationEnvFixture(scenario, self.identities, self.external)


def test_fixed_pool_profiles_share_all_300_base_geometries(pool: FixedValidationPool) -> None:
    """同一固定基底派生空/CV；至少半数垂向条件不误写为恰好半数。"""
    manifest = pool.compact_manifest()
    assert manifest['base_indices'] == list(range(300))
    assert manifest['monitor_indices'] == list(range(30))
    assert len(manifest['scenarios']) == 300
    assert len(set(pool.environment_seeds)) == 300
    vertical = 0
    for index in range(300):
        empty, cv = pool.scenario('obstacle_free', index), pool.scenario('cv_train_v1', index)
        assert states_equal(empty.initial_auv_state, cv.initial_auv_state)
        np.testing.assert_array_equal(empty.goal_position_ned_m, cv.goal_position_ned_m)
        assert not empty.initial_obstacle_states and 1 <= len(cv.initial_obstacle_states) <= 4
        assert empty.scenario_id != cv.scenario_id
        vertical += int(abs(empty.goal_position_ned_m[2]
                            - empty.initial_auv_state.position_ned_m[2]) >= 4.0)
    assert vertical >= 150


def test_fixed_pool_reconstructed_manifest_exactly_equal(
    pool: FixedValidationPool, project_config: ProjectConfig, scenario_config: Any,
) -> None:
    """重建固定池同root/index直接逐值一致，无文件摘要或fresh-copy封存。"""
    other = FixedValidationPool(project_config, scenario_config)
    assert states_equal(pool.compact_manifest(), other.compact_manifest())


@pytest.mark.parametrize('profile', ['obstacle_free', 'cv_train_v1'])
def test_monitor_reuses_same_subset_and_seeds_across_training_seed_and_calls(
    pool: FixedValidationPool, project_config: ProjectConfig, profile: str,
) -> None:
    """每次0..29且跨11/22/33固定几何环境seed，不消费训练source或Replay。"""
    identities = []
    for training_seed in (11, 22, 33):
        harness = ValidationHarnessFixture(project_config, training_seed)
        before = harness.state_dict()
        first = pool.evaluate(harness, profile)
        second = pool.evaluate(harness, profile)
        assert first['count'] == 30 and first['indices'] == list(range(30))
        assert first['evaluation_env_transitions'] == 60
        assert first['warmup_control_transitions'] == 150
        assert first['training_state_unchanged']
        assert states_equal(first['episodes'], second['episodes'])
        assert harness.source.next_scenario_index == 77
        assert states_equal(before, harness.state_dict())
        assert harness.identities[:30] == harness.identities[30:]
        identities.append(harness.identities[:30])
    assert identities[0] == identities[1] == identities[2]


def test_full_count_raw_and_trajectory_retention(
    pool: FixedValidationPool, project_config: ProjectConfig,
) -> None:
    """Val300保留全部episode原始行，仅0/1/2和升序最早失败5保留逐节点轨迹。"""
    record = pool.evaluate(ValidationHarnessFixture(project_config), 'cv_train_v1', full=True)
    assert record['count'] == len(record['episodes']) == 300
    assert record['evaluation_env_transitions'] == 600
    assert record['warmup_control_transitions'] == 1500
    assert [episode['index'] for episode in record['episodes'] if 'trajectory' in episode] \
        == [0, 1, 2, 5]
    assert record['episodes'][5]['trajectory_retention_reasons'] == ['earliest_failure_by_index']
    assert record['summary']['physical_complete_episodes'] == 300
    assert record['summary']['collision_count'] == 2


def test_validation_progress_is_actual_completed_prefix(
    pool: FixedValidationPool, project_config: ProjectConfig,
) -> None:
    """每完成一例发出真实夹具前缀，允许外部日志观测且不改变训练状态。"""
    progress = []
    record = pool.evaluate(ValidationHarnessFixture(project_config), 'obstacle_free',
                           progress_callback=lambda *values: progress.append(values))
    assert progress == [(index, index + 1, 2 * (index + 1), 5 * (index + 1))
                        for index in range(30)]
    assert record['evaluation_env_transitions'] == progress[-1][2]
    assert record['warmup_control_transitions'] == progress[-1][3]


def test_empty_clearance_is_unmeasured_not_fake_zero(
    pool: FixedValidationPool, project_config: ProjectConfig,
) -> None:
    """空场景真实无目标的净间距不写0，评估episode仍有当前/最终距离和进展。"""
    record = pool.evaluate(ValidationHarnessFixture(project_config), 'obstacle_free')
    assert all(episode['minimum_clearance_m'] is None for episode in record['episodes'])
    assert all(math_values_are_finite(episode) for episode in record['episodes'])


def math_values_are_finite(episode: dict[str, Any]) -> bool:
    """距离与进展采用数值字段，而非未测量哨兵或字符串。"""
    return bool(np.all(np.isfinite([episode['initial_distance_m'], episode['final_distance_m'],
                                   episode['progress_m']])))


@pytest.mark.parametrize('error', ['torch', 'numpy', 'local_rng'])
def test_random_state_mutation_is_rejected_and_failure_saved(
    pool: FixedValidationPool, project_config: ProjectConfig, error: str,
) -> None:
    """不回滚随机流掩盖错误；保存实际合成步数和当前episode故障位置。"""
    harness = ValidationHarnessFixture(project_config)
    harness.agent.error = error
    before_torch, before_numpy = torch.get_rng_state().clone(), deepcopy(np.random.get_state())
    try:
        with pytest.raises(RuntimeError, match='训练完整状态或随机流'):
            pool.evaluate(harness, 'obstacle_free')
        assert harness.failure_metadata['operation'] == 'fixed_validation'
        assert not harness.failure_metadata['checkpoint_safe']
        assert harness.failure_metadata['evaluation_env_transitions'] == 60
    finally:
        # 错误测试自身不污染其他测试的全局流；生产验证没有此恢复逻辑。
        torch.set_rng_state(before_torch)
        np.random.set_state(before_numpy)


def test_training_scene_index_mutation_is_rejected(
    pool: FixedValidationPool, project_config: ProjectConfig,
) -> None:
    """训练发行器状态也直接比较，不能仅比Actor/Q权重。"""
    harness = ValidationHarnessFixture(project_config)
    harness.mutate_source = True
    with pytest.raises(RuntimeError, match='训练完整状态或随机流'):
        pool.evaluate(harness, 'cv_train_v1')
    assert harness.failure_metadata['completed_validation_episodes'] == 30


@pytest.mark.parametrize('problem', ['external', 'nonfinite', 'inside'])
def test_validation_rejects_external_truncation_nonfinite_and_half_transition(
    pool: FixedValidationPool, project_config: ProjectConfig, problem: str,
) -> None:
    """真实时域、数值有限性和保存边界不能由工程夹具绕过。"""
    harness = ValidationHarnessFixture(project_config)
    harness.external = problem == 'external'
    harness._inside_transition = problem == 'inside'
    harness.agent.error = 'nonfinite' if problem == 'nonfinite' else None
    with pytest.raises((RuntimeError, FloatingPointError)):
        pool.evaluate(harness, 'cv_train_v1')
    if problem != 'inside':
        assert harness.failure_metadata is not None


@pytest.mark.parametrize('profile,index', [('unknown', 0), ('obstacle_free', 300),
                                          ('cv_train_v1', -1), ('cv_train_v1', True)])
def test_unregistered_profile_or_index_fails(
    pool: FixedValidationPool, profile: str, index: int,
) -> None:
    """固定池不悄悄追加或替换失败场景。"""
    with pytest.raises(ValueError):
        pool.scenario(profile, index)


def current_state_and_targets() -> tuple[AUVState, GroundTruthObstacleState,
                                         GroundTruthObstacleState]:
    """互不遮挡的当前几何；不访问未来或推进真实环境。"""
    own = AUVState(np.array([20., 50., 20.]), 0., 0., .3, 0., 0.)
    target = GroundTruthObstacleState(7, np.array([32., 50., 20.]), np.zeros(3), .5)
    extra = GroundTruthObstacleState(0, np.array([28., 55., 20.]), np.zeros(3), .5)
    return own, target, extra


def capture_signature(config: ProjectConfig, targets: tuple[GroundTruthObstacleState, ...],
                      tick: int, *, seed: int = 91) -> dict[int, Any]:
    """仅冻结传感器函数调用；噪声/dropout由时间与ID定位，而非顺序流。"""
    own, _, _ = current_state_and_targets()
    with paired_validation_sensor_streams(seed):
        result = local_perception.capture_center_detections(
            own, targets, tick*config.dynamics.control_dt_s, tick, config,
            np.random.default_rng(1), np.random.default_rng(2))
    return {item.obstacle_id: item for item in result}


def test_paired_sensor_target_noise_not_changed_by_other_visibility(
    project_config: ProjectConfig,
) -> None:
    """另一个低ID目标在FOV内外变化，不改变同tick/ID目标的高斯噪声与协方差。"""
    _, target, extra = current_state_and_targets()
    config = replace(project_config, sensor=replace(project_config.sensor, dropout_probability=0.))
    hidden = replace(extra, position_ned_m=np.array([10., 55., 20.]))
    first = capture_signature(config, (extra, target), 11)[7]
    second = capture_signature(config, (hidden, target), 11)[7]
    assert states_equal(first, second)
    assert first.measurement_control_tick == first.arrival_control_tick == 11


def test_paired_dropout_and_noise_do_not_depend_on_other_targets_or_previous_ticks(
    project_config: ProjectConfig,
) -> None:
    """20个预定tick直接比较潜在漏检与噪声；不筛选有利tick或强迫相同检测序列。"""
    own, target, extra = current_state_and_targets()
    full, fewer = [], []
    with paired_validation_sensor_streams(91):
        for tick in range(20):
            detections = local_perception.capture_center_detections(
                own, (extra, target), tick*.2, tick, project_config,
                np.random.default_rng(100), np.random.default_rng(200))
            full.append({item.obstacle_id: item for item in detections}.get(7))
    with paired_validation_sensor_streams(91):
        for tick in range(20):
            # 另一路上偶数时刻目标不可见；奇数时刻必须仍用对应tick潜在随机量。
            obstacles = (target,) if tick % 2 else ()
            detections = local_perception.capture_center_detections(
                own, obstacles, tick*.2, tick, project_config,
                np.random.default_rng(100), np.random.default_rng(200))
            fewer.append({item.obstacle_id: item for item in detections}.get(7))
    for tick in range(1, 20, 2):
        assert states_equal(full[tick], fewer[tick])
    assert any(item is not None for item in full)


def test_paired_capture_preserves_occlusion_and_restores_original_on_exception(
    project_config: ProjectConfig,
) -> None:
    """适配器不关闭真实遮挡，异常退出后训练/默认感知前端恢复原函数对象。"""
    own, target, _ = current_state_and_targets()
    blocker = GroundTruthObstacleState(0, np.array([25., 50., 20.]), np.zeros(3), 1.)
    config = replace(project_config, sensor=replace(project_config.sensor, dropout_probability=0.))
    original = local_perception.capture_center_detections
    with pytest.raises(RuntimeError, match='fixture error'):
        with paired_validation_sensor_streams(91):
            assert local_perception.capture_center_detections is not original
            actual = local_perception.capture_center_detections(
                own, (blocker, target), 1., 5, config,
                np.random.default_rng(1), np.random.default_rng(2))
            assert [item.obstacle_id for item in actual] == [0]
            raise RuntimeError('fixture error')
    assert local_perception.capture_center_detections is original
