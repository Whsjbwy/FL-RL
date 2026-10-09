"""B0 MVP 固定 Val300；独立场景与时间/目标索引传感流不消耗训练状态。"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from threading import RLock
from time import perf_counter
from typing import Any

import numpy as np
import torch

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.local_task import LocalTaskConfig
from auv_risk_rl.env.scenario_generator import (
    LocalTrainingScenarioGenerator,
    TrainingScenario,
    TrainingScenarioConfig,
)
from auv_risk_rl.runtime import local_perception
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.sensors.occlusion import unoccluded_centers
from auv_risk_rl.sensors.sonar import generate_sonar_detections
from auv_risk_rl.training.harness import B0TrainingHarness, states_equal
from auv_risk_rl.types import AUVState, GroundTruthObstacleState, SensorDetection

_CAPTURE_LOCK = RLock()


@contextmanager
def paired_validation_sensor_streams(environment_seed: int) -> Iterator[None]:
    """仅同步验证期间替换传感前端；复用遮挡/FOV/噪声数学，退出恢复默认函数。"""
    if isinstance(environment_seed, bool) or not isinstance(environment_seed, int) \
            or environment_seed < 0:
        raise ValueError('配对验证环境种子必须是非负整数。')

    def paired_capture(auv: AUVState, obstacles: tuple[GroundTruthObstacleState, ...],
                       timestamp_s: float, tick: int, config: ProjectConfig,
                       noise_rng: np.random.Generator, dropout_rng: np.random.Generator,
                       ) -> tuple[SensorDetection, ...]:
        """潜在随机量由世界整数tick与稳定ID定位，不依赖别的目标是否可见或漏检。"""
        seeds = SeedManager(environment_seed)
        detections = []
        for obstacle in sorted(unoccluded_centers(auv, obstacles), key=lambda o: o.obstacle_id):
            namespace = f'paired_validation/tick-{tick}/obstacle-{obstacle.obstacle_id}'
            detections.extend(generate_sonar_detections(
                auv, (obstacle,), timestamp_s, config.sensor, config.dynamics,
                seeds.get_rng(namespace + '/noise'), measurement_control_tick=tick,
                dropout_rng=seeds.get_rng(namespace + '/dropout')))
        return tuple(detections)

    # 验证为单线程同步调用；锁防止多个该context互相覆盖，未并行推进训练环境。
    with _CAPTURE_LOCK:
        original = local_perception.capture_center_detections
        local_perception.capture_center_detections = paired_capture
        try:
            yield
        finally:
            local_perception.capture_center_detections = original


class FixedValidationPool:
    """运行前固定0..299基底；monitor复用0..29，不按策略结果筛选或重抽。"""

    PROFILES = ('obstacle_free', 'cv_train_v1')
    FULL_COUNT = 300
    MONITOR_COUNT = 30
    TRAJECTORY_INDICES = (0, 1, 2)

    def __init__(self, project: ProjectConfig, scenario: TrainingScenarioConfig,
                 root_seed: int = 20261006) -> None:
        """两个profile共享起终点、初态与固定环境种子，独立于训练seed和场景索引。"""
        if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
            raise ValueError('固定验证root_seed必须是非负整数。')
        self.project, self.scenario_config, self.root_seed = project, scenario, root_seed
        generator = LocalTrainingScenarioGenerator(scenario, project)
        self.base_scenarios = tuple(generator.generate(root_seed, index)
                                    for index in range(self.FULL_COUNT))
        self.environment_seeds = tuple(int(SeedManager(root_seed).get_rng(
            f'b0/fixed-validation/environment/episode-{index}').integers(0, 2**63))
            for index in range(self.FULL_COUNT))
        self._scenarios = {
            'cv_train_v1': self.base_scenarios,
            'obstacle_free': tuple(self._without_obstacles(item) for item in self.base_scenarios),
        }

    @staticmethod
    def _without_obstacles(base: TrainingScenario) -> TrainingScenario:
        """独立无障碍身份；不改变TRAIN_SCENARIO_V1本身的1–4障碍支持。"""
        version = 'B0_OBSTACLE_FREE_V1'
        identity = f'b0-empty-validation-seed-{base.root_seed}-idx-{base.scenario_index}'
        return replace(base, scenario_id=identity, distribution_version=version,
                       initial_obstacle_states=(), metadata=replace(
                           base.metadata, obstacle_count=0, obstacles=(),
                           initial_obstacle_overlap_count=0, scenario_generation_version=version))

    def scenario(self, profile: str, index: int) -> TrainingScenario:
        """返回只读固定场景；拒绝未登记profile和池外索引。"""
        if profile not in self.PROFILES:
            raise ValueError('固定验证只接受已登记的两个B0 profile。')
        if (isinstance(index, bool) or not isinstance(index, int)
                or not 0 <= index < self.FULL_COUNT):
            raise ValueError('固定验证索引必须属于0..299。')
        return self._scenarios[profile][index]

    def compact_manifest(self) -> dict[str, Any]:
        """保存一份共享基底几何和ID；无摘要、未来查询、评估结果或重复两份场景。"""
        entries = []
        for index, base in enumerate(self.base_scenarios):
            geometry = base.to_dict()
            entries.append(dict(
                index=index, base_scenario_id=base.scenario_id,
                actual_scenario_ids={profile: self.scenario(profile, index).scenario_id
                                     for profile in self.PROFILES},
                environment_seed=self.environment_seeds[index],
                initial_auv_state=geometry['initial_auv_state'],
                initial_obstacle_states=geometry['initial_obstacle_states'],
                goal_position_ned_m=geometry['goal_position_ned_m'],
                vertical_separation_stratum=base.metadata.vertical_separation_stratum,
                initial_obstacle_overlap_count=base.metadata.initial_obstacle_overlap_count))
        return dict(format='b0-fixed-validation-v1', root_seed=self.root_seed,
                    distribution_version=self.scenario_config.distribution_version,
                    base_indices=list(range(self.FULL_COUNT)),
                    monitor_indices=list(range(self.MONITOR_COUNT)),
                    environment_seed_namespace='b0/fixed-validation/environment/episode-{index}',
                    sensor_stream_namespace=(
                        'paired_validation/tick-{tick}/obstacle-{id}/{noise|dropout}'),
                    sensor_stream_key='fixed_environment_seed/world_tick/obstacle_id',
                    sensor_visibility='实际遮挡、range、FOV和dropout，不强迫相同检测序列',
                    profiles=list(self.PROFILES), scenarios=entries)

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        """直接快照全局随机状态；验证不通过恢复覆盖来掩盖消耗。"""
        initialized = torch.cuda.is_initialized()
        return dict(torch=torch.get_rng_state().clone(), numpy=deepcopy(np.random.get_state()),
                    cuda_initialized=initialized,
                    cuda=torch.cuda.get_rng_state_all() if initialized else None)

    @staticmethod
    def _training_state(harness: B0TrainingHarness) -> dict[str, Any]:
        """完整harness恢复状态和发行器都比较，含Replay、优化器、各环境及触发位置。"""
        return dict(harness=harness.state_dict(), source=deepcopy(vars(harness.source)))

    @staticmethod
    def _summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
        """仅完整物理episode作分母；原始episode另存，不以汇总代替证据。"""
        complete = [episode for episode in episodes if episode['complete']]
        count = len(complete)
        return dict(physical_complete_episodes=count,
                    success_count=sum(episode['success'] for episode in complete),
                    collision_count=sum(episode['collision'] for episode in complete),
                    boundary_count=sum(episode['boundary'] for episode in complete),
                    task_timeout_count=sum(episode['task_timeout'] for episode in complete),
                    mean_reward=(sum(episode['reward'] for episode in complete) / count
                                 if count else None),
                    mean_progress_m=(sum(episode['progress_m'] for episode in complete) / count
                                     if count else None))

    def evaluate(self, harness: B0TrainingHarness, profile: str,
                 full: bool = False,
                 progress_callback: Callable[[int, int, int, int], None] | None = None,
                 ) -> dict[str, Any]:
        """固定子集/全池、真实任务时域；仅返回记录，调用方一次emit并累计独立验证计数。"""
        if harness._inside_transition or harness.failure_metadata is not None:
            raise RuntimeError('固定验证必须在无失败的完整控制步/update边界执行。')
        self.scenario(profile, 0)
        if not isinstance(full, bool):
            raise TypeError('full必须是明确布尔值。')
        if not states_equal(self.project, harness.project_config):
            raise ValueError('固定验证与训练必须复用相同共同动力学/传感/任务配置。')
        before, before_rng = self._training_state(harness), self._rng_state()
        count = self.FULL_COUNT if full else self.MONITOR_COUNT
        episodes: list[dict[str, Any]] = []
        environment_steps = warmup_steps = 0
        first_failure_retained = False
        current: dict[str, Any] | None = None
        started_at = perf_counter()
        task_config = asdict(getattr(harness.config, 'task', LocalTaskConfig()))
        try:
            for index in range(count):
                scenario = self.scenario(profile, index)
                environment_seed = self.environment_seeds[index]
                current = dict(index=index, base_scenario_id=self.base_scenarios[index].scenario_id,
                               actual_scenario_id=scenario.scenario_id,
                               environment_seed=environment_seed, trajectory=[])
                env = harness._make_env(scenario)
                with paired_validation_sensor_streams(environment_seed):
                    obs, warmup = env.reset(seed=environment_seed,
                                            options={'external_max_steps': None})
                    warmup_duration = warmup['warmup_duration_s']
                    warmup_steps += round(warmup_duration / self.project.dynamics.control_dt_s)
                    current.update(
                        scenario_id=scenario.scenario_id, scenario_index=index,
                        scenario_root_seed=scenario.root_seed, split='validation',
                        training_seed=harness.config.training_seed, episode_id=index, env_slot=None,
                        task_profile=profile, steps=0, physical_time_s=0.0, reward=0.0,
                        reward_components={}, path_length_m=0.0,
                        path_length_definition='CONTROL_NODE_POLYLINE', minimum_clearance_m=None,
                        action_saturation_count=0, warmup=deepcopy(warmup),
                        task_config=task_config.copy(),
                        initial_position_ned_m=env.world.auv_state.position_ned_m.tolist(),
                        goal_position_ned_m=scenario.goal_position_ned_m.tolist())
                    initial_distance = float(np.linalg.norm(
                        scenario.goal_position_ned_m-env.world.auv_state.position_ned_m))
                    terminated = truncated = False
                    while not (terminated or truncated):
                        if current['steps'] >= self.project.environment.max_episode_control_steps:
                            raise RuntimeError('固定验证环境未在真实任务时域内终止。')
                        if not np.all(np.isfinite(obs)):
                            raise FloatingPointError('固定验证观察含非有限值。')
                        start = env.world.auv_state.position_ned_m.copy()
                        action = harness.agent.deterministic_action(obs)
                        if not np.all(np.isfinite(action)):
                            raise FloatingPointError('固定验证Actor动作含非有限值。')
                        obs, reward, terminated, truncated, info = env.step(action)
                        environment_steps += 1
                        current['steps'] += 1
                        if not math.isfinite(reward):
                            raise FloatingPointError('固定验证reward含非有限值。')
                        current['physical_time_s'] += info['elapsed_s']
                        current['reward'] += reward
                        for key, value in info['reward_components'].items():
                            previous = current['reward_components'].get(key, 0.0)
                            current['reward_components'][key] = previous + value
                        end = env.world.auv_state.position_ned_m
                        current['path_length_m'] += float(np.linalg.norm(end-start))
                        clearance = info['minimum_clearance']
                        if math.isfinite(clearance):
                            previous = current['minimum_clearance_m']
                            current['minimum_clearance_m'] = (clearance if previous is None
                                                             else min(clearance, previous))
                        current['action_saturation_count'] += int(
                            np.any(np.abs(action) >= 1.0-1.0e-6))
                        current['trajectory'].append(dict(
                            task_step=info['task_control_step'], position_ned_m=end.tolist(),
                            action=action.tolist(), reward=reward,
                            reward_components=deepcopy(info['reward_components']),
                            elapsed_s=info['elapsed_s'],
                            minimum_clearance_m=(clearance if math.isfinite(clearance) else None),
                            failure_type=info['failure_type']))
                    failure = info['failure_type']
                    if truncated or failure == 'external_truncation':
                        raise RuntimeError('科研固定验证禁止工程外部截断。')
                    final_distance = float(np.linalg.norm(
                        scenario.goal_position_ned_m-env.world.auv_state.position_ned_m))
                    current.update(
                        terminated=terminated, truncated=truncated, failure_type=failure,
                        complete=terminated, success=failure == 'goal_success',
                        collision=failure == 'collision',
                        boundary=failure == 'operational_boundary_failure',
                        task_timeout=failure == 'task_horizon', external_truncation=False,
                        initial_distance_m=initial_distance, final_distance_m=final_distance,
                        progress_m=initial_distance-final_distance)
                reasons = ['preregistered_index'] if index in self.TRAJECTORY_INDICES else []
                if failure != 'goal_success' and not first_failure_retained:
                    reasons.append('earliest_failure_by_index')
                    first_failure_retained = True
                if reasons:
                    current['trajectory_retention_reasons'] = reasons
                else:
                    del current['trajectory']
                episodes.append(current)
                if progress_callback is not None:
                    progress_callback(index, len(episodes), environment_steps, warmup_steps)
            if not (states_equal(before, self._training_state(harness))
                    and states_equal(before_rng, self._rng_state())):
                raise RuntimeError('固定验证改变了训练完整状态或随机流。')
        except BaseException as error:
            harness.failure_metadata = dict(
                transition=harness.transitions, error=repr(error),
                operation='fixed_validation', profile=profile, full=full,
                evaluation_env_transitions=environment_steps,
                warmup_control_transitions=warmup_steps,
                completed_validation_episodes=len(episodes),
                current_validation_episode=deepcopy(current), checkpoint_safe=False)
            raise
        return dict(
            at_transition=harness.transitions, profile=profile, task_profile=profile,
            count=count, full=full, validation_kind='Val300' if full else 'monitor30',
            validation_root_seed=self.root_seed, indices=list(range(count)), episodes=episodes,
            task_config=task_config,
            evaluation_env_transitions=environment_steps, warmup_control_transitions=warmup_steps,
            wall_clock_seconds=perf_counter()-started_at, training_state_unchanged=True,
            potential_sensor_rng='fixed_environment_seed/world_tick/obstacle_id',
            summary=self._summary(episodes))
