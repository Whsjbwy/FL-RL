"""登记的一次B0无障碍R1复测；复用普通SAC，只增加调度和只读诊断记录。"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_risk_rl.config import ProjectConfig, load_project_config
from auv_risk_rl.env.b0_navigation import B0NavigationEnv
from auv_risk_rl.env.scenario_generator import (
    TrainingScenarioConfig,
    load_training_scenario_config,
)
from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.training.config import B0HarnessConfig, derived_sac_config
from auv_risk_rl.training.fixed_validation import FixedValidationPool
from auv_risk_rl.training.harness import B0TrainingHarness, EnvFactory, LogSink, states_equal
from auv_risk_rl.training.mvp_batch import (
    BatchLock,
    SegmentLog,
    atomic_json,
    resource_identity,
    utc,
)
from auv_risk_rl.training.mvp_registration import git
from auv_risk_rl.training.repair_registration import APPROVED, REGISTRATION_ID, verify_code_identity
from auv_risk_rl.types import AUVState

EvaluationCallback = Callable[['B0RepairHarness', str, bool], dict[str, Any]]


def _limit_fraction(start: float, end: float, lower: float, upper: float) -> float | None:
    """只读线性首次越界参考；不参与原世界事件决策。"""
    if start < lower or start > upper:
        return 0.0
    if end > upper:
        return (upper - start) / (end - start)
    if end < lower:
        return (lower - start) / (end - start)
    return None


def boundary_labels(start: AUVState, end: AUVState, config: ProjectConfig,
                    ) -> list[str]:
    """从未提交小步两端独立细分最早约束；同刻标签均保留，不改优先级。"""
    radius = config.risk.auv_radius_m
    lower = np.asarray(config.environment.position_lower_bound_ned_m) + radius
    upper = np.asarray(config.environment.position_upper_bound_ned_m) - radius
    values = []
    for axis, name in enumerate(('N', 'E', 'D')):
        values.append((f'position_{name}_lower', float(start.position_ned_m[axis]),
                       float(end.position_ned_m[axis]), float(lower[axis]), float(upper[axis])))
    dynamics = config.dynamics
    for name, start_value, end_value, low, high in (
        ('pitch', start.pitch_rad, end.pitch_rad, -dynamics.max_pitch_rad,
         dynamics.max_pitch_rad),
        ('surge', start.surge_speed_mps, end.surge_speed_mps,
         dynamics.min_surge_speed_mps, dynamics.max_surge_speed_mps),
        ('yaw_rate', start.yaw_rate_rad_s, end.yaw_rate_rad_s,
         -dynamics.max_yaw_rate_rad_s, dynamics.max_yaw_rate_rad_s),
        ('pitch_rate', start.pitch_rate_rad_s, end.pitch_rate_rad_s,
         -dynamics.max_pitch_rate_rad_s, dynamics.max_pitch_rate_rad_s),
    ):
        values.append((name + '_lower', start_value, end_value, low, high))
    candidates = []
    for name, first, last, low, high in values:
        fraction = _limit_fraction(first, last, low, high)
        if fraction is not None:
            label = name if last < low or first < low else name.removesuffix('_lower') + '_upper'
            candidates.append((fraction, label))
    if not candidates:
        return ['unclassified_boundary']
    first = min(value for value, _ in candidates)
    return sorted(label for value, label in candidates if abs(value - first) <= 1.0e-8)


def new_episode_diagnostics(env: B0NavigationEnv) -> dict[str, Any]:
    """合法暖机之后初始化，只存简单可恢复的标量/列表，不保存闭包。"""
    distance = float(np.linalg.norm(env.world.auv_state.position_ned_m-goal(env)))
    return dict(initial_distance_m=distance, final_distance_m=distance,
                minimum_goal_distance_m=distance, minimum_goal_distance_time_s=0.0,
                first_within_10m_time_s=0.0 if distance <= 10.0 else None,
                first_goal_entry_time_s=0.0 if distance <= env.config.environment.goal_radius_m
                else None, boundary_constraints=[], boundary_subtype=None,
                episode_origin_timestamp_s=env.world.timestamp_s,
                diagnostic_scope='ACTUALLY_EXECUTED_0.05S_LINEAR_SEGMENTS')


def goal(env: B0NavigationEnv) -> np.ndarray:
    """读取共同当前任务目标；不查询未来或修改任务。"""
    return np.asarray(env.goal_position_ned_m, dtype=np.float64)


def _first_ball_entry(start: np.ndarray, end: np.ndarray, center: np.ndarray,
                      radius: float) -> float | None:
    """独立二次方程球进入分数，仅用于首次接近日志，不决定成功事件。"""
    offset, motion = start-center, end-start
    c = float(offset @ offset-radius*radius)
    if c <= 0.0:
        return 0.0
    a = float(motion @ motion)
    if a == 0.0:
        return None
    b = 2.0*float(offset @ motion)
    discriminant = b*b-4.0*a*c
    if discriminant < 0.0:
        return None
    fraction = (-b-math.sqrt(discriminant))/(2.0*a)
    return fraction if 0.0 <= fraction <= 1.0 else None


@contextmanager
def observe_world_step(env: B0NavigationEnv, diagnostic: dict[str, Any]) -> Iterator[None]:
    """仅调用期间包装原proposal，原返回对象原样交回，退出移除不可pickle闭包。"""
    world = env.world
    original = world._propose_integration_step
    original_instance = world.__dict__.get('_propose_integration_step')

    def observe(start: AUVState, obstacles: Any, command: Any, timestamp: float) -> Any:
        """读取实际执行前缀；不能把事件之后的未执行提议加入距离统计。"""
        proposal = original(start, obstacles, command, timestamp)
        fraction = 1.0 if proposal.event is None else proposal.event.fraction
        end = start.position_ned_m + fraction*(proposal.auv_end_state.position_ned_m
                                               - start.position_ned_m)
        center = goal(env)
        motion = end-start.position_ned_m
        length = float(motion @ motion)
        projection = (0.0 if length == 0.0 else min(1.0, max(0.0,
                      float((center-start.position_ned_m) @ motion)/length)))
        minimum = float(np.linalg.norm(start.position_ned_m+projection*motion-center))
        dt = env.config.dynamics.integration_dt_s*fraction
        origin = diagnostic['episode_origin_timestamp_s']
        if minimum < diagnostic['minimum_goal_distance_m']:
            diagnostic['minimum_goal_distance_m'] = minimum
            diagnostic['minimum_goal_distance_time_s'] = timestamp-origin+projection*dt
        for field, radius in (('first_within_10m_time_s', 10.0),
                              ('first_goal_entry_time_s', env.config.environment.goal_radius_m)):
            if diagnostic[field] is None:
                entry = _first_ball_entry(start.position_ned_m,
                                          proposal.auv_end_state.position_ned_m, center, radius)
                if entry is not None and entry <= fraction+1.0e-8:
                    diagnostic[field] = (timestamp-origin
                                         + entry*env.config.dynamics.integration_dt_s)
        if proposal.event is not None and proposal.event.event.reason == 'boundary':
            labels = boundary_labels(start, proposal.auv_end_state, env.config)
            diagnostic['boundary_constraints'] = labels
            diagnostic['boundary_subtype'] = labels[0] if len(labels) == 1 else 'simultaneous'
        return proposal

    world._propose_integration_step = observe
    try:
        yield
    finally:
        if original_instance is None:
            world.__dict__.pop('_propose_integration_step', None)
        else:
            world._propose_integration_step = original_instance


def diagnostic_step(env: B0NavigationEnv, diagnostic: dict[str, Any],
                    action: np.ndarray, original_step: Callable[..., Any]) -> Any:
    """观察既有动作执行，输入输出逐项保持；这里只加日志，不替换控制。"""
    with observe_world_step(env, diagnostic):
        result = original_step(action)
    diagnostic['final_distance_m'] = float(np.linalg.norm(env.world.auv_state.position_ned_m
                                                         - goal(env)))
    return result


def diagnostic_validation(pool: FixedValidationPool, harness: B0TrainingHarness,
                          full: bool, progress_callback: Callable[..., None] | None = None,
                          ) -> dict[str, Any]:
    """临时验证工厂只返回原B0类，退出恢复方法；验证episode细分字段不进Replay。"""
    original_factory = harness._make_env
    original_instance = harness.__dict__.get('_make_env')
    diagnostics: dict[str, dict[str, Any]] = {}

    def factory(scenario: Any) -> B0NavigationEnv:
        """评价专用实例包装，无训练环境/状态/RNG改写。"""
        env = original_factory(scenario)
        original_reset, original_step = env.reset, env.step

        def reset(*args: Any, **kwargs: Any) -> Any:
            """原暖机完成以后建立评价独立诊断。"""
            result = original_reset(*args, **kwargs)
            diagnostics[scenario.scenario_id] = new_episode_diagnostics(env)
            return result

        def step(action: np.ndarray) -> Any:
            """执行原评价动作，增加只读字段。"""
            return diagnostic_step(env, diagnostics[scenario.scenario_id], action, original_step)

        env.reset, env.step = reset, step
        return env

    harness._make_env = factory
    try:
        result = pool.evaluate(harness, 'obstacle_free', full, progress_callback)
    finally:
        if original_instance is None:
            harness.__dict__.pop('_make_env', None)
        else:
            harness._make_env = original_instance
    for episode in result['episodes']:
        episode.update(deepcopy(diagnostics[episode['scenario_id']]))
    return result


class B0RepairHarness(B0TrainingHarness):
    """固定100k空场景；SAC及随机派生路径复用V1，没有CV课程切换。"""

    FORMAT = 'b0-repair-r1-full-resume-v1'
    END = 100000

    def __init__(self, config: B0HarnessConfig, project_config: ProjectConfig,
                 scenario_config: TrainingScenarioConfig, *, code_version: str,
                 evaluation_callback: EvaluationCallback,
                 agent: OrdinarySACAgent | None = None, env_factory: EnvFactory | None = None,
                 log_sink: LogSink | None = None) -> None:
        """严格接受唯一登记候选；构造不执行真实环境或梯度更新。"""
        if (config.run_kind != 'scientific_training' or config.task_profile != 'obstacle_free'
                or config.research_registration != REGISTRATION_ID
                or config.training_seed not in (11, 22, 33) or config.num_envs != 2
                or config.transition_budget != self.END or config.sac.learning_rate != 0.0001
                or config.validation_interval != 25000 or config.checkpoint_interval != 25000
                or config.validation_max_steps != 1000 or config.validation_episodes != 30):
            raise ValueError('R1必须是登记的100k无障碍、三seed、公共lr1e-4候选。')
        self.completed_validation_keys: set[str] = set()
        self.final_budget_logged = False
        self.evaluation_warmup_control_transitions = 0
        self.evaluation_callback = evaluation_callback
        super().__init__(config, project_config, scenario_config, code_version=code_version,
                         agent=agent, env_factory=env_factory, log_sink=log_sink)

    @property
    def current_profile(self) -> str:
        """本轮全程空场景，不隐式推进CV。"""
        return 'obstacle_free'

    def _reset_slot(self, slot_index: int) -> None:
        """原发行/暖机完全复用，再初始化独立只读诊断。"""
        super()._reset_slot(slot_index)
        slot = self.slots[slot_index]
        slot['repair_diagnostics'] = (new_episode_diagnostics(slot['env'])
                                      if type(slot['env']) is B0NavigationEnv else None)

    def _episode_record(self, slot: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """实际诊断字段缺失时保持null，不让合成调度夹具冒充测量。"""
        record = super()._episode_record(slot, **kwargs)
        diagnostic = slot.get('repair_diagnostics')
        record.update(deepcopy(diagnostic) if diagnostic is not None else
                      dict(boundary_subtype=None, minimum_goal_distance_m=None,
                           diagnostic_scope='NOT_MEASURED_SYNTHETIC_FIXTURE'))
        return record

    def step(self) -> dict[str, Any]:
        """原harness完整step调用期间观察env.step，评价/保存前已移除临时闭包。"""
        self._assert_can_step()
        slot_index = self.next_slot
        if self.slots[slot_index]['needs_reset']:
            try:
                self._reset_slot(slot_index)
            except BaseException as error:
                self.failure_metadata = dict(transition=self.transitions, operation='reset',
                                             error=repr(error), checkpoint_safe=False)
                raise
        slot = self.slots[slot_index]
        env, diagnostic = slot['env'], slot['repair_diagnostics']
        if diagnostic is None:
            return super().step()
        original_step = env.step
        original_instance = env.__dict__.get('step')

        def restore() -> None:
            """恢复原实例/类方法，使checkpoint不含临时闭包。"""
            if original_instance is None:
                env.__dict__.pop('step', None)
            else:
                env.step = original_instance

        def step(action: np.ndarray) -> Any:
            """原动作执行后立即移除临时env方法。"""
            try:
                return diagnostic_step(env, diagnostic, action, original_step)
            finally:
                restore()

        env.step = step
        try:
            return super().step()
        finally:
            restore()

    def _training_state(self) -> dict[str, Any]:
        """固定验证幂等身份也直接比较，禁止改变训练触发/片段标记。"""
        return {**super()._training_state(), 'repair': self._repair_state()}

    def _repair_state(self) -> dict[str, Any]:
        """可恢复的简单状态，保留唯一修复配置与固定验证完成位置。"""
        return dict(registration_id=REGISTRATION_ID, branch='L2_COMMON_LEARNING_RATE',
                    learning_rate=self.config.sac.learning_rate, endpoint=self.END,
                    completed_validation_keys=sorted(self.completed_validation_keys),
                    final_budget_logged=self.final_budget_logged,
                    evaluation_warmup_control_transitions=self.evaluation_warmup_control_transitions)

    def evaluate(self) -> dict[str, Any]:
        """复用原自动25k触发，仅替换固定池日程。"""
        return self.ensure_validation()

    def ensure_validation(self, full: bool | None = None) -> dict[str, Any]:
        """step0/25k/50k/75k monitor，100k Val300；重复恢复不重做评价。"""
        if self._inside_transition or self.failure_metadata is not None:
            raise RuntimeError('固定R1验证要求无失败的完整控制步边界。')
        full = self.transitions == self.END if full is None else full
        if not isinstance(full, bool) or (full and self.transitions != self.END) or (
                not full and self.transitions not in (0, 25000, 50000, 75000)):
            raise ValueError('R1验证只能在登记时刻/规模执行。')
        key = f'obstacle_free:{self.transitions}:{"full" if full else "monitor"}'
        if key in self.completed_validation_keys:
            return dict(evaluation_key=key, already_completed=True,
                        at_transition=self.transitions, full=full)
        before = self._training_state()
        numpy_rng, torch_rng = deepcopy(np.random.get_state()), torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            result = self.evaluation_callback(self, 'obstacle_free', full)
            if not (states_equal(before, self._training_state())
                    and states_equal(numpy_rng, np.random.get_state())
                    and torch.equal(torch_rng, torch.get_rng_state())
                    and (cuda_rng is None
                         or states_equal(cuda_rng, torch.cuda.get_rng_state_all()))):
                raise RuntimeError('R1固定验证改变训练状态或随机流。')
            if result.get('count') != (300 if full else 30):
                raise ValueError('R1验证案例数不符。')
            self.evaluation_env_transitions += result['evaluation_env_transitions']
            self.evaluation_warmup_control_transitions += result['warmup_control_transitions']
            self.validation_count += 1
            result.update(profile='obstacle_free', full=full, at_transition=self.transitions,
                          evaluation_key=key, training_state_unchanged=True, diagnostic_only=False)
            self._emit('validation', result)
            self.completed_validation_keys.add(key)
            return result
        except BaseException as error:
            metadata = deepcopy(self.failure_metadata or {})
            metadata.update(transition=self.transitions, evaluation_key=key, error=repr(error))
            self.failure_metadata = metadata
            raise

    def record_final_budget_stop(self) -> None:
        """严格100k停止，未完成片段不伪造物理事件、不更改Replay标志。"""
        if self.transitions != self.END or 'obstacle_free:100000:full' \
                not in self.completed_validation_keys:
            raise RuntimeError('100k预算片段须在固定终点Val300以后登记。')
        if self.final_budget_logged:
            return
        for slot in self.slots:
            if not slot['needs_reset']:
                record = self._episode_record(slot, complete=False, failure_type='none',
                                              budget_stop=True)
                self.budget_stop_records.append(record)
                self._emit('episode', record)
        self.final_budget_logged = True

    def state_dict(self) -> dict[str, Any]:
        """包含原完整SAC/Replay/RNG/环境和新的验证/诊断恢复状态。"""
        return {**super().state_dict(), 'repair': self._repair_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """先验证候选/预算/验证位置，不能先修改Agent后发现错误身份。"""
        repair = state.get('repair', {})
        identity = self._repair_state()
        static = ('registration_id', 'branch', 'learning_rate', 'endpoint')
        if any(repair.get(key) != identity[key] for key in static):
            raise ValueError('R1恢复候选/登记/学习率/预算不一致。')
        transition = state.get('scheduler', {}).get('transitions')
        keys = repair.get('completed_validation_keys')
        if (isinstance(transition, bool) or not isinstance(transition, int)
                or not 0 <= transition <= self.END or not isinstance(keys, list)
                or len(keys) != len(set(keys))):
            raise ValueError('R1恢复计数/验证键损坏。')
        allowed = {f'obstacle_free:{point}:{"full" if point == self.END else "monitor"}'
                   for point in (0, 25000, 50000, 75000, self.END) if point <= transition}
        if not set(keys).issubset(allowed) or (transition > 0 and set(keys) != allowed):
            raise ValueError('R1恢复缺失或包含未来验证键。')
        if (not isinstance(repair.get('final_budget_logged'), bool)
                or (repair['final_budget_logged'] and transition != self.END)
                or state['scheduler']['validation_count'] != len(keys)):
            raise ValueError('R1恢复验证计数/预算片段不一致。')
        warmup = repair.get('evaluation_warmup_control_transitions')
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError('R1恢复暖机计数损坏。')
        super().load_state_dict(state)
        self.completed_validation_keys = set(keys)
        self.final_budget_logged = repair['final_budget_logged']
        self.evaluation_warmup_control_transitions = warmup


def harness_config(registration: dict[str, Any], seed: int) -> B0HarnessConfig:
    """只改变公共学习率，显式复用V1 scientific_training随机命名空间。"""
    if seed not in registration['training_seeds']:
        raise ValueError('R1 seed必须属于已批准的11/22/33。')
    sac = derived_sac_config(SACConfig(device=registration['device'],
                                      **registration['sac']),
                             training_seed=seed, run_kind='scientific_training')
    return B0HarnessConfig(run_kind='scientific_training', task_profile='obstacle_free',
                           training_seed=seed, transition_budget=100000, num_envs=2,
                           validation_interval=25000, validation_episodes=30,
                           validation_max_steps=1000, checkpoint_interval=25000,
                           research_registration=REGISTRATION_ID, sac=sac)


def save_network_snapshot(harness: B0RepairHarness, output: Path, seed: int) -> Path:
    """25k小模型保留Actor/两个Q/targets/alpha，不复制Replay、不选择best。"""
    models = output / 'models'
    models.mkdir(exist_ok=True)
    path = models / f'seed_{seed}_{harness.transitions}.pt'
    networks = {name: {key: tensor.detach().cpu() for key, tensor
                      in getattr(harness.agent, name).state_dict().items()}
                for name in ('actor', 'q1', 'q2', 'target_q1', 'target_q2')}
    value = dict(format='b0-repair-r1-network-snapshot-v1', models=networks,
                 actor=networks['actor'], log_alpha=harness.agent.log_alpha.detach().cpu(),
                 sac_config=asdict(harness.agent.config), counters=harness.agent.counters.copy(),
                 seed=seed, transition=harness.transitions, code_version=harness.code_version,
                 run_kind='scientific_training', method=harness.config.method,
                 registration_id=REGISTRATION_ID, branch='L2_COMMON_LEARNING_RATE')
    needed = sum(tensor.numel()*tensor.element_size() for model in networks.values()
                 for tensor in model.values()) + 1024**2
    if shutil.disk_usage(output).free < needed:
        raise OSError('没有足够临时空间保存小模型。')
    temporary = path.with_suffix('.pt.partial')
    try:
        with temporary.open('xb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def run_repair_batch(root: Path, registration: dict[str, Any], *,
                     resume: bool = False) -> dict[str, Any]:
    """真实worker串行完成三seed100k；不继续CV、不自动失败重试或加预算。"""
    if registration != APPROVED:
        raise ValueError('R1批次必须使用完整canonical登记。')
    output = root / registration['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'batch_state.json'
    if state_path.exists() != resume:
        raise RuntimeError('已有状态须显式resume；没有状态不能恢复。')
    orphan_dirs = [output / f'seed_{seed}' for seed in (11, 22, 33)] + [output / 'models']
    if not resume and any(path.exists() for path in orphan_dirs):
        raise RuntimeError('新run输出存在旧seed产物，拒绝混入。')
    state = (json.loads(state_path.read_text(encoding='utf-8')) if resume else dict(
        registration_id=REGISTRATION_ID, registration=registration,
        experiment_code_commit=git(root, 'rev-parse', 'HEAD'), created_at=utc(),
        status='PENDING', completed_seeds=[], seeds={}))
    if state['registration'] != registration or state['registration_id'] != REGISTRATION_ID:
        raise ValueError('R1恢复登记不符。')
    if state['status'] == 'FAILED':
        raise RuntimeError('失败R1不自动重试；须先核对故障及重算预算。')
    verify_code_identity(root, state['experiment_code_commit'])
    if state['status'] == 'COMPLETED':
        return state
    lock = BatchLock(output / 'batch.lock', resume=resume)
    logger: SegmentLog | None = None
    active: B0RepairHarness | None = None
    try:
        resources = resource_identity(output)
        row_bytes = resources['replay_bytes_per_transition']
        resources.update(
            inherited_conservative_v1_storage_note=resources['storage_note'],
            r1_replay100k_raw_bytes=100000*row_bytes,
            r1_replay100k_chunk_bytes=25*4096*row_bytes,
            r1_three_latest_plus_one_temporary_chunk_bytes=4*25*4096*row_bytes,
            storage_note='R1: three 100k latest resumes plus one temporary; raw chunk estimate '
                         'excludes network/optimizer/Python serialization. Inherited 8GiB disk '
                         'and 5GiB available RAM preflight remains conservative; no benchmark.')
        atomic_json(output / 'environment_resources.json', resources)
        project = load_project_config(root / registration['project_config'])
        scenario = load_training_scenario_config(root / registration['scenario_config'])
        pool = FixedValidationPool(project, scenario, root_seed=20261006)
        atomic_json(output / 'validation_manifest.json', pool.compact_manifest())
        state.update(status='RUNNING', pid=os.getpid(), started_at=utc())
        atomic_json(state_path, state)
        for seed in registration['training_seeds']:
            if seed in state['completed_seeds']:
                continue
            seed_dir = output / f'seed_{seed}'
            seed_dir.mkdir(exist_ok=True)
            checkpoint = seed_dir / 'latest_resume.pt'
            row = state['seeds'].setdefault(str(seed), dict(training_seconds=0.0,
                                                          evaluation_seconds=0.0,
                                                          checkpoint_seconds=0.0))
            row['status'] = 'RUNNING'
            state.update(current_seed=seed, current_operation='INITIALIZING')

            def evaluate(current: B0RepairHarness, profile: str, full: bool,
                         row: dict[str, Any] = row) -> dict[str, Any]:
                """验证真实进展单列，回调不更新任何训练状态或Replay。"""
                if profile != 'obstacle_free':
                    raise ValueError('R1不运行CV验证。')
                state.update(current_operation='EVALUATING', evaluation_full=full,
                             evaluation_at_transition=current.transitions, last_progress_at=utc())
                atomic_json(state_path, state)
                start = time.perf_counter()

                def on_episode(index: int, count: int, steps: int, warmup: int) -> None:
                    """保存实际验证进展，不能把计划数填成实测。"""
                    state.update(current_validation_index=index,
                                 current_validation_completed_episodes=count,
                                 current_validation_transitions=steps,
                                 current_validation_warmup_transitions=warmup,
                                 last_progress_at=utc())
                    atomic_json(state_path, state)

                try:
                    return diagnostic_validation(pool, current, full, on_episode)
                finally:
                    row['evaluation_seconds'] += time.perf_counter()-start
                    state['current_operation'] = 'TRAINING'

            harness = B0RepairHarness(harness_config(registration, seed), project, scenario,
                                      code_version=state['experiment_code_commit'],
                                      evaluation_callback=evaluate)
            active = harness
            if checkpoint.exists():
                harness.load_checkpoint(checkpoint, trusted_local=True)
            logger = SegmentLog(output, seed, parent=checkpoint if checkpoint.exists() else None,
                                cutoff=harness.log_sequence if checkpoint.exists() else None,
                                registration_id=REGISTRATION_ID)
            harness.log_sink = logger

            def progress(harness: B0RepairHarness = harness,
                         row: dict[str, Any] = row, seed: int = seed) -> None:
                """只从实际harness状态更新计数和剩余时间估计。"""
                row.update(transitions=harness.transitions,
                           updates=harness.agent.counters['gradient_updates'],
                           optimizer_steps=sum(harness.agent.counters[key]
                                               for key in ('actor', 'q1', 'q2', 'alpha')),
                           profile='obstacle_free', replay_size=len(harness.agent.replay),
                           evaluation_env_transitions=harness.evaluation_env_transitions,
                           evaluation_warmup_control_transitions=(
                               harness.evaluation_warmup_control_transitions),
                           training_warmup_transitions=harness.started_episodes*5,
                           completed_episodes=harness.completed_episodes,
                           last_progress_at=utc())
                done = sum(item.get('transitions', 0) for item in state['seeds'].values())
                seconds = sum(item['training_seconds'] for item in state['seeds'].values())
                state.update(actual_training_transitions=done,
                             actual_sac_updates=sum(item.get('updates', 0)
                                                    for item in state['seeds'].values()),
                             last_progress_at=utc(), remaining_training_hours_estimate=(
                                 (300000-done)*seconds/done/3600 if done else None),
                             estimate_excludes_future_evaluation_and_save=True)
                atomic_json(state_path, state)
                print(json.dumps(dict(seed=seed, transition=harness.transitions,
                                      updates=row['updates'], operation=state['current_operation'],
                                      time=utc())), flush=True)

            def save(status: str = 'RUNNING', harness: B0RepairHarness = harness,
                     logger: SegmentLog = logger, row: dict[str, Any] = row,
                     checkpoint: Path = checkpoint,
                     progress: Callable[..., None] = progress) -> None:
                """先flush并安全替换恢复点，再确认有效日志前缀。"""
                logger.flush(sync=True)
                state['current_operation'] = 'CHECKPOINT_SAVE'
                atomic_json(state_path, state)
                start = time.perf_counter()
                harness.save_checkpoint(checkpoint)
                row['checkpoint_seconds'] += time.perf_counter()-start
                logger.confirm(harness.log_sequence, status)
                row.update(checkpoint=str(checkpoint), checkpoint_transition=harness.transitions)
                state['current_operation'] = 'TRAINING'
                progress()

            if harness.transitions == 0:
                harness.ensure_validation()
                save()
            while harness.transitions < harness.END:
                before_eval = row['evaluation_seconds']
                start = time.perf_counter()
                step = harness.step()
                row['training_seconds'] += max(0.0, time.perf_counter()-start
                                                - (row['evaluation_seconds']-before_eval))
                if step['checkpoint_due']:
                    save_network_snapshot(harness, output, seed)
                    if harness.transitions < harness.END:
                        save()
                elif harness.transitions % 100 == 0:
                    progress()
            harness.record_final_budget_stop()
            row['status'] = 'COMPLETED'
            save('COMPLETED')
            state['completed_seeds'].append(seed)
            progress()
            logger.close()
            logger = active = None
            del harness, evaluate, progress, save
            torch.cuda.empty_cache()
        state.update(status='COMPLETED', completed_at=utc(), current_operation='BATCH_FINISHED',
                     scientific_stage2_decision='AWAITING_RESULT_ANALYSIS')
        atomic_json(state_path, state)
        return state
    except BaseException:
        state.update(status='FAILED', failed_at=utc(), traceback=traceback.format_exc(),
                     recovery_note='Do not auto-retry; inspect last safe checkpoint and budget.')
        if active is not None:
            state['failure_actual_counters'] = active.summary()
            state['failure_metadata'] = deepcopy(active.failure_metadata)
        if logger is not None:
            logger.flush(sync=True)
            logger.metadata[-1]['status'] = 'FAILED'
            atomic_json(logger.metadata_path, logger.metadata)
        atomic_json(state_path, state)
        raise
    finally:
        if logger is not None:
            logger.close()
        lock.close()
