"""B0最后一次两组300k无障碍对照；复用R1只读诊断和原普通SAC调度。"""

from __future__ import annotations

import json
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
from auv_risk_rl.env.local_task import LocalTaskConfig
from auv_risk_rl.env.scenario_generator import (
    TrainingScenarioConfig,
    load_training_scenario_config,
)
from auv_risk_rl.env.world import _interpolate_auv_state
from auv_risk_rl.frames import rotation_body_to_ned
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
from auv_risk_rl.training.r2_registration import APPROVED, REGISTRATION_ID, verify_code_identity
from auv_risk_rl.training.repair_r1 import (
    B0RepairHarness,
    _first_ball_entry,
    diagnostic_step,
    new_episode_diagnostics,
)
from auv_risk_rl.types import AUVState, ControlCommand

EvaluationCallback = Callable[['B0R2Harness', str, bool], dict[str, Any]]


def _r2_state_sample(env: B0NavigationEnv, state: AUVState, timestamp_s: float,
                     origin_s: float, *, basis: str,
                     action: np.ndarray | None = None,
                     command: ControlCommand | None = None) -> dict[str, Any]:
    """只读八维NED状态和当前目标向量；不额外采样Actor或查询未来。"""
    vector = np.asarray(env.goal_position_ned_m)-state.position_ned_m
    distance = float(np.linalg.norm(vector))
    return dict(
        state_ned_8d=[*state.position_ned_m.tolist(), state.yaw_rad, state.pitch_rad,
                      state.surge_speed_mps, state.yaw_rate_rad_s, state.pitch_rate_rad_s],
        position_ned_m=state.position_ned_m.tolist(), yaw_rad=state.yaw_rad,
        pitch_rad=state.pitch_rad, surge_speed_mps=state.surge_speed_mps,
        yaw_rate_rad_s=state.yaw_rate_rad_s, pitch_rate_rad_s=state.pitch_rate_rad_s,
        timestamp_s=timestamp_s, physical_time_s=timestamp_s-origin_s, state_basis=basis,
        goal_vector_ned_m=vector.tolist(), goal_distance_m=distance,
        goal_direction_ned_unit=(vector/distance).tolist() if distance > 0.0 else None,
        goal_vector_body_m=(rotation_body_to_ned(state.yaw_rad, state.pitch_rad).T@vector).tolist(),
        action=None if action is None else action.tolist(),
        physical_command=None if command is None else asdict(command))


def new_r2_episode_diagnostics(env: B0NavigationEnv) -> dict[str, Any]:
    """复用R1标量，并登记5m/3m和最小距离时的实际执行线段插值状态。"""
    diagnostic = new_episode_diagnostics(env)
    initial = _r2_state_sample(env, env.world.auv_state, env.world.timestamp_s,
                               env.world.timestamp_s, basis='ACTUAL_CONTROL_NODE')
    distance = diagnostic['initial_distance_m']
    diagnostic.update(first_within_5m_time_s=0.0 if distance <= 5.0 else None,
                      first_within_3m_time_s=0.0 if distance <= 3.0 else None,
                      _r2_initial_state=initial, _r2_minimum_state=deepcopy(initial),
                      _r2_minimum_distance_m=distance, _r2_trajectory=[])
    return diagnostic


@contextmanager
def _observe_r2_world_step(env: B0NavigationEnv, diagnostic: dict[str, Any]) -> Iterator[None]:
    """只观察原proposal的已执行前缀，绝不把终止后的提议算为接近或轨迹。"""
    world = env.world
    original = world._propose_integration_step
    original_instance = world.__dict__.get('_propose_integration_step')

    def observe(start: AUVState, obstacles: Any, command: ControlCommand,
                timestamp_s: float) -> Any:
        """原proposal对象原样交回；插值只用于离线诊断，不改变世界提交状态。"""
        proposal = original(start, obstacles, command, timestamp_s)
        fraction = 1.0 if proposal.event is None else proposal.event.fraction
        end = start.position_ned_m+fraction*(proposal.auv_end_state.position_ned_m
                                            - start.position_ned_m)
        motion = end-start.position_ned_m
        length = float(motion@motion)
        goal = np.asarray(env.goal_position_ned_m)
        projection = (0.0 if length == 0.0 else min(1.0, max(0.0,
                      float((goal-start.position_ned_m)@motion)/length)))
        minimum = float(np.linalg.norm(start.position_ned_m+projection*motion-goal))
        if minimum < diagnostic['_r2_minimum_distance_m']:
            raw_fraction = projection*fraction
            state = _interpolate_auv_state(start, proposal.auv_end_state, raw_fraction)
            diagnostic['_r2_minimum_distance_m'] = minimum
            diagnostic['_r2_minimum_state'] = _r2_state_sample(
                env, state, timestamp_s+raw_fraction*env.config.dynamics.integration_dt_s,
                diagnostic['episode_origin_timestamp_s'],
                basis='LINEAR_INTERPOLATED_EXECUTED_RK2_SEGMENT', command=command)
        for key, radius in (('first_within_5m_time_s', 5.0), ('first_within_3m_time_s', 3.0)):
            if diagnostic[key] is None:
                entry = _first_ball_entry(start.position_ned_m,
                                          proposal.auv_end_state.position_ned_m, goal, radius)
                if entry is not None and entry <= fraction+1.0e-8:
                    diagnostic[key] = (timestamp_s-diagnostic['episode_origin_timestamp_s']
                                       + entry*env.config.dynamics.integration_dt_s)
        return proposal

    world._propose_integration_step = observe
    try:
        yield
    finally:
        if original_instance is None:
            world.__dict__.pop('_propose_integration_step', None)
        else:
            world._propose_integration_step = original_instance


def r2_diagnostic_step(env: B0NavigationEnv, diagnostic: dict[str, Any], action: np.ndarray,
                       original_step: Callable[..., Any]) -> Any:
    """原动作只执行一次，复用R1事件诊断，记录实际控制节点八维快照。"""
    with _observe_r2_world_step(env, diagnostic):
        result = diagnostic_step(env, diagnostic, action, original_step)
    info = result[4]
    sample = _r2_state_sample(
        env, env.world.auv_state, env.world.timestamp_s, diagnostic['episode_origin_timestamp_s'],
        basis='ACTUAL_CONTROL_NODE', action=action, command=info['executed_action_physical'])
    clearance = info['minimum_clearance']
    sample.update(task_step=info['task_control_step'], reward=result[1],
                  reward_components=deepcopy(info['reward_components']),
                  elapsed_s=info['elapsed_s'],
                  minimum_clearance_m=float(clearance) if np.isfinite(clearance) else None,
                  failure_type=info['failure_type'])
    diagnostic['_r2_trajectory'].append(sample)
    return result


def r2_diagnostic_validation(pool: FixedValidationPool, harness: B0TrainingHarness,
                             full: bool, progress_callback: Callable[..., None] | None = None,
                             ) -> dict[str, Any]:
    """固定选择0/1/2、升序最早失败/成功，其他案例只保存标量；方法在finally恢复。"""
    before, before_rng = pool._training_state(harness), pool._rng_state()
    original_factory = harness._make_env
    factory_instance = harness.__dict__.get('_make_env')
    diagnostics: dict[str, dict[str, Any]] = {}
    reasons: dict[str, list[str]] = {}
    originals: dict[int, tuple[Any, Any, Any]] = {}
    first: set[str] = set()

    def restore_env(env: B0NavigationEnv) -> None:
        """每例结束即释放环境包装，避免为恢复方法保留300份感知历史。"""
        _, reset, step = originals.pop(id(env))
        for key, value in (('reset', reset), ('step', step)):
            if value is None:
                env.__dict__.pop(key, None)
            else:
                setattr(env, key, value)

    def factory(scenario: Any) -> B0NavigationEnv:
        """只包装本次评价的新环境；不碰训练槽、动作或随机流。"""
        env = original_factory(scenario)
        original_reset, original_step = env.reset, env.step
        originals[id(env)] = (env, env.__dict__.get('reset'), env.__dict__.get('step'))

        def reset(*args: Any, **kwargs: Any) -> Any:
            """原合法暖机后记录正式初态，暖机不混入策略trajectory。"""
            result = original_reset(*args, **kwargs)
            diagnostics[scenario.scenario_id] = new_r2_episode_diagnostics(env)
            return result

        def step(action: np.ndarray) -> Any:
            """原deterministic动作执行一次；结束后只保留预登记或首次事件轨迹。"""
            diagnostic = diagnostics[scenario.scenario_id]
            result = r2_diagnostic_step(env, diagnostic, action, original_step)
            if result[2] or result[3]:
                selected = (['preregistered_index']
                            if scenario.scenario_index in pool.TRAJECTORY_INDICES else [])
                event_kind = 'success' if result[4]['failure_type'] == 'goal_success' else 'failure'
                if event_kind not in first:
                    selected.append(f'earliest_{event_kind}_by_index')
                    first.add(event_kind)
                reasons[scenario.scenario_id] = selected
                if not selected:
                    for key in ('_r2_trajectory', '_r2_initial_state', '_r2_minimum_state'):
                        diagnostic.pop(key)
                restore_env(env)
            return result

        env.reset, env.step = reset, step
        return env

    harness._make_env = factory
    try:
        result = pool.evaluate(harness, 'obstacle_free', full, progress_callback)
    except BaseException:
        metadata = harness.failure_metadata
        current = None if metadata is None else metadata.get('current_validation_episode')
        if isinstance(current, dict) and current.get('scenario_id') in diagnostics:
            metadata['r2_current_diagnostics'] = deepcopy(diagnostics[current['scenario_id']])
        raise
    finally:
        for env, _, _ in tuple(originals.values()):
            restore_env(env)
        if factory_instance is None:
            harness.__dict__.pop('_make_env', None)
        else:
            harness._make_env = factory_instance
    if not (states_equal(before, pool._training_state(harness))
            and states_equal(before_rng, pool._rng_state())):
        harness.failure_metadata = dict(operation='r2_diagnostic_validation',
                                         error='diagnostic_wrapper_changed_training_state',
                                         checkpoint_safe=False)
        raise RuntimeError('R2只读诊断改变训练状态或随机流。')
    for episode in result['episodes']:
        identity = episode['scenario_id']
        diagnostic = diagnostics[identity]
        episode.update({key: deepcopy(value) for key, value in diagnostic.items()
                        if not key.startswith('_r2_')})
        if reasons[identity]:
            episode.update(trajectory=diagnostic['_r2_trajectory'],
                           formal_initial_state=diagnostic['_r2_initial_state'],
                           minimum_goal_state=diagnostic['_r2_minimum_state'],
                           trajectory_retention_reasons=reasons[identity])
    result['r2_diagnostic_state_unchanged'] = True
    result['r2_state_order'] = ['N', 'E', 'D', 'yaw', 'pitch', 'surge', 'yaw_rate', 'pitch_rate']
    return result


def harness_config(registration: dict[str, Any], group: str, seed: int) -> B0HarnessConfig:
    """组身份不进入随机派生；仅goal权重不同，所有原科学SAC参数保持。"""
    if group not in registration['groups'] or seed not in registration['training_seeds']:
        raise ValueError('R2组或seed未授权。')
    sac = derived_sac_config(SACConfig(device=registration['device'], **registration['sac']),
                            training_seed=seed, run_kind='scientific_training')
    return B0HarnessConfig(
        run_kind='scientific_training', task_profile='obstacle_free', training_seed=seed,
        transition_budget=300000, num_envs=2, validation_episodes=30,
        research_registration=REGISTRATION_ID, sac=sac,
        task=LocalTaskConfig(w_goal=registration['groups'][group]['w_goal']))


class B0R2Harness(B0RepairHarness):
    """只扩展登记、300k固定验证及reward身份，继承同一真实step与诊断路径。"""

    FORMAT = 'b0-r2-full-resume-v1'
    END = 300000
    MONITOR_POINTS = tuple(range(0, END + 1, 25000))
    FULL_POINTS = (100000, END)

    def __init__(self, config: B0HarnessConfig, project_config: ProjectConfig,
                 scenario_config: TrainingScenarioConfig, *, group: str, code_version: str,
                 evaluation_callback: EvaluationCallback,
                 agent: OrdinarySACAgent | None = None, env_factory: EnvFactory | None = None,
                 log_sink: LogSink | None = None) -> None:
        """严格同组reward/预算/方法；初始化不执行任何真实transition或更新。"""
        expected = APPROVED['groups'].get(group)
        if (expected is None or config.run_kind != 'scientific_training'
                or config.research_registration != REGISTRATION_ID
                or config.task_profile != 'obstacle_free'
                or config.training_seed not in (11, 22, 33)
                or config.transition_budget != self.END or config.num_envs != 2
                or config.task != LocalTaskConfig(w_goal=expected['w_goal'])
                or config.sac.learning_rate != 0.0003
                or config.validation_interval != 25000 or config.checkpoint_interval != 25000
                or config.validation_max_steps != 1000 or config.validation_episodes != 30):
            raise ValueError('R2必须匹配登记组reward、原学习率、300k无障碍和固定验证。')
        self.group = group
        self.completed_validation_keys: set[str] = set()
        self.final_budget_logged = False
        self.evaluation_warmup_control_transitions = 0
        self.evaluation_callback = evaluation_callback
        B0TrainingHarness.__init__(self, config, project_config, scenario_config,
                                   code_version=code_version, agent=agent,
                                   env_factory=env_factory, log_sink=log_sink)

    def _emit(self, kind: str, record: dict[str, Any]) -> None:
        """各日志保留实际reward版本，不能把不同效用未说明地直接比较。"""
        record.update(group=self.group, reward_config=asdict(self.config.task))
        super()._emit(kind, record)

    def _repair_state(self) -> dict[str, Any]:
        """完整恢复包含组、共同任务reward和全部固定验证位置。"""
        return dict(registration_id=REGISTRATION_ID, group=self.group,
                    reward_config=asdict(self.config.task), endpoint=self.END,
                    completed_validation_keys=sorted(self.completed_validation_keys),
                    final_budget_logged=self.final_budget_logged,
                    evaluation_warmup_control_transitions=self.evaluation_warmup_control_transitions)

    @classmethod
    def required_validation_keys(cls, transition: int) -> set[str]:
        """100k/300k同时有monitor30及额外Val300，恢复不能重复或遗漏。"""
        return ({f'obstacle_free:{point}:monitor' for point in cls.MONITOR_POINTS
                 if point <= transition}
                | {f'obstacle_free:{point}:full' for point in cls.FULL_POINTS
                   if point <= transition})

    def ensure_validation(self, full: bool | None = None) -> dict[str, Any]:
        """每25k monitor，100/300k追加full；已完成键幂等，状态直接比较。"""
        if full is None:
            monitor = self.ensure_validation(False)
            if self.transitions in self.FULL_POINTS:
                return self.ensure_validation(True)
            return monitor
        if self._inside_transition or self.failure_metadata is not None:
            raise RuntimeError('R2固定验证要求无失败的完整控制步边界。')
        points = self.FULL_POINTS if full else self.MONITOR_POINTS
        if not isinstance(full, bool) or self.transitions not in points:
            raise ValueError('R2验证只能在登记时刻/规模执行。')
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
                raise RuntimeError('R2评价改变训练状态或随机流。')
            if result.get('count') != (300 if full else 30):
                raise ValueError('R2固定验证分母不符。')
            self.evaluation_env_transitions += result['evaluation_env_transitions']
            self.evaluation_warmup_control_transitions += result['warmup_control_transitions']
            self.validation_count += 1
            result.update(profile='obstacle_free', full=full, at_transition=self.transitions,
                          evaluation_key=key, training_state_unchanged=True, diagnostic_only=False)
            self._emit('validation', result)
            self.completed_validation_keys.add(key)
            return result
        except BaseException as error:
            self.failure_metadata = {**deepcopy(self.failure_metadata or {}),
                                     'transition': self.transitions, 'evaluation_key': key,
                                     'error': repr(error), 'checkpoint_safe': False}
            raise

    def record_final_budget_stop(self) -> None:
        """预算不制造物理结束，也不多采样等待episode结束。"""
        if (self.transitions != self.END or self.completed_validation_keys
                != self.required_validation_keys(self.END)):
            raise RuntimeError('R2预算片段须在全部固定终点评价完成以后登记。')
        if not self.final_budget_logged:
            for slot in self.slots:
                if not slot['needs_reset']:
                    record = self._episode_record(slot, complete=False, failure_type='none',
                                                  budget_stop=True)
                    self.budget_stop_records.append(record)
                    self._emit('episode', record)
            self.final_budget_logged = True

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """先检查组/reward/验证及预算，再恢复任何模型/RNG，拒绝跨组续点。"""
        repair = state.get('repair', {})
        expected = self._repair_state()
        if any(repair.get(key) != expected[key]
               for key in ('registration_id', 'group', 'reward_config', 'endpoint')):
            raise ValueError('R2恢复组/reward/登记/预算不一致。')
        transition = state.get('scheduler', {}).get('transitions')
        keys = repair.get('completed_validation_keys')
        if (isinstance(transition, bool) or not isinstance(transition, int)
                or not 0 <= transition <= self.END or not isinstance(keys, list)
                or len(keys) != len(set(keys))
                or set(keys) != self.required_validation_keys(transition)):
            raise ValueError('R2恢复计数或验证键不完整。')
        if (not isinstance(repair.get('final_budget_logged'), bool)
                or (repair['final_budget_logged'] and transition != self.END)
                or state['scheduler']['validation_count'] != len(keys)):
            raise ValueError('R2恢复终点评价/片段计数不一致。')
        warmup = repair.get('evaluation_warmup_control_transitions')
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError('R2恢复评价暖机计数损坏。')
        B0TrainingHarness.load_state_dict(self, state)
        self.completed_validation_keys = set(keys)
        self.final_budget_logged = repair['final_budget_logged']
        self.evaluation_warmup_control_transitions = warmup


def save_network_snapshot(harness: B0R2Harness, output: Path) -> Path:
    """固定25k小模型保留五网络及alpha，非best选择，不复制Replay。"""
    models = output / 'models'
    models.mkdir(exist_ok=True)
    path = models / f'seed_{harness.config.training_seed}_{harness.transitions}.pt'
    networks = {name: {key: tensor.detach().cpu() for key, tensor
                      in getattr(harness.agent, name).state_dict().items()}
                for name in ('actor', 'q1', 'q2', 'target_q1', 'target_q2')}
    value = dict(format='b0-r2-network-snapshot-v1', models=networks, actor=networks['actor'],
                 log_alpha=harness.agent.log_alpha.detach().cpu(),
                 sac_config=asdict(harness.agent.config), harness_config=harness.config.to_dict(),
                 counters=harness.agent.counters.copy(), seed=harness.config.training_seed,
                 transition=harness.transitions, code_version=harness.code_version,
                 registration_id=REGISTRATION_ID, group=harness.group,
                 run_kind='scientific_training', method=harness.config.method)
    needed = sum(t.numel()*t.element_size() for model in networks.values()
                 for t in model.values()) + 1024**2
    if shutil.disk_usage(output).free < needed:
        raise OSError('R2小模型临时写入空间不足。')
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


def validate_resume_inventory(output: Path, state: dict[str, Any],
                              registration: dict[str, Any]) -> None:
    """恢复前只读核对既有run身份/计数/可靠点，禁止掉点后隐式从零重训。"""
    order = registration['run_order']
    completed, runs = state.get('completed_runs'), state.get('runs')
    if (not isinstance(completed, list) or not isinstance(runs, dict)
            or completed != order[:len(completed)] or not set(runs).issubset(order)
            or not set(completed).issubset(runs)):
        raise ValueError('R2恢复run清单或顺序损坏。')
    endpoint = registration['transition_budget_per_run']
    learning_starts = registration['sac']['learning_starts']
    for run_id in order:
        group, seed_text = run_id.split('_seed')
        seed = int(seed_text)
        seed_dir = output / group / f'seed_{seed}'
        checkpoint = seed_dir / 'latest_resume.pt'
        row = runs.get(run_id)
        if (row is not None or seed_dir.exists()) and not checkpoint.is_file():
            raise RuntimeError(f'R2已有run的可靠checkpoint缺失：{run_id}，禁止隐式从零重训。')
        if row is None:
            if checkpoint.exists():
                raise ValueError(f'R2恢复checkpoint缺少对应run记录：{run_id}。')
            continue
        if (not isinstance(row, dict) or row.get('group') != group or row.get('seed') != seed
                or row.get('reward_config') != registration['groups'][group]):
            raise ValueError(f'R2恢复run组/seed/reward身份损坏：{run_id}。')
        transition, updates = row.get('transitions'), row.get('updates')
        if (type(transition) is not int or not 0 <= transition <= endpoint
                or type(updates) is not int
                or not 0 <= updates <= max(0, transition-learning_starts+1)
                or row.get('status') not in ('RUNNING', 'COMPLETED')):
            raise ValueError(f'R2恢复run计数或状态损坏：{run_id}。')
        if (row['status'] == 'COMPLETED'
                and (transition != endpoint or updates != endpoint-learning_starts+1
                     or row.get('checkpoint_transition') != endpoint)):
            raise ValueError(f'R2恢复完成run缺少完整预算/更新/终点checkpoint：{run_id}。')
        if run_id in completed and row['status'] != 'COMPLETED':
            raise ValueError(f'R2恢复completed_runs与实际run状态不符：{run_id}。')
    if state.get('status') == 'COMPLETED' and completed != order:
        raise ValueError('R2完成批次未包含全部六个完整run。')


def run_r2_batch(root: Path, registration: dict[str, Any], *,
                 resume: bool = False) -> dict[str, Any]:
    """单worker串行六run，正常低SR不提前停，故障不自动重试或追加预算。"""
    if registration != APPROVED:
        raise ValueError('R2必须使用完整canonical登记。')
    output = root / registration['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'batch_state.json'
    if state_path.exists() != resume:
        raise RuntimeError('R2已有状态须显式resume；无状态不能恢复。')
    if not resume and any((output / group).exists() for group in registration['groups']):
        raise RuntimeError('R2新run输出存在旧组产物，拒绝混入。')
    state = (json.loads(state_path.read_text(encoding='utf-8')) if resume else dict(
        registration_id=REGISTRATION_ID, registration=registration,
        experiment_code_commit=git(root, 'rev-parse', 'HEAD'), created_at=utc(),
        status='PENDING', completed_runs=[], runs={}))
    if state['registration'] != registration or state['registration_id'] != REGISTRATION_ID:
        raise ValueError('R2恢复登记不符。')
    if state['status'] == 'FAILED':
        raise RuntimeError('失败R2不自动恢复；先核对故障、可靠恢复点和重算预算。')
    if resume:
        validate_resume_inventory(output, state, registration)
    verify_code_identity(root, state['experiment_code_commit'])
    if state['status'] == 'COMPLETED':
        return state
    lock = BatchLock(output / 'batch.lock', resume=resume)
    logger: SegmentLog | None = None
    active: B0R2Harness | None = None
    try:
        resource_identity(output)
        project = load_project_config(root / registration['project_config'])
        scenario = load_training_scenario_config(root / registration['scenario_config'])
        pool = FixedValidationPool(project, scenario, root_seed=20261006)
        atomic_json(output / 'validation_manifest.json', pool.compact_manifest())
        state.update(status='RUNNING', pid=os.getpid(), started_at=utc())
        atomic_json(state_path, state)
        for run_id in registration['run_order']:
            if run_id in state['completed_runs']:
                continue
            group, seed_text = run_id.split('_seed')
            seed = int(seed_text)
            group_dir = output / group
            seed_dir = group_dir / f'seed_{seed}'
            seed_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = seed_dir / 'latest_resume.pt'
            row = state['runs'].setdefault(run_id, dict(
                group=group, seed=seed, training_seconds=0.0, evaluation_seconds=0.0,
                checkpoint_seconds=0.0, reward_config=registration['groups'][group],
                transitions=0, updates=0))
            row['status'] = 'RUNNING'
            state.update(current_run=run_id, current_operation='INITIALIZING')

            def evaluate(current: B0R2Harness, profile: str, full: bool,
                         row: dict[str, Any] = row) -> dict[str, Any]:
                """独立固定评价计时和实际进展，不参与科研更新。"""
                if profile != 'obstacle_free':
                    raise ValueError('R2不运行CV验证。')
                state.update(current_operation='EVALUATING', evaluation_full=full,
                             evaluation_at_transition=current.transitions, last_progress_at=utc())
                atomic_json(state_path, state)
                start = time.perf_counter()

                def on_episode(index: int, count: int, steps: int, warmup: int) -> None:
                    """实际案例完成后更新，不能用计划数字冒充运行。"""
                    state.update(current_validation_index=index,
                                 current_validation_completed_episodes=count,
                                 current_validation_transitions=steps,
                                 current_validation_warmup_transitions=warmup,
                                 last_progress_at=utc())
                    atomic_json(state_path, state)

                try:
                    return r2_diagnostic_validation(pool, current, full, on_episode)
                finally:
                    row['evaluation_seconds'] += time.perf_counter()-start
                    state['current_operation'] = 'TRAINING'

            harness = B0R2Harness(harness_config(registration, group, seed), project, scenario,
                                  group=group, code_version=state['experiment_code_commit'],
                                  evaluation_callback=evaluate)
            active = harness
            if checkpoint.exists():
                harness.load_checkpoint(checkpoint, trusted_local=True)
            logger = SegmentLog(group_dir, seed,
                                parent=checkpoint if checkpoint.exists() else None,
                                cutoff=harness.log_sequence if checkpoint.exists() else None,
                                registration_id=REGISTRATION_ID)
            harness.log_sink = logger

            def progress(harness: B0R2Harness = harness, row: dict[str, Any] = row) -> None:
                """只读取真实计数给出剩余训练估算，不包含未来评价或保存。"""
                row.update(transitions=harness.transitions,
                           updates=harness.agent.counters['gradient_updates'],
                           optimizer_steps=sum(harness.agent.counters[key]
                                               for key in ('actor', 'q1', 'q2', 'alpha')),
                           replay_size=len(harness.agent.replay),
                           evaluation_env_transitions=harness.evaluation_env_transitions,
                           evaluation_warmup_control_transitions=(
                               harness.evaluation_warmup_control_transitions),
                           training_warmup_transitions=harness.started_episodes*5,
                           completed_episodes=harness.completed_episodes, last_progress_at=utc())
                done = sum(item['transitions'] for item in state['runs'].values())
                seconds = sum(item['training_seconds'] for item in state['runs'].values())
                state.update(actual_training_transitions=done,
                             actual_sac_updates=sum(item['updates']
                                                    for item in state['runs'].values()),
                             last_progress_at=utc(), remaining_training_hours_estimate=(
                                 (1800000-done)*seconds/done/3600 if done else None),
                             estimate_excludes_future_evaluation_and_save=True)
                atomic_json(state_path, state)
                print(json.dumps(dict(run=state['current_run'], transition=harness.transitions,
                                      updates=row['updates'], operation=state['current_operation'],
                                      time=utc())), flush=True)

            def save(status: str = 'RUNNING', harness: B0R2Harness = harness,
                     logger: SegmentLog = logger, row: dict[str, Any] = row,
                     checkpoint: Path = checkpoint,
                     progress: Callable[..., None] = progress) -> None:
                """flush后安全替换完整恢复点，再确认日志有效前缀。"""
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
                    save_network_snapshot(harness, group_dir)
                    if harness.transitions < harness.END:
                        save()
                elif harness.transitions % 100 == 0:
                    progress()
            harness.record_final_budget_stop()
            row['status'] = 'COMPLETED'
            save('COMPLETED')
            state['completed_runs'].append(run_id)
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
                     recovery_note='No automatic retry; inspect reliable checkpoint and budget.')
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
