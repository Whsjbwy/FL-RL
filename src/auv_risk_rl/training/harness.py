"""B0 普通 SAC 组合式调度、独立验证与控制步边界的完整恢复。"""

from __future__ import annotations

import math
import os
import shutil
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.b0_navigation import B0NavigationEnv
from auv_risk_rl.env.local_task import LocalTaskConfig
from auv_risk_rl.env.scenario_generator import TrainingScenario, TrainingScenarioConfig
from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.training.config import B0HarnessConfig
from auv_risk_rl.training.scenarios import B0ScenarioSource

LogSink = Callable[[str, dict[str, Any]], None]
EnvFactory = Callable[[TrainingScenario], B0NavigationEnv]


def states_equal(left: Any, right: Any) -> bool:
    """直接比较离散、数组、随机流与环境对象状态；不生成任何摘要。"""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right, equal_nan=True))
    if isinstance(left, np.random.Generator):
        return states_equal(left.bit_generator.state, right.bit_generator.state)
    if isinstance(left, np.random.SeedSequence):
        return states_equal(left.state, right.state)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(states_equal(left[k], right[k]) for k in left)
    if isinstance(left, list | tuple):
        return len(left) == len(right) and all(
            states_equal(a, b) for a, b in zip(left, right, strict=True))
    if hasattr(left, '__dict__'):
        return states_equal(vars(left), vars(right))
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    return bool(left == right)


class B0TrainingHarness:
    """固定 round-robin 调度；预算耗尽不伪造环境终止或科研结论。"""

    FORMAT = 'b0-harness-full-resume-v1'

    def __init__(self, config: B0HarnessConfig, project_config: ProjectConfig,
                 scenario_config: TrainingScenarioConfig, *, code_version: str,
                 agent: OrdinarySACAgent | None = None, env_factory: EnvFactory | None = None,
                 log_sink: LogSink | None = None) -> None:
        """组合既有普通 Agent 与 B0 环境，初始不执行 warm-up、采样或更新。"""
        if not code_version:
            raise ValueError('必须登记实际 Git 代码版本。')
        self.config, self.project_config = config, project_config
        self.scenario_config, self.code_version = scenario_config, code_version
        self.agent = (OrdinarySACAgent(config.sac, source_fingerprint=f'git:{code_version}')
                      if agent is None else agent)
        # B0身份不能仅凭相同配置判断；成本组合器和覆盖Actor的子类不是普通SAC。
        if type(self.agent) is not OrdinarySACAgent:
            schedule_fixture = getattr(self.agent, '_b0_schedule_fixture', False) is True
            if not schedule_fixture or env_factory is None or hasattr(self.agent, 'actor'):
                raise TypeError('B0必须使用原OrdinarySACAgent；夹具须显式标记且无Actor模块。')
        if self.agent.config != config.sac:
            raise ValueError('Agent 与 harness SAC 配置必须一致。')
        self.source = B0ScenarioSource(config, project_config, scenario_config)
        self.env_factory = env_factory or self.source.make_env
        self.log_sink = log_sink
        self.slots: list[dict[str, Any]] = [dict(needs_reset=True) for _ in range(config.num_envs)]
        self.next_slot = self.next_scenario_index = self.validation_scenario_index = 0
        self.transitions = self.completed_episodes = self.started_episodes = 0
        self.next_validation_transition = config.validation_interval
        self.next_checkpoint_transition = config.checkpoint_interval
        self.validation_count = self.log_sequence = 0
        self.evaluation_env_transitions = 0
        self.episode_log: list[dict[str, Any]] = []
        self.update_log: list[dict[str, Any]] = []
        self.validation_log: list[dict[str, Any]] = []
        self.budget_stop_records: list[dict[str, Any]] = []
        self._inside_transition = False
        self.failure_metadata: dict[str, Any] | None = None

    def _emit(self, kind: str, record: dict[str, Any]) -> None:
        """完整日志可流式写出；内存只保留最近一千条诊断。"""
        self.log_sequence += 1
        record.update(log_sequence=self.log_sequence, code_version=self.code_version,
                      method=self.config.method, run_kind=self.config.run_kind)
        if self.log_sink is not None:
            self.log_sink(kind, deepcopy(record))
        target = getattr(self, kind + '_log', None)
        if target is not None:
            target.append(deepcopy(record))
            del target[:-1000]

    def _make_env(self, scenario: TrainingScenario) -> B0NavigationEnv:
        """真实B0必须使用专用环境；在reset/warm-up前阻止误用有限感知验证路径。"""
        env = self.env_factory(scenario)
        if type(self.agent) is OrdinarySACAgent and type(env) is not B0NavigationEnv:
            raise TypeError('真实B0必须使用原B0NavigationEnv，不能启用执行过滤环境。')
        if type(env) is B0NavigationEnv and env.task_config != self.config.task:
            raise ValueError('B0环境任务奖励必须与已登记训练配置一致。')
        return env

    def _task_profile(self) -> str:
        """原入口为固定profile；课程适配层可以显式覆盖发行身份。"""
        return self.config.task_profile

    def _reset_slot(self, slot_index: int) -> None:
        """连续发行场景；warm-up 不计正式 transition、Replay 或起步。"""
        index = self.next_scenario_index
        profile = self._task_profile()
        scenario = self.source.scenario(index, task_profile=profile)
        env = self._make_env(scenario)
        options = {'external_max_steps': self.config.external_max_steps}
        obs, warmup = env.reset(seed=self.source.environment_seed(index), options=options)
        self.slots[slot_index] = dict(
            env=env, obs=obs.copy(), needs_reset=False, scenario_index=index,
            scenario_id=scenario.scenario_id, scenario_root_seed=scenario.root_seed,
            episode_id=self.started_episodes, env_slot=slot_index, steps=0,
            task_profile=profile, start_transition=self.transitions + 1,
            last_transition=self.transitions,
            physical_time_s=0.0, reward=0.0, reward_components={}, path_length_m=0.0,
            minimum_clearance_m=None, action_saturation_count=0,
            initial_position_ned_m=env.world.auv_state.position_ned_m.copy(),
            warmup=deepcopy(warmup), trajectory=[])
        self.next_scenario_index += 1
        self.started_episodes += 1

    def _episode_record(self, slot: dict[str, Any], *, complete: bool,
                        failure_type: str, budget_stop: bool = False) -> dict[str, Any]:
        """完成、外部截断和预算停机分开；无障碍间距或未测量指标使用 None。"""
        keys = ('scenario_index', 'scenario_id', 'scenario_root_seed', 'episode_id', 'env_slot',
                'steps', 'physical_time_s', 'reward', 'reward_components', 'path_length_m',
                'minimum_clearance_m', 'action_saturation_count', 'warmup',
                'start_transition', 'last_transition')
        return {**{k: deepcopy(slot[k]) for k in keys}, 'training_seed': self.config.training_seed,
                'split': 'train', 'task_profile': slot['task_profile'],
                'at_transition': self.transitions,
                'failure_type': failure_type, 'complete': complete, 'budget_stop': budget_stop,
                'success': failure_type == 'goal_success', 'collision': failure_type == 'collision',
                'boundary': failure_type == 'operational_boundary_failure',
                'task_timeout': failure_type == 'task_horizon',
                'external_truncation': failure_type == 'external_truncation',
                'path_length_definition': 'CONTROL_NODE_POLYLINE',
                'task_config': asdict(self.config.task),
                'config': self.config.to_dict()}

    def _assert_can_step(self) -> None:
        """无显式运行身份/预算不采样；工程更新硬上限在计算前检查。"""
        if self.failure_metadata is not None:
            raise RuntimeError('当前运行已有失败，必须保存故障证据并停止。')
        if self.config.run_kind == 'preflight':
            raise RuntimeError('preflight 不执行采样或更新。')
        if self.config.run_kind == 'scientific_training' and not self.config.research_registration:
            raise RuntimeError('科研运行安排尚未登记，不能开始训练。')
        if self.transitions >= int(self.config.transition_budget or 0):
            raise RuntimeError('已达到显式 transition 预算。')
        prospective = self.agent.counters['environment_steps'] + 1
        eligible = (prospective >= self.config.sac.learning_starts
                    and min(len(self.agent.replay) + 1, self.config.sac.replay_capacity)
                    >= self.config.sac.batch_size)
        if (self.config.run_kind == 'engineering_smoke' and eligible
                and self.agent.counters['gradient_updates'] >= 128):
            raise RuntimeError('已达到工程完整 SAC update 上限，拒绝追加采样。')

    def step(self) -> dict[str, Any]:
        """一次真实控制 transition 后一次 eligible SAC update，不提前 reset。"""
        self._assert_can_step()
        self._inside_transition = True
        try:
            slot_index = self.next_slot
            if self.slots[slot_index]['needs_reset']:
                self._reset_slot(slot_index)
            slot = self.slots[slot_index]
            env, obs = slot['env'], slot['obs']
            before = env.world.auv_state.position_ned_m.copy()
            nominal = self.agent.sample_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(nominal)
            self.agent.store_transition(
                obs=obs, nominal_action=nominal, executed_action=info['executed_action_normalized'],
                reward=reward, cost=info['cost'], next_obs=next_obs, terminated=terminated,
                truncated=truncated, failure_type=info['failure_type'],
                episode_id=slot['episode_id'], task_step=info['task_control_step'])
            slot['obs'] = next_obs.copy()
            self.transitions += 1
            slot['last_transition'] = self.transitions
            slot['steps'] += 1
            slot['physical_time_s'] += info['elapsed_s']
            slot['reward'] += reward
            for key, value in info['reward_components'].items():
                slot['reward_components'][key] = slot['reward_components'].get(key, 0.0) + value
            after = env.world.auv_state.position_ned_m.copy()
            slot['path_length_m'] += float(np.linalg.norm(after - before))
            clearance = info['minimum_clearance']
            if math.isfinite(clearance):
                previous = slot['minimum_clearance_m']
                slot['minimum_clearance_m'] = (clearance if previous is None
                                               else min(previous, clearance))
            # 归一化边缘 1e-6 仅用于动作饱和诊断，不修改动作或模型。
            slot['action_saturation_count'] += int(np.any(np.abs(nominal) >= 1.0 - 1.0e-6))
            slot['trajectory'].append(dict(task_step=info['task_control_step'],
                                           position_ned_m=after.tolist(),
                                           action=nominal.tolist(),
                                           failure_type=info['failure_type']))
            metrics = self.agent.update() if self.agent.eligible() else {}
            if metrics:
                self._emit('update', dict(transition=self.transitions, metrics=metrics,
                                          counters=self.agent.counters.copy()))
            if terminated or truncated:
                slot['needs_reset'] = True
                self.agent.counters['episodes'] += 1
                self.completed_episodes += int(terminated)
                record = self._episode_record(slot, complete=terminated,
                                              failure_type=info['failure_type'])
                if info['failure_type'] != 'goal_success':
                    record['trajectory'] = deepcopy(slot['trajectory'])
                self._emit('episode', record)
            self.next_slot = (slot_index + 1) % self.config.num_envs
        except BaseException as error:
            self.failure_metadata = dict(transition=self.transitions, error=repr(error),
                                         agent_failure=getattr(self.agent,
                                                               'last_failure_metadata', None))
            raise
        finally:
            self._inside_transition = False
        validation = None
        if self.transitions >= self.next_validation_transition:
            validation = self.evaluate()
            self.next_validation_transition += self.config.validation_interval
        checkpoint_due = self.transitions >= self.next_checkpoint_transition
        if checkpoint_due:
            self.next_checkpoint_transition += self.config.checkpoint_interval
        return dict(transition=self.transitions, slot=slot_index, scenario_id=slot['scenario_id'],
                    metrics=metrics, terminated=terminated, truncated=truncated, info=info,
                    validation=validation, checkpoint_due=checkpoint_due)

    def run(self, checkpoint_callback: Callable[[B0TrainingHarness], None] | None = None,
            ) -> dict[str, Any]:
        """执行已明确预算；只记录预算停止，不把半段 episode 改为物理 timeout。"""
        self._assert_can_step()
        while self.transitions < int(self.config.transition_budget or 0):
            result = self.step()
            if result['checkpoint_due'] and checkpoint_callback is not None:
                checkpoint_callback(self)
        for slot in self.slots:
            if not slot['needs_reset']:
                record = self._episode_record(slot, complete=False, failure_type='none',
                                              budget_stop=True)
                self.budget_stop_records.append(record)
                self._emit('episode', record)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        """分别返回真实采样、完整 update 和 Adam 操作计数，不推断科学性能。"""
        return dict(transitions=self.transitions, updates=self.agent.counters['gradient_updates'],
                    evaluation_env_transitions=self.evaluation_env_transitions,
                    failure_metadata=deepcopy(self.failure_metadata),
                    optimizer_steps=sum(self.agent.counters[name]
                                        for name in ('actor', 'q1', 'q2', 'alpha')),
                    scientific_training_steps=(self.transitions
                                               if self.config.run_kind == 'scientific_training'
                                               else 0),
                    scientific_training_updates=(self.agent.counters['gradient_updates']
                                                 if self.config.run_kind == 'scientific_training'
                                                 else 0))

    def _training_state(self) -> dict[str, Any]:
        """验证前后的直接比较快照，不包括独立验证发行位置与其日志。"""
        return dict(agent=self.agent.state_dict(), slots=deepcopy(self.slots),
                    transitions=self.transitions, completed_episodes=self.completed_episodes,
                    started_episodes=self.started_episodes, next_slot=self.next_slot,
                    next_scenario_index=self.next_scenario_index)

    def evaluate(self) -> dict[str, Any]:
        """独立环境与 deterministic_action；直接确认训练状态及全局随机流未变。"""
        if self._inside_transition:
            raise RuntimeError('验证必须位于完整控制步边界。')
        before = self._training_state()
        torch_rng = torch.get_rng_state().clone()
        numpy_rng = deepcopy(np.random.get_state())
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        episodes = []
        for _ in range(self.config.validation_episodes):
            index = self.validation_scenario_index
            scenario = self.source.scenario(index, 'validation')
            env = self._make_env(scenario)
            env_seed = self.source.environment_seed(index, 'validation')
            options = {'external_max_steps': self.config.validation_max_steps}
            obs, warmup = env.reset(seed=env_seed,
                                    options=options)
            terminated = truncated = False
            episode = dict(scenario_id=scenario.scenario_id, scenario_index=index,
                           scenario_root_seed=scenario.root_seed, environment_seed=env_seed,
                           split='validation', training_seed=self.config.training_seed,
                           episode_id=index, env_slot=None, task_profile=self.config.task_profile,
                           steps=0, physical_time_s=0.0, reward=0.0, reward_components={},
                           path_length_m=0.0, path_length_definition='CONTROL_NODE_POLYLINE',
                           minimum_clearance_m=None, action_saturation_count=0,
                           warmup=deepcopy(warmup), config=self.config.to_dict(),
                           task_config=asdict(self.config.task),
                           method=self.config.method, run_kind=self.config.run_kind,
                           code_version=self.code_version)
            trajectory = []
            while not (terminated or truncated):
                start = env.world.auv_state.position_ned_m.copy()
                action = self.agent.deterministic_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                self.evaluation_env_transitions += 1
                episode['steps'] += 1
                episode['physical_time_s'] += info['elapsed_s']
                episode['reward'] += reward
                for key, value in info['reward_components'].items():
                    previous = episode['reward_components'].get(key, 0.0)
                    episode['reward_components'][key] = previous + value
                end = env.world.auv_state.position_ned_m
                episode['path_length_m'] += float(np.linalg.norm(end - start))
                clearance = info['minimum_clearance']
                if math.isfinite(clearance):
                    previous = episode['minimum_clearance_m']
                    episode['minimum_clearance_m'] = (clearance if previous is None
                                                       else min(clearance, previous))
                episode['action_saturation_count'] += int(np.any(np.abs(action) >= 1.0 - 1.0e-6))
                trajectory.append(dict(task_step=info['task_control_step'],
                                       position_ned_m=end.tolist(), action=action.tolist(),
                                       failure_type=info['failure_type']))
            failure = info['failure_type']
            episode.update(terminated=terminated, truncated=truncated, failure_type=failure,
                           complete=terminated, success=failure == 'goal_success',
                           collision=failure == 'collision',
                           boundary=failure == 'operational_boundary_failure',
                           task_timeout=failure == 'task_horizon',
                           external_truncation=failure == 'external_truncation')
            if failure != 'goal_success':
                episode['trajectory'] = trajectory
            episodes.append(episode)
            self.validation_scenario_index += 1
        after = self._training_state()
        preserved = (states_equal(before, after) and torch.equal(torch_rng, torch.get_rng_state())
                     and states_equal(numpy_rng, np.random.get_state())
                     and (cuda_rng is None
                          or states_equal(cuda_rng, torch.cuda.get_rng_state_all())))
        if not preserved:
            self.failure_metadata = dict(transition=self.transitions,
                                         error='evaluation_changed_training_state')
            raise RuntimeError('独立验证改变了训练状态或随机流，拒绝继续。')
        self.validation_count += 1
        record = dict(at_transition=self.transitions, episodes=episodes,
                      training_state_unchanged=True,
                      diagnostic_only=self.config.run_kind == 'engineering_smoke')
        self._emit('validation', record)
        return record

    def state_dict(self) -> dict[str, Any]:
        """保存各环境/感知历史、随机流、完整 Agent、计数和下一触发位置。"""
        if self._inside_transition or self.failure_metadata is not None:
            raise RuntimeError('只能在无失败的完整 transition/update 边界保存。')
        names = ('slots', 'next_slot', 'next_scenario_index', 'validation_scenario_index',
                 'transitions', 'completed_episodes', 'started_episodes',
                 'next_validation_transition', 'next_checkpoint_transition', 'validation_count',
                 'log_sequence', 'episode_log', 'update_log', 'validation_log',
                 'budget_stop_records', 'evaluation_env_transitions')
        return dict(format=self.FORMAT, config=self.config.to_dict(),
                    torch_version=str(torch.__version__),
                    project_config=self.project_config.to_dict(),
                    scenario_config=asdict(self.scenario_config), code_version=self.code_version,
                    agent=self.agent.state_dict(),
                    scheduler=deepcopy({k: getattr(self, k) for k in names}))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """先检验方法、run_kind、配置和 Git 版本；不接受工程产物作为科研续点。"""
        recorded_config = deepcopy(state.get('config'))
        if isinstance(recorded_config, dict) and 'task' not in recorded_config:
            # 历史默认续点仅补原默认奖励身份；w_goal=200仍与该状态不匹配。
            recorded_config['task'] = asdict(LocalTaskConfig())
        if (state.get('format') != self.FORMAT or recorded_config != self.config.to_dict()
                or state.get('torch_version') != str(torch.__version__)
                or state.get('project_config') != self.project_config.to_dict()
                or state.get('scenario_config') != asdict(self.scenario_config)
                or state.get('code_version') != self.code_version):
            raise ValueError('完整恢复的运行身份、配置、方法、Torch 及 Git 版本必须一致。')
        for slot in state['scheduler']['slots']:
            env = slot.get('env')
            if type(env) is B0NavigationEnv and env.task_config != self.config.task:
                raise ValueError('完整恢复中的B0环境奖励与登记配置不一致。')
        self.agent.load_state_dict(state['agent'])
        for key, value in state['scheduler'].items():
            setattr(self, key, deepcopy(value))

    def save_checkpoint(self, path: str | Path) -> None:
        """可信本机快照临时写完后原子替换；只保留调用者指定的最近恢复点。"""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        replay_bytes = sum(arr.nbytes for block in self.agent.replay.chunks.values()
                           for arr in block.values())
        if shutil.disk_usage(destination.parent).free < replay_bytes + 16 * 1024**2:
            raise OSError('没有足够的临时 checkpoint 写入空间。')
        temporary = destination.with_name(destination.name + '.partial')
        with temporary.open('xb') as stream:
            try:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                stream.close()
                temporary.unlink()
                raise
        os.replace(temporary, destination)

    def load_checkpoint(self, path: str | Path, *, trusted_local: bool = False) -> None:
        """反序列化完整 Python 环境前，必须由调用者确认可信本项目本机来源。"""
        if not trusted_local:
            raise ValueError('拒绝加载来源未确认的完整 checkpoint。')
        # CPU反序列化保留Adam步数标量；既有加载器将模型/矩送回原设备，不是CPU执行回退。
        state = torch.load(Path(path), map_location='cpu', weights_only=False)
        self.load_state_dict(state)
