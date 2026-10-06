"""已登记B0 MVP的100k无障碍→200k CV调度；不改变普通SAC或Replay数学。"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.scenario_generator import TrainingScenarioConfig
from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.training.config import B0HarnessConfig
from auv_risk_rl.training.harness import B0TrainingHarness, EnvFactory, LogSink, states_equal

EvaluationCallback = Callable[['B0CurriculumHarness', str, bool], dict[str, Any]]


class B0CurriculumHarness(B0TrainingHarness):
    """显式课程切换、固定验证幂等键和完整恢复；原工程入口保持原语义。"""

    FORMAT = 'b0-mvp-curriculum-full-resume-v1'
    REGISTRATION_ID = 'STAGE2_B0_MVP_BATCH_V1'
    OBSTACLE_FREE_END = 100000
    TOTAL_END = 300000

    def __init__(self, config: B0HarnessConfig, project_config: ProjectConfig,
                 scenario_config: TrainingScenarioConfig, *, code_version: str,
                 evaluation_callback: EvaluationCallback,
                 agent: OrdinarySACAgent | None = None, env_factory: EnvFactory | None = None,
                 log_sink: LogSink | None = None) -> None:
        """只接受已登记的三seed固定预算；构造不执行采样、验证或更新。"""
        if (config.run_kind != 'scientific_training'
                or config.research_registration != self.REGISTRATION_ID
                or config.transition_budget != self.TOTAL_END
                or config.training_seed not in (11, 22, 33)
                or config.task_profile != 'obstacle_free' or config.num_envs != 2
                or config.validation_interval != 25000 or config.checkpoint_interval != 25000):
            raise ValueError('课程配置必须与STAGE2_B0_MVP_BATCH_V1登记一致。')
        self.phase = 'obstacle_free'
        self.switched = False
        self.phase_boundary_logged = self.final_budget_logged = False
        self.completed_validation_keys: set[str] = set()
        self.evaluation_warmup_control_transitions = 0
        self.evaluation_callback = evaluation_callback
        super().__init__(config, project_config, scenario_config, code_version=code_version,
                         agent=agent, env_factory=env_factory, log_sink=log_sink)

    @property
    def current_profile(self) -> str:
        """100k后仍须显式切换，不能根据计数偷偷改变环境。"""
        return self.phase

    @property
    def phase_transition_counts(self) -> dict[str, int]:
        """一条真实transition只计一次，两个课程段从实际全局计数确定。"""
        return dict(obstacle_free=min(self.transitions, self.OBSTACLE_FREE_END),
                    cv_train_v1=max(self.transitions - self.OBSTACLE_FREE_END, 0))

    def _task_profile(self) -> str:
        """明确选择当前课程的场景发行profile。"""
        return self.phase

    def _assert_can_step(self) -> None:
        """100k边界必须先验证/保存和显式切换，再允许第100001条transition。"""
        super()._assert_can_step()
        if self.transitions >= self.OBSTACLE_FREE_END and not self.switched:
            raise RuntimeError('100k课程边界尚未显式切换，禁止继续空场景采样。')
        if (self.switched and self.transitions == self.OBSTACLE_FREE_END
                and f'cv_train_v1:{self.OBSTACLE_FREE_END}:monitor'
                not in self.completed_validation_keys):
            raise RuntimeError('CV首条transition前必须完成100k固定monitor验证。')

    def _training_state(self) -> dict[str, Any]:
        """直接比较还包括课程状态和已完成验证键，避免回调修改调度。"""
        state = super()._training_state()
        state['curriculum'] = dict(phase=self.phase, switched=self.switched,
                                   validation_keys=sorted(self.completed_validation_keys),
                                   counts=self.phase_transition_counts,
                                   next_validation=self.next_validation_transition,
                                   next_checkpoint=self.next_checkpoint_transition,
                                   log_sequence=self.log_sequence,
                                   phase_boundary_logged=self.phase_boundary_logged,
                                   final_budget_logged=self.final_budget_logged)
        return state

    def evaluate(self, profile: str | None = None, full: bool | None = None) -> dict[str, Any]:
        """覆盖原增量场景验证，使用登记的固定pool回调。"""
        return self.ensure_validation(profile, full)

    def ensure_validation(self, profile: str | None = None,
                          full: bool | None = None) -> dict[str, Any]:
        """相同课程/计数/验证规模最多执行一次；成功后才写完成键。"""
        if self._inside_transition or self.failure_metadata is not None:
            raise RuntimeError('固定验证要求无失败的完整控制步边界。')
        profile = self.phase if profile is None else profile
        if profile != self.phase:
            raise ValueError('验证profile必须与当前显式课程一致。')
        endpoint = self.OBSTACLE_FREE_END if profile == 'obstacle_free' else self.TOTAL_END
        full = self.transitions == endpoint if full is None else full
        monitor_start = 0 if profile == 'obstacle_free' else self.OBSTACLE_FREE_END
        allowed = self.transitions == endpoint if full else self.transitions in range(
            monitor_start, endpoint, self.config.validation_interval)
        if not allowed:
            raise ValueError('固定验证只能在登记的课程/全局transition时刻执行。')
        key = f'{profile}:{self.transitions}:{"full" if full else "monitor"}'
        if key in self.completed_validation_keys:
            return dict(evaluation_key=key, already_completed=True, at_transition=self.transitions,
                        profile=profile, full=full)
        before = self._training_state()
        numpy_rng = deepcopy(np.random.get_state())
        torch_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            record = self.evaluation_callback(self, profile, full)
            preserved = (states_equal(before, self._training_state())
                         and states_equal(numpy_rng, np.random.get_state())
                         and torch.equal(torch_rng, torch.get_rng_state())
                         and (cuda_rng is None
                              or states_equal(cuda_rng, torch.cuda.get_rng_state_all())))
            if not preserved:
                raise RuntimeError('固定验证回调改变了训练状态或随机流。')
            if record.get('count') != (300 if full else 30):
                raise ValueError('固定验证实际案例数与登记的monitor30/Val300不符。')
            self.evaluation_env_transitions += int(record['evaluation_env_transitions'])
            self.evaluation_warmup_control_transitions += int(
                record.get('warmup_control_transitions', 0))
            self.validation_count += 1
            record.update(profile=profile, full=full, at_transition=self.transitions,
                          evaluation_key=key, training_state_unchanged=True, diagnostic_only=False)
            self._emit('validation', record)
            self.completed_validation_keys.add(key)
            return record
        except BaseException as error:
            # 保留固定池已记录的故障episode、轨迹和实际计算计数，不覆盖原始反例。
            metadata = {} if self.failure_metadata is None else deepcopy(self.failure_metadata)
            metadata.update(transition=self.transitions, error=repr(error), evaluation_key=key)
            self.failure_metadata = metadata
            raise

    def switch_to_cv(self) -> dict[str, Any]:
        """记录未完成片段后丢弃两槽；不改Replay flags/reward，不额外推进。"""
        if self.switched:
            return dict(switched=True, already_switched=True, at_transition=self.transitions)
        if self._inside_transition or self.failure_metadata is not None:
            raise RuntimeError('课程切换要求无失败的完整控制步边界。')
        key = f'obstacle_free:{self.OBSTACLE_FREE_END}:full'
        if self.transitions != self.OBSTACLE_FREE_END or key not in self.completed_validation_keys:
            raise RuntimeError('课程切换必须在100k完成空场景Val300以后。')
        if not self.phase_boundary_logged:
            for slot in self.slots:
                if not slot['needs_reset']:
                    record = self._episode_record(slot, complete=False, failure_type='none')
                    record.update(phase_boundary=True, boundary_reason='curriculum_switch')
                    self._emit('episode', record)
            self.phase_boundary_logged = True
        self.slots = [dict(needs_reset=True) for _ in range(self.config.num_envs)]
        self.phase, self.switched = 'cv_train_v1', True
        record = dict(switched=True, already_switched=False, at_transition=self.transitions,
                      from_profile='obstacle_free', task_profile=self.phase,
                      next_scenario_index=self.next_scenario_index,
                      retained_agent_optimizer_replay=True)
        self._emit('course_switch', record)
        return record

    def run_until(self, stop_transition: int,
                  checkpoint_callback: Callable[[B0TrainingHarness], None] | None = None,
                  ) -> dict[str, Any]:
        """仅执行登记的100k/300k段，不自动切换课程或追加预算。"""
        if stop_transition not in (self.OBSTACLE_FREE_END, self.TOTAL_END):
            raise ValueError('批次段终点只能是100k或300k。')
        if stop_transition < self.transitions:
            raise ValueError('不能回退训练计数。')
        if stop_transition == self.TOTAL_END and not self.switched:
            raise RuntimeError('执行CV段前必须显式切换课程。')
        if stop_transition == self.OBSTACLE_FREE_END and self.switched:
            raise RuntimeError('已切换CV，不能重新执行无障碍段终点。')
        if self.transitions == 0:
            self.ensure_validation('obstacle_free', False)
        if self.switched and self.transitions == self.OBSTACLE_FREE_END:
            self.ensure_validation('cv_train_v1', False)
        while self.transitions < stop_transition:
            result = self.step()
            if result['checkpoint_due'] and checkpoint_callback is not None:
                checkpoint_callback(self)
        self.ensure_validation()
        if stop_transition == self.TOTAL_END:
            self.record_final_budget_stop()
        return self.summary()

    def record_final_budget_stop(self) -> None:
        """记录300k未完成片段一次，不能把总预算停机变成物理终止。"""
        key = f'cv_train_v1:{self.TOTAL_END}:full'
        if (self.transitions != self.TOTAL_END or not self.switched
                or key not in self.completed_validation_keys):
            raise RuntimeError('最终预算片段必须在300k固定CV Val300完成后登记。')
        if not self.final_budget_logged:
            for slot in self.slots:
                if not slot['needs_reset']:
                    record = self._episode_record(slot, complete=False, failure_type='none',
                                                  budget_stop=True)
                    record['phase_boundary'] = False
                    self.budget_stop_records.append(record)
                    self._emit('episode', record)
            self.final_budget_logged = True

    def run(self, checkpoint_callback: Callable[[B0TrainingHarness], None] | None = None,
            ) -> dict[str, Any]:
        """拒绝原单段入口自动越过100k课程边界。"""
        raise RuntimeError('固定课程必须使用显式run_until，不自动切换或追加训练。')

    def summary(self) -> dict[str, Any]:
        """报告实际计数、课程段和固定验证完成键。"""
        result = super().summary()
        result.update(phase=self.phase, switched=self.switched,
                      phase_transition_counts=self.phase_transition_counts,
                      completed_validation_keys=sorted(self.completed_validation_keys),
                      evaluation_warmup_control_transitions=self.evaluation_warmup_control_transitions)
        return result

    def state_dict(self) -> dict[str, Any]:
        """完整恢复包含课程/验证幂等状态；Agent和Replay仍沿用可信原实现。"""
        state = super().state_dict()
        state['curriculum'] = dict(phase=self.phase, switched=self.switched,
                                   phase_boundary_logged=self.phase_boundary_logged,
                                   final_budget_logged=self.final_budget_logged,
                                   completed_validation_keys=sorted(self.completed_validation_keys),
                                   evaluation_warmup_control_transitions=
                                   self.evaluation_warmup_control_transitions,
                                   phase_transition_counts=self.phase_transition_counts)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """先拒绝课程状态与计数冲突，再加载相同代码/config/Torch的完整状态。"""
        course = state.get('curriculum', {})
        transitions = state.get('scheduler', {}).get('transitions', -1)
        phase, switched = course.get('phase'), course.get('switched')
        if (phase not in ('obstacle_free', 'cv_train_v1')
                or switched is not (phase == 'cv_train_v1')
                or not 0 <= transitions <= self.TOTAL_END
                or (not switched and transitions > self.OBSTACLE_FREE_END)
                or (switched and transitions < self.OBSTACLE_FREE_END)):
            raise ValueError('课程恢复状态与全局计数不一致。')
        expected_counts = dict(obstacle_free=min(transitions, self.OBSTACLE_FREE_END),
                               cv_train_v1=max(transitions - self.OBSTACLE_FREE_END, 0))
        keys = course.get('completed_validation_keys', [])
        allowed_keys = {
            f'{profile}:{step}:monitor'
            for profile, start, end in (
                ('obstacle_free', 0, self.OBSTACLE_FREE_END),
                ('cv_train_v1', self.OBSTACLE_FREE_END, self.TOTAL_END))
            for step in range(start, end, self.config.validation_interval)
            if step <= transitions and (profile != 'cv_train_v1' or switched)
        }
        if transitions >= self.OBSTACLE_FREE_END:
            allowed_keys.add(f'obstacle_free:{self.OBSTACLE_FREE_END}:full')
        if transitions == self.TOTAL_END:
            allowed_keys.add(f'cv_train_v1:{self.TOTAL_END}:full')
        if (not isinstance(keys, list) or any(not isinstance(key, str) for key in keys)
                or len(keys) != len(set(keys)) or not set(keys) <= allowed_keys
                or course.get('phase_transition_counts') != expected_counts
                or course.get('phase_boundary_logged') is not switched
                or (switched and f'obstacle_free:{self.OBSTACLE_FREE_END}:full' not in keys)
                or (transitions > self.OBSTACLE_FREE_END
                    and f'cv_train_v1:{self.OBSTACLE_FREE_END}:monitor' not in keys)
                or (course.get('final_budget_logged')
                    and f'cv_train_v1:{self.TOTAL_END}:full' not in keys)):
            raise ValueError('课程恢复的分项、验证完成键或边界记录不一致。')
        super().load_state_dict(state)
        self.phase, self.switched = phase, switched
        self.phase_boundary_logged = course['phase_boundary_logged']
        self.final_budget_logged = course['final_budget_logged']
        self.completed_validation_keys = set(course['completed_validation_keys'])
        self.evaluation_warmup_control_transitions = course['evaluation_warmup_control_transitions']
