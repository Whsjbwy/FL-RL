"""LOCAL无学习器环境薄层；复用世界、KF、风险与验证器，提供reset/step接口。"""

from copy import deepcopy
from typing import Any

import numpy as np

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.local_task import (
    LocalTaskConfig,
    command_to_normalized,
    instantaneous_cost,
    normalized_to_command,
    task_reward,
    termination_flags,
)
from auv_risk_rl.env.observation import ObservationBuilder
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.runtime.local_perception import PerceptionSession, legal_warmup
from auv_risk_rl.safety.validator import validate_nominal_action
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.sensors.rays import cast_sonar_rays
from auv_risk_rl.types import AUVState, GroundTruthObstacleState, ValidationDecision


class LocalNavigationEnv:
    """第15章无RL子集；外部提供物理t=0场景，输出234维观察与独立即时成本。"""

    def __init__(self, config: ProjectConfig, initial_auv_state: AUVState,
                 initial_obstacle_states: tuple[GroundTruthObstacleState, ...],
                 goal_position_ned_m: np.ndarray, scenario_id: str,
                 task_config: LocalTaskConfig | None = None) -> None:
        self.config = config
        self.task_config = task_config or LocalTaskConfig()
        self._initial_auv = deepcopy(initial_auv_state)
        self._initial_obstacles = deepcopy(initial_obstacle_states)
        self.goal_position_ned_m = np.array(goal_position_ned_m, dtype=np.float64, copy=True)
        self.scenario_id = scenario_id
        self.builder = ObservationBuilder(config)
        self._done = True
        self._seed = 0

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """重建随机流/历史/世界，实际执行一秒合法历史；正式任务从t=1的步0开始。"""

        self._done = True
        if seed is not None:
            self._seed = seed
        self.seeds = SeedManager(self._seed)
        self.seeds.get_rng('scenario')
        self.seeds.get_rng('environment')
        self.external_max_steps = (options or {}).get('external_max_steps')
        if self.external_max_steps is not None and (
            isinstance(self.external_max_steps, bool)
            or not isinstance(self.external_max_steps, int) or self.external_max_steps <= 0
        ):
            raise ValueError("外部截断步数必须为正整数。")
        warm_world = AUVWorld(self.config, deepcopy(self._initial_auv),
                              deepcopy(self._initial_obstacles), self.goal_position_ned_m)
        self.perception = PerceptionSession(
            self.config, self.seeds, {o.obstacle_id: o.radius_m for o in self._initial_obstacles})
        self.rays, self.previous_command = legal_warmup(warm_world, self.perception, self.config)
        self.world = AUVWorld(self.config, warm_world.auv_state, warm_world.obstacle_states,
                              self.goal_position_ned_m, initial_timestamp_s=1.0)
        self.previous_normalized = command_to_normalized(
            self.previous_command, self.config.dynamics)
        self._sensor_tick_offset = self.perception.last_tick
        self._collision_seen = False
        self._done = False
        return self._observation(), {
            'scenario_id': self.scenario_id, 'policy_version': None,
            'warmup_duration_s': 1.0, 'warmup_measurement_times_s':
                tuple(self.perception.capture_times_s),
            'episode_start_timestamp_s': self.world.timestamp_s, 'root_seed': self._seed,
            'rng_namespaces': ('scenario', 'sensor_noise', 'dropout', 'environment'),
            'epsilon_safe': self.config.risk.short_horizon_risk_budget, 'd_C': self.task_config.d_C,
        }

    def _observation(self) -> np.ndarray:
        """只有ObservationBuilder组装输入；所有后验先传入，builder内部取六槽。"""

        return self.builder.build(
            self.world.auv_state, self.goal_position_ned_m, self.previous_normalized,
            self.config.environment.max_episode_control_steps-self.world.control_step_index,
            self.rays, self.perception.tracks(self.world.timestamp_s), self.world.timestamp_s,
        )

    def _decision_info(self, decision: ValidationDecision, nominal: np.ndarray,
                       executed: np.ndarray) -> dict[str, Any]:
        """双动作与验证器诊断；信息不进入Observation，U为所选动作未截断界。"""

        return {
            'nominal_action_normalized': nominal.copy(),
            'nominal_action_physical': decision.nominal_action,
            'executed_action_normalized': executed.copy(),
            'executed_action_physical': decision.executed_action,
            'intervention': decision.decision_type != 'nominal',
            'fallback': decision.decision_type == 'fallback',
            'fallback_type': decision.reason if decision.decision_type == 'fallback' else None,
            'unclipped_risk_U': decision.selected_untruncated_union_bound,
            'scenario_id': self.scenario_id, 'policy_version': None,
            'epsilon_safe': self.config.risk.short_horizon_risk_budget, 'd_C': self.task_config.d_C,
        }

    def step(self, nominal_action: np.ndarray,
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """名义→原验证器→执行→真实转移；独立reward/cost/终止，不运行训练。"""

        if self._done:
            raise RuntimeError("请先reset；终止或截断之后不能继续step。")
        nominal = np.asarray(nominal_action, dtype=np.float64)
        command = normalized_to_command(nominal, self.config.dynamics)
        before_s = self.world.timestamp_s
        distance_before_m = float(np.linalg.norm(
            self.goal_position_ned_m-self.world.auv_state.position_ned_m))
        decision = validate_nominal_action(
            self.world.auv_state, before_s, command, self.previous_command,
            list(self.perception.tracks(before_s)), self.config,
        )
        result = self.world.step(decision.executed_action)
        executed = command_to_normalized(decision.executed_action, self.config.dynamics)
        distance_after_m = float(np.linalg.norm(
            self.goal_position_ned_m-result.auv_state.position_ned_m))
        reward, reward_parts = task_reward(
            distance_before_m, distance_after_m, result.event.reason == 'success',
            result.timestamp_s-before_s, self.config.dynamics.control_dt_s,
            self.previous_normalized, executed, self.task_config,
        )
        costs = instantaneous_cost(decision.decision_type != 'nominal',
                                   result.event.reason, self._collision_seen)
        self._collision_seen |= result.event.reason == 'collision'
        external = (self.external_max_steps is not None
                    and result.control_step_index >= self.external_max_steps)
        terminated, truncated, failure = termination_flags(result.event.reason, external)
        self.previous_command, self.previous_normalized = decision.executed_action, executed
        if result.is_terminated:
            self.rays = cast_sonar_rays(
                result.auv_state, result.obstacle_states, self.config.sensor)
        else:
            self.rays = self.perception.capture(
                self.world, self._sensor_tick_offset+result.control_step_index)
        info = self._decision_info(decision, nominal, executed)
        info.update(cost=costs['c_train'], cost_components=costs, reward_components=reward_parts,
                    minimum_clearance=result.event.minimum_clearance_m, failure_type=failure,
                    timestamp_s=result.timestamp_s, elapsed_s=result.timestamp_s-before_s,
                    task_control_step=result.control_step_index)
        self._done = terminated or truncated
        return self._observation(), reward, terminated, truncated, info
