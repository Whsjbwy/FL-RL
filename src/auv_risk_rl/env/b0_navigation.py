"""LOCAL §18 B0当前真值适配层；共同世界/reward不变，不进行风险训练或执行过滤。"""

from typing import Any

import numpy as np

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.local_navigation import LocalNavigationEnv
from auv_risk_rl.env.local_task import LocalTaskConfig, normalized_to_command, task_reward
from auv_risk_rl.env.local_task import termination_flags as task_termination_flags
from auv_risk_rl.env.observation import ObservationBuilder
from auv_risk_rl.frames import rotation_body_to_ned
from auv_risk_rl.sensors.rays import RayFrame, cast_sonar_rays
from auv_risk_rl.types import AUVState, GroundTruthObstacleState


class B0ObservationBuilder:
    """234维当前真值编码；地速/1m/s和年龄0为已登记工程选择，无未来预测。"""

    velocity_scale_mps = 1.0

    def __init__(self, config: ProjectConfig) -> None:
        """保存共同配置；有限感知builder只负责自身/任务与射线部分。"""

        self.config = config
        self.finite_builder = ObservationBuilder(config)

    def build(self, state: AUVState, goal_position_ned_m: np.ndarray,
              previous_executed_normalized: np.ndarray, remaining_control_steps: int,
              rays: RayFrame, obstacles: tuple[GroundTruthObstacleState, ...],
              timestamp_s: float) -> np.ndarray:
        """复用自身/任务和真实射线，当前距离/稳定ID排序六个真值槽；不写回输入。"""

        output = self.finite_builder.build(
            state, goal_position_ned_m, previous_executed_normalized,
            remaining_control_steps, rays, (), timestamp_s,
        )
        identifiers = [obstacle.obstacle_id for obstacle in obstacles]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("B0当前真值障碍ID必须唯一。")
        for obstacle in obstacles:
            position = np.asarray(obstacle.position_ned_m)
            velocity = np.asarray(obstacle.velocity_ned_mps)
            if (position.shape != (3,) or velocity.shape != (3,)
                    or not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity))
                    or not np.isfinite(obstacle.radius_m) or obstacle.radius_m <= 0):
                raise ValueError("B0当前障碍位置/地速/物理半径必须合法且有限。")
        rotation = rotation_body_to_ned(state.yaw_rad, state.pitch_rad).T
        ordered = sorted(obstacles, key=lambda obstacle: (
            float(np.linalg.norm(obstacle.position_ned_m - state.position_ned_m)),
            obstacle.obstacle_id,
        ))
        for index, obstacle in enumerate(ordered[:6]):
            slot = np.zeros(21, dtype=np.float64)
            slot[:3] = rotation @ (obstacle.position_ned_m - state.position_ned_m)
            slot[:3] /= self.config.sensor.range_m
            # 障碍NED地速转Body；不是相对地速，也不是旋转坐标的相对位置导数。
            slot[3:6] = rotation @ obstacle.velocity_ned_mps / self.velocity_scale_mps
            slot[18:] = (obstacle.radius_m, 0.0, 1.0)
            output[108+21*index:129+21*index] = slot
        if not np.all(np.isfinite(output)):
            raise ValueError("B0观察包含非有限数据。")
        return output


class B0NavigationEnv(LocalNavigationEnv):
    """独立B0入口：当前真值、名义等于执行指令；执行器响应/物理失败照常。"""

    def __init__(self, config: ProjectConfig, initial_auv_state: AUVState,
                 initial_obstacle_states: tuple[GroundTruthObstacleState, ...],
                 goal_position_ned_m: np.ndarray, scenario_id: str,
                 task_config: LocalTaskConfig | None = None) -> None:
        """登记固定场景，保持原世界和任务配置；不改变原环境类。"""

        super().__init__(config, initial_auv_state, initial_obstacle_states,
                         goal_position_ned_m, scenario_id, task_config)
        self.b0_builder = B0ObservationBuilder(config)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """复用真实一秒warm-up及射线；KF历史不作为B0目标槽输入。"""

        observation, info = super().reset(seed=seed, options=options)
        info.pop('epsilon_safe')
        info.pop('d_C')
        info.update(method='B0_FULL_STATE_SAC', observation_method='CURRENT_TRUTH',
                    risk_training=False, safety_validation=False,
                    truth_age_s=0.0, cost_semantics='physical_failure_diagnostic_only')
        return observation, info

    def _observation(self) -> np.ndarray:
        """每个控制时刻读取世界当前快照；不访问KF后验、未来轨迹或预测器。"""

        return self.b0_builder.build(
            self.world.auv_state, self.goal_position_ned_m, self.previous_normalized,
            self.config.environment.max_episode_control_steps-self.world.control_step_index,
            self.rays, self.world.obstacle_states, self.world.timestamp_s,
        )

    def step(self, nominal_action: np.ndarray,
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """原动作映射后直接推进共同世界；Replay成本仅真实失败诊断，不用于普通SAC。"""

        if self._done:
            raise RuntimeError("请先reset；终止或截断之后不能继续step。")
        nominal = np.asarray(nominal_action, dtype=np.float64)
        command = normalized_to_command(nominal, self.config.dynamics)
        before_s = self.world.timestamp_s
        distance_before_m = float(np.linalg.norm(
            self.goal_position_ned_m-self.world.auv_state.position_ned_m))
        result = self.world.step(command)
        executed = nominal.copy()
        distance_after_m = float(np.linalg.norm(
            self.goal_position_ned_m-result.auv_state.position_ned_m))
        reward, reward_parts = task_reward(
            distance_before_m, distance_after_m, result.event.reason == 'success',
            result.timestamp_s-before_s, self.config.dynamics.control_dt_s,
            self.previous_normalized, executed, self.task_config,
        )
        collision = result.event.reason == 'collision'
        physical_failure = collision or result.event.reason == 'boundary'
        costs = {'c_risk': None, 'c_real': int(collision and not self._collision_seen),
                 'f_t': int(physical_failure), 'c_train': int(physical_failure)}
        self._collision_seen |= collision
        external = (self.external_max_steps is not None
                    and result.control_step_index >= self.external_max_steps)
        terminated, truncated, failure = task_termination_flags(result.event.reason, external)
        self.previous_command, self.previous_normalized = command, executed
        if result.is_terminated:
            self.rays = cast_sonar_rays(
                result.auv_state, result.obstacle_states, self.config.sensor)
        else:
            self.rays = self.perception.capture(
                self.world, self._sensor_tick_offset+result.control_step_index)
        info = {
            'method': 'B0_FULL_STATE_SAC', 'observation_method': 'CURRENT_TRUTH',
            'nominal_action_normalized': nominal.copy(), 'nominal_action_physical': command,
            'executed_action_normalized': executed.copy(), 'executed_action_physical': command,
            'risk_training': False, 'safety_validation': False,
            'intervention': None, 'fallback': None, 'fallback_type': None,
            'unclipped_risk_U': None, 'scenario_id': self.scenario_id, 'policy_version': None,
            'cost': costs['c_train'], 'cost_components': costs,
            'cost_semantics': 'physical_failure_diagnostic_only',
            'reward_components': reward_parts,
            'minimum_clearance': result.event.minimum_clearance_m,
            'failure_type': failure, 'timestamp_s': result.timestamp_s,
            'elapsed_s': result.timestamp_s-before_s,
            'task_control_step': result.control_step_index,
        }
        self._done = terminated or truncated
        return self._observation(), reward, terminated, truncated, info
