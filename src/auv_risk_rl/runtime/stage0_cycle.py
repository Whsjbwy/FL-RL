"""
Stage 0 无 RL 控制周期集成编排器。

功能：
1. 将当前 KF tracks 交给 validator 检查名义动作；
2. 只把 executed_action 交给真实环境推进；
3. 在控制时刻生成简化声呐检测、执行固定延迟队列并更新 KF；
4. 同时返回策略侧感知帧与独立真值诊断，避免接口混用。

说明：
这是 Stage 0 的集成验收 harness，不是后续 Gymnasium 环境，也不实现 reward、Replay 或 SAC。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.types import PolicyPerceptionFrame, WorldStepResult
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.safety.validator import validate_nominal_action
from auv_risk_rl.sensors.delay_queue import DetectionDelayQueue
from auv_risk_rl.sensors.sonar import generate_sonar_detections
from auv_risk_rl.tracking.pose_history import AUVPoseHistory
from auv_risk_rl.tracking.track_manager import MultiTargetCVTracker
from auv_risk_rl.types import ControlCommand, TrackedObstacle, ValidationDecision


@dataclass(frozen=True)
class Stage0ControlCycleResult:
    """
    Stage 0 单控制周期的集成输出。

    对应技术协议：
        第 15 章完整算法流程的无 RL 子集。

    参数：
        validation_decision:
            validator 对名义动作的决定。
        perception_frame:
            策略侧允许访问的数据，不含障碍 Ground Truth。
        world_diagnostics:
            环境/离线评价专用真值结果，必须与 perception_frame 分开使用。

    关键假设：
        Stage 0 不训练策略，nominal_action 由测试脚本或上层诊断器提供。

    重要限制：
        world_diagnostics 绝不能拼接到后续普通 Policy Observation。
    """

    validation_decision: ValidationDecision
    perception_frame: PolicyPerceptionFrame
    world_diagnostics: WorldStepResult


class Stage0ControlCycleRunner:
    """
    Stage 0 感知—估计—验证—执行集成 reference runner。

    对应技术协议：
        第 5、9、10、14、15、25.1 章。

    输入/输出：
        输入名义动作和上一执行动作；输出 Stage0ControlCycleResult。

    shape/单位/坐标系：
        AUV/track/风险计算统一 NED；SensorDetection 为 Body；时间单位 s。

    关键假设：
        A1、A3–A10；障碍半径作为已知保守几何属性单独登记。

    重要限制：
        当前不含 45 条测距射线、复杂遮挡、Observation 234 维编码或 RL 更新。
    """

    def __init__(
        self,
        config: ProjectConfig,
        world: AUVWorld,
        sensor_rng: np.random.Generator,
        obstacle_radius_by_id_m: dict[int, float],
    ) -> None:
        """
        组合 Stage 0 已冻结的环境、传感、跟踪和验证模块。

        参数：
            config:
                Stage 0 配置。
            world:
                真值环境。
            sensor_rng:
                显式 sensor 随机流。
            obstacle_radius_by_id_m:
                已知保守障碍包络半径，单位 m；只含几何属性，不含真值位置/速度。

        返回：
            无。

        关键假设：
            radius registry 覆盖所有可能被维护的 obstacle_id。

        重要限制：
            不允许从 world 真值动态读取半径后再伪装成策略估计；场景初始化时应固定登记。
        """

        self._config = config
        self._world = world
        self._sensor_rng = sensor_rng
        self._obstacle_radius_by_id_m = dict(obstacle_radius_by_id_m)
        self._delay_queue = DetectionDelayQueue(
            control_dt_s=config.dynamics.control_dt_s,
            clock_origin_s=(
                world.timestamp_s - world.control_step_index * config.dynamics.control_dt_s
            ),
        )
        self._pose_history = AUVPoseHistory()
        self._tracker = MultiTargetCVTracker(config)
        self._pose_history.add(world.timestamp_s, world.auv_state)

    def _capture_and_update_perception(self) -> PolicyPerceptionFrame:
        """
        在当前世界时刻生成检测、释放已到达测量并更新 KF。

        该函数严格在 measurement_timestamp_s 对应的历史 AUV 位姿上完成 Body→NED 转换。
        """

        generated_detections = generate_sonar_detections(
            auv_state=self._world.auv_state,
            obstacle_states=self._world.obstacle_states,
            measurement_timestamp_s=self._world.timestamp_s,
            sensor_config=self._config.sensor,
            dynamics_config=self._config.dynamics,
            sensor_rng=self._sensor_rng,
            measurement_control_tick=self._world.control_step_index,
        )
        self._delay_queue.enqueue_many(generated_detections)
        arrived_detections = self._delay_queue.pop_arrived(
            current_control_tick=self._world.control_step_index
        )
        self._tracker.add_arrived_detections(arrived_detections, self._pose_history)
        track_states = self._tracker.get_track_states(self._world.timestamp_s)
        return PolicyPerceptionFrame(
            timestamp_s=self._world.timestamp_s,
            auv_state=self._world.auv_state,
            arrived_detections=arrived_detections,
            track_states=track_states,
        )

    def initialize_perception(self) -> PolicyPerceptionFrame:
        """
        在世界初始控制时刻生成首个策略侧感知帧。

        对应技术协议：
            第 9 章初始化/Delay 与第 15 章步骤 1–2。

        返回：
            PolicyPerceptionFrame。

        关键假设：
            调用一次后再进入 run_control_cycle。

        重要限制：
            若配置有正延迟，初始帧可能没有任何已到达检测，这是正确行为而非初始化失败。
        """

        return self._capture_and_update_perception()

    def _tracked_obstacles(self) -> list[TrackedObstacle]:
        """把当前 KF 状态与预登记物理半径组合为 validator 输入，不读取真值运动状态。"""

        tracked_obstacles: list[TrackedObstacle] = []
        for track_state in self._tracker.get_track_states(self._world.timestamp_s):
            if track_state.obstacle_id not in self._obstacle_radius_by_id_m:
                raise KeyError(
                    "track 缺少预登记障碍包络半径："
                    f"obstacle_id={track_state.obstacle_id}。"
                )
            tracked_obstacles.append(
                TrackedObstacle(
                    track_state=track_state,
                    radius_m=self._obstacle_radius_by_id_m[track_state.obstacle_id],
                )
            )
        return tracked_obstacles

    def run_control_cycle(
        self,
        nominal_action: ControlCommand,
        previous_executed_action: ControlCommand,
    ) -> Stage0ControlCycleResult:
        """
        执行一次无 RL 的完整控制周期。

        对应技术协议：
            第 15 章步骤 3–7 的 Stage 0 子集，validator 对应 Eq. (55)–(58)。

        参数：
            nominal_action:
                上层诊断脚本给出的名义动作。
            previous_executed_action:
                上一周期执行动作，用于候选库。

        返回：
            Stage0ControlCycleResult。

        关键假设：
            环境尚未终止；策略侧不直接读取 world_diagnostics。

        重要限制：
            终止发生在控制周期中途时不生成虚假的“下一控制时刻测量”。
        """

        validation_decision = validate_nominal_action(
            auv_state=self._world.auv_state,
            current_timestamp_s=self._world.timestamp_s,
            nominal_action=nominal_action,
            previous_executed_action=previous_executed_action,
            tracked_obstacles=self._tracked_obstacles(),
            config=self._config,
        )
        world_result = self._world.step(validation_decision.executed_action)
        self._pose_history.add(world_result.timestamp_s, world_result.auv_state)

        if world_result.is_terminated:
            track_states = self._tracker.get_track_states(world_result.timestamp_s)
            perception_frame = PolicyPerceptionFrame(
                timestamp_s=world_result.timestamp_s,
                auv_state=world_result.auv_state,
                arrived_detections=tuple(),
                track_states=track_states,
            )
        else:
            perception_frame = self._capture_and_update_perception()

        return Stage0ControlCycleResult(
            validation_decision=validation_decision,
            perception_frame=perception_frame,
            world_diagnostics=world_result,
        )
