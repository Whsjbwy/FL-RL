"""
Stage 0 环境专用数据结构。

功能：
1. 区分世界真值、终止事件、感知侧公开数据与诊断数据；
2. 为无 RL 环境提供稳定接口；
3. 明确禁止把 Oracle 未来或障碍真值混入策略侧感知帧。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from auv_risk_rl.types import AUVState, GroundTruthObstacleState, KFTrackState, SensorDetection

TerminationReason = Literal["none", "collision", "boundary", "success", "timeout"]


@dataclass(frozen=True)
class EnvironmentEvent:
    """
    表示控制周期内最早发生的环境事件。

    对应技术协议：
        Eq. (27)、第 7、15、21、25.1–25.2 章。

    参数：
        reason:
            none/collision/boundary/success/timeout。
        event_timestamp_s:
            事件世界时间，单位 s。
        event_fraction_of_integration_step:
            事件在当前积分小步中的归一化位置，范围 [0,1]。
        obstacle_id:
            碰撞目标 ID；非碰撞事件为 None。
        minimum_clearance_m:
            本控制周期截至真实事件的已执行轨迹最小净间距，单位 m；无障碍为正无穷。
        detail:
            结构化可读原因。

    关键假设：
        A8；积分节点之间的位置按线性插值定义。

    重要限制：
        事件时间只对本协议的分段线性位置几何精确，不表示完整连续水动力轨迹事件时间。
    """

    reason: TerminationReason
    event_timestamp_s: float
    event_fraction_of_integration_step: float
    obstacle_id: int | None
    minimum_clearance_m: float
    detail: str


@dataclass(frozen=True)
class WorldStepResult:
    """
    表示环境真值推进一个控制周期后的结果。

    对应技术协议：
        Eq. (9)、Eq. (11)–(12)、Eq. (27) 与第 15 章执行顺序。

    参数：
        auv_state:
            推进后或最早事件时刻的 AUV 真值状态。
        obstacle_states:
            同一时刻障碍真值元组，仅供环境与评价使用。
        timestamp_s:
            结果时刻，单位 s。
        control_step_index:
            已完成控制周期数；若周期中途终止，则仍记该次尝试对应的索引。
        is_terminated:
            是否发生真实任务终止。
        event:
            最早事件。

    关键假设：
        A1、A3、A8。

    重要限制：
        obstacle_states 属于 Ground Truth，禁止直接传给普通策略或 KF 更新接口。
    """

    auv_state: AUVState
    obstacle_states: tuple[GroundTruthObstacleState, ...]
    timestamp_s: float
    control_step_index: int
    is_terminated: bool
    event: EnvironmentEvent


@dataclass(frozen=True)
class PolicyPerceptionFrame:
    """
    策略侧可访问的 Stage 0 感知帧。

    对应技术协议：
        第 5、9、13.2、15、30 章数据流要求；无独立编号公式。

    参数：
        timestamp_s:
            当前世界时间，单位 s。
        auv_state:
            主实验中允许使用的自身状态；位置 NED，其他单位见 AUVState。
        arrived_detections:
            当前周期已经到达的 SensorDetection，只含测量，不含障碍真实速度或未来真值。
        track_states:
            KF 当前估计，不含 Ground Truth。

    关键假设：
        A1、A5、A6。

    重要限制：
        本结构故意不含 GroundTruthObstacleState 和 OracleFutureState，作为 Stage 0 真值泄漏边界。
    """

    timestamp_s: float
    auv_state: AUVState
    arrived_detections: tuple[SensorDetection, ...]
    track_states: tuple[KFTrackState, ...]
