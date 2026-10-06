"""
项目关键数据结构定义。

功能：
1. 显式隔离 Ground Truth、Sensor Measurement、KF Estimate 与 Validator 输出；
2. 在类型层面降低真值泄漏风险；
3. 为后续 Stage 1–14 保留稳定接口。

说明：
这些数据结构本身不执行物理或概率计算；它们只表达数据语义与坐标/单位约定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class AUVState:
    """
    表示 AUV 的八维降阶导航状态。

    对应技术协议：
        Eq. (5)

    参数：
        position_ned_m:
            AUV 位置，shape=(3,)，单位 m，坐标系 NED。
        yaw_rad:
            航向角，单位 rad。
        pitch_rad:
            俯仰角，单位 rad；正俯仰表示抬头，因此前向运动具有负 Down 分量。
        surge_speed_mps:
            Body 前向相对水体速度，单位 m/s。
        yaw_rate_rad_s:
            Euler yaw 变化率，单位 rad/s。
        pitch_rate_rad_s:
            Euler pitch 变化率，单位 rad/s。

    关键假设：
        A1、A2。

    重要限制：
        这是降阶导航模型状态，不是完整 5-DOF/6-DOF 水动力学状态。
    """

    position_ned_m: FloatVector
    yaw_rad: float
    pitch_rad: float
    surge_speed_mps: float
    yaw_rate_rad_s: float
    pitch_rate_rad_s: float


@dataclass(frozen=True)
class ControlCommand:
    """
    表示导航层物理控制指令。

    对应技术协议：
        Eq. (43)、Eq. (55)

    参数：
        surge_speed_command_mps:
            前向速度指令，单位 m/s。
        yaw_rate_command_rad_s:
            Euler yaw-rate 指令，单位 rad/s。
        pitch_rate_command_rad_s:
            Euler pitch-rate 指令，单位 rad/s。

    输出/坐标：
        本结构不产生输出；坐标语义属于导航指令空间，而不是 NED 向量。

    关键假设：
        动作范围由配置冻结。

    重要限制：
        该指令不是推进器推力，也不是完整机体系角速度控制量。
    """

    surge_speed_command_mps: float
    yaw_rate_command_rad_s: float
    pitch_rate_command_rad_s: float


@dataclass(frozen=True)
class GroundTruthObstacleState:
    """
    表示仿真障碍真值状态，仅允许环境和离线评价访问。

    对应技术协议：
        Eq. (11)

    参数：
        obstacle_id:
            障碍唯一标识。
        position_ned_m:
            真值位置，shape=(3,)，单位 m，坐标系 NED。
        velocity_ned_mps:
            真值地速，shape=(3,)，单位 m/s，坐标系 NED。
        radius_m:
            物理包络半径，单位 m。

    关键假设：
        A3、A6。

    重要限制：
        该结构不得直接进入普通策略 Observation 或 KF 测量更新接口。
    """

    obstacle_id: int
    position_ned_m: FloatVector
    velocity_ned_mps: FloatVector
    radius_m: float


@dataclass(frozen=True)
class SensorDetection:
    """
    表示简化声呐前端产生的相对笛卡尔位置测量。

    对应技术协议：
        Eq. (16)–(17)

    参数：
        obstacle_id:
            正确关联条件下的目标标识；MVP 默认关联已知。
        relative_position_body_m:
            相对位置测量，shape=(3,)，单位 m，坐标系 Body。
        measurement_covariance_body_m2:
            测量协方差，shape=(3,3)，单位 m^2，坐标系 Body。
        measurement_timestamp_s:
            测量实际发生时刻，单位 s。
        arrival_timestamp_s:
            测量到达滤波器时刻，单位 s。

    关键假设：
        A1、A5、A6。

    重要限制：
        本结构不包含真实障碍速度或未来真值。
    """

    obstacle_id: int
    relative_position_body_m: FloatVector
    measurement_covariance_body_m2: FloatMatrix
    measurement_timestamp_s: float
    arrival_timestamp_s: float
    measurement_control_tick: int | None = None
    arrival_control_tick: int | None = None


@dataclass(frozen=True)
class KFTrackState:
    """
    表示单目标 CV-KF 的后验估计状态。

    对应技术协议：
        Eq. (19)–(25)

    参数：
        obstacle_id:
            目标标识。
        state_mean_ned:
            后验均值，shape=(6,)，单位 [m,m,m,m/s,m/s,m/s]，坐标系 NED。
        state_covariance_ned:
            后验协方差，shape=(6,6)，分块单位，坐标系 NED。
        state_timestamp_s:
            该后验对应时刻，单位 s。
        last_measurement_timestamp_s:
            最近一次被吸收的测量发生时刻，单位 s。

    关键假设：
        A3–A6。

    重要限制：
        协方差仅表达已声明模型条件下的不确定性，不覆盖错误关联、未知偏置等未建模因素。
    """

    obstacle_id: int
    state_mean_ned: FloatVector
    state_covariance_ned: FloatMatrix
    state_timestamp_s: float
    last_measurement_timestamp_s: float


@dataclass(frozen=True)
class TrackedObstacle:
    """
    将 KF 估计与已知保守障碍包络半径组合为验证器输入。

    对应技术协议：
        Eq. (27)–(28)、Eq. (57)

    参数：
        track_state:
            KF 后验估计，NED；均值 shape=(6,)，协方差 shape=(6,6)。
        radius_m:
            已知保守物理包络半径，单位 m。

    关键假设：
        A6、A9。

    重要限制：
        该结构只包含已维护目标；未探测目标不在模型条件风险覆盖范围内。
    """

    track_state: KFTrackState
    radius_m: float


@dataclass(frozen=True)
class PredictionResult:
    """
    表示未来障碍位置高斯边缘分布。

    对应技术协议：
        Eq. (24)–(25)

    参数：
        obstacle_id:
            目标标识。
        prediction_horizon_s:
            从当前 KF 后验向未来传播的时长，单位 s。
        position_mean_ned_m:
            未来位置均值，shape=(3,)，单位 m，坐标系 NED。
        position_covariance_ned_m2:
            未来位置协方差，shape=(3,3)，单位 m^2，坐标系 NED。

    关键假设：
        A3–A5。

    重要限制：
        输出不包含未建模机动、错误关联、自身定位误差或未知海流造成的附加不确定性。
    """

    obstacle_id: int
    prediction_horizon_s: float
    position_mean_ned_m: FloatVector
    position_covariance_ned_m2: FloatMatrix


@dataclass(frozen=True)
class RiskResult:
    """
    表示模型条件风险上界的计算结果。

    对应技术协议：
        Eq. (34)、Eq. (37)–(39)

    参数：
        risk_upper_bound:
            截断到 [0,1] 的模型条件碰撞概率上界。
        untruncated_union_bound:
            未截断的并集上界和，用于候选排序和保守性诊断。
        is_numerically_valid:
            是否完成了无数值异常的有效计算。
        detail:
            简短原因标记，不用于替代结构化日志。

    关键假设：
        A3–A9。

    重要限制：
        该数值不是无条件真实 collision probability，也不是整任务安全保证。
    """

    risk_upper_bound: float
    untruncated_union_bound: float
    is_numerically_valid: bool
    detail: str


@dataclass(frozen=True)
class ValidationDecision:
    """
    表示在线安全验证模块最终动作选择。

    对应技术协议：
        Eq. (57)–(58)

    参数：
        nominal_action:
            策略或上层模块提出的名义动作。
        executed_action:
            经验证后实际执行的动作。
        decision_type:
            nominal 表示名义动作通过；modified 表示选用了通过验证的备选动作；
            fallback 表示没有通过验证的候选，执行协议定义的回退动作。
        selected_risk_upper_bound:
            被选动作的截断风险上界。
        selected_untruncated_union_bound:
            被选动作的未截断风险和。
        reason:
            结构化原因摘要。

    关键假设：
        A3–A10。

    重要限制：
        fallback 不应被称为安全控制；它只是在定义域内按预注册规则选择动作。
    """

    nominal_action: ControlCommand
    executed_action: ControlCommand
    decision_type: Literal["nominal", "modified", "fallback"]
    selected_risk_upper_bound: float
    selected_untruncated_union_bound: float
    reason: str
