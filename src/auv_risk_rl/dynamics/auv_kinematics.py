"""
AUV 八状态降阶三维导航模型与 RK2 积分模块。

功能：
1. 根据 Eq. (6)–(8) 计算状态导数；
2. 根据 Eq. (9) 进行显式中点 RK2 积分；
3. 统一执行速度、角率与俯仰操作约束检查。

说明：
该模型是导航层工程模型，不是完整水动力学。主模型中海流默认为零，
但接口保留 NED 海流输入以支持后续 OOD 压力测试。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from auv_risk_rl.config import DynamicsConfig
from auv_risk_rl.exceptions import InvalidActionError
from auv_risk_rl.frames import rotation_body_to_ned
from auv_risk_rl.types import AUVState, ControlCommand

FloatVector = NDArray[np.float64]


def _saturate_rate(requested_rate: float, absolute_limit: float) -> float:
    """按对称变化率上限截断导数；该截断表达执行器能力而不是数值修补。"""

    return float(np.clip(requested_rate, -absolute_limit, absolute_limit))


def validate_control_command(command: ControlCommand, config: DynamicsConfig) -> None:
    """
    检查导航层动作是否落在协议定义域内。

    对应技术协议：
        Eq. (43)、Eq. (55)

    参数：
        command:
            物理动作，三个标量分别为 m/s、rad/s、rad/s。
        config:
            动作范围配置。

    返回：
        无；非法时抛出 InvalidActionError。

    shape/坐标系：
        不适用；动作属于导航指令空间。

    关键假设：
        配置范围已经冻结。

    重要限制：
        本检查只验证动作定义域，不保证未来轨迹满足操作边界或风险预算。
    """

    command_values = np.array(
        [
            command.surge_speed_command_mps,
            command.yaw_rate_command_rad_s,
            command.pitch_rate_command_rad_s,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(command_values)):
        raise InvalidActionError(f"动作包含非有限值：{command_values}。")

    if not (
        config.min_surge_speed_mps
        <= command.surge_speed_command_mps
        <= config.max_surge_speed_mps
    ):
        raise InvalidActionError(
            "前向速度指令越界："
            f"actual={command.surge_speed_command_mps}, "
            f"expected=[{config.min_surge_speed_mps}, {config.max_surge_speed_mps}]。"
        )
    if abs(command.yaw_rate_command_rad_s) > config.max_yaw_rate_rad_s:
        raise InvalidActionError(
            "yaw-rate 指令越界："
            f"actual={command.yaw_rate_command_rad_s}, "
            f"limit={config.max_yaw_rate_rad_s}。"
        )
    if abs(command.pitch_rate_command_rad_s) > config.max_pitch_rate_rad_s:
        raise InvalidActionError(
            "pitch-rate 指令越界："
            f"actual={command.pitch_rate_command_rad_s}, "
            f"limit={config.max_pitch_rate_rad_s}。"
        )


def auv_state_derivative(
    state: AUVState,
    command: ControlCommand,
    current_velocity_ned_mps: FloatVector,
    config: DynamicsConfig,
) -> FloatVector:
    """
    计算八维降阶 AUV 状态的一阶导数。

    对应技术协议：
        Eq. (6)–(8)

    数学模型：
        约束三维运动学 + 一阶指令响应 + 变化率限制。

    参数：
        state:
            当前八维 AUV 状态；位置 NED，单位见 AUVState。
        command:
            固定控制指令，单位 m/s、rad/s、rad/s。
        current_velocity_ned_mps:
            海流速度，shape=(3,)，单位 m/s，NED。
        config:
            动力学与执行器参数。

    返回：
        state_derivative:
            shape=(8,)，单位依次为 [m/s,m/s,m/s,rad/s,rad/s,m/s^2,rad/s^2,rad/s^2]。

    关键假设：
        A1、A2。

    重要限制：
        Eq. (8) 属于协议定义的工程启发式执行响应，不是严格水动力学推导。
    """

    validate_control_command(command, config)
    if current_velocity_ned_mps.shape != (3,) or not np.all(np.isfinite(current_velocity_ned_mps)):
        raise ValueError("current_velocity_ned_mps 必须是有限的 shape=(3,) NED 向量。")

    rotation_matrix = rotation_body_to_ned(state.yaw_rad, state.pitch_rad)
    surge_velocity_body_mps = np.array([state.surge_speed_mps, 0.0, 0.0], dtype=np.float64)
    position_rate_ned_mps = rotation_matrix @ surge_velocity_body_mps + current_velocity_ned_mps

    surge_accel_mps2 = _saturate_rate(
        (command.surge_speed_command_mps - state.surge_speed_mps) / config.surge_time_constant_s,
        config.max_surge_accel_mps2,
    )
    yaw_accel_rad_s2 = _saturate_rate(
        (command.yaw_rate_command_rad_s - state.yaw_rate_rad_s) / config.yaw_rate_time_constant_s,
        config.max_yaw_accel_rad_s2,
    )
    pitch_accel_rad_s2 = _saturate_rate(
        (command.pitch_rate_command_rad_s - state.pitch_rate_rad_s)
        / config.pitch_rate_time_constant_s,
        config.max_pitch_accel_rad_s2,
    )

    return np.array(
        [
            *position_rate_ned_mps,
            state.yaw_rate_rad_s,
            state.pitch_rate_rad_s,
            surge_accel_mps2,
            yaw_accel_rad_s2,
            pitch_accel_rad_s2,
        ],
        dtype=np.float64,
    )


def _state_to_vector(state: AUVState) -> FloatVector:
    """将 AUVState 转为 Eq. (5) 的八维向量，便于数值积分。"""

    return np.array(
        [
            *state.position_ned_m,
            state.yaw_rad,
            state.pitch_rad,
            state.surge_speed_mps,
            state.yaw_rate_rad_s,
            state.pitch_rate_rad_s,
        ],
        dtype=np.float64,
    )


def _vector_to_state(state_vector: FloatVector) -> AUVState:
    """将八维数值向量恢复为强类型 AUVState。"""

    return AUVState(
        position_ned_m=state_vector[0:3].copy(),
        yaw_rad=float(state_vector[3]),
        pitch_rad=float(state_vector[4]),
        surge_speed_mps=float(state_vector[5]),
        yaw_rate_rad_s=float(state_vector[6]),
        pitch_rate_rad_s=float(state_vector[7]),
    )


def integrate_rk2_step(
    state: AUVState,
    command: ControlCommand,
    current_velocity_ned_mps: FloatVector,
    integration_dt_s: float,
    config: DynamicsConfig,
) -> AUVState:
    """
    使用显式中点 RK2 推进一个积分小步。

    对应技术协议：
        Eq. (9)

    数学模型：
        控制指令在积分小步内保持固定，先计算中点预测，再用中点导数更新全部八维状态。

    参数：
        state:
            当前 AUV 状态；位置 NED，混合单位。
        command:
            本积分步固定物理指令。
        current_velocity_ned_mps:
            海流，shape=(3,)，单位 m/s，NED。
        integration_dt_s:
            积分步长，单位 s。
        config:
            动力学配置。

    返回：
        next_state:
            下一积分节点的 AUVState。

    关键假设：
        A1、A8。

    重要限制：
        RK2 是数值近似；在执行器饱和切换点不保证经典光滑二阶误差率。
        本函数不会把超出空间/俯仰边界的状态夹回合法范围，边界检查由验证器显式处理。
    """

    if integration_dt_s <= 0.0:
        raise ValueError(f"integration_dt_s 必须为正，actual={integration_dt_s}。")

    state_vector = _state_to_vector(state)
    derivative_start = auv_state_derivative(state, command, current_velocity_ned_mps, config)
    midpoint_vector = state_vector + 0.5 * integration_dt_s * derivative_start
    midpoint_state = _vector_to_state(midpoint_vector)
    derivative_midpoint = auv_state_derivative(
        midpoint_state,
        command,
        current_velocity_ned_mps,
        config,
    )
    next_vector = state_vector + integration_dt_s * derivative_midpoint
    return _vector_to_state(next_vector)


def rollout_constant_command(
    initial_state: AUVState,
    command: ControlCommand,
    current_velocity_ned_mps: FloatVector,
    horizon_s: float,
    integration_dt_s: float,
    config: DynamicsConfig,
) -> list[AUVState]:
    """
    在固定导航指令下滚动预测 AUV 轨迹。

    对应技术协议：
        Eq. (56)，内部积分使用 Eq. (9)。

    参数：
        initial_state:
            起始八维状态。
        command:
            整个滚动时域保持不变的物理指令。
        current_velocity_ned_mps:
            预测模型使用的海流，shape=(3,)，单位 m/s，NED；主验证器为零流。
        horizon_s:
            滚动时长，单位 s。
        integration_dt_s:
            数值积分步长，单位 s。
        config:
            动力学配置。

    返回：
        states:
            含起点与每个积分节点的 AUVState 列表，长度 = horizon_s/integration_dt_s + 1。

    关键假设：
        A7、A8、A10。

    重要限制：
        该模块属于协议定义的工程启发式固定指令延续，不代表未来 SAC 策略会保持动作不变。
    """

    step_count_float = horizon_s / integration_dt_s
    step_count = int(round(step_count_float))
    if not np.isclose(step_count * integration_dt_s, horizon_s, atol=1e-12, rtol=0.0):
        raise ValueError(
            "验证时域必须是积分步长的整数倍："
            f"horizon_s={horizon_s}, integration_dt_s={integration_dt_s}。"
        )

    states = [initial_state]
    current_state = initial_state
    for _ in range(step_count):
        current_state = integrate_rk2_step(
            state=current_state,
            command=command,
            current_velocity_ned_mps=current_velocity_ned_mps,
            integration_dt_s=integration_dt_s,
            config=config,
        )
        states.append(current_state)
    return states
