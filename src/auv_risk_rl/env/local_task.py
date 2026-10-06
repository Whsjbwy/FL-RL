"""LOCAL表13/Eq43–44任务接口；与风险阈值和有限时域成本账本独立。"""

from dataclasses import dataclass

import numpy as np

from auv_risk_rl.config import DynamicsConfig
from auv_risk_rl.types import ControlCommand


@dataclass(frozen=True)
class LocalTaskConfig:
    """冻结四项任务权重和独立d_C预算；不含乘子或学习器。"""

    w_progress: float = 1.0
    w_goal: float = 100.0
    w_time: float = 0.01
    w_smooth: float = 0.02
    d_C: float = 0.05


def normalized_to_command(action: np.ndarray, dynamics: DynamicsConfig) -> ControlCommand:
    """Eq43：有限(3,)归一化动作映射到物理指令；拒绝越界，不静默裁剪。"""

    action = np.asarray(action, dtype=np.float64)
    if action.shape != (3,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1):
        raise ValueError("名义动作必须有限shape=(3,)且位于[-1,1]。")
    span = dynamics.max_surge_speed_mps - dynamics.min_surge_speed_mps
    return ControlCommand(float(dynamics.min_surge_speed_mps + span * (action[0]+1)/2),
                          float(action[1] * dynamics.max_yaw_rate_rad_s),
                          float(action[2] * dynamics.max_pitch_rate_rad_s))


def command_to_normalized(command: ControlCommand, dynamics: DynamicsConfig) -> np.ndarray:
    """Eq43精确逆仿射映射；输出(3,)float64，用于执行动作日志和reward。"""

    span = dynamics.max_surge_speed_mps - dynamics.min_surge_speed_mps
    return np.array([
        2*(command.surge_speed_command_mps-dynamics.min_surge_speed_mps)/span-1,
        command.yaw_rate_command_rad_s/dynamics.max_yaw_rate_rad_s,
        command.pitch_rate_command_rad_s/dynamics.max_pitch_rate_rad_s,
    ], dtype=np.float64)


def task_reward(distance_before_m: float, distance_after_m: float, goal_reached: bool,
                elapsed_s: float, control_dt_s: float, previous_executed: np.ndarray,
                executed: np.ndarray, task: LocalTaskConfig) -> tuple[float, dict[str, float]]:
    """Eq44四项：进展/1m、到达、实际时长比例、归一化执行动作差平方范数。"""

    difference = executed - previous_executed
    components = {
        'progress': task.w_progress * (distance_before_m-distance_after_m),
        'goal': task.w_goal * int(goal_reached),
        'time': -task.w_time * elapsed_s/control_dt_s,
        'smoothness': -task.w_smooth * float(difference @ difference),
    }
    return float(sum(components.values())), components


def instantaneous_cost(nominal_rejected: bool, event_reason: str,
                       collision_seen: bool = False) -> dict[str, int]:
    """Eq45即时指标；首次碰撞单独记录，训练成本是拒绝与真实安全失败的max。"""

    collision = event_reason == 'collision'
    failure = collision or event_reason == 'boundary'
    risk = int(nominal_rejected)
    return {'c_risk': risk, 'c_real': int(collision and not collision_seen),
            'f_t': int(failure), 'c_train': max(risk, int(failure))}


def termination_flags(event_reason: str, external_cutoff: bool) -> tuple[bool, bool, str]:
    """保留世界最早真实事件，独立报告外部截断；回退不是终止事件。"""

    names = {'none': 'none', 'success': 'goal_success', 'collision': 'collision',
             'boundary': 'operational_boundary_failure', 'timeout': 'task_horizon'}
    if event_reason not in names:
        raise ValueError(f"未知世界事件：{event_reason}")
    terminated = event_reason != 'none'
    failure_type = names[event_reason]
    if external_cutoff and not terminated:
        failure_type = 'external_truncation'
    return terminated, bool(external_cutoff), failure_type
