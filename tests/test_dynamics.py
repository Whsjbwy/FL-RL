"""验证 AUV Eq. (6)–(9) 的方向符号与 RK2 数值行为。"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.dynamics.auv_kinematics import integrate_rk2_step, rollout_constant_command
from auv_risk_rl.types import AUVState, ControlCommand


def test_positive_pitch_moves_auv_up_in_ned_convention(project_config) -> None:
    """
    验证正 pitch 前进时 Depth 减小。

    该测试与旋转矩阵符号独立地穿过完整动力学路径，防止 Eq. (6) 集成接口反向。
    """

    initial_state = AUVState(
        position_ned_m=np.array([50.0, 50.0, 20.0], dtype=np.float64),
        yaw_rad=0.0,
        pitch_rad=np.deg2rad(15.0),
        surge_speed_mps=1.0,
        yaw_rate_rad_s=0.0,
        pitch_rate_rad_s=0.0,
    )
    command = ControlCommand(1.0, 0.0, 0.0)
    next_state = integrate_rk2_step(
        initial_state,
        command,
        np.zeros(3, dtype=np.float64),
        project_config.dynamics.integration_dt_s,
        project_config.dynamics,
    )
    assert next_state.position_ned_m[2] < initial_state.position_ned_m[2], (
        "expected Depth to decrease for positive pitch, "
        f"initial={initial_state.position_ned_m[2]}, actual={next_state.position_ned_m[2]}"
    )


def test_rk2_step_halving_reduces_smooth_trajectory_error(project_config) -> None:
    """
    验证光滑无饱和场景中 RK2 减半步长显著降低误差。

    这里比较相同 1 秒固定指令轨迹与更细参考步长，不在饱和切换点宣称严格二阶收敛。
    """

    initial_state = AUVState(
        position_ned_m=np.array([40.0, 40.0, 20.0], dtype=np.float64),
        yaw_rad=0.2,
        pitch_rad=0.1,
        surge_speed_mps=0.8,
        yaw_rate_rad_s=0.05,
        pitch_rate_rad_s=-0.03,
    )
    command = ControlCommand(0.9, 0.06, -0.02)
    zero_current = np.zeros(3, dtype=np.float64)
    horizon_s = 1.0

    coarse = rollout_constant_command(
        initial_state,
        command,
        zero_current,
        horizon_s,
        0.05,
        project_config.dynamics,
    )[-1]
    medium = rollout_constant_command(
        initial_state,
        command,
        zero_current,
        horizon_s,
        0.025,
        project_config.dynamics,
    )[-1]
    reference = rollout_constant_command(
        initial_state,
        command,
        zero_current,
        horizon_s,
        0.00625,
        project_config.dynamics,
    )[-1]

    coarse_error = float(np.linalg.norm(coarse.position_ned_m - reference.position_ned_m))
    medium_error = float(np.linalg.norm(medium.position_ned_m - reference.position_ned_m))
    assert medium_error < coarse_error, (
        f"expected smaller error after step halving, coarse={coarse_error}, medium={medium_error}"
    )
