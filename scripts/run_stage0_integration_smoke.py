"""
Stage 0 感知—KF—validator—env 一控制周期冒烟运行。

用途：
让用户在本地 pytest 之外，直接看到真实代码路径完成一次无 RL 控制周期。
该脚本只展示接口连通性，不产生论文导航性能结论。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from auv_risk_rl.config import load_project_config
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.runtime.stage0_cycle import Stage0ControlCycleRunner
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.types import AUVState, ControlCommand, GroundTruthObstacleState


def main() -> None:
    """
    构造一个可审计的简单场景并运行一次完整 Stage 0 控制周期。

    对应技术协议：
        第 15 章无 RL 子集；不对应新的编号公式。

    输入：
        无；全部参数来自 stage0.yaml 和脚本内明确的诊断场景初值。

    输出：
        终端 JSON 摘要，包括验证决定、时间、track 数量与真实事件类型。

    shape/单位/坐标系：
        场景位置为 shape=(3,) m，NED；速度 m/s；时间 s。

    关键假设：
        该脚本只是 Stage 0 冒烟测试，不是随机场景统计实验。

    重要限制：
        输出不能解释为成功率、碰撞率或实时性性能。
    """

    repository_root = Path(__file__).resolve().parents[1]
    config = load_project_config(repository_root / "configs" / "stage0.yaml")
    auv_state = AUVState(
        position_ned_m=np.array([20.0, 20.0, 20.0], dtype=np.float64),
        yaw_rad=0.0,
        pitch_rad=0.0,
        surge_speed_mps=0.8,
        yaw_rate_rad_s=0.0,
        pitch_rate_rad_s=0.0,
    )
    obstacle_state = GroundTruthObstacleState(
        obstacle_id=1,
        position_ned_m=np.array([40.0, 20.0, 20.0], dtype=np.float64),
        velocity_ned_mps=np.array([0.2, 0.0, 0.0], dtype=np.float64),
        radius_m=0.5,
    )
    world = AUVWorld(
        config=config,
        initial_auv_state=auv_state,
        initial_obstacle_states=(obstacle_state,),
        goal_position_ned_m=np.array([90.0, 80.0, 20.0], dtype=np.float64),
    )
    runner = Stage0ControlCycleRunner(
        config=config,
        world=world,
        sensor_rng=SeedManager(root_seed=20260917).get_rng("sensor"),
        obstacle_radius_by_id_m={1: obstacle_state.radius_m},
    )
    initial_perception = runner.initialize_perception()
    nominal_action = ControlCommand(0.8, 0.0, 0.0)
    result = runner.run_control_cycle(
        nominal_action=nominal_action,
        previous_executed_action=nominal_action,
    )
    summary = {
        "initial_track_count": len(initial_perception.track_states),
        "decision_type": result.validation_decision.decision_type,
        "executed_action": {
            "surge_speed_command_mps": (
                result.validation_decision.executed_action.surge_speed_command_mps
            ),
            "yaw_rate_command_rad_s": (
                result.validation_decision.executed_action.yaw_rate_command_rad_s
            ),
            "pitch_rate_command_rad_s": (
                result.validation_decision.executed_action.pitch_rate_command_rad_s
            ),
        },
        "world_timestamp_s": result.world_diagnostics.timestamp_s,
        "event_reason": result.world_diagnostics.event.reason,
        "policy_track_count": len(result.perception_frame.track_states),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
