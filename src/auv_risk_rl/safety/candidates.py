"""
在线验证器有限候选动作库。

说明：
候选离散化是协议定义的工程启发式机制，不是连续动作空间最优性证明。
主网格由 3 个速度、5 个 yaw-rate、3 个 pitch-rate 组成，共 45 个固定候选，
并额外加入名义动作和上一执行动作后去重。
"""

from __future__ import annotations

from auv_risk_rl.config import ValidatorConfig
from auv_risk_rl.types import ControlCommand


def build_candidate_actions(
    nominal_action: ControlCommand,
    previous_executed_action: ControlCommand,
    validator_config: ValidatorConfig,
) -> list[ControlCommand]:
    """
    构造并去重有限候选动作集合。

    对应技术协议：
        Eq. (55)

    参数：
        nominal_action:
            名义物理动作，单位 m/s、rad/s、rad/s。
        previous_executed_action:
            上一实际执行动作，单位同上。
        validator_config:
            固定候选网格配置。

    返回：
        candidates:
            ControlCommand 列表；顺序固定为名义、上一动作、随后固定网格顺序。

    shape/坐标系：
        每个动作等价于 3 维指令向量；属于导航指令空间。

    关键假设：
        网格范围与 AUV 动作定义域一致。

    重要限制：
        该模块属于协议定义的工程启发式机制，不是严格理论推导；
        候选不通过不能推出连续动作空间无解。
    """

    candidates: list[ControlCommand] = []
    seen_keys: set[tuple[float, float, float]] = set()

    def append_unique(command: ControlCommand) -> None:
        """按物理动作三元组去重，并保持首次出现对应的确定性候选编号。"""

        # 候选值直接来自冻结配置或上游动作，因此使用精确 tuple 去重可保持确定性候选编号。
        key = (
            float(command.surge_speed_command_mps),
            float(command.yaw_rate_command_rad_s),
            float(command.pitch_rate_command_rad_s),
        )
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(command)

    append_unique(nominal_action)
    append_unique(previous_executed_action)
    for surge_speed_mps in validator_config.speed_grid_mps:
        for yaw_rate_rad_s in validator_config.yaw_rate_grid_rad_s:
            for pitch_rate_rad_s in validator_config.pitch_rate_grid_rad_s:
                append_unique(
                    ControlCommand(
                        surge_speed_command_mps=surge_speed_mps,
                        yaw_rate_command_rad_s=yaw_rate_rad_s,
                        pitch_rate_command_rad_s=pitch_rate_rad_s,
                    )
                )
    return candidates
