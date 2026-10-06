"""LOCAL普通SAC冻结参数与显式工程设备选择。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SACConfig:
    """附录B默认值；小容量/起步数仅供工程测试显式覆盖。"""

    gamma: float = 0.999
    tau: float = 0.005
    learning_rate: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 500_000
    learning_starts: int = 10_000
    utd: int = 1
    initial_alpha: float = 0.2
    target_entropy: float = -3.0
    device: str = "cpu"
    initialization_seed: int = 0
    actor_seed: int = 1
    replay_seed: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma/tau必须在(0,1]。")
        if self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("Replay必须容纳完整batch。")
        if self.learning_starts < 0 or self.utd != 1:
            raise ValueError("起步数非负，LOCAL冻结UTD=1。")
        if not 0 < self.learning_rate < 1 or not 0 < self.initial_alpha < float('inf'):
            raise ValueError("学习率与初始温度必须有限且为正。")
        if self.target_entropy != -3:
            raise ValueError("LOCAL三维动作目标熵冻结为-3。")
