"""LOCAL合法感知历史编排；冻结KF/延迟模块保持原样，不向估计器传递真速度。"""

import numpy as np

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.sensors.delay_queue import DetectionDelayQueue
from auv_risk_rl.sensors.occlusion import capture_center_detections
from auv_risk_rl.sensors.rays import RayFrame, cast_sonar_rays
from auv_risk_rl.tracking.pose_history import AUVPoseHistory
from auv_risk_rl.tracking.track_manager import MultiTargetCVTracker
from auv_risk_rl.types import ControlCommand, SensorDetection, TrackedObstacle


class PerceptionSession:
    """连续传感时钟与合法检测历史；半径来自预登记几何，tracks全部维护。"""

    def __init__(self, config: ProjectConfig, seeds: SeedManager,
                 radius_by_id_m: dict[int, float]) -> None:
        self.config = config
        self.radius_by_id_m = dict(radius_by_id_m)
        self.noise_rng = seeds.get_rng('sensor_noise')
        self.dropout_rng = seeds.get_rng('dropout')
        self.queue = DetectionDelayQueue(control_dt_s=config.dynamics.control_dt_s)
        self.poses = AUVPoseHistory()
        self.tracker = MultiTargetCVTracker(config)
        self.generated: list[SensorDetection] = []
        self.arrived: list[SensorDetection] = []
        self.capture_times_s: list[float] = []
        self.last_tick = -1

    def capture(self, world: AUVWorld, tick: int) -> RayFrame:
        """在唯一整数采样tick生成中心检测，释放已到达测量；射线独立测物理表面。"""

        if tick != self.last_tick + 1:
            raise ValueError("感知tick必须连续且不能重复。")
        timestamp_s = tick * self.config.dynamics.control_dt_s
        if not np.isclose(world.timestamp_s, timestamp_s, rtol=0, atol=1e-10):
            raise ValueError("世界时刻与传感整数tick不一致。")
        self.poses.add(timestamp_s, world.auv_state)
        detections = capture_center_detections(
            world.auv_state, world.obstacle_states, timestamp_s, tick, self.config,
            self.noise_rng, self.dropout_rng,
        )
        self.queue.enqueue_many(detections)
        arrived = self.queue.pop_arrived(current_control_tick=tick)
        self.tracker.add_arrived_detections(arrived, self.poses)
        self.generated.extend(detections)
        self.arrived.extend(arrived)
        self.capture_times_s.append(timestamp_s)
        self.last_tick = tick
        return cast_sonar_rays(world.auv_state, world.obstacle_states, self.config.sensor)

    def tracks(self, timestamp_s: float) -> tuple[TrackedObstacle, ...]:
        """返回所有已维护后验与登记半径；缺测只预测，绝不裁剪到六个。"""

        return tuple(TrackedObstacle(t, self.radius_by_id_m[t.obstacle_id])
                     for t in self.tracker.get_track_states(timestamp_s))


def legal_warmup(world: AUVWorld, session: PerceptionSession,
                 config: ProjectConfig) -> tuple[RayFrame, ControlCommand]:
    """SIMULATION IMPLEMENTATION DETAIL：物理[0,1]秒六端点采样，固定最低前进指令。"""

    duration_s = 1.0
    intervals = round(duration_s / config.dynamics.control_dt_s)
    if not np.isclose(intervals * config.dynamics.control_dt_s, duration_s):
        raise ValueError("一秒历史必须由完整控制周期覆盖。")
    if world.timestamp_s != 0 or world.control_step_index != 0:
        raise ValueError("warm-up必须从物理时刻0的全新世界开始。")
    command = ControlCommand(config.dynamics.min_surge_speed_mps, 0.0, 0.0)
    rays = session.capture(world, 0)
    for tick in range(1, intervals + 1):
        result = world.step(command)
        if result.is_terminated:
            raise ValueError(f"warm-up场景发生真实事件：{result.event.reason}")
        rays = session.capture(world, tick)
    return rays, command
