"""LOCAL表13-1/Eq42唯一234维编码；仅接收自身导航、合法感知和KF后验。"""

import numpy as np

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.frames import rotation_body_to_ned
from auv_risk_rl.prediction.predictor import predict_position_distribution
from auv_risk_rl.sensors.rays import RayFrame
from auv_risk_rl.types import AUVState, TrackedObstacle


class ObservationBuilder:
    """输出(234,)float32；NED后验保持float64不回写，不接收世界或未来真值。"""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def _slot(self, target: TrackedObstacle, state: AUVState,
              rotation: np.ndarray, timestamp_s: float) -> np.ndarray:
        """编码21维：当前/五秒均值，一秒/五秒协方差，半径/年龄/mask。"""

        track = target.track_state
        if not np.isclose(track.state_timestamp_s, timestamp_s, rtol=0, atol=1e-10):
            raise ValueError("Observation要求当前时刻的KF后验。")
        slot = np.zeros(21, dtype=np.float64)
        slot[:3] = rotation @ (track.state_mean_ned[:3] - state.position_ned_m)
        spectral_density = np.diag(self.config.kf.acceleration_spectral_density_m2_s3)
        for horizon_s, offset in ((1.0, 6), (5.0, 12)):
            predicted = predict_position_distribution(
                track, horizon_s, spectral_density,
                self.config.kf.covariance_symmetry_tolerance,
                self.config.kf.covariance_psd_tolerance,
            )
            covariance_body_m2 = rotation @ predicted.position_covariance_ned_m2 @ rotation.T
            slot[offset:offset+6] = np.arcsinh(covariance_body_m2[np.triu_indices(3)])
            if horizon_s == 5.0:
                slot[3:6] = rotation @ (predicted.position_mean_ned_m - state.position_ned_m)
        slot[:6] /= self.config.sensor.range_m
        age_s = timestamp_s - track.last_measurement_timestamp_s
        if age_s < 0:
            raise ValueError("观测年龄不能为负，禁止未来测量。")
        slot[18:] = [target.radius_m, np.arcsinh(age_s), 1.0]
        return slot

    def build(self, state: AUVState, goal_position_ned_m: np.ndarray,
              previous_executed_normalized: np.ndarray, remaining_control_steps: int,
              rays: RayFrame, tracks: tuple[TrackedObstacle, ...],
              timestamp_s: float) -> np.ndarray:
        """按表13-1构造新数组；mask外padding全零，距离排序稳定ID打破并列。"""

        if rays.ranges_m.shape != (45,) or rays.valid_masks.shape != (45,):
            raise ValueError("射线距离和mask必须45维。")
        if not np.all(np.isin(rays.valid_masks, [0, 1])):
            raise ValueError("射线mask必须0/1。")
        if previous_executed_normalized.shape != (3,):
            raise ValueError("上一执行动作必须3维。")
        horizon = self.config.environment.max_episode_control_steps
        if not 0 <= remaining_control_steps <= horizon:
            raise ValueError("剩余任务步数越界。")
        rotation = rotation_body_to_ned(state.yaw_rad, state.pitch_rad).T
        output = np.zeros(234, dtype=np.float64)
        output[:3] = rotation @ (goal_position_ned_m - state.position_ned_m) / 100.0
        lower = np.asarray(self.config.environment.position_lower_bound_ned_m)
        upper = np.asarray(self.config.environment.position_upper_bound_ned_m)
        output[3:6] = 2 * (state.position_ned_m - lower) / (upper - lower) - 1
        dynamics = self.config.dynamics
        output[6] = (2 * (state.surge_speed_mps - dynamics.min_surge_speed_mps)
                     / (dynamics.max_surge_speed_mps - dynamics.min_surge_speed_mps) - 1)
        output[7:9] = [state.yaw_rate_rad_s / dynamics.max_yaw_rate_rad_s,
                       state.pitch_rate_rad_s / dynamics.max_pitch_rate_rad_s]
        output[9:13] = [np.sin(state.yaw_rad), np.cos(state.yaw_rad),
                        np.sin(state.pitch_rad), np.cos(state.pitch_rad)]
        output[13:16] = previous_executed_normalized
        output[16] = remaining_control_steps / horizon
        output[17] = self.config.risk.short_horizon_risk_budget / 0.2
        output[18:63] = rays.ranges_m / self.config.sensor.range_m
        output[63:108] = rays.valid_masks
        ordered = sorted(tracks, key=lambda t: (
            float(np.linalg.norm(t.track_state.state_mean_ned[:3] - state.position_ned_m)),
            t.track_state.obstacle_id,
        ))
        for index, target in enumerate(ordered[:6]):
            output[108+21*index:129+21*index] = self._slot(target, state, rotation, timestamp_s)
        result = output.astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("Observation包含非有限数据。")
        return result
